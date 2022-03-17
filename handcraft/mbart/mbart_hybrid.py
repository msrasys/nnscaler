"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    handcraft/mbart/mbart.py --pp-size 4 --tp-size 1 --nmb 4
"""

from typing import Optional
import argparse
import math
import torch

import cube
from cube.runtime.device import DeviceGroup
from cube.runtime.syndata import SynTextDataLoader
from cube.profiler import CudaTimer
from cube.profiler.memory import memory_summary, model_summary
from cube.profiler.timer import print_each_rank

from handcraft.mbart.schedule import schedule_naive, schedule_1f1b, schedule_tp_1f1b_pack
from handcraft.mbart.tp import AllGatherScatter, AllReduceIdentity, BroadcastReduce, IdentityAllreduce, ReduceBroadcast

_tp_group = -1
_pp_group = -1
_pp_embed_group = -1
_pp_next_rank = None
_pp_prev_rank = None

# fairseq task
# translation_from_pretrained_bart

# fairseq criterion
# label_smoothed_cross_entropy, --label_smoothing = 0.2

class Config:

    num_embeddings = 250027

    encoder_embed_path = None
    encoder_embed_dim = 1024
    encoder_ffn_embed_dim = 4 * 1024
    encoder_layers = 12
    encoder_attention_heads = 16
    encoder_normalize_before = True
    encoder_learned_pos = True

    decoder_embed_path = None
    decoder_embed_dim = 1024
    decoder_ffn_embed_dim = 4 * 1024
    decoder_layers = 12
    decoder_attention_heads = 16
    decoder_normalize_before = True
    decoder_learned_pos = True
    cross_self_attention = False
    no_cross_attention = False

    attention_dropout = 0.0
    activation_dropout = 0.0
    dropout = 0.1

    max_target_positions = 1024
    max_source_positions = 1024
    adaptive_softmax_cutoff = None
    adaptive_softmax_dropout = 0

    share_decoder_input_output_embed = True
    share_all_embeddings = True

    decoder_output_dim = 1024  # same with decorder_embed_dim
    decoder_input_dim = 1024   # same with decorder_embed_dim

    no_scale_embedding = False # True in bart large
    layernorm_embedding = True
    activation_fn = 'gelu'
    pooler_activation_fn = 'tanh'
    pooler_dropout = 0.0

    # classification task
    num_classes = 2


def attn_fn(query: torch.Tensor, key: torch.Tensor,
            wq: torch.Tensor, wq_bias: Optional[torch.Tensor],
            wk: torch.Tensor, wk_bias: Optional[torch.Tensor],
            wv: torch.Tensor, wv_bias: Optional[torch.Tensor],
            wout: torch.Tensor, wout_bias: Optional[torch.Tensor],
            h: int, scale: float, dropout: float, mask=True):
    """
    query, key: (L, N, E) = (seqlen, batch size, embed_dim)
    wq, wk, wv weight: [(num_head * dim_head), E]
    dropout: float
    h: int: number of heads
    """
    num_head = h
    L, N = query.size(0), query.size(1)
    dim_head = wq.size(0) // num_head

    q = torch.nn.functional.linear(query, wq, wq_bias) # L N E, (h d) E -> L N (h d)
    k = torch.nn.functional.linear(key, wk, wk_bias)   # L N E, (h d) E -> L N (h d)
    v = torch.nn.functional.linear(key, wv, wv_bias)   # L N E, (h d) E -> L N (h d)
    q = q.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    k = k.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    v = v.contiguous().view(L, (N * num_head), dim_head) # L N (h d) -> L (N h) d
    q = q.transpose(0, 1)  # L (N h) d -> (N h) L d
    k = k.transpose(0, 1)  # L (N h) d -> (N h) L d
    v = v.transpose(0, 1)  # L (N h) d -> (N h) L d
    q = q * scale          # (N h) L d, 1 -> (N h) L d
    k = k.transpose(1, 2)  # (N h) L d -> (N h) d L
    attn = torch.bmm(q, k) # (N h) L d, (N h) d L -> (N h) L L

    # attention mask
    if mask: # (N h) L L -> (N h) L L
        attn = attn.view(N, num_head, L, L)
        ones = torch.ones((N, L, L), device=attn.device)
        mask = torch.tril(ones)
        mask = mask.view(N, 1, L, L)
        mask = (mask < 0.5)
        attn = attn.masked_fill_(mask, -10000.0)
        attn = attn.view((N * num_head), L, L)

    attn = torch.nn.functional.softmax(attn, dim=-1) # (N h) L L -> (N h) L L
    attn = torch.nn.functional.dropout(attn, dropout, True, False) # (N h) L L -> (N h) L L
    output = torch.bmm(attn, v) # (N h) L L, (N h) L d -> (N h) L d
    output = output.transpose(0, 1).contiguous()     # (N h) L d -> L (N h) d
    output = output.view(L, N, num_head * dim_head)  # (N h) L d -> L N (h d)
    output = torch.nn.functional.linear(output, wout, wout_bias) # L N (h d), E E  -> L N E
    return output


class MultiheadAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, dropout=0.0, bias=True):
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)

        self.qdim = embed_dim
        self.kdim = embed_dim
        self.vdim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.num_heads = num_heads
        self.dropout_p = dropout
        # K
        self.k_proj = torch.nn.Parameter(torch.empty(embed_dim // self.tp_size, self.kdim))
        if bias:
            self.k_bias = torch.nn.Parameter(torch.empty(embed_dim // self.tp_size))
        else:
            self.k_bias = None
        # V
        self.v_proj = torch.nn.Parameter(torch.empty(embed_dim // self.tp_size, self.vdim))
        if bias:
            self.v_bias = torch.nn.Parameter(torch.empty(embed_dim // self.tp_size))
        else:
            self.v_bias = None
        # Q
        self.q_proj = torch.nn.Parameter(torch.empty(embed_dim // self.tp_size, self.qdim))
        if bias:
            self.q_bias = torch.nn.Parameter(torch.empty(embed_dim // self.tp_size))
        else:
            self.q_bias = None
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, embed_dim // self.tp_size))
        if bias:
            self.out_bias = torch.nn.Parameter(torch.empty(embed_dim))
        else:
            self.out_bias = None

    def forward(self, query: torch.Tensor, key: torch.Tensor):
        if key is not query:
            key = IdentityAllreduce.apply(key, self.tp_group)
        query = IdentityAllreduce.apply(query, self.tp_group)
        attn = attn_fn(query, key, 
               self.q_proj, self.q_bias,
               self.k_proj, self.k_bias,
               self.v_proj, self.v_bias,
               self.out_proj, self.out_bias,
               self.num_heads, self.scaling, self.dropout_p)
        attn = AllReduceIdentity.apply(attn, self.tp_group)
        return attn


class EncoderLayer(torch.nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)

        self.cfg = cfg
        self.self_attn = MultiheadAttention(cfg.encoder_embed_dim, cfg.encoder_attention_heads, cfg.attention_dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(cfg.encoder_embed_dim)
        self.dropout = torch.nn.Dropout(p=cfg.dropout)
        self.activation_dropout = torch.nn.Dropout(p=cfg.activation_dropout)
        self.fc1 = torch.nn.Linear(cfg.encoder_embed_dim, cfg.encoder_ffn_embed_dim // self.tp_size)
        self.fc2 = torch.nn.Linear(cfg.encoder_ffn_embed_dim // self.tp_size, cfg.encoder_embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.encoder_embed_dim)

    def input_shape(self):
        # L, N, E
        return (self.cfg.max_source_positions, 1, self.cfg.encoder_embed_dim)

    def output_shape(self):
        # L N E
        return (self.cfg.max_source_positions, 1, self.cfg.encoder_embed_dim)

    def input_dtype(self):
        return torch.float32

    def output_dtype(self):
        return torch.float32

    def forward(self, x):  # , encoder_padding_mask: Optional[torch.Tensor], attn_mask: Optional[torch.Tensor] = None):
        # print(f'encoder layer: x: {x.size()}')
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, x)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        if self.tp_size > 1:
            x = IdentityAllreduce.apply(x, self.tp_group)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.activation_dropout(x)
        x = self.fc2(x)
        if self.tp_size > 1:
            x = AllReduceIdentity.apply(x, self.tp_group)

        x = self.dropout(x)
        x = x + residual
        return x


class DecoderLayer(torch.nn.Module):

    def __init__(self, cfg: Config):

        super().__init__()

        self.tp_group = _tp_group
        self.tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)

        self.cfg = cfg
        self.dropout = torch.nn.Dropout(p=cfg.dropout)
        self.self_attn = MultiheadAttention(cfg.decoder_embed_dim, cfg.decoder_attention_heads, cfg.attention_dropout)
        self.activation_dropout = torch.nn.Dropout(p=cfg.activation_dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(cfg.decoder_embed_dim)
        # encoder atten
        self.encoder_attn = MultiheadAttention(cfg.encoder_embed_dim, cfg.decoder_attention_heads, cfg.attention_dropout)
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(cfg.decoder_embed_dim)

        self.fc1 = torch.nn.Linear(cfg.decoder_embed_dim, cfg.decoder_ffn_embed_dim // self.tp_size)
        self.fc2 = torch.nn.Linear(cfg.decoder_ffn_embed_dim // self.tp_size, cfg.decoder_embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.decoder_embed_dim)

    def input_shape(self):
        return (
            (self.cfg.max_target_positions, 1, self.cfg.decoder_embed_dim),
            (self.cfg.max_source_positions, 1, self.cfg.encoder_embed_dim)
        )

    def output_shape(self):
        return (
            (self.cfg.max_target_positions, 1, self.cfg.decoder_embed_dim),
            (self.cfg.max_source_positions, 1, self.cfg.encoder_embed_dim)
        )

    def input_dtype(self):
        return (torch.float32, torch.float32)

    def output_dtype(self):
        return (torch.float32, torch.float32)

    def forward(self, x, encoder_out):  # encoder_padding_mask):
        # print(f'decoder layer: x: {x.size()}, encoder_out: {encoder_out.size()}')
        residual = x
        # normalize before
        x = self.self_attn_layer_norm(x)

        # self attention
        x = self.self_attn(x, x)
        x = self.dropout(x)
        x = residual + x
    
        # encoder attn
        residual = x
        # normalize before
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(x, encoder_out)
        x = self.dropout(x)
        x = x + residual

        residual = x
        # normalize before
        x = self.final_layer_norm(x)
        if self.tp_size > 1:
            x = IdentityAllreduce.apply(x, self.tp_group)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.activation_dropout(x)
        x = self.fc2(x)
        if self.tp_size > 1:
            x = AllReduceIdentity.apply(x, self.tp_group)
        x = self.dropout(x)
        x = x + residual
        return x, encoder_out


class MBartClassificationHead(torch.nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)
        
        self.num_classes = num_classes
        self.dense = torch.nn.Linear(input_dim, inner_dim // self.tp_size)
        self.dropout = torch.nn.Dropout(p=pooler_dropout)
        self.out_proj = torch.nn.Linear(inner_dim // self.tp_size, num_classes)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, dec: torch.Tensor, labels):
        # sentence_represent = dec[eos_mask,:].view(dec.size(0), -1, hidden_states.size(-1))[:,-1,:]
        dec = dec.transpose(0, 1)[:,-1,:]
        sentence_represent = dec
        hidden_states = self.dropout(sentence_represent)
        if self.tp_size > 1:
            hidden_states = IdentityAllreduce.apply(hidden_states, self.tp_group)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.out_proj(hidden_states)
        if self.tp_size > 1:
            logits = AllReduceIdentity.apply(logits, self.tp_group)
        loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        return (loss,)


class ShardHeadTail(torch.nn.Module):

    def __init__(self, cfg: Config):
        """
        group = -1 means no tensor parallelism
        """
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)
        self.tp_idx = 0 if _tp_group == -1 else torch.distributed.get_rank(_tp_group)

        self.cfg = cfg
        if self.tp_size > 0:
            print(f'[{torch.distributed.get_rank()}]: initialize sharding embed (x{self.tp_size})')
        
        self.vocab_start_index = self.cfg.num_embeddings // self.tp_size * self.tp_idx
        self.vocab_end_index = self.cfg.num_embeddings // self.tp_size * (self.tp_idx + 1)
        self.weight = torch.nn.Parameter(torch.ones((self.cfg.num_embeddings // self.tp_size, self.cfg.encoder_embed_dim)))

        # encoder-preprocess
        self.embed_positions_encoder = torch.nn.Embedding(cfg.max_source_positions, cfg.encoder_embed_dim)
        self.embed_scale_encoder = math.sqrt(cfg.encoder_embed_dim)
        self.layernorm_embedding_encoder = torch.nn.LayerNorm(cfg.encoder_embed_dim)

        # decoder-preprocess
        self.embed_scale_decoder = math.sqrt(cfg.decoder_embed_dim)
        self.embed_positions_decoder = torch.nn.Embedding(cfg.max_source_positions, cfg.decoder_embed_dim)
        self.layernorm_embedding_decoder = torch.nn.LayerNorm(cfg.decoder_embed_dim)

        self._inputs = (None, )

    def set_inputs(self, *inputs):
        self._inputs = inputs

    def embed_lookup(self, tokens):
        if self.tp_size > 1:
            mask = (tokens < self.vocab_start_index) | \
                        (tokens >= self.vocab_end_index)
            tokens = tokens.clone() - self.vocab_start_index
            tokens[mask] = 0
            embed = torch.nn.functional.embedding(tokens, self.weight)
            embed[mask, :] = 0.0
            embed = AllReduceIdentity.apply(embed, self.tp_group)
        else:
            embed = torch.nn.functional.embedding(tokens, self.weight)
        return embed

    def encoder_preprocess(self):
        source_tokens = self._inputs[0]
        source_embed = self.embed_lookup(source_tokens)
        embed = self.embed_scale_encoder * source_embed
        x = embed + self.embed_positions_encoder.weight
        x = self.layernorm_embedding_encoder(x)
        x = torch.nn.functional.dropout(x, p=0.0)
        enc = x.transpose(0, 1)
        return (enc,)

    def decoder_preprocess(self):
        prev_output_tokens = self._inputs[0]
        target_emb = self.embed_lookup(prev_output_tokens)
        embed = self.embed_scale_decoder * target_emb
        embed = embed + self.embed_positions_decoder.weight
        embed = self.layernorm_embedding_decoder(embed)
        embed = torch.nn.functional.dropout(embed, p=0.0)
        dec = embed.transpose(0, 1)
        return (dec,)


class mBARTFull(torch.nn.Module):

    def __init__(self, cfg: Config,
                 encoder_preprocess=True,
                 decoder_preprocess=True,
                 post_process=True):
        super().__init__()
        self.cfg = cfg
        self.dummy_labels = torch.tensor([1]).cuda()
        self._preprocess = [None, None] # enc, dec

        self.rank = DeviceGroup().rank

        global _pp_group
        self.pp_group = _pp_group
        self.total_layers = cfg.encoder_layers + cfg.decoder_layers

        self.pp_stage = torch.distributed.get_rank(_pp_group)
        self.num_stages = torch.distributed.get_world_size(_pp_group)

        self.layer_start = self.total_layers // self.num_stages * self.pp_stage
        self.layer_end = self.total_layers // self.num_stages * (self.pp_stage + 1)

        self.encoder_preprocess  = encoder_preprocess
        self.encoder_forward     = (self.layer_start < cfg.encoder_layers)

        self.decoder_preprocess  = decoder_preprocess
        self.decoder_first_stage = self.layer_start == cfg.encoder_layers
        self.decoder_forward     = (self.layer_end > cfg.encoder_layers)
        self.decoder_last_stage  = (self.layer_end == cfg.encoder_layers + cfg.decoder_layers)

        self.postprocess         = post_process

        self.encoder_layer_start = self.layer_start
        self.encoder_layer_end = min(self.layer_end, cfg.encoder_layers)

        self.decoder_layer_start = max(cfg.encoder_layers, self.layer_start)
        self.decoder_layer_end = self.layer_end

        if encoder_preprocess or decoder_preprocess or post_process:
            self.headtail = ShardHeadTail(cfg)
        else:
            self.headtail = None

        # encoders
        if self.encoder_forward:
            print(f'[{self.rank}]: initializing {self.encoder_layer_end - self.encoder_layer_start} encoder layers')
            self.encoders = torch.nn.ModuleList([EncoderLayer(cfg) for _ in range(self.encoder_layer_end - self.encoder_layer_start)])
            if self.encoder_layer_end == cfg.encoder_layers:
                self.layer_norm_encoder = torch.nn.LayerNorm(cfg.encoder_embed_dim)
            else:
                self.layer_norm_encoder = None

        # decoders
        if self.decoder_forward:
            print(f'[{self.rank}]: initializing {self.decoder_layer_end - self.decoder_layer_start} decoder layers')
            self.decoders = torch.nn.ModuleList([DecoderLayer(cfg) for _ in range(self.decoder_layer_end - self.decoder_layer_start)])
            if self.decoder_layer_end == cfg.encoder_layers + cfg.decoder_layers:
                self.layer_norm_decoder = torch.nn.LayerNorm(cfg.decoder_embed_dim)
            else:
                self.layer_norm_decoder = None

        # postpross
        if self.postprocess:
            print(f'[{self.rank}]: will compute loss')
            self.head = MBartClassificationHead(cfg.decoder_embed_dim, 1024, cfg.num_classes, 0.0)

    def input_shape(self):
        if self.encoder_preprocess:
            return ()
        elif self.encoder_forward:
            return (
                (self.cfg.max_source_positions, 1, self.cfg.encoder_embed_dim),
            )
        elif self.decoder_preprocess:
            return (
                (self.cfg.max_source_positions, 1, self.cfg.encoder_embed_dim),
            )
        elif self.decoder_first_stage:
            return (
                (self.cfg.max_source_positions, 1, self.cfg.encoder_embed_dim),
            )
        elif self.decoder_forward:
            return (
                (self.cfg.max_source_positions, 1, self.cfg.encoder_embed_dim),
                (self.cfg.max_target_positions, 1, self.cfg.decoder_embed_dim),
            )
        elif self.postprocess:
            return ((1,),)
        assert False

    def output_shape(self):
        shape = None
        if self.encoder_preprocess or self.encoder_forward:
            shape = (self.cfg.max_source_positions, 1, self.cfg.encoder_embed_dim),
        # decoder preprocess is not allowed to be a single stage
        if self.decoder_preprocess or self.decoder_forward:
            shape = (
                (self.cfg.max_source_positions, 1, self.cfg.encoder_embed_dim),
                (self.cfg.max_target_positions, 1, self.cfg.decoder_embed_dim),
            )
        if self.postprocess:
            shape = (
                (1,),
            )
        assert shape is not None
        return shape

    def input_dtype(self):
        if self.encoder_preprocess:
            return ()
        elif self.encoder_forward:
            return (torch.float32,)
        elif self.decoder_preprocess:
            return (torch.float32,)
        elif self.decoder_forward:
            return (torch.float32, torch.float32)
        else:
            assert False

    def output_dtype(self):
        dtype = None
        if self.encoder_preprocess or self.encoder_forward:
            dtype = (torch.float32,)
        if self.decoder_preprocess or self.decoder_forward:
            if self.pp_stage == self.num_stages - 1:
                dtype = (torch.float32,)
            else:
                dtype = (torch.float32, torch.float32)
        if self.postprocess:
            dtype = ((torch.float32,),)
        assert dtype is not None
        return dtype

    def set_inputs(self, *inputs):
        assert len(inputs) == 1
        if self.headtail is not None:
            self.headtail.set_inputs(*inputs)

    def set_preprocess(self, enc=None, dec=None):
        if enc is not None:
            self._preprocess[0] = enc
        if dec is not None:
            self._preprocess[1] = dec

    def forward_encoder_preprocess(self):
        return self.headtail.encoder_preprocess()

    def forward_decoder_preprocess(self):
        return self.headtail.decoder_preprocess()

    def forward_postprocess(self, dec):
        return self.head(dec, self.dummy_labels)

    def forward(self, enc=None, dec=None):
        """
        enc: encoder input/output
        dec: decoder output/input
        """
        pre_enc, pre_dec = self._preprocess
        enc = pre_enc if enc is None else enc
        dec = pre_dec if dec is None else dec

        # encoder preprocess
        if self.encoder_preprocess:
            output = self.forward_encoder_preprocess()
            enc = output[0]

        # forward encoder
        if self.encoder_forward:
            for layer in self.encoders:
                enc = layer(enc) # encoder_padding_mask if has_pads else None)
            if self.layer_norm_encoder is not None:
                enc = self.layer_norm_encoder(enc)
            output = (enc,)

        # decoder preprocess
        if self.decoder_preprocess:
            output = self.forward_decoder_preprocess()
            dec = output[0]

        # forward decoder
        if self.decoder_forward:
            dec = pre_dec if dec is None else dec
            for layer in self.decoders:
                dec, enc = layer(dec, enc)
            if self.layer_norm_decoder is not None:
                dec = self.layer_norm_decoder(dec)
                output = (dec,)
            else:
                output = (enc, dec)

        # postprocess
        if self.postprocess:
            output = self.forward_postprocess(dec)
            loss = output[0]

        return output


def reduce_embed(model, pp_embed_group):
    """
    Embedding gradients needs to be reduced across pipeline stages
    """
    if isinstance(model.headtail, torch.nn.Module):
        grad = model.headtail.weight.grad
    else:
        grad = None
    if grad is not None:
        torch.distributed.all_reduce(grad, group=pp_embed_group)
    torch.cuda.synchronize()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='swin')
    parser.add_argument('--nmb', type=int, default=4,
                        help='num of micro batch')
    parser.add_argument('--pp-size', type=int, default=1,
                        help='use pipeline parallelism')
    parser.add_argument('--tp-size', type=int, default=1,
                    help='use tensor parallelism')
    args = parser.parse_args()

    print(args)

    cube.init()
    pp_ranks, tp_ranks = DeviceGroup().create_hybrid([args.pp_size, args.tp_size])
    print_each_rank(f'my pp ranks: {pp_ranks}')
    print_each_rank(f'my tp ranks: {tp_ranks}')

    if _tp_group == -1:
        _tp_group = DeviceGroup().get_group(tp_ranks)

    if _pp_group == -1:
        _pp_group = DeviceGroup().get_group(pp_ranks)
        idx = pp_ranks.index(DeviceGroup().rank)
        _pp_next_rank = pp_ranks[(idx+1) % len(pp_ranks)]
        _pp_prev_rank = pp_ranks[(idx-1) % len(pp_ranks)]
        is_first_stage = idx == 0
        is_first_decoder_stage = idx == len(pp_ranks) // 2
        is_last_stage = idx == len(pp_ranks) - 1

    if len(pp_ranks) > 1:
        pranks = [torch.zeros((args.pp_size,), dtype=torch.int, device=torch.cuda.current_device()) for _ in range(args.tp_size)]
        prank = torch.tensor(pp_ranks, dtype=torch.int).cuda()
        pranks[torch.distributed.get_rank(_pp_group)] = prank
        torch.distributed.all_gather(pranks, prank, group=_tp_group)
        torch.cuda.synchronize()
        print_each_rank(f'allgather-pp ranks: {pranks}')

        for prank in pranks:
            prank = prank.tolist()
            embed_ranks = [prank[0], prank[len(prank) // 2]]
            embed_ranks = list(set(embed_ranks))
            group = DeviceGroup().get_group(embed_ranks)
            if torch.distributed.get_rank(_tp_group) in prank:
                print(f'embedding group: {embed_ranks}')
                _pp_embed_group = group
        assert _pp_embed_group != -1

    cfg = Config()
    dataloader = SynTextDataLoader(
        shapes=(
            [1, cfg.max_source_positions],
        ),
        dtypes=(torch.int64,),
        batch_dims=(0,)
    )
    dataloader = iter(dataloader)


    if args.pp_size > 1:
        encoder_preprocess = is_first_stage
        decoder_preprocess = is_first_decoder_stage
        postprocess = is_last_stage
        model = mBARTFull(cfg, encoder_preprocess, decoder_preprocess, postprocess).cuda()
    else:
        model = mBARTFull(cfg, True, True, True).cuda()

    print_each_rank('model weight consumpition:')
    memory_summary()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))

    CudaTimer(enable=False).warmup()
    iter_num = 32
    for step in range(iter_num):
        if step >= 10:
            CudaTimer(enable=True).start('e2e')
        if args.pp_size > 1:
            schedule_naive(model, dataloader, args.nmb, (_pp_prev_rank, _pp_next_rank))
            reduce_embed(model, _pp_embed_group)
        else:
            for _ in range(args.nmb):
                model.set_inputs(next(dataloader))
                loss = model()[0]
                loss.backward()
        if step == 0:
            print('passed 1st iteration')
            memory_summary()
        optimizer.step()
        optimizer.zero_grad()
        if step >= 10:
            CudaTimer().stop('e2e')
        if (step + 1) % 10 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-10, field_name='e2e')))
    memory_summary()
