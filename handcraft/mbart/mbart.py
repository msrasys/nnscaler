"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    handcraft/mbart/mbart.py \
        --layers 12 --hidden-size 1024 --heads 16 \
        --use-1f1b --nmb 4 --iter-nmb 4 \
        --use-recompute --use-swap
"""

from typing import Optional
import argparse
import math
import torch

import cube
from cube.runtime.device import DeviceGroup
from cube.runtime.syndata import SynTextDataLoader
from cube.profiler import CudaTimer
from cube.profiler.memory import memory_summary
from cube.profiler.timer import print_each_rank

from handcraft.mbart.schedule import schedule_1f1b, schedule_tp_1f1b_pack
from handcraft.mbart.swap import SwapEmbed, get_swap_parameters
from handcraft.mbart.tp import ReduceBroadcast


_pp_group = -1
_pp_embed_group = -1
_pp_next_rank = None
_pp_prev_rank = None
_layer_divisions = []

parser = argparse.ArgumentParser(description='swin')
# model arch
parser.add_argument('--layers', type=int, default=12,
                    help='number encoder/decoder of layers')
parser.add_argument('--hidden-size', type=int, default=1024,
                    help='hidden size')
parser.add_argument('--heads', type=int, default=16,
                    help='number of heads')
# training config
parser.add_argument('--nmb', type=int,
                    help='num of micro batch')
parser.add_argument('--iter-nmb', type=int, default=0,
                    help='num of micro batch per scheduling iteration (1f1b only)')
# parallelism
parser.add_argument('--use-1f1b', action='store_true',
                help='use 1f1b scheduling')
parser.add_argument('--use-tp1f1b-pack', action='store_true',
                help='use tensor parallel 1f1b')
parser.add_argument('--use-recompute', action='store_true',
                    help='use recompute for a stage')
parser.add_argument('--use-swap', action='store_true',
                    help='use embedding swap (1f1b only)')
args = parser.parse_args()
print(args)

cube.init()
pp_ranks = list(range(DeviceGroup().world_size))
print_each_rank(f'my pp ranks: {pp_ranks}')

if _pp_group == -1:
    _pp_group = DeviceGroup().get_group(pp_ranks)
    idx = pp_ranks.index(DeviceGroup().rank)
    _pp_next_rank = pp_ranks[(idx+1) % len(pp_ranks)]
    _pp_prev_rank = pp_ranks[(idx-1) % len(pp_ranks)]

    encoder_time = [1] * args.layers
    decoder_time = [1] * args.layers
    times = encoder_time + decoder_time
    num_stages = torch.distributed.get_world_size(_pp_group)
    budget = sum(times) // num_stages
    print_each_rank(f'budget: {budget}', rank_only=0)
    start, end = 0, 1
    for idx in range(num_stages):
        accum = times[start]
        assert end <= args.layers * 2
        while end != args.layers * 2:
            accum += times[end]
            if accum > budget:
                break
            end += 1
        if idx == num_stages - 1:
            end = args.layers * 2
        _layer_divisions.append((start, end))
        start, end = end, end+1

    # uniform division algorithm:
    # num_stages = torch.distributed.get_world_size(_pp_group)
    # chunk = args.layers // (num_stages // 2)
    # encoder_nlayers = [chunk] * (num_stages // 2)
    # for idx in range(args.layers % (num_stages // 2)):
    #     encoder_nlayers[-idx] += 1
    # encoder_layers = [
    #     (sum(encoder_nlayers[:rank]),
    #      sum(encoder_nlayers[:rank+1])) for rank in range(num_stages // 2)
    # ]
    # decoder_layers = [
    #     (args.layers + sum(encoder_nlayers[:rank]),
    #      args.layers + sum(encoder_nlayers[:rank+1])) for rank in range(num_stages // 2)
    # ]
    # _layer_divisions = encoder_layers + decoder_layers

    print_each_rank(f'layer divisions: {_layer_divisions}', rank_only=0)

    
# create embed group: first encoder, first decoder
if args.use_1f1b:
    encoder_preprocess = 0
    decoder_preprocess = None
    for rank in range(len(pp_ranks)):
        start, end = _layer_divisions[rank]
        if start <= args.layers and end > args.layers:
            decoder_preprocess = rank
            break
    assert decoder_preprocess is not None
    embed_ranks = [encoder_preprocess, decoder_preprocess]
    _pp_embed_group = DeviceGroup().get_group(embed_ranks)


class Config:


    # 610M model -> original setting
    # num_embeddings = 250027
    # decoder_layers = 12
    # encoder_layers = 12
    # embed_dim = 1024
    # attention_heads = 16
    # attention_inner_dim = attention_heads * 64
    # ffn_dim = 4 * embed_dim

    num_embeddings = 500000 # 250027 + int(250027*scale_p)
    # decoder_layers = 12 + int(12*scale_p)
    # encoder_layers = 12 + int(12*scale_p)
    # embed_dim = 1024 + int(1024*scale_p)
    # attention_heads = 16 + int(16*scale_p) if scale < 6 else 40 + 8*(scale-6)
    decoder_layers = args.layers
    encoder_layers = args.layers
    embed_dim = args.hidden_size
    attention_heads = args.heads

    attention_inner_dim = attention_heads * 64
    ffn_dim = 4 * embed_dim

    attention_dropout = 0.0
    activation_dropout = 0.0
    dropout = 0.1

    max_target_positions = 1024
    max_source_positions = 1024

    # classification task
    pooler_dropout = 0.0
    num_classes = 3


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


class PositionalEmbedding(torch.nn.Embedding):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, seq_len: int):
        positions = torch.arange(
            0, seq_len, dtype=torch.long, device=torch.cuda.current_device()
        )
        return super().forward(positions + self.offset)


class MultiheadAttention(torch.nn.Module):

    def __init__(self, embed_dim: int, num_heads: int, inner_dim: int, dropout=0.0, bias=True):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # K
        self.k_proj = torch.nn.Parameter(torch.empty(self.inner_dim, embed_dim))
        self.k_bias = torch.nn.Parameter(torch.empty(self.inner_dim)) if bias else None
        # V
        self.v_proj = torch.nn.Parameter(torch.empty(self.inner_dim, embed_dim))
        self.v_bias = torch.nn.Parameter(torch.empty(self.inner_dim)) if bias else None
        # Q
        self.q_proj = torch.nn.Parameter(torch.empty(self.inner_dim, embed_dim))
        self.q_bias = torch.nn.Parameter(torch.empty(self.inner_dim)) if bias else None
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, self.inner_dim))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim)) if bias else None

    def forward(self, query: torch.Tensor, key: torch.Tensor):
        return attn_fn(query, key, 
                       self.q_proj, self.q_bias,
                       self.k_proj, self.k_bias,
                       self.v_proj, self.v_bias,
                       self.out_proj, self.out_bias,
                       self.num_heads, self.scaling, self.dropout_p)

    def forward_encoder_decoder_attn(self, query: torch.Tensor, key: torch.Tensor):
        return attn_fn(query, key, 
                       self.q_proj, self.q_bias,
                       self.k_proj, self.k_bias,
                       self.v_proj, self.v_bias,
                       self.out_proj, self.out_bias,
                       self.num_heads, self.scaling, self.dropout_p)

    def forward_self_attn(self, query):
        return attn_fn(query, query,
                       self.q_proj, self.q_bias,
                       self.k_proj, self.k_bias,
                       self.v_proj, self.v_bias,
                       self.out_proj, self.out_bias,
                       self.num_heads, self.scaling, self.dropout_p)


class EncoderLayer(torch.nn.Module):

    def __init__(self, cfg: Config):

        super().__init__()
        self.cfg = cfg
        self.self_attn = MultiheadAttention(cfg.embed_dim, cfg.attention_heads, cfg.attention_inner_dim, cfg.attention_dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(cfg.embed_dim)
        self.dropout = torch.nn.Dropout(p=cfg.dropout)
        self.activation_dropout = torch.nn.Dropout(p=cfg.activation_dropout)
        self.fc1 = torch.nn.Linear(cfg.embed_dim, cfg.ffn_dim)
        self.fc2 = torch.nn.Linear(cfg.ffn_dim, cfg.embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.embed_dim)

    def input_shape(self):
        # L, N, E
        return (self.cfg.max_source_positions, 1, self.cfg.embed_dim)

    def output_shape(self):
        # L N E
        return (self.cfg.max_source_positions, 1, self.cfg.embed_dim)

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
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.activation_dropout(x)
        x = self.fc2(x)

        x = self.dropout(x)
        x = x + residual
        return x


class DecoderLayer(torch.nn.Module):

    def __init__(self, cfg: Config):

        super().__init__()
        self.cfg = cfg
        self.dropout = torch.nn.Dropout(p=cfg.dropout)
        self.self_attn = MultiheadAttention(cfg.embed_dim, cfg.attention_heads, cfg.attention_inner_dim, cfg.attention_dropout)
        self.activation_dropout = torch.nn.Dropout(p=cfg.activation_dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(cfg.embed_dim)
        # encoder atten
        self.encoder_attn = MultiheadAttention(cfg.embed_dim, cfg.attention_heads, cfg.attention_inner_dim, cfg.attention_dropout)
        self.encoder_attn_layer_norm = torch.nn.LayerNorm(cfg.embed_dim)

        self.fc1 = torch.nn.Linear(cfg.embed_dim, cfg.ffn_dim)
        self.fc2 = torch.nn.Linear(cfg.ffn_dim, cfg.embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.embed_dim)

    def input_shape(self):
        return (
            (self.cfg.max_target_positions, 1, self.cfg.embed_dim),
            (self.cfg.max_source_positions, 1, self.cfg.embed_dim)
        )

    def output_shape(self):
        return (
            (self.cfg.max_target_positions, 1, self.cfg.embed_dim),
            (self.cfg.max_source_positions, 1, self.cfg.embed_dim)
        )

    def input_dtype(self):
        return (torch.float32, torch.float32)

    def output_dtype(self):
        return (torch.float32, torch.float32)

    def forward(self, x, encoder_out):
        # print(f'decoder layer: x: {x.size()}, encoder_out: {encoder_out.size()}')
        residual = x
        x = self.self_attn_layer_norm(x)

        # self attention
        x = self.self_attn(x, x)
        x = self.dropout(x)
        x = residual + x
    
        # encoder attn
        residual = x
        x = self.encoder_attn_layer_norm(x)
        x = self.encoder_attn(x, encoder_out)
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.activation_dropout(x)
        x = self.fc2(x)
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
        self.num_classes = num_classes
        self.dense = torch.nn.Linear(input_dim, inner_dim)
        self.dropout = torch.nn.Dropout(p=pooler_dropout)
        self.out_proj = torch.nn.Linear(inner_dim, num_classes)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, dec: torch.Tensor, labels):
        # sentence_represent = dec[eos_mask,:].view(dec.size(0), -1, hidden_states.size(-1))[:,-1,:]
        dec = dec.transpose(0, 1)[:,-1,:]
        sentence_represent = dec
        hidden_states = self.dropout(sentence_represent)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.out_proj(hidden_states)
        loss = self.loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        return (loss,)


class ShardEmbed(torch.nn.Module):

    def __init__(self, cfg: Config, group=-1, swap=False):
        """
        group = -1 means no tensor parallelism
        """
        super().__init__()
        self.cfg = cfg
        self.group = group
        self.shard_num = torch.distributed.get_world_size(group) if group != -1 else 1
        self.shard_idx = torch.distributed.get_rank(group) if group != -1 else 0
        if self.shard_num > 0:
            print(f'[{torch.distributed.get_rank()}]: initialize sharding embed (x{self.shard_num})')
        assert not (swap and self.shard_idx > 1), "only 1f1b can use swap"
        
        self.swap = swap
        if swap:
            self.embed = SwapEmbed(self.cfg.num_embeddings, self.cfg.embed_dim)
        else:
            self.vocab_start_index = self.cfg.num_embeddings // self.shard_num * self.shard_idx
            self.vocab_end_index = self.cfg.num_embeddings // self.shard_num * (self.shard_idx + 1)
            self.weight = torch.nn.Parameter(torch.ones((self.cfg.num_embeddings // self.shard_num, self.cfg.embed_dim)))

        # encoder-preprocess
        self.embed_positions_encoder = PositionalEmbedding(cfg.max_source_positions, cfg.embed_dim)
        self.embed_scale_encoder = math.sqrt(cfg.embed_dim)
        self.layernorm_embedding_encoder = torch.nn.LayerNorm(cfg.embed_dim)

        # decoder-preprocess
        self.embed_scale_decoder = math.sqrt(cfg.embed_dim)
        self.embed_positions_decoder = PositionalEmbedding(cfg.max_target_positions, cfg.embed_dim)
        self.layernorm_embedding_decoder = torch.nn.LayerNorm(cfg.embed_dim)

        self._inputs = (None, )

    def set_inputs(self, *inputs):
        self._inputs = inputs

    def embed_lookup(self, tokens, dst: Optional[int] = None):
        if self.swap:
            embed = self.embed(tokens)
        elif self.shard_num > 1:
            mask = (tokens < self.vocab_start_index) | \
                        (tokens >= self.vocab_end_index)
            tokens = tokens.clone() - self.vocab_start_index
            tokens[mask] = 0
            embed = torch.nn.functional.embedding(tokens, self.weight)
            embed[mask, :] = 0.0
            embed = ReduceBroadcast.apply(embed, dst, self.group)
        else:
            embed = torch.nn.functional.embedding(tokens, self.weight)
        return embed

    def encoder_preprocess(self, dst: Optional[int] = None):
        source_tokens = self._inputs[0]
        source_embed = self.embed_lookup(source_tokens, dst)
        embed = self.embed_scale_encoder * source_embed
        x = embed + self.embed_positions_encoder(source_tokens.size(1))
        x = self.layernorm_embedding_encoder(x)
        x = torch.nn.functional.dropout(x, p=0.0)
        enc = x.transpose(0, 1)
        return (enc,)

    def decoder_preprocess(self, dst: Optional[int] = None):
        prev_output_tokens = self._inputs[0]
        target_emb = self.embed_lookup(prev_output_tokens, dst)
        embed = self.embed_scale_decoder * target_emb
        embed = embed + self.embed_positions_decoder(prev_output_tokens.size(1))
        embed = self.layernorm_embedding_decoder(embed)
        embed = torch.nn.functional.dropout(embed, p=0.0)
        dec = embed.transpose(0, 1)
        return (dec,)


class mBARTFull(torch.nn.Module):

    def __init__(self, cfg: Config, shard=True):
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
        self.layer_start, self.layer_end = _layer_divisions[self.pp_stage]

        self.encoder_preprocess  = self.layer_start == 0 if not shard else False
        self.encoder_forward     = (self.layer_start < cfg.encoder_layers)

        self.decoder_first_stage = self.layer_start <= cfg.encoder_layers and self.layer_end > cfg.encoder_layers
        self.decoder_preprocess  = self.decoder_first_stage if not shard else False
        self.decoder_forward     = (self.layer_end > cfg.encoder_layers)
        self.decoder_last_stage  = (self.layer_end == cfg.encoder_layers + cfg.decoder_layers)

        self.postprocess         = self.decoder_last_stage

        self.encoder_layer_start = self.layer_start
        self.encoder_layer_end = min(self.layer_end, cfg.encoder_layers)

        self.decoder_layer_start = max(cfg.encoder_layers, self.layer_start)
        self.decoder_layer_end = self.layer_end

        if self.encoder_preprocess or self.decoder_preprocess or shard:
            self.headtail = ShardEmbed(cfg, group = None if shard else -1, swap=args.use_swap)
        else:
            self.headtail = None

        # encoders
        if self.encoder_forward:
            print(f'[{self.rank}]: initializing {self.encoder_layer_end - self.encoder_layer_start} encoder layers')
            self.encoders = torch.nn.ModuleList([EncoderLayer(cfg) for _ in range(self.encoder_layer_end - self.encoder_layer_start)])
            if self.encoder_layer_end == cfg.encoder_layers:
                self.layer_norm_encoder = torch.nn.LayerNorm(cfg.embed_dim)
            else:
                self.layer_norm_encoder = None

        # decoders
        if self.decoder_forward:
            print(f'[{self.rank}]: initializing {self.decoder_layer_end - self.decoder_layer_start} decoder layers')
            self.decoders = torch.nn.ModuleList([DecoderLayer(cfg) for _ in range(self.decoder_layer_end - self.decoder_layer_start)])
            if self.decoder_layer_end == cfg.encoder_layers + cfg.decoder_layers:
                self.layer_norm_decoder = torch.nn.LayerNorm(cfg.embed_dim)
            else:
                self.layer_norm_decoder = None

        # postpross
        if self.postprocess:
            print(f'[{self.rank}]: will compute loss')
            self.head = MBartClassificationHead(cfg.embed_dim, 1024, cfg.num_classes, 0.0)

    def input_shape(self):
        if self.encoder_preprocess:
            return ()
        elif self.encoder_forward:
            return (
                (self.cfg.max_source_positions, 1, self.cfg.embed_dim),
            )
        elif self.decoder_preprocess:
            return (
                (self.cfg.max_source_positions, 1, self.cfg.embed_dim),
            )
        elif self.decoder_first_stage:
            return (
                (self.cfg.max_source_positions, 1, self.cfg.embed_dim),
            )
        elif self.decoder_forward:
            return (
                (self.cfg.max_source_positions, 1, self.cfg.embed_dim),
                (self.cfg.max_target_positions, 1, self.cfg.embed_dim),
            )
        elif self.postprocess:
            return ((1,),)
        assert False

    def output_shape(self):
        shape = None
        if self.encoder_preprocess or self.encoder_forward:
            shape = (self.cfg.max_source_positions, 1, self.cfg.embed_dim),
        # decoder preprocess is not allowed to be a single stage
        if self.decoder_preprocess or self.decoder_forward:
            shape = (
                (self.cfg.max_source_positions, 1, self.cfg.embed_dim),
                (self.cfg.max_target_positions, 1, self.cfg.embed_dim),
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

    def forward_encoder_preprocess(self, dst=None):
        return self.headtail.encoder_preprocess(dst)

    def forward_decoder_preprocess(self, dst=None):
        return self.headtail.decoder_preprocess(dst)

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
            output = self.forward_encoder_preprocess(dst=None)
            enc = output[0]

        # forward encoder
        if self.encoder_forward:
            for layer in self.encoders:
                # enc = checkpoint.checkpoint(layer, enc)
                enc = layer(enc)
            if self.layer_norm_encoder is not None:
                enc = self.layer_norm_encoder(enc)
            output = (enc,)

        # decoder preprocess
        if self.decoder_preprocess:
            output = self.forward_decoder_preprocess(dst=None)
            dec = output[0]

        # forward decoder
        if self.decoder_forward:
            dec = pre_dec if dec is None else dec
            for layer in self.decoders:
                # dec, enc = checkpoint.checkpoint(layer, dec, enc)
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
        if model.headtail.swap:
            with torch.no_grad():
                grad = model.headtail.embed.weight.grad
                grad = grad.cuda()
        else:
            grad = model.headtail.weight.grad
    else:
        grad = None
    if grad is not None:
        CudaTimer().start('comm')
        torch.distributed.all_reduce(grad, group=pp_embed_group)
        torch.cuda.synchronize()
        CudaTimer().stop('comm')
    if isinstance(model.headtail, torch.nn.Module):
        if model.headtail.swap:
            with torch.no_grad():
                model.headtail.embed.weight.grad.copy_(grad)
    torch.cuda.synchronize()


if __name__ == '__main__':


    cfg = Config()
    print_each_rank(f'enc/dec layer#: {cfg.encoder_layers}, embed_dim#: {cfg.embed_dim}, heads#: {cfg.attention_heads}, ffn_dim#: {cfg.ffn_dim}', rank_only=0)
    dataloader = SynTextDataLoader(
        shapes=(
            [1, cfg.max_source_positions],
        ),
        dtypes=(torch.int64,),
        batch_dims=(0,)
    )
    if args.use_1f1b:
        model = mBARTFull(cfg, shard=False).cuda()
    else:
        model = mBARTFull(cfg, shard=True).cuda()

    print_each_rank('model weight consumpition:')
    memory_summary()

    if args.use_swap:
        parameters = get_swap_parameters() + list(model.parameters())
    else:
        parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=3e-05, betas=(0.9, 0.98))

    CudaTimer(enable=False).warmup()
    iter_num = 6
    for step in range(iter_num):
        if step >= 2:
            CudaTimer(enable=True).start('e2e')
        if args.use_1f1b:
            for _ in range(args.nmb // args.iter_nmb):
                schedule_1f1b(
                    model, iter(dataloader),
                    args.iter_nmb, len(pp_ranks),
                    (_pp_prev_rank, _pp_next_rank),
                    recompute=args.use_recompute,
                )
            reduce_embed(model, _pp_embed_group)
        if args.use_tp1f1b_pack:
            schedule_tp_1f1b_pack(
                model, iter(dataloader),
                args.nmb, len(pp_ranks),
                (_pp_prev_rank, _pp_next_rank),
                recompute=args.use_recompute,
            )
        if step == 0:
            print('passed 1st iteration')
            memory_summary()
        optimizer.step()
        optimizer.zero_grad()
        if step == 0:
            print('memory after optimizer')
            memory_summary()
        if step >= 2:
            CudaTimer().stop('e2e')
        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-2, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-2)
    memory_summary()
