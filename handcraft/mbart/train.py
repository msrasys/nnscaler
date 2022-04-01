"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    handcraft/mbart/train.py \
        --layers 12 --hidden-size 1024 --heads 16 \
        --dp-size 1 --pp-size 4 --tp-size 1 \
        --bs 4 --micro-bs 1 --schedule 1f1b
"""

from typing import Optional
import argparse
import math
import numpy as np
import torch

import cube
from cube.runtime.device import DeviceGroup
from cube.runtime.adapter.reducer import Reducer
from cube.runtime.adapter.distnn import ReduceBroadcast, AllReduceIdentity, IdentityAllreduce

from cube.profiler import CudaTimer
from cube.profiler.memory import memory_summary
from cube.profiler.timer import print_each_rank


from handcraft.module.schedule import schedule_1f1b, schedule_tp1f1b
from handcraft.module.stage import PipeStage
from handcraft.mbart.swap import SwapEmbed, get_swap_parameters

torch.manual_seed(0)
np.random.seed(0)


parser = argparse.ArgumentParser(description='mbart')
# model arch
parser.add_argument('--layers', type=int, default=12,
                    help='number encoder/decoder of layers')
parser.add_argument('--hidden-size', type=int, default=1024,
                    help='hidden size')
parser.add_argument('--heads', type=int, default=16,
                    help='number of heads')
# training config
parser.add_argument('--bs', type=int, default=256,
                    help='num of micro batch')
parser.add_argument('--micro-bs', type=int, default=1,
                    help='micro batch size')
parser.add_argument('--fp16', action='store_true', default=False)
# parallelism
parser.add_argument('--pp-size', type=int, default=1,
                    help='pipeline parallelism size')
parser.add_argument('--tp-size', type=int, default=1,
                    help='tensor parallelism size')
parser.add_argument('--dp-size', type=int, default=1,
                    help='data parallelism size')
parser.add_argument('--schedule', type=str, default='1f1b', choices=['1f1b', 'tp1f1b'],
                    help='scheduling algorithm')
parser.add_argument('--use-swap', action='store_true', default=False,
                    help='swap on embedding weight')

args = parser.parse_args()
print(args)

_tp_group = -1

_dp_group = -1
_dp_reducer = None

_pp_group = -1
_pp_global_ranks = ()
_first_encoder_stage = 0
_first_decoder_stage = 0
_layer_divisions = []

_schedule = schedule_1f1b if args.schedule == '1f1b' else schedule_tp1f1b
if args.schedule == 'tp1f1b':
    assert args.tp_size == 1 and args.dp_size == 1, "tp1f1b only supports pure pipeline"

_pp_embed_group = -1

cube.init()
dp_ranks, pp_ranks, tp_ranks= DeviceGroup().create_hybrid(
    [args.dp_size, args.pp_size, args.tp_size]
)

if len(dp_ranks) != 1:
    assert False, "DP is not supported yet"
    print_each_rank(f'initializing dp ranks: {dp_ranks}')
    _dp_group = DeviceGroup().get_group(dp_ranks)
    _dp_reducer = Reducer(dp_ranks)

if len(tp_ranks) != 1:
    print_each_rank(f'initializing tp ranks: {tp_ranks}')
    _tp_group = DeviceGroup().get_group(tp_ranks)

if len(pp_ranks) != 1:
    print_each_rank(f'initializing pp ranks: {pp_ranks}')
    _pp_group = DeviceGroup().get_group(pp_ranks)
    _pp_global_ranks = tuple(pp_ranks)

    # layer division
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
        if start <= args.layers and end > args.layers:
            _first_decoder_stage = idx
        start, end = end, end+1
    assert _first_decoder_stage != _first_encoder_stage, "Not supported yet"
else:
    _layer_divisions = [(0, args.layers * 2)]
print_each_rank(
    f"layer divisions: {_layer_divisions} | "
    f"first encoder stage: {_first_encoder_stage} | "
    f"first decoder stage: {_first_decoder_stage}", rank_only=0
)

    
# create embed group: first encoder, first decoder
if args.schedule == '1f1b' and args.pp_size > 1:
    grid = np.arange(
        args.pp_size * args.tp_size).reshape((args.pp_size, args.tp_size))
    encoder_preprocess = grid[_first_encoder_stage,:]
    decoder_preprocess = grid[_first_decoder_stage,:]
    embed_ranks = np.vstack((encoder_preprocess, decoder_preprocess))
    grank = torch.distributed.get_rank()
    for gid in range(args.tp_size):
        embed_rank = embed_ranks[:,gid]
        embed_rank = np.squeeze(embed_rank).tolist()
        print_each_rank(f'creating embed group: {embed_rank}')
        group = DeviceGroup().get_group(embed_rank)
        if grank in embed_rank:
            print(f'rank [{grank}]: embedding group: {embed_rank}')
            _pp_embed_group = group


class Config:

    num_embeddings = 500000
    decoder_layers = args.layers
    encoder_layers = args.layers
    embed_dim = args.hidden_size
    attention_heads = args.heads

    attention_inner_dim = attention_heads * 64
    ffn_dim = 4 * embed_dim

    attention_dropout = 0.0  # for correctness veirfication
    activation_dropout = 0.0 # for correctness veirfication
    dropout = 0.0            # for correctness veirfication

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
        self.tp_group = _tp_group
        self.tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)

        self.inner_dim = inner_dim
        self.head_dim = inner_dim // num_heads
        self.num_heads = num_heads // self.tp_size
        self.scaling = self.head_dim ** -0.5
        self.dropout_p = dropout
        # K
        self.k_proj = torch.nn.Parameter(torch.empty(self.inner_dim // self.tp_size, embed_dim))
        self.k_bias = torch.nn.Parameter(torch.empty(self.inner_dim // self.tp_size)) if bias else None
        # V
        self.v_proj = torch.nn.Parameter(torch.empty(self.inner_dim // self.tp_size, embed_dim))
        self.v_bias = torch.nn.Parameter(torch.empty(self.inner_dim // self.tp_size)) if bias else None
        # Q
        self.q_proj = torch.nn.Parameter(torch.empty(self.inner_dim // self.tp_size, embed_dim))
        self.q_bias = torch.nn.Parameter(torch.empty(self.inner_dim // self.tp_size)) if bias else None
        # Out
        self.out_proj = torch.nn.Parameter(torch.empty(embed_dim, self.inner_dim // self.tp_size))
        self.out_bias = torch.nn.Parameter(torch.empty(embed_dim)) if bias else None

    def forward(self, query: torch.Tensor, key: torch.Tensor):
        if self.tp_size > 1:
            if key is not query:
                key = IdentityAllreduce.apply(key, self.tp_group)
            query = IdentityAllreduce.apply(query, self.tp_group)
        attn = attn_fn(query, key, 
                       self.q_proj, self.q_bias,
                       self.k_proj, self.k_bias,
                       self.v_proj, self.v_bias,
                       self.out_proj, self.out_bias,
                       self.num_heads, self.scaling, self.dropout_p)
        if self.tp_size > 1:
            attn = AllReduceIdentity.apply(attn, self.tp_group)
        return attn


class EncoderLayer(PipeStage):

    def __init__(self, cfg: Config):
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)

        self.cfg = cfg
        self.self_attn = MultiheadAttention(cfg.embed_dim, cfg.attention_heads, cfg.attention_inner_dim, cfg.attention_dropout)
        self.self_attn_layer_norm = torch.nn.LayerNorm(cfg.embed_dim)
        self.dropout = torch.nn.Dropout(p=cfg.dropout)
        self.activation_dropout = torch.nn.Dropout(p=cfg.activation_dropout)
        self.fc1 = torch.nn.Linear(cfg.embed_dim, cfg.ffn_dim)
        self.fc2 = torch.nn.Linear(cfg.ffn_dim, cfg.embed_dim)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.embed_dim)

        self.inputs_info = (
            ((self.cfg.max_source_positions, 1, self.cfg.embed_dim),),
            (torch.float32 if not args.fp16 else torch.float16,)
        )
        self.outputs_info = (
            ((self.cfg.max_source_positions, 1, self.cfg.embed_dim),),
            (torch.float32 if not args.fp16 else torch.float16,)
        )

    def forward(self, x):
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


class DecoderLayer(PipeStage):

    def __init__(self, cfg: Config):
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)

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

        self.inputs_info = (
            ((self.cfg.max_target_positions, 1, self.cfg.embed_dim),
             (self.cfg.max_source_positions, 1, self.cfg.embed_dim)),
            (torch.float32 if not args.fp16 else torch.float16,
             torch.float32 if not args.fp16 else torch.float16,)
        )
        self.outputs_info = (
            ((self.cfg.max_target_positions, 1, self.cfg.embed_dim),
             (self.cfg.max_source_positions, 1, self.cfg.embed_dim)),
            (torch.float32 if not args.fp16 else torch.float16,
             torch.float32 if not args.fp16 else torch.float16,)
        )

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
        return loss


class ShardEmbed(torch.nn.Module):

    def __init__(self, cfg: Config):
        super().__init__()
        self.tp_group = None if args.schedule == 'tp1f1b' else _tp_group
        self.tp_size = 1 if self.tp_group == -1 else torch.distributed.get_world_size(self.tp_group)
        self.tp_id = 0 if self.tp_group == -1 else torch.distributed.get_rank(self.tp_group)

        self.cfg = cfg
        print(f'initialize sharding embed (x{self.tp_size})')

        self.swap = args.use_swap
        if self.swap:
            assert args.schedule == '1f1b', "only 1f1b can use swap"
            self.embed = SwapEmbed(self.cfg.num_embeddings, self.cfg.embed_dim)
        else:
            self.vocab_start_index = self.cfg.num_embeddings // self.tp_size * self.tp_id
            self.vocab_end_index = self.cfg.num_embeddings // self.tp_size * (self.tp_id + 1)
            self.weight = torch.nn.Parameter(
                torch.ones((self.cfg.num_embeddings // self.tp_size, self.cfg.embed_dim))
            )

        # encoder-preprocess
        self.embed_positions_encoder = PositionalEmbedding(cfg.max_source_positions, cfg.embed_dim)
        self.embed_scale_encoder = math.sqrt(cfg.embed_dim)
        self.layernorm_embedding_encoder = torch.nn.LayerNorm(cfg.embed_dim)

        # decoder-preprocess
        self.embed_scale_decoder = math.sqrt(cfg.embed_dim)
        self.embed_positions_decoder = PositionalEmbedding(cfg.max_target_positions, cfg.embed_dim)
        self.layernorm_embedding_decoder = torch.nn.LayerNorm(cfg.embed_dim)

        self.init_weight()

    def embed_lookup(self, tokens, dst: Optional[int] = None):
        """
        Embedding lookup
        if dst is None, use all
        """
        if self.swap:
            embed = self.embed(tokens)
        elif self.tp_size > 1:
            mask = (tokens < self.vocab_start_index) | \
                        (tokens >= self.vocab_end_index)
            tokens = tokens.clone() - self.vocab_start_index
            tokens[mask] = 0
            embed = torch.nn.functional.embedding(tokens, self.weight)
            embed[mask, :] = 0.0
            if dst is None:
                assert _tp_group != -1
                embed = AllReduceIdentity.apply(embed, self.tp_group)
            else:
                assert self.tp_group is None  # args.sharding = True
                embed = ReduceBroadcast.apply(embed, dst, None)
        else:
            embed = torch.nn.functional.embedding(tokens, self.weight)
        return embed

    def forward(self, tokens, encoder=False, decoder=False, dst: Optional[int] = None):
        """
        If dst is not None: the embedding is sharded across all devices
        using tp1f1b, and hence requires a Reduce on the target rank.
        """
        assert encoder ^ decoder, "can only be either encoder or decoder"
        embed = self.embed_lookup(tokens, dst)
        x = embed + self.embed_positions_encoder(embed.size(1))
        if encoder:
            x = self.layernorm_embedding_encoder(x)
        if decoder:
            x = self.layernorm_embedding_decoder(x)
        x = torch.nn.functional.dropout(x, p=0.0)
        x = x.transpose(0, 1)
        return x

    def init_weight(self):
        for param in self.parameters():
            torch.nn.init.constant_(param, 0.1)


class MBart(PipeStage):

    def __init__(self, cfg: Config):
        super().__init__()
        self.set_pipeline(_pp_global_ranks)
        self.first_encoder_stage = _first_encoder_stage
        self.first_decoder_stage = _first_decoder_stage

        self.cfg = cfg
        encoders = [EncoderLayer(cfg) for _ in range(self.cfg.encoder_layers)]
        decoders = [DecoderLayer(cfg) for _ in range(self.cfg.decoder_layers)]

        start, end = _layer_divisions[self.stage_local_rank]
        print_each_rank(f'initializing layer ranging from [{start}, {end})')

        self.encoder_preprocess = self.is_first_stage
        self.encoder_forward = start < cfg.encoder_layers
        self.decoder_preprocess = start <= cfg.encoder_layers and end > cfg.encoder_layers
        self.decoder_forward = end > cfg.encoder_layers
        self.sharding = args.schedule == 'tp1f1b'
        print_each_rank(
            f"encoder: (pre: {self.encoder_preprocess}) {self.encoder_forward} | "
            f"decoder (pre: {self.decoder_preprocess}) {self.decoder_forward} | "
            f"post-process: {self.is_last_stage} | sharding {self.sharding}"
        )

        inputs_info = None
        outputs_info = None

        self.embed: ShardEmbed = None
        if self.encoder_preprocess:
            self.embed = ShardEmbed(cfg) if self.embed is None else self.embed
            inputs_info = ((), ()) if inputs_info is None else inputs_info

        if self.encoder_forward:
            self.encoders = torch.nn.ModuleList(
                encoders[start:min(end, cfg.encoder_layers)]
            )
            self.layer_norm_encoder = None
            if self.decoder_preprocess or self.decoder_forward:
                self.layer_norm_encoder = torch.nn.LayerNorm(cfg.embed_dim)

            inputs_info = self.encoders[0].inputs_info if inputs_info is None else inputs_info
            outputs_info = self.encoders[-1].outputs_info

        if self.decoder_preprocess:
            self.embed = ShardEmbed(cfg) if self.embed is None else self.embed
            inputs_info = encoders[-1].outputs_info if inputs_info is None else inputs_info

        if self.decoder_forward:
            self.decoders = torch.nn.ModuleList(
                decoders[max(0, start-cfg.encoder_layers): end-cfg.encoder_layers]
            )
            self.layer_norm_decoder = None
            if self.is_last_stage:
                self.layer_norm_decoder = torch.nn.LayerNorm(cfg.embed_dim)
            
            inputs_info = self.decoders[0].inputs_info if inputs_info is None else inputs_info
            outputs_info = self.decoders[-1].outputs_info

        if self.is_last_stage:
            self.head = MBartClassificationHead(cfg.embed_dim, 1024, cfg.num_classes, 0.0)
            outputs_info = ((1,), torch.float32 if not args.fp16 else torch.float16)
        
        if self.sharding:
            self.embed = ShardEmbed(cfg) if self.embed is None else self.embed

        assert inputs_info is not None
        assert outputs_info is not None
        self.inputs_info = inputs_info
        self.outputs_info = outputs_info
        print_each_rank(f'stage: inputs: {inputs_info} | outputs: {outputs_info}')
        self.init_weight()

    def init_weight(self):
        for param in self.parameters():
            torch.nn.init.constant_(param, 0.01)

    def forward_encoder_shard(self):
        """
        Return detached outputs with enabled gradient
        """
        source_tokens, _, _ = self.data
        enc = self.embed(source_tokens, encoder=True, dst=self.first_encoder_stage)
        model.push(enc, 'encoder_sharding_output')
        if self.stage_global_rank == self.first_encoder_stage:
            enc = enc.detach().requires_grad_()
            self.push(enc, 'encoder_preprocess')
        return enc

    def forward_decoder_shard(self):
        """
        Return detached outputs with enabled gradient
        """
        _, prev_tokens, _ = self.data
        dec = self.embed(prev_tokens, decoder=True, dst=self.first_decoder_stage)
        model.push(dec, 'decoder_sharding_output')
        if self.stage_global_rank == self.first_decoder_stage:
            dec = dec.detach().requires_grad_()
            self.push(dec, 'decoder_preprocess')
        return dec

    def forward(self, enc=None, dec=None, recompute=False):
        """
        enc: encoder input
        dec: decoder input
        recompute: outside control for tp1f1b
        """
        if self.encoder_preprocess:
            if self.sharding:
                if recompute:
                    enc = self.get_last('encoder_preprocess')
                else:
                    enc = self.pop('encoder_preprocess')
            else:
                source_tokens, _, _ = self.data
                enc = self.embed(source_tokens, encoder=True)

        if self.encoder_forward:
            for layer in self.encoders:
                enc = layer(enc)
            if self.layer_norm_encoder is not None:
                enc = self.layer_norm_encoder(enc)
            output = enc

        if self.decoder_preprocess:
            if self.sharding:
                if recompute:
                    dec = self.get_last('decoder_preprocess')
                else:
                    dec = self.pop('decoder_preprocess')
            else:
                _, prev_tokens, _ = self.data
                dec = self.embed(prev_tokens, decoder=True)

        if self.decoder_forward:
            assert enc is not None
            for layer in self.decoders:
                dec, enc = layer(dec, enc)
            if self.layer_norm_decoder is not None:
                dec = self.layer_norm_decoder(dec)
            output = (enc, dec)

        if self.is_last_stage:
            _, _, label = self.data
            loss = self.head(dec, label)
            output = loss

        return output


def reduce_embed(model: MBart, pp_embed_group):
    """
    Embedding gradients needs to be reduced across pipeline stages
    """
    if pp_embed_group == -1:
        return
    if isinstance(model.embed, torch.nn.Module):
        if model.embed.swap:
            with torch.no_grad():
                grad = model.embed.weight.grad
                grad = grad.cuda()
        else:
            grad = model.embed.weight.grad
    else:
        grad = None
    if grad is not None:
        CudaTimer().start('comm')
        torch.distributed.all_reduce(grad, group=pp_embed_group)
        torch.cuda.synchronize()
        CudaTimer().stop('comm')
    if isinstance(model.embed, torch.nn.Module):
        if model.embed.swap:
            with torch.no_grad():
                model.embed.embed.weight.grad.copy_(grad)
    torch.cuda.synchronize()


class MBartDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int, cfg: Config):
        self.bs = batch_size
        self.cfg = cfg
        super().__init__(
            shapes=(
                [batch_size, cfg.max_source_positions,],
                [batch_size, cfg.max_target_positions,],
                [batch_size,]
            ),
            dtypes=(
                torch.int64,
                torch.int64,
                torch.int,
            ),
            batch_dims=(0, 0, 0)
        )
        self.samples = [self.random_sample()]

    def random_sample(self):
        source_token = torch.randint(
            0, 25000,
            size=(self.bs, cfg.max_source_positions,),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        target_token = torch.randint(
            0, 25000,
            size=(self.bs, cfg.max_target_positions,),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        labels = torch.randint(
            0, self.cfg.num_classes,
            size=(self.bs,),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        return (source_token, target_token, labels)

    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]


if __name__ == '__main__':


    cfg = Config()
    print_each_rank(f'enc/dec layer#: {cfg.encoder_layers}, embed_dim#: {cfg.embed_dim}, heads#: {cfg.attention_heads}, ffn_dim#: {cfg.ffn_dim}', rank_only=0)

    model = MBart(cfg)
    model = model.half().cuda() if args.fp16 else model.cuda()

    dataloader = MBartDataLoader(args.micro_bs, cfg)

    parameters = get_swap_parameters() + list(model.parameters()) if args.use_swap else model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=3e-05, betas=(0.9, 0.98))

    print_each_rank('model weight consumpition:')
    memory_summary()

    CudaTimer(enable=False).warmup()
    iter_num = 6
    for step in range(iter_num):
        if step >= 2:
            CudaTimer(enable=True).start('e2e')

        # train 1 step
        num_microbatch = args.bs // args.micro_bs
        if args.pp_size > 1:
            _schedule(model, dataloader, num_microbatch, recompute=True)
            reduce_embed(model, _pp_embed_group)
        else:
            for _ in range(num_microbatch):
                model.data = next(dataloader)
                loss = model()
                loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step >= 2:
            CudaTimer().stop('e2e')

        if step == 0:
            print_each_rank('memory after optimizer:', rank_only=0)
            memory_summary()

        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-2, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-2)
    memory_summary()
