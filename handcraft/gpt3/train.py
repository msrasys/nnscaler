"""
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    handcraft/gpt3/train.py \
        --layers 24 --hidden-size 2048 --heads 32 \
        --dp-size 1 --tp-size 1 --pp-size 1 \
        --seqlen 8192 --bs 8 --micro-bs 1 --fp16

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    handcraft/gpt3/train.py \
        --layers 32 --hidden-size 4096 --heads 32 \
        --dp-size 1 --tp-size 1 --pp-size 4 \
        --seqlen 1024 --bs 8 --micro-bs 1 --fp16

350M: --layers 24 --hidden-size 1024 --heads 16 \
1.3B: --layers 24 --hidden-size 2048 --heads 32 \
2.6B: --layers 32 --hidden-size 2560 --heads 32 \
6.7B: --layers 32 --hidden-size 4096 --heads 32 \
15 B: --layers 48 --hidden-size 5120 --heads 32 \
39 B: --layers 48 --hidden-size 8192 --heads 64 \
"""

import torch
import torch.utils.checkpoint as checkpoint
import cube
import math
import numpy as np

from cube.runtime.device import DeviceGroup
from cube.runtime.adapter.reducer import Reducer
from cube.runtime.adapter.distnn import AllReduceIdentity, IdentityAllreduce, AllGatherSplit

from cube.profiler import CudaTimer
from cube.profiler.memory import memory_summary
from cube.profiler.timer import print_each_rank

from handcraft.module.schedule import schedule_1f1b
from handcraft.module.stage import PipeStage, layer_division

import argparse

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser(description='gpt3')
# model arch
parser.add_argument('--layers', type=int, default=12,
                    help='number encoder/decoder of layers')
parser.add_argument('--hidden-size', type=int, default=1024,
                    help='hidden size')
parser.add_argument('--heads', type=int, default=16,
                    help='number of heads')
parser.add_argument('--seqlen', type=int, default=1024,
                    help='sequence length')
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
parser.add_argument('--schedule', type=str, default='1f1b', choices=['1f1b'],
                    help='scheduling algorithm')
parser.add_argument('--use-coshard', action='store_true', default=False)
parser.add_argument('--coshard-num', type=int, default=4,
                    help='if use coshard, the coshard number')

args = parser.parse_args()
print(args)

_tp_group = -1

_dp_group = -1
_dp_reducer = None

_pp_group = -1
_pp_global_ranks = ()
_layer_divisions = []

_schedule = schedule_1f1b

_pp_embed_group = -1
_pp_embed_reducer = None
cube.init()
dp_ranks, pp_ranks, tp_ranks= DeviceGroup().create_hybrid(
    [args.dp_size, args.pp_size, args.tp_size]
)

if len(dp_ranks) != 1:
    print_each_rank(f'initializing dp ranks: {dp_ranks}')
    _dp_group = DeviceGroup().get_group(dp_ranks)
    _dp_reducer = Reducer(dp_ranks)

if len(tp_ranks) != 1:
    print_each_rank(f'initializing tp ranks: {tp_ranks}')
    _tp_group = DeviceGroup().get_group(tp_ranks)
    assert args.heads % args.tp_size == 0, "cannot be divided by tp-size"

if len(pp_ranks) != 1:
    print_each_rank(f'initializing pp ranks: {pp_ranks}')
    _pp_group = DeviceGroup().get_group(pp_ranks)
    _pp_global_ranks = tuple(pp_ranks)
    _layer_divisions = layer_division([1] * args.layers, args.pp_size)
else:
    _layer_divisions = [(0, args.layers)]
print_each_rank(f'layer divisions: {_layer_divisions}')

if args.schedule == '1f1b' and args.pp_size > 1:
    grid = np.arange(args.dp_size * args.pp_size * args.tp_size).reshape(
        (args.dp_size, args.pp_size, args.tp_size))
    for dp_rank in range(args.dp_size):
        embed_ranks = np.vstack((grid[dp_rank, 0,  :], grid[dp_rank, -1, :]))
        grank = torch.distributed.get_rank()
        for gid in range(args.tp_size):
            embed_rank = embed_ranks[:,gid]
            embed_rank = np.squeeze(embed_rank).tolist()
            print_each_rank(f'creating embed group: {embed_rank}')
            group = DeviceGroup().get_group(embed_rank)
            if grank in embed_rank:
                print(f'rank [{grank}]: embedding group: {embed_rank}')
                _pp_embed_group = group
                _pp_embed_reducer = Reducer(embed_rank)


class Config:
    vocab_size = 50273
    seqlen = args.seqlen
    layers = args.layers
    heads = args.heads
    hidden_size = args.hidden_size

config = Config()


class MLP(torch.nn.Module):

    def __init__(self, hidden_dim: int = None):
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if self.tp_group == -1 else torch.distributed.get_world_size(self.tp_group)
        
        hidden_dim = config.hidden_size * 4 if hidden_dim is None else hidden_dim
        self.dense_h_to_4h = torch.nn.Linear(
            config.hidden_size, hidden_dim // self.tp_size
        )

        self.dense_4h_to_h = torch.nn.Linear(
            hidden_dim // self.tp_size, config.hidden_size
        )

    def forward_(self, hidden_states):
        if self.tp_size > 1:
            hidden_states = IdentityAllreduce.apply(hidden_states, self.tp_group)
        x = self.dense_h_to_4h(hidden_states)
        x = torch.nn.functional.gelu(x)
        x = self.dense_4h_to_h(x)
        if self.tp_size > 1:
            x = AllReduceIdentity.apply(x, self.tp_group)
        return x

    def forward(self, hidden_states, recompute=False):
        if recompute:
            x = checkpoint.checkpoint(self.forward_, hidden_states)
        else:
            x = self.forward_(hidden_states)
        return x


class SeqMLP(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if self.tp_group == -1 else torch.distributed.get_world_size(self.tp_group)
        
        coshard = args.coshard_num
        assert (config.hidden_size * 4) % (self.tp_size * coshard) == 0
        hidden_dim = config.hidden_size * 4 // coshard
        self.mlps = torch.nn.ModuleList([MLP(hidden_dim) for _ in range(coshard)])
        for mlp in self.mlps:
            mlp.tp_size = 1

    def forward(self, x, recompute=True):
        if self.tp_size > 1:
            x = IdentityAllreduce.apply(x, self.tp_group)

        outs = None
        for mlp in self.mlps:
            x_out = mlp(x, recompute=recompute)
            outs = x_out if outs is None else outs + x_out

        if self.tp_size > 1:
            outs = AllReduceIdentity.apply(outs, self.tp_group)
        return outs


class Attention(torch.nn.Module):

    def __init__(self, num_heads: int = None):
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if self.tp_group == -1 else torch.distributed.get_world_size(self.tp_group)
        
        self.num_heads = (config.heads if num_heads is None else num_heads) // self.tp_size
        self.head_dim = config.hidden_size // config.heads
        projection_size = self.num_heads * self.head_dim

        self.query_key_value = torch.nn.Linear(
            config.hidden_size,
            3 * projection_size,
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        self.norm_factor = math.sqrt(self.head_dim)
        self.dense = torch.nn.Linear(
            projection_size, config.hidden_size
        )

    def forward_(self, x, mask):
        # x: [seqlen, bs, hidden], np: head num | hn: head dim
        if self.tp_size > 1:
            x = IdentityAllreduce.apply(x, self.tp_group)

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(x)
        new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_heads, 3 * self.head_dim)
        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)
        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        query_layer, key_layer, value_layer = \
            torch.chunk(mixed_x_layer, 3, dim=-1)
        
        # [b, np, seqlen, seqlen]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [seqlen, b, np, hn] -> [seqlen, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)

        # [seqlen, b, np, hn] -> [seqlen, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        matmul_result = torch.empty(
            output_size[0]*output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device())

        # Raw attention scores. [b * np, seqlen, seqlen]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),   # [b * np, seqlen, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, seqlen]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, seqlen, seqlen]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, seqlen, seqlen]
        if mask is not None:
            attention_scores.masked_fill_(mask, -10000.0)
        attention_probs = self.softmax(attention_scores)

        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [seqlen, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, seqlen, seqlen]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, seqlen, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, seqlen, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, seqlen, hn] --> [seqlen, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [seqlen, b, np, hn] --> [seqlen, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.head_dim * self.num_heads,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [seqlen, b, h]
        # =================
        output = self.dense(context_layer)
        if self.tp_size > 1:
            output = AllReduceIdentity.apply(output, self.tp_group)
        return output

    def forward(self, x, mask, recompute=False):
        if recompute:
            x = checkpoint.checkpoint(self.forward_, x, mask)
        else:
            x = self.forward_(x, mask)
        return x


class SeqAttention(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if _tp_group == -1 else torch.distributed.get_world_size(_tp_group)

        coshard = args.coshard_num
        assert config.heads % (coshard * self.tp_size) == 0
        self.shard_num_heads = config.heads // coshard
        self.attns = torch.nn.ModuleList(
            [Attention(self.shard_num_heads) for _ in range(coshard)]
        )
        for attn in self.attns:
            attn.tp_size = 1

    def forward(self, x, mask, recompute=True):
        if self.tp_size > 1:
            x = IdentityAllreduce.apply(x, self.tp_group)

        outs = None
        for attn in self.attns:
            x_out = attn(x, mask, recompute)
            outs = x_out if outs is None else outs + x_out

        if self.tp_size > 1:
            outs = AllReduceIdentity.apply(outs, self.tp_group)
        return outs


class Embedding(torch.nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_group = _tp_group
        self.tp_size = 1 if self.tp_group == -1 else torch.distributed.get_world_size(self.tp_group)
        self.tp_id = 0 if self.tp_group == -1 else torch.distributed.get_rank(self.tp_group)

        self.vocab_start_index = num_embeddings // self.tp_size * self.tp_id
        self.vocab_end_index = num_embeddings // self.tp_size * (self.tp_id + 1)
        self.weight = torch.nn.Parameter(
            torch.ones((num_embeddings // self.tp_size, embedding_dim))
        )

    def forward(self, tokens):
        """
        Embedding lookup
        if dst is None, use all
        """
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


class TransformerLayer(PipeStage):

    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(config.hidden_size)
        if args.use_coshard:
            # print('use cosharding attention...')
            self.self_attention = SeqAttention()
        else:
            self.self_attention = Attention()

        self.hidden_dropout = 0.0
        self.post_attention_layernorm = torch.nn.LayerNorm(config.hidden_size)
        if args.use_coshard:
            # print('use cosharding mlp...')
            self.mlp = SeqMLP()
        else:
            self.mlp = MLP()

        # seqlen, b, h
        self.inputs_info = (
            ((config.seqlen, args.micro_bs, config.hidden_size),),
            (torch.float16 if args.fp16 else torch.float32,)
        )
        self.outputs_info = (
            ((config.seqlen, args.micro_bs, config.hidden_size),),
            (torch.float16 if args.fp16 else torch.float32,)
        )


    def forward(self, hidden_states, attention_mask):

        layernrom_output = self.input_layernorm(hidden_states)
        
        attention_output = self.self_attention(layernrom_output, attention_mask)
        
        residual = hidden_states
        layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        layernorm_input = layernorm_input + residual
        layernrom_output = self.post_attention_layernorm(layernorm_input)

        mlp_output = self.mlp(layernrom_output)

        residual = layernorm_input
        output = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)
        output = layernorm_input + residual
        return output


class Pooler(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.dense = troch.nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, sequence_index=0):
        pooled = hidden_states[:, sequence_index, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class GPT3(PipeStage):

    def __init__(self):
        super().__init__()
        self.set_pipeline(pp_ranks)
        self.tp_group = _tp_group
        self.tp_size = 1 if self.tp_group == -1 else torch.distributed.get_world_size(self.tp_group)

        inputs_info = None
        outputs_info = None

        self.word_embeddings = None
        if self.is_first_stage:
            print(f'rank [{torch.distributed.get_rank()}]: initializing preprocess...')
            self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
            self.position_embeddings = torch.nn.Embedding(
                config.seqlen, config.hidden_size
            )
            self.embedding_dropout = torch.nn.Dropout(0.0)
            
            inputs_info = ((), ()) if inputs_info is None else inputs_info

        start, end = _layer_divisions[self.stage_local_rank]
        print_each_rank(f'initializing layers [{start}, {end})...')
        layers = [TransformerLayer() for _ in range(end - start)]
        self.layers = torch.nn.ModuleList(layers)

        inputs_info = self.layers[0].inputs_info if inputs_info is None else inputs_info
        outputs_info = self.layers[-1].outputs_info

        if self.is_last_stage:
            print(f'rank [{torch.distributed.get_rank()}]: initializing postprocess...')
            self.word_embeddings = Embedding(config.vocab_size, config.hidden_size) if self.word_embeddings is None else self.word_embeddings
            self.final_layernorm = torch.nn.LayerNorm(config.hidden_size)
            outputs_info = ((1,), (torch.float32,))
        
        assert inputs_info is not None
        assert outputs_info is not None
        self.inputs_info = inputs_info
        self.outputs_info = outputs_info
        print_each_rank(f'stage: inputs: {inputs_info} | outputs: {outputs_info}')

    def forward(self, hidden_states = None):
        # data
        # input_ids, position_ids, atten_mask, loss_mask

        # preprocess
        if self.is_first_stage:
            input_ids, position_ids, _, _ = self.data
            word_embeddings = self.word_embeddings(input_ids)
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
            embeddings = self.embedding_dropout(embeddings)
            hidden_states = embeddings
            # [seqlen, bs, hidden]
            hidden_states = hidden_states.transpose(0, 1).contiguous()


        assert hidden_states is not None
        _, _, attention_mask, _ = self.data
        for layer in self.layers:
            if args.use_coshard:
                # inner recompute
                hidden_states = layer(hidden_states, attention_mask)
            else:
                # block recompute
                hidden_states = checkpoint.checkpoint(layer, hidden_states, attention_mask)
        outputs = hidden_states

        # postprocess
        if self.is_last_stage:
            labels, _, _, loss_mask = self.data

            hidden_states = hidden_states.transpose(0, 1).contiguous()
            hidden_states = self.final_layernorm(hidden_states)
            
            if self.tp_size > 1:
                hidden_states = IdentityAllreduce.apply(hidden_states, self.tp_group)
            logits = torch.nn.functional.linear(hidden_states, self.word_embeddings.weight)
            if self.tp_size > 1:
                logits = AllGatherSplit.apply(logits, -1, self.tp_group)

            # minor changes from
            # https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/pretrain_gpt.py#L75
            logits = logits.float()
            logits = logits.view(args.micro_bs * config.seqlen, -1)
            labels = labels.view(-1)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            outputs = loss

        return outputs


class GPT3DataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, batch_size: int):
        self.bs = batch_size
        super().__init__(
            shapes=(
                [batch_size, config.seqlen,],
                [batch_size, config.seqlen,],
                [batch_size, config.seqlen,],
                [batch_size, config.seqlen,],
            ),
            dtypes=(
                torch.int64,
                torch.int64,
                torch.float16 if args.fp16 else torch.float,
                torch.float16 if args.fp16 else torch.float,
            ),
            batch_dims=(0, 0, 0, 0)
        )
        self.samples = [self.random_sample()]

    def __iter__(self):
        return self

    def __next__(self):
        return self.samples[0]

    def random_sample(self):
        input_ids = torch.randint(
            0, 25000,
            size=(self.bs, config.seqlen,),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        attention_mask, loss_mask, position_ids = self.get_ltor_masks_and_position_ids(input_ids)
        return (input_ids, position_ids, attention_mask, loss_mask)

    def get_ltor_masks_and_position_ids(self, input_ids):
        """
        Build masks and position id for left to right model.
        https://github.com/NVIDIA/Megatron-LM/blob/e156d2fea7fc5c98e645f7742eb86b643956d840/megatron/utils.py#L81
        """
        # Extract batch size and sequence length.
        seq_length = config.seqlen
        # Attention mask (lower triangular).
        mask_dtype = torch.float16 if args.fp16 else torch.float32
        attention_mask = torch.tril(
            torch.ones((args.micro_bs, seq_length, seq_length), dtype=mask_dtype, device=torch.cuda.current_device())
        ).view(args.micro_bs, 1, seq_length, seq_length)
    
        # Loss mask.
        loss_mask = torch.ones(input_ids.size(), device=input_ids.device)
        eod_token = 2
        loss_mask[input_ids == eod_token] = 0.0
    
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # Convert attention mask to binary:
        attention_mask = (attention_mask < 0.5)
        return attention_mask, loss_mask, position_ids


if __name__ == '__main__':

    model = GPT3()
    nparams = sum([param.numel() for param in model.parameters()])
    # forward_flops = model.flops()
    tflops = 0 # forward_flops * 4 / 1e12 # forward + re-compute forward + backward (=2 forward flops)
    print_each_rank(f'model params (M): {nparams / 1e6}  | TFLOPs: {tflops}.  Launching model...')
    model = model.half().cuda() if args.fp16 else model.cuda()

    dataloader = GPT3DataLoader(args.micro_bs)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-05, betas=(0.9, 0.98))
    if _pp_embed_reducer is not None:
        _pp_embed_reducer.add_param(model.word_embeddings.weight)
    if _dp_reducer is not None:
        for param in model.parameters():
            _dp_reducer.add_param(param)

    print_each_rank('model weight consumpition:')
    memory_summary()

    CudaTimer(enable=False)
    torch.distributed.barrier()
    iter_num = 6
    for step in range(iter_num):
        if step >= 2:
            CudaTimer(enable=True).start('e2e')

        # train 1 step
        num_microbatch = args.bs // (args.micro_bs * args.dp_size)
        if args.pp_size > 1:
            _schedule(model, dataloader, num_microbatch)
        else:
            for _ in range(num_microbatch):
                model.data = next(dataloader)
                loss = model()
                loss.backward()

        if _pp_embed_reducer is not None:
            _pp_embed_reducer.allreduce()
        
        if _dp_reducer is not None:
            _dp_reducer.allreduce()

        optimizer.step()
        optimizer.zero_grad()

        if step >= 2:
            CudaTimer().stop('e2e')

        torch.cuda.empty_cache()
        torch.distributed.barrier()

        if step == 0:
            print_each_rank('memory after optimizer:', rank_only=0)
            memory_summary()

        if (step + 1) % 2 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-2, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-2)
    memory_summary()
