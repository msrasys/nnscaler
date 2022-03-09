"""
Dummy model

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    handcraft/pipeline/dummy_hybrid.py --tp-size 2 --pp-size 2

OMP_NUM_THREADS=4 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    handcraft/pipeline/dummy_hybrid.py --use-naive
"""
import torch
import torch.nn.functional as F
import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from cube.profiler.memory import memory_summary
from cube.runtime.device import DeviceGroup
from cube.runtime.syndata import SynDataLoader, SynTextDataLoader

from handcraft.pipeline.schedule import schedule_tp_1f1b, schedule_naive, schedule_tp_1f1b_pack
import argparse

"""
Stage0:
    Embedding [M, 1], [N, E] -> [M, E]
    Linear    [M, E], [E, E] -> [M, E]

Stage Else:
    Linear [M, E], [E, E] -> [M, E]
        
Condition: N > 8M - E
"""

_tp_group = None
_pp_group = None
_pp_next_rank = None
_pp_prev_rank = None

io_input = input

class ReduceEmbed(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        global _tp_group
        torch.distributed.all_reduce(input, group=_tp_group)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class IdentityFoward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        global _tp_group
        torch.distributed.all_reduce(grad_output, group=_tp_group)
        return grad_output


class DummyModelEmbed(torch.nn.Module):

    def __init__(self, M: int, N: int, E: int, tp_group = None, pp_group = None):
        super().__init__()
        self.M = M
        self.N = N
        self.E = E

        self.tp_group = tp_group
        self.tp_size = torch.distributed.get_world_size(tp_group)
        self.pp_group = pp_group

        shard_id = torch.distributed.get_rank(tp_group)
        self.vocab_start_index = N // self.tp_size * shard_id
        self.vocab_end_index = N // self.tp_size * (shard_id + 1)
        self.embed_weight = torch.nn.Parameter(torch.ones((N // self.tp_size, E)))

    def input_shape(self):
        return (self.M, )

    def input_dtype(self):
        return torch.int64

    def output_shape(self):
        return (self.M, self.E)

    def output_dtype(self):
        return torch.float32

    def forward(self, input: torch.Tensor):
        if self.tp_size > 1:
            mask = (input < self.vocab_start_index) | \
                        (input >= self.vocab_end_index)
            input = input.clone() - self.vocab_start_index
            input[mask] = 0
            input = F.embedding(input, self.embed_weight)
            input[mask, :] = 0.0
            input = ReduceEmbed.apply(input)
        else:
            input = F.embedding(input, self.embed_weight)
        return input


class DummyModel(torch.nn.Module):

    def __init__(self, M: int, N: int, E: int):

        super().__init__()
        self.M = M
        self.N = N
        self.E = E

        # group
        global _tp_group
        self.tp_group = _tp_group
        global _pp_group
        self.pp_group = _pp_group

        self.pp_stage = torch.distributed.get_rank(_pp_group)
        self.is_first_pp_stage = self.pp_stage == 0
        self.is_last_stage = self.pp_stage == torch.distributed.get_world_size(_pp_group) - 1
    
        self.tp_size = torch.distributed.get_world_size(_tp_group)

        # mebed module
        if self.is_first_pp_stage:
            self.embed = DummyModelEmbed(M, N, E, self.tp_group, self.pp_group)
        else:
            self.embed = None

        total_fc_num = torch.distributed.get_world_size()
        fc_weights = list()
        input_shapes = list()
        output_shapes = list()
        shard_types = list()
        for idx in range(total_fc_num):
            if idx % 2 == 0:
                fc_weights.append(
                    torch.nn.Parameter(torch.ones((E // self.tp_size, E)) / 10000)
                )
                input_shapes.append((M, E))
                output_shapes.append((M, E // self.tp_size))
                shard_types.append('col')
            else:
                fc_weights.append(
                    torch.nn.Parameter(torch.ones((E, E // self.tp_size)) / 10000)
                )
                input_shapes.append((M, E // self.tp_size))
                output_shapes.append((M, E))
                shard_types.append('row')
        
        self.fc_num = total_fc_num // torch.distributed.get_world_size(_pp_group)
        self.fc_weights = torch.nn.ParameterList(
            fc_weights[self.fc_num * self.pp_stage : self.fc_num * (self.pp_stage + 1)]
        )
        self.ins = input_shapes[self.fc_num * self.pp_stage : self.fc_num * (self.pp_stage + 1)]
        self.ous = output_shapes[self.fc_num * self.pp_stage : self.fc_num * (self.pp_stage + 1)]
        self.shard_types = shard_types[self.fc_num * self.pp_stage : self.fc_num * (self.pp_stage + 1)]
        print_each_rank(f'initializing with {self.fc_num} fcs: {self.shard_types}')


    def input_shape(self):
        if self.embed:
            return self.embed.input_shape()
        else:
            return self.ins[0]

    def output_shape(self):
        if self.is_last_stage:
            return (1,)
        else:
            return self.ous[-1]

    def input_dtype(self):
        if self.embed:
            return self.embed.input_dtype()
        else:
            return torch.float32

    def output_dtype(self):
        return torch.float32

    def forward(self, input: torch.Tensor):
        # print(f'[{DeviceGroup().rank}] input: {input}, shape={input.size()}')
        if self.embed:
            x = self.embed(input)
        else:
            x = input

        for stype, weight in zip(self.shard_types, self.fc_weights):
            # column partition
            if stype == 'col':
                x = IdentityFoward.apply(x)
                x = F.linear(x, weight)
            elif stype == 'row':
                x = F.linear(x, weight)
                x = ReduceEmbed.apply(x)
            else:
                assert False

        if self.is_last_stage:
            x = torch.sum(x)
            # print(x)
        # print(f'[{DeviceGroup().rank}] output: {output}, shape={output.size()}')
        return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='swin')
    parser.add_argument('--tp-size', type=int,
                        help='tensor parallelism size')
    parser.add_argument('--pp-size', type=int,
                        help='pipeline parallelism size')
    parser.add_argument('--nmb', type=int, default=4,
                        help='num of micro batch')
    parser.add_argument('--M', type=int, default=4096,
                        help='M dimension length = sequence length')
    parser.add_argument('--N', type=int, default=50257,
                        help='word number')
    parser.add_argument('--E', type=int, default=2048,
                        help='E dimension length = hidden dimension length')
    args = parser.parse_args()

    print(args)

    cube.init()
    rank = DeviceGroup().rank
    pp_ranks, tp_ranks = DeviceGroup().create_hybrid([args.pp_size, args.tp_size])

    if _pp_group is None:
        _pp_group = DeviceGroup().get_group(pp_ranks)
        idx = pp_ranks.index(DeviceGroup().rank)
        _pp_next_rank = pp_ranks[(idx+1) % len(pp_ranks)]
        _pp_prev_rank = pp_ranks[(idx-1) % len(pp_ranks)]
    if _tp_group is None:
        _tp_group = DeviceGroup().get_group(tp_ranks)


    model = DummyModel(args.M, args.N, args.E).cuda()
    optimizer = torch.optim.Adam(model.parameters())

    # 0.11GB
    print_each_rank('model consumption')
    memory_summary()

    dataloader = SynTextDataLoader(
        shapes=([args.M],),
        dtypes=(torch.int64, ),
        batch_dims=(0,),
        length=128000
    )

    # 0.11GB
    print_each_rank('model + dataloader consumption')
    memory_summary()

    iter_num = 64
    CudaTimer(enable=False).warmup()
    for step in range(iter_num):
        if step >= 20:
            CudaTimer(enable=True).start('e2e')

        schedule_naive(model, dataloader, args.nmb, [_pp_prev_rank, _pp_next_rank])
        optimizer.step()
        optimizer.zero_grad()

        if step >= 20:
            CudaTimer().stop('e2e')
        if (step+1) % 10 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
    
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-20, field_name='e2e')))
    memory_summary()
