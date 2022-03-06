"""
Dummy model

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    handcraft/pipeline/dummy.py --use-naive

OMP_NUM_THREADS=4 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    handcraft/pipeline/dummy.py --use-naive
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

class ReduceEmbed(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        torch.distributed.all_reduce(input)
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
        torch.distributed.all_reduce(grad_output)
        return grad_output


class DummyModelEmbed(torch.nn.Module):

    def __init__(self, M: int, N: int, E: int, stage_id: int, sharding=False):
        super().__init__()
        self.M = M
        self.N = N
        self.E = E
        self.sharding = sharding
        chunk_num = torch.distributed.get_world_size() if sharding else 1
        self.vocab_start_index = N // chunk_num * stage_id
        self.vocab_end_index = N // chunk_num * (stage_id + 1)
        self.embed_weight = torch.nn.Parameter(torch.zeros((N // chunk_num, E)))

    def input_shape(self):
        return (self.M, )

    def input_dtype(self):
        return torch.int64

    def output_shape(self):
        return (self.M, self.E)

    def output_dtype(self):
        return torch.float32

    def forward(self, input: torch.Tensor):
        if self.sharding:
            mask = (input < self.vocab_start_index) | \
                        (input >= self.vocab_end_index)
            input = input.clone() - self.vocab_start_index
            input[mask] = 0
            input = F.embedding(input, self.embed_weight)
            input = ReduceEmbed.apply(input)
        else:
            input = F.embedding(input, self.embed_weight)
        return input


class DummyModel(torch.nn.Module):

    def __init__(self, M: int, N: int, E: int, stage_id: int,
                 sharding=False, embed: torch.nn.Module = None):

        super().__init__()
        self.M = M
        self.N = N
        self.E = E
        self.is_last_stage = stage_id == DeviceGroup().world_size - 1
        self.sharding = sharding
        # mebed module
        self.embed = embed
        # first stage
        chunk_num = torch.distributed.get_world_size() if sharding else 1
        self.fc_weight = torch.nn.Parameter(torch.zeros((E // chunk_num, E)))

    def input_shape(self):
        if self.embed:
            return self.embed.input_shape()
        else:
            return (self.M, self.E)

    def output_shape(self):
        if self.is_last_stage:
            return (1,)
        else:
            return (self.M, self.E)

    def input_dtype(self):
        if self.embed:
            return self.embed.input_dtype()
        else:
            return torch.float32

    def output_dtype(self):
        return torch.float32

    def forward(self, input: torch.Tensor):
        if self.embed:
            input = self.embed(input)

        if self.sharding:
            input = IdentityFoward.apply(input)
        output = F.linear(input, self.fc_weight)

        if self.is_last_stage:
            output = torch.sum(output)
        return output


class DummyModelTP(torch.nn.Module):

    def __init__(self, M: int, N: int, E: int, stage_id: int):
        super().__init__()
        self.M = M
        self.N = N
        self.E = E
        self.stages = DeviceGroup().world_size

        self.vocab_start_index = N // self.stages * stage_id
        self.vocab_end_index = N // self.stages * (stage_id + 1)
        self.embed_weight = torch.nn.Parameter(torch.zeros((N // self.stages, E)))
        self.fc_weights = torch.nn.ParameterList()
        for idx in range(self.stages):
            if idx % 2 == 0:
                self.fc_weights.append(
                    torch.nn.Parameter(torch.zeros((E // self.stages, E)))
                )
            else:
                self.fc_weights.append(
                    torch.nn.Parameter(torch.zeros((E, E // self.stages)))
                )

    def forward(self, input: torch.Tensor):
        mask = (input < self.vocab_start_index) | \
                        (input >= self.vocab_end_index)
        input = input.clone() - self.vocab_start_index
        input[mask] = 0
        input = F.embedding(input, self.embed_weight)
        x = ReduceEmbed.apply(input)
        for idx in range(self.stages):
            # column partition
            if idx % 2 == 0:
                x = IdentityFoward.apply(x)
                x = F.linear(x, self.fc_weights[idx])
            else:
                x = ReduceEmbed.apply(x)
                x = F.linear(x, self.fc_weights[idx])
        # reduce
        if self.stages % 2 == 0:
            x = ReduceEmbed.apply(x)
        else:
            raise RuntimeError("number of stages only supported to be mod 2 == 0")
        return torch.sum(x)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='swin')
    parser.add_argument('--use-naive', action='store_true',
                        help='use naive pipeline')
    parser.add_argument('--use-tp1f1b', action='store_true',
                        help='use tensor parallel 1f1b')
    parser.add_argument('--use-tp1f1b-pack', action='store_true',
                        help='use tensor parallel 1f1b')
    parser.add_argument('--use-tp', action='store_true',
                        help='use pure tensor parallelism')
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

    # tp 1f1b
    if args.use_tp1f1b:
        embed = DummyModelEmbed(args.M, args.N, args.E, 0, sharding=True).cuda()
        first_stage_model = DummyModel(args.M, args.N, args.E, 0, sharding=True, embed=embed).cuda()
        if rank == 0:
            model = None
        else:
            model = DummyModel(args.M, args.N, args.E, rank, sharding=False).cuda()

    if args.use_naive:
        # naive pipleline
        embed = None
        if rank == 0:
            embed = DummyModelEmbed(args.M, args.N, args.E, 0, sharding=False).cuda()
        model = DummyModel(args.M, args.N, args.E, rank, sharding=False, embed=embed).cuda()

    if args.use_tp1f1b_pack:
        embed = DummyModelEmbed(args.M, args.N, args.E, 0, sharding=True).cuda()
        model = DummyModel(args.M, args.N, args.E, rank, sharding=False).cuda()

    if args.use_tp:
        model = DummyModelTP(args.M, args.N, args.E, rank).cuda()

    dataloader = SynTextDataLoader(
        shapes=([args.M],),
        dtypes=(torch.int64, ),
        batch_dims=(0,),
        length=128000
    )

    iter_num = 64
    CudaTimer(enable=False).warmup()
    for step in range(iter_num):
        if step >= 20:
            CudaTimer(enable=True).start('e2e')

        if args.use_tp1f1b:
            schedule_tp_1f1b(model, first_stage_model, dataloader, args.nmb, DeviceGroup().world_size)
        if args.use_naive:
            schedule_naive(model, dataloader, args.nmb)
        if args.use_tp1f1b_pack:
            schedule_tp_1f1b_pack(model, embed, dataloader, args.nmb, DeviceGroup().world_size)
        if args.use_tp:
            for _ in range(args.nmb):
                data = next(dataloader)
                loss = model(data)
                loss.backward()

        if step >= 20:
            CudaTimer().stop('e2e')
        if (step+1) % 10 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
    
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-20, field_name='e2e')))
    memory_summary()
