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

from handcraft.pipeline.schedule import schedule_tp_1f1b, schedule_naive
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



class DummyModel(torch.nn.Module):

    def __init__(self, M: int, N: int, E: int, stage_id: int, sharding=False):

        super().__init__()
        self.M = M
        self.N = N
        self.E = E
        self.is_last_stage = stage_id == DeviceGroup().world_size - 1
        self.is_first_stage = stage_id == 0
        self.sharding = sharding
        # first stage
        chunk_num = torch.distributed.get_world_size() if sharding else 1
        if self.is_first_stage:
            self.vocab_start_index = N // chunk_num * stage_id
            self.vocab_end_index = N // chunk_num * (stage_id + 1)
            self.embed_weight = torch.nn.Parameter(torch.zeros((N // chunk_num, E)))
            self.fc_weight = torch.nn.Parameter(torch.zeros((E // chunk_num, E)))
        else:
            self.fc_weight = torch.nn.Parameter(torch.zeros((E // chunk_num, E)))

    def input_shape(self):
        if self.is_first_stage:
            return (self.M,)
        else:
            return (self.M, self.E)

    def output_shape(self):
        if self.is_last_stage:
            return (1,)
        else:
            return (self.M, self.E)

    def input_dtype(self):
        if self.is_first_stage:
            return torch.int64
        else:
            return torch.float32

    def output_dtype(self):
        return torch.float32

    def forward(self, input: torch.Tensor):
        if self.is_first_stage:
            if self.sharding:
                mask = (input < self.vocab_start_index) | \
                        (input >= self.vocab_end_index)
                input = input.clone() - self.vocab_start_index
                input[mask] = 0
                input = F.embedding(input, self.embed_weight)
                input = ReduceEmbed.apply(input)
            else:
                input = F.embedding(input, self.embed_weight)

        if self.sharding:
            input = IdentityFoward.apply(input)
        output = F.linear(input, self.fc_weight)

        if self.is_last_stage:
            output = torch.sum(output)
        return output


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='swin')
    parser.add_argument('--use-naive', action='store_true',
                        help='use naive pipeline')
    parser.add_argument('--use-tp1f1b', action='store_true',
                        help='use tensor parallel 1f1b')
    parser.add_argument('--nmb', type=int, default=4,
                        help='num of micro batch')
    parser.add_argument('--M', type=int, default=4096,
                        help='M dimension length = sequence length')
    parser.add_argument('--N', type=int, default=50257,
                        help='word number')
    parser.add_argument('--E', type=int, default=2048,
                        help='E dimension length = hidden dimension length')
    args = parser.parse_args()
    assert args.use_naive ^ args.use_tp1f1b, "Specify (only) 1 way pipeline"
    print(args)

    cube.init()
    rank = DeviceGroup().rank

    # tp 1f1b
    if args.use_tp1f1b:
        first_stage_model = DummyModel(args.M, args.N, args.E, 0, sharding=True).cuda()
        if rank == 0:
            model = None
        else:
            model = DummyModel(args.M, args.N, args.E, rank, sharding=False).cuda()

    if args.use_naive:
        # naive pipleline
        model = DummyModel(args.M, args.N, args.E, rank, sharding=False).cuda()

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

        if step >= 20:
            CudaTimer().stop('e2e')
        if (step+1) % 10 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
    
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-20, field_name='e2e')))
    memory_summary()
