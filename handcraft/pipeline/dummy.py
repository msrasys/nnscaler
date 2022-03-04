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
from cube.runtime.device import DeviceGroup
from cube.runtime.syndata import SynDataLoader

from handcraft.pipeline.schedule import schedule_tp_1f1b, schedule_naive
import argparse


class DummyModel(torch.nn.Module):

    def __init__(self, dim: int, bs: int, stage_id: int, sharding=False):

        super().__init__()
        self.bs = bs
        self.dim = dim
        self.is_last_stage = stage_id == DeviceGroup().world_size - 1
        if sharding:
            chunk_num = torch.distributed.get_world_size()
            self.weight = torch.nn.Parameter(torch.zeros((dim // chunk_num, dim)))
        else:
            self.weight = torch.nn.Parameter(torch.zeros((dim, dim)))

    def input_shape(self):
        return (self.bs, self.dim, self.dim)

    def output_shape(self):
        return (1,) if self.is_last_stage else (self.bs, self.dim, self.dim)

    def input_dtype(self):
        return torch.float32

    def output_dtype(self):
        return torch.float32

    def forward(self, input):
        output = F.linear(input, self.weight)
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
    args = parser.parse_args()
    assert args.use_naive ^ args.use_tp1f1b, "Specify (only) 1 way pipeline"

    cube.init()
    rank = DeviceGroup().rank

    dim = 2048
    mbs = 8
    gbs = mbs * args.nmb

    # tp 1f1b
    if args.use_tp1f1b:
        first_stage_model = DummyModel(dim, mbs, 0, sharding=True).cuda()
        if rank == 0:
            model = None
        else:
            model = DummyModel(dim, mbs, rank, sharding=False).cuda()

    if args.use_naive:
        # naive pipleline
        model = DummyModel(dim, mbs, rank, sharding=False).cuda()

    dataloader = SynDataLoader(
        shapes=([mbs, dim, dim],),
        dtypes=(torch.float32, ),
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
