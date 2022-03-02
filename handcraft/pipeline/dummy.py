"""
Dummy model

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    handcraft/pipeline/dummy.py
"""
import torch
import torch.nn.functional as F
import cube
from cube.runtime.device import DeviceGroup
from cube.runtime.syndata import SynDataLoader

from handcraft.pipeline.schedule import schedule_tp_1f1b, schedule_naive


class DummyModel(torch.nn.Module):

    def __init__(self, dim: int, bs: int, stage_id: int, sharding=False):

        super().__init__()
        self.bs = bs
        self.dim = dim
        self.is_last_stage = stage_id == DeviceGroup().world_size
        if sharding:
            chunk_num = torch.distributed.get_world_size()
            self.weight = torch.nn.Parameter(torch.zeros((dim // chunk_num, dim)))
        else:
            self.weight = torch.nn.Parameter(torch.zeros((dim, dim)))

    def input_shape(self):
        return (self.bs, self.dim, self.dim)

    def input_dtype(self):
        return torch.float32

    def forward(self, input):
        output = F.linear(input, self.weight)
        if self.is_last_stage:
            output = torch.sum(output)
        return output
        


if __name__ == '__main__':

    cube.init()
    rank = DeviceGroup().rank

    dim = 1024
    gbs = 32
    mbs = 8

    # tp 1f1b
    first_stage_model = DummyModel(dim, mbs, 0, sharding=True).cuda()
    if rank == 0:
        model = None
    else:
        model = DummyModel(dim, mbs, rank, sharding=False).cuda()

    # naive pipleline
    # model = DummyModel(dim, mbs, sharding=False).cuda()

    dataloader = SynDataLoader(
        shapes=([mbs, dim, dim],),
        dtypes=(torch.float32, ),
        batch_dims=(0,)
    )

    for step in range(128):
        # schedule_naive(model, dataloader, gbs // mbs)
        schedule_tp_1f1b(model, first_stage_model, dataloader, gbs // mbs)
        if (step+1) % 10 == 0:
            print(f'iteration: {step+1}/128')
    