"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py
"""

import torch
from torch import nn

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
from examples.mlp.policy.spmd import PASMegatron as PAS

# =================== Semantic Model Description ====================

class MLP(nn.Module):
    def __init__(self, dim, mult=1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.linear3 = nn.Linear(dim, dim * mult)
        self.linear4 = nn.Linear(dim * mult, dim)
        # self.linear5 = nn.Linear(dim, dim * mult)
        # self.linear6 = nn.Linear(dim * mult, dim)
        # self.linear7 = nn.Linear(dim, dim * mult)
        # self.linear8 = nn.Linear(dim * mult, dim)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        # output = self.linear5(output)
        # output = self.linear6(output)
        # output = self.linear7(output)
        # output = self.linear8(output)
        loss = torch.sum(output)
        return loss


def train():
    batch_size = 256
    dim = 8192

    model = MLP(dim=dim)
    model = cube.SemanticModel(
        model, input_shapes=([batch_size, dim],),
    )

    dataloader = cube.runtime.syndata.SynDataLoader(
        shapes=([batch_size, dim],),
        dtypes=(torch.float32,),
        batch_dims=(0,)
    )

    @cube.compile(model, dataloader, PAS=PAS)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
    model = model.get_gen_module()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    CudaTimer(enable=False).warmup()
    torch.distributed.barrier()
    iter_num = 64
    warmup = 20
    for step in range(iter_num):
        if step >= warmup:
            CudaTimer(enable=True).start('e2e')
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        if step >= warmup:
            CudaTimer().stop('e2e')
        if (step + 1) % 20 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)


if __name__ == '__main__':

    cube.init()
    train()