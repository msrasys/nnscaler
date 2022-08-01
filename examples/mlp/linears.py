"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/linears.py --policy PASMegatron
"""

import torch
from torch import nn

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank

import examples.mlp.policy.spmd as spmd
import examples.mlp.policy.mpmd as mpmd

import argparse

parser = argparse.ArgumentParser(description='comm primitive')
parser.add_argument('--policy', type=str, help='PAS policy choice, starting with "PAS"')
args = parser.parse_args()

cube.init()

# set up policy
PAS = None
policies = list(spmd.__dict__.keys()) + list(mpmd.__dict__.keys())
if args.policy in spmd.__dict__:
    PAS = spmd.__dict__[args.policy]
    print_each_rank(f'using policy from spmd.{args.policy}')
elif args.policy in mpmd.__dict__:
    PAS = mpmd.__dict__[args.policy]
    print_each_rank(f'using policy from mpmd.{args.policy}')
else:
    raise ValueError(f"policy {args.policy} not found. Candidates: {policies}")


# =================== Semantic Model Description ====================

class MLP(nn.Module):
    def __init__(self, dim, mult=1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.linear3 = nn.Linear(dim, dim * mult)
        self.linear4 = nn.Linear(dim * mult, dim)
        self.linear5 = nn.Linear(dim, dim * mult)
        self.linear6 = nn.Linear(dim * mult, dim)
        self.linear7 = nn.Linear(dim, dim * mult)
        self.linear8 = nn.Linear(dim * mult, dim)
        self.linear9 = nn.Linear(dim, dim * mult)
        self.linear10 = nn.Linear(dim * mult, dim)
        self.linear11 = nn.Linear(dim, dim * mult)
        self.linear12 = nn.Linear(dim * mult, dim)
        self.linear13 = nn.Linear(dim, dim * mult)
        self.linear14 = nn.Linear(dim * mult, dim)
        self.linear15 = nn.Linear(dim, dim * mult)
        self.linear16 = nn.Linear(dim * mult, dim)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        output = self.linear5(output)
        output = self.linear6(output)
        output = self.linear7(output)
        output = self.linear8(output)
        output = self.linear9(output)
        output = self.linear10(output)
        output = self.linear11(output)
        output = self.linear12(output)
        output = self.linear13(output)
        output = self.linear14(output)
        output = self.linear15(output)
        output = self.linear16(output)
        loss = torch.sum(output)
        return loss


def train():
    batch_size = 128
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
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    iter_num = 500
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


train()