"""
example:

ASYNC_COMM=1 OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/mlp/infer.py --policy PASMegatron
"""

import torch
from torch import nn
import time

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank

import examples.mlp.policy.spmd as spmd
import examples.mlp.policy.mpmd as mpmd

import argparse

parser = argparse.ArgumentParser(description='MLP example')
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
    def __init__(self, dim, mult=1, nlayers=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for lid in range(nlayers):
            if lid % 2 == 0:
                self.layers.append(nn.Linear(dim, dim * mult, bias=False))
            else:
                self.layers.append(nn.Linear(dim * mult, dim, bias=False))

    def forward(self, data):
        x = data
        for layer in self.layers:
            x = layer(x)
        loss = torch.sum(x)
        return loss


def infer():
    batch_size = 128
    dim = 4096

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
    def infer_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
    model = model.get_gen_module()

    CudaTimer(enable=False).warmup()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    iter_num = 16
    warmup = 4
    for step in range(iter_num):
        if step >= warmup:
            CudaTimer(enable=True, predefined=True).start('e2e')
        infer_iter(model, dataloader)
        if step >= warmup:
            CudaTimer().stop('e2e')
        if (step + 1) % 20 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
        
        torch.distributed.barrier()

    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)


infer()