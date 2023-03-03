"""
example:

//torchscript based DAG capture
PYTHONPATH=.:..:$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=1  examples/mlp/linearsfx.py --policy PASData
//torch.fx based DAG capture
USE_TORCHFX=1 PYTHONPATH=.:..:$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=1  examples/mlp/linearsfx.py --policy PASData
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
parser.add_argument('--local_rank', type=int, default=0)
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
                last_dim = dim * mult
            else:
                self.layers.append(nn.Linear(dim * mult, dim, bias=False))
                last_dim = dim

        # self.layer_norm = nn.LayerNorm(last_dim) #TODO CHECK torch.fx ignores LayerNorm
        # self.p = 0.5
        self.drop_out = nn.Dropout()
        # self.y = torch.nn.Parameter(torch.empty(128, last_dim))

    def forward(self, data, mask):
        x = data.masked_fill(mask, 0.0)
        x = x.fill_(0.0)
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.relu(x)
        # x = self.layer_norm(x)
        x = x.type_as(data)
        x = x.unsqueeze(1)
        x = self.drop_out(x)
        x = x.squeeze()
        x = torch.triu(x, 1)
        x = torch.nan_to_num(x)
        # ne cannot backward
        # x = torch.ne(x, 1.0)
        # x = torch.nn.functional.dropout(x, self.p)
        # x = x * self.y
        x = torch.cumsum(x, -1)
        # y = torch.eq(x, 1.0)
        loss = torch.sum(x)
        # long cannot backward
        # loss = loss.long()
        return loss


def train():
    batch_size = 32
    dim = 1024

    dataloader = cube.runtime.syndata.SynDataLoader(
        shapes=([batch_size, dim, dim], [batch_size, dim, dim],),
        dtypes=(torch.float32, torch.bool,),
        batch_dims=(0, 0,)
    )

    model = MLP(dim=dim)

    # shape based input (will trigger standard torch.fx.Tracer)
    # model = cube.SemanticModel(
    #     model, input_shapes=([batch_size, dim, dim], [batch_size, dim, dim]),
    # )

    # dummy based input (will trigger concrete Tracer)
    device = next(model.parameters()).device
    dummy_input = next(dataloader)
    dummy_input = tuple([input.to(device) for input in dummy_input])
    model = cube.SemanticModel(
        model, dummy_input=dummy_input,
    )

    @cube.compile(model, dataloader, PAS=PAS, load_content=False)
    def train_iter(model, dataloader):
        data, mask = next(dataloader)
        loss = model(data, mask)
        loss.backward()

    model = model.get_gen_module()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    CudaTimer(enable=False).warmup()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    iter_num = 2 #32
    warmup = 0 #8
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
        CudaTimer().duration(iter_num - warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num - warmup)


train()