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

        self.layer_norm = nn.LayerNorm(last_dim) #TODO CHECK torch.fx ignores LayerNorm
        # self.p = 0.5
        self.drop_out = nn.Dropout()
        # self.y = torch.nn.Parameter(torch.empty(128, last_dim))

    def forward(self, data, mask):
        x = data.masked_fill(mask, 0.0)
        y = torch._shape_as_tensor(x)
        z = torch.gt(x, x)
        x = x.fill_(0.0)
        x = torch.nn.functional.softmax(x, dim=-1)
        x = torch.bmm(x, x)
        adder = torch.sum(x, dim=2, keepdim=True)
        x = torch.baddbmm(adder, batch1=x, batch2=x, alpha=0.125, beta=1.0)
        x = torch.tanh(x)
        x = torch.pow(x, x)
        for layer in self.layers:
            x = layer(x)
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.gelu(x)
        x = self.layer_norm(x)
        type_x = torch.pow(x, 1.0)
        x = x.type_as(type_x)
        x = x.unsqueeze(1)
        x = self.drop_out(x)
        x = x.squeeze()
        x = torch.triu(x, 1)
        x = torch.nan_to_num(x)
        ne_var = x.detach()
        ne_var = torch.ne(ne_var, 1.0)
        eq_var = x.detach()
        eq_var = torch.eq(eq_var, 1.0)
        long_var = x.detach()
        long_var = long_var.long()
        floor_div_var = x.detach()
        floor_div_var = torch.floor_divide(floor_div_var, 2.0)
        x = torch.true_divide(x, 1.0)
        x = torch.cumsum(x, dim=-1)
        x = x.permute(0, 2, 1)
        x = x.transpose(1, 2)
        x = torch.div(x, 1.0)
        # concrete_trace not support
        # x = torch.Tensor.view(x, [32 * 1024, 1024])
        x = x.view(32 * 1024, 1024)
        x = x.reshape(32, 1024, 1024)
        neg_x = torch.neg(x)
        x = torch.einsum('a b c, a c d -> a b d', x, neg_x)
        # TODO(yizhu1): uncomment and check
        # bs = x.size(1)
        # indices = torch.arange(bs, dtype=torch.int64)
        # x = torch.index_select(x, 1, indices)
        p = torch.div(x, 2.0)
        x = torch.stack((x, p), dim=1)
        x = torch.flatten(x, 2, 3)
        x = x.repeat(1, 2, 1)
        loss = torch.sum(x)
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
    model = cube.SemanticModel(model)

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