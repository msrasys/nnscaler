# OMP_NUM_THREADS=12 USE_TORCHFX=1 PYTHONPATH=.:..:$PYTHONPATH python -m torch.distributed.launch --nproc_per_node=1 --master_port=25648 tests/parser/test_torchscale_basic.py --policy PASData

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

class SimpleNLP(nn.Module):
    def __init__(self):
        super().__init__()
        self._tensor_constant0 = 1
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, src_tokens, num):
        _shape_as_tensor = torch._shape_as_tensor(src_tokens)
        getitem_1 = _shape_as_tensor[1]
        add = 2 + getitem_1
        arange = torch.arange(add, dtype=torch.float32)
        unsqueeze = arange.unsqueeze(1)
        _tensor_constant0 = self._tensor_constant0
        mul = unsqueeze * _tensor_constant0
        sin = torch.sin(mul)
        cos = torch.cos(mul)
        cat = torch.cat([sin, cos], dim=1)
        view = cat.view(add, -1)
        linear = self.linear(view)
        return linear

def run():
    dataloader = cube.runtime.syndata.SynDataLoader(
        shapes=([4, 16], [2],),
        dtypes=(torch.int64, torch.int64,),
        batch_dims=(0, 0,)
    )

    sample_input = next(dataloader)
    print(f'next(dataloader) = {sample_input}')

    model = SimpleNLP()
    output = model(*sample_input)
    print(f'output = {output}')

    device = next(model.parameters()).device
    sample_input = next(dataloader)
    sample_input_cpu = tuple([input.to(device) for input in sample_input])
    model = cube.SemanticModel(
        model, dummy_input=sample_input_cpu,
    )

    # @cube.compile(model, dataloader, PAS=PAS, load_content=False)
    def train_iter(model, dataloader):
        data = next(dataloader)
        out = model(*data)
        return out

    train_iter(model, dataloader)

run()