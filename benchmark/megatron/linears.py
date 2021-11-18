"""
example:

python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=8004 \
    --use_env \
    benchmark/megatron/linears.py
"""

import argparse

import torch
from torch import nn
from benchmark.megatron.layers import ColumnParallelLinear, RowParallelLinear

import cube
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank


class ColumnMLP(nn.Module):
    def __init__(self, dim, mult=16):
        super().__init__()
        self.linear1 = ColumnParallelLinear(dim, dim * mult, full_input=True, full_output=True)
        self.linear2 = ColumnParallelLinear(dim * mult, dim, full_input=True, full_output=True)
        self.linear3 = ColumnParallelLinear(dim, dim * mult, full_input=True, full_output=True)
        self.linear4 = ColumnParallelLinear(dim * mult, dim, full_input=True, full_output=True)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        loss = torch.sum(output)
        return loss


class RowMLP(nn.Module):
    def __init__(self, dim, mult=16):
        super().__init__()
        self.linear1 = RowParallelLinear(dim, dim * mult, full_input=True, full_output=True)
        self.linear2 = RowParallelLinear(dim * mult, dim, full_input=True, full_output=True)
        self.linear3 = RowParallelLinear(dim, dim * mult, full_input=True, full_output=True)
        self.linear4 = RowParallelLinear(dim * mult, dim, full_input=True, full_output=True)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        loss = torch.sum(output)
        return loss


class HybridMLP(nn.Module):
    def __init__(self, dim, mult=16):
        super().__init__()
        self.linear1 = ColumnParallelLinear(dim, dim * mult, full_input=True, full_output=False)
        self.linear2 = RowParallelLinear(dim * mult, dim, full_input=False, full_output=True)
        self.linear3 = ColumnParallelLinear(dim, dim * mult, full_input=True, full_output=False)
        self.linear4 = RowParallelLinear(dim * mult, dim, full_input=False, full_output=True)

    def forward(self, data):
        output = self.linear1(data)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        loss = torch.sum(output)
        return loss


def train(args):

    batch_size = 128
    dim = 1024

    # model = ColumnMLP(dim=dim).cuda()
    # model = RowMLP(dim=dim).cuda()
    model = HybridMLP(dim=dim).cuda()

    for param in model.parameters():
        torch.nn.init.uniform_(param)

    dataloader = cube.runtime.syndata.SynDataLoader(1280, [batch_size, dim])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    def train_iter(model, dataloader):
        data = next(dataloader)
        # torch.distributed.broadcast(data, 0)
        # torch.cuda.synchronize()
        loss = model(data)
        loss.backward()

    CudaTimer().warmup(seconds=1.0)
    torch.distributed.barrier()
    iter_num = 128
    for step in range(iter_num):
        if step >= 40:
            CudaTimer().start('e2e')
        train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        if step >= 40:
            CudaTimer().stop('e2e')
        if (step + 1) % 20 == 0:
            print_each_rank(f'iter [{step + 1}/{iter_num}]', rank_only=0)
    
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-40, field_name='e2e')))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='inspect')
    parser.add_argument('--bs', type=int, default=128)
    args = parser.parse_args()

    cube.init()
    train(args)
