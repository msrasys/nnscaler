"""
example:

ASYNC_REDUCER=0 USE_ZERO=1 OMP_NUM_THREADS=4 torchrun --nproc_per_node=4 \
    tests/runtime/test_reducer.py
"""
from typing import List
from functools import partial

import torch
import random
from torch import nn

import cube
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank
# from cube.tools.debug import DebugTool

cube.init()


class MLP(nn.Module):
    def __init__(self, dim, nlayers=16):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(nn.Linear(dim, dim, bias=False))
        self.param = torch.nn.Parameter(torch.ones([1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x * self.param  # for padding test
        loss = torch.sum(x)
        return loss


class MLPDataLoader(cube.runtime.syndata.CubeDataLoader):

    def __init__(self, bs: int, dim: int):
        super().__init__(bs, [0])
        self.sample = None
        self.dim = dim
        self.set_batch_size(bs)

    def __iter__(self):
        return self

    def __next__(self):
        return self.sample
    
    def set_batch_size(self, batch_size: int):
        torch.random.manual_seed(0)
        self.batch_size = batch_size
        self.sample = torch.randn(
            [batch_size, self.dim], dtype=torch.float32,
            device=torch.cuda.current_device()
        )
        self.sample = (self.sample - 1) * 1e3


def init_model_dataloader():
    batch_size = 4
    dim = 4096
    torch.random.manual_seed(0)
    random.seed(0)
    model = MLP(dim=dim)
    # torch.random.manual_seed(0)
    dataloader = MLPDataLoader(batch_size, dim)
    return model, dataloader


def policy(graph: IRGraph, resource):

    # tensor parallelism
    def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int], **configs):
        algo = node.algorithms('dim')
        sub_nodes = graph.partition(node, algo, **configs)
        assert sub_nodes is not None
        for devid, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, devid)
        return sub_nodes

    # replicate
    def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
        sub_nodes = graph.replicate(node, times=len(devs))
        for devid, sub_node in zip(devs, sub_nodes):
            graph.assign(sub_node, devid)
        return sub_nodes
    
    devs = list(range(resource.ngpus))
    for node in graph.select(ntype=IRDataOperation):
        _replica(graph, node, devs)
    for node in graph.select(ntype=IRFwOperation):
        _tp(graph, node, devs, idx=0, dim=0, num=resource.ngpus)
        # if node.name == 'linear':
        #     _tp(graph, node, devs, idx=0, dim=0, num=resource.ngpus)
        # else:
        #     _replica(graph, node, devs)
    return graph


def get_baseline():

    model, dataloader = init_model_dataloader()
    model = model.cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
        return loss
    
    wsz = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    niters = 4
    losses = []
    for idx in range(niters):
        loss = train_iter(model, dataloader)
        # loss = DebugTool.record(
        #     model, 
        #     partial(train_iter, model, dataloader), 
        #     filename=f'base-{wsz}gpus-{rank}.iter{idx}.log'
        # )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    
    for idx, loss in enumerate(losses):
        print_each_rank(f'baseline loss[{idx}]: {loss}', rank_only=0)

    return losses


baseline_losses = get_baseline()


def test_reducer():

    # nonlocal baseline_losses

    model, dataloader = init_model_dataloader()

    @cube.compile(model, dataloader, PAS=policy)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
        return loss
    
    model = cube.load_model()
    # optimizer = torch.optim.Adam(model.parameters_for_optimizer(), lr=1e-4) # not match for adam
    optimizer = torch.optim.SGD(model.parameters_for_optimizer(), lr=1e-2)

    def post_hook(grad):
        grad.mul_(0.1)
    for reducer in model.reducers:
        reducer.register_post_hook(post_hook)

    wsz = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    niters = 4
    losses = []
    for idx in range(niters):
        loss = train_iter(model, dataloader)
        # loss = DebugTool.record(
        #     model, 
        #     partial(train_iter, model, dataloader), 
        #     filename=f'reducer-{wsz}gpus-{rank}.iter{idx}.log'
        # )
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()
        model.gather_params()
        losses.append(loss.item())

    for idx, loss in enumerate(losses):
        print_each_rank(f'reducer loss[{idx}]: {loss}', rank_only=0)


if __name__ == '__main__':

    test_reducer()