

"""
example:

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    tests/codegen/test_scale.py

OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    tests/codegen/test_scale.py
"""

from typing import List
import torch
from torch import nn

import cube
from cube.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.profiler import CudaTimer
from cube.profiler.timer import print_each_rank


cube.init()


class MLP(nn.Module):
    def __init__(self, dim, mult=1, nlayers=16):
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
        self.batch_size = batch_size
        self.sample = torch.rand(
            [batch_size, self.dim], dtype=torch.float32,
            device=torch.cuda.current_device()
        )


# tensor parallelism
def _tp(graph: IRGraph, node: IRFwOperation, devs: List[int],
        idx: int, dim: int, tag='dim'):
    algo = node.algorithms(tag)
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
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


def _run(train_iter, model, dataloader, optimizer):
    iter_num, warmup = 5, 2
    for step in range(iter_num):
        if step >= warmup:
            CudaTimer(enable=True).start('e2e')
        loss = train_iter(model, dataloader)
        optimizer.step()
        optimizer.zero_grad()
        # model.zero_grad()
        # model.gather_params()
        if step >= warmup:
            CudaTimer().stop('e2e')
        print_each_rank(f'loss: {loss.item()}', rank_only=0)
    print_each_rank('e2e time (ms) per iteration: {} ms'.format(
          CudaTimer().duration(iter_num-warmup, field_name='e2e')))
    CudaTimer().print_all(times=iter_num-warmup)


def test_scale_full_dp():

    model = MLP(dim=4096)
    dataloader = MLPDataLoader(bs=8, dim=4096)

    def policy(graph: IRGraph, resource):
        assert resource.ngpus > 2
        ngpus = 2
        for dl in graph.select(ntype=IRDataOperation):
            _replica(graph, dl, list(range(ngpus)))
        for node in graph.select(ntype=IRFwOperation):
            if node.name == 'linear':
                _tp(graph, node, list(range(ngpus)), idx=0, dim=0, tag='dim')
            else:
                _replica(graph, node, list(range(ngpus)))
        return graph

    @cube.compile(model, dataloader, PAS=policy, scale=True)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
        return loss
    model = cube.load_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _run(train_iter, model, dataloader, optimizer)


def test_scale_partial_dp():

    model = MLP(dim=4096)
    dataloader = MLPDataLoader(bs=8, dim=4096)

    def policy(graph: IRGraph, resource):
        assert resource.ngpus > 2
        ngpus = 2
        for dl in graph.select(ntype=IRDataOperation):
            _replica(graph, dl, list(range(ngpus)))
        for idx, node in enumerate(graph.select(ntype=IRFwOperation)):
            if node.name == 'linear':
                if idx % 4 == 0:
                    _tp(graph, node, list(range(ngpus)), idx=0, dim=0, tag='dim')
                if idx % 4 == 1:  # partition weight, partition input (reduction)
                    _tp(graph, node, list(range(ngpus)), idx=0, dim=1, tag='dim')
                if idx % 4 == 2: # partition weight, replicate input
                    _tp(graph, node, list(range(ngpus)), idx=1, dim=0, tag='dim')
                if idx % 4 == 3:  # replicate
                    _replica(graph, node, list(range(ngpus)))
            else:
                _replica(graph, node, list(range(ngpus)))
        return graph

    @cube.compile(model, dataloader, PAS=policy, scale=True)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
        return loss
    model = cube.load_model()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _run(train_iter, model, dataloader, optimizer)


def test_scale_no_dp():
    
    model = MLP(dim=4096)
    dataloader = MLPDataLoader(bs=8, dim=4096)

    def policy(graph: IRGraph, resource):
        assert resource.ngpus > 2
        ngpus = 2
        for dl in graph.select(ntype=IRDataOperation):
            _replica(graph, dl, list(range(ngpus)))
        for node in graph.select(ntype=IRFwOperation):
            _replica(graph, node, list(range(ngpus)))
        return graph
    
    @cube.compile(model, dataloader, PAS=policy, scale=True)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
        return loss
    model = cube.load_model()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    _run(train_iter, model, dataloader, optimizer)


if __name__ == '__main__':

    # test_scale_full_dp()
    # test_scale_partial_dp()
    test_scale_no_dp()