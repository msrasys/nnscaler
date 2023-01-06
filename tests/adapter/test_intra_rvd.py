"""
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    tests/adapter/test_intra_rvd.py
"""
from typing import List, Tuple
import cube
from cube.ir.operator import IRFwOperation, IRDataOperation
from cube.ir.tensor import IRFullTensor
from cube.graph.graph import IRGraph
from cube.graph.gener.layout import GridLayout
from cube.graph.function.dimops import IRDimops
from cube.algorithm.generics import GenericDistAlgo
import torch
import numpy as np

cube.init()


class RVDSplit(GenericDistAlgo):

    def __init__(self, node: IRDimops):
        super().__init__(node)

    def satisfy(self, in_rvd: Tuple[int], out_rvd: Tuple[int]):
        return True
    
    def instantiate(self, in_rvd: Tuple[int], out_rvd: Tuple[int]) -> List[IRFwOperation]:
        assert np.prod(np.array(in_rvd, dtype=int)) == np.prod(np.array(out_rvd, dtype=int)), \
            f"tensor number not match: {in_rvd}, {out_rvd}"
        assert tuple(in_rvd)[2:] == tuple(out_rvd)[2:], f"input /  output shape should be same"
        
        node: IRDimops = self.node
        iftensor: IRFullTensor = node.input(0).parent
        itensors = GridLayout.grid(iftensor, r=in_rvd[0], v=in_rvd[1], dims=in_rvd[2:]).mat.flatten()
        oftensor: IRFullTensor = node.output(0).parent
        otensors = GridLayout.grid(oftensor, r=out_rvd[0], v=out_rvd[1], dims=out_rvd[2:]).mat.flatten()
        subnodes = []
        for itensor, otensor in zip(itensors, otensors):
            subnode = node.new([itensor, 2], [otensor])
            subnodes.append(subnode)
        return subnodes
    

class TestModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(1024, 1024))

    def forward(self, x):
        x = torch.matmul(x, self.param)
        # residual = x
        x = x * 2
        x = x * 2
        x = x * 2
        x = x * 2
        x = x * 2
        x = x * 2
        x = x * 2
        # x = x + residual
        x = torch.sum(x)
        return x


def _ntp(graph, node: IRDimops, idx: int, dim: int, devs: List[int]):
    algo = node.algorithms('dim')
    nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
    assert nodes is not None
    for devid, node in zip(devs, nodes):
        graph.assign(node, devid)
    return nodes


def _tp(graph, node: IRFwOperation, in_rvd: Tuple[int], out_rvd: Tuple[int], devs: List[int]):
    algo = RVDSplit(node)
    nodes = graph.partition(node, algo, in_rvd=in_rvd, out_rvd=out_rvd)
    assert nodes is not None
    for devid, node in zip(devs, nodes):
        graph.assign(node, devid)
    return nodes


def _replica(graph: IRGraph, node: IRFwOperation, devs: List[int]):
    nodes = graph.replicate(node, times=len(devs))
    assert nodes is not None
    for devid, node in zip(devs, nodes):
        graph.assign(node, devid)
    return nodes
    

def test_multiref_intra_rvd():

    model =  TestModel()
    dataloader = cube.runtime.syndata.SynDataLoader(
        shapes=([1024,1024],),
        dtypes=(torch.float32,),
        batch_dims=(0,)
    )

    def policy(graph: IRGraph, resource):
        print(graph.extra_repr())
        devs = list(range(resource.ngpus))

        for ftensor in graph.full_tensors():
            if len(graph.consumers(ftensor)) > 1:
                graph.multiref(ftensor, [[n] for n in graph.consumers(ftensor)])

        for dl in graph.select(ntype=IRDataOperation):
            _replica(graph, dl, devs)

        for node in graph.select(ntype=IRFwOperation):
            if node.name == 'multiref': continue
            if node.name == 'mul':
                _ntp(graph, node, idx=0, dim=0, devs=devs)
            elif node.name == 'add':
                _ntp(graph, node, idx=0, dim=1, devs=devs)
            else:
                _replica(graph, node, devs)
        print(graph.extra_repr())
        return graph

    model = cube.SemanticModel(model)
    @cube.compile(model, dataloader, PAS=policy)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
    model = model.get_gen_module()

    for _ in range(4):
        train_iter(model, dataloader)


def test_intra_rvd():

    model =  TestModel()
    dataloader = cube.runtime.syndata.SynDataLoader(
        shapes=([1024,1024],),
        dtypes=(torch.float32,),
        batch_dims=(0,)
    )

    def policy(graph: IRGraph, resource):
        assert resource.ngpus == 4
        print(graph.extra_repr())
        devs = list(range(resource.ngpus))

        # for ftensor in graph.full_tensors():
        #     if len(graph.consumers(ftensor)) > 1:
        #         graph.multiref(ftensor, [[n] for n in graph.consumers(ftensor)])

        for dl in graph.select(ntype=IRDataOperation):
            _replica(graph, dl, devs)

        for idx, node in enumerate(graph.select(name='mul')):
            if idx == 0:  # out: R(4)V(1)D(1,1) -> in: R(1)V(1)D(4,1): schunk
                _tp(graph, node, in_rvd=(1,1,4,1), out_rvd=(1,1,4,1), devs=devs)
            elif idx == 1:  # out: R(1)V(1)D(4,1) -> in: R(1)V(1)D(1,4): all-to-all wil FAIL. expected!!
                _tp(graph, node, in_rvd=(1,1,1,4), out_rvd=(1,1,1,4), devs=devs)
            elif idx == 2:  # out: R(1)V(1)D(1,4) -> in: R(1)V(1)D(2,2): schunk
                _tp(graph, node, in_rvd=(1,1,2,2), out_rvd=(1,1,2,2), devs=devs)
            elif idx == 3:  # out: R(1)V(1)D(2,2) -> in: R(4)V(1)D(1,1): all-gather + all-gather
                _tp(graph, node, in_rvd=(4,1,1,1), out_rvd=(1,4,1,1), devs=devs)
            elif idx == 4:  # out: R(1)V(4)D(1,1) -> in: R(1)V(1)D(4,1): reduce-scatter
                _tp(graph, node, in_rvd=(1,1,4,1), out_rvd=(1,1,4,1), devs=devs)
            elif idx == 5:  # out: R(1)V(1)D(4,1) -> in R(4)V(1)D(1,1): all-gather
                _tp(graph, node, in_rvd=(4,1,1,1), out_rvd=(1,4,1,1), devs=devs)
            elif idx == 6:  # out: R(1)V(4)D(1,1) -> in R(1)V(1)D(2,2): reduce-scatter + reduce-scatter
                _tp(graph, node, in_rvd=(1,1,2,2), out_rvd=(1,1,2,2), devs=devs)
            else:
                assert False

        for node in graph.select(ntype=IRFwOperation):
            if len(node.device) == 0:
                _replica(graph, node, devs)

        print(graph.extra_repr())
        return graph
    
    model = cube.SemanticModel(model)
    @cube.compile(model, dataloader, PAS=policy)
    def train_iter(model, dataloader):
        data = next(dataloader)
        loss = model(data)
        loss.backward()
    model = model.get_gen_module()

    for _ in range(4):
        train_iter(model, dataloader)



if __name__ == '__main__':

    # test_multiref_intra_rvd()
    test_intra_rvd()
