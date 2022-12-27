"""
OMP_NUM_THREADS=4 torchrun \
    --nproc_per_node=4 \
    tests/adapter/test_rvd.py
"""
from typing import List
import cube
from cube.graph.graph import IRGraph
from cube.ir.operator import IRFwOperation, IRDataOperation
import torch

cube.init()


class TestModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.empty(1024, 1024))

    def forward(self, x):
        x = self.param * x
        residual = x
        x = x * 2
        x = x + residual
        x = torch.sum(x)
        return x


def _tp(graph, node: IRFwOperation, idx, dim, devs: List[int]):
    algo = node.algorithms('dim')
    nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
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
            if node.name == 'mul':
                _tp(graph, node, idx=0, dim=0, devs=devs)
            elif node.name == 'add':
                _tp(graph, node, idx=0, dim=1, devs=devs)
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


if __name__ == '__main__':

    test_multiref_intra_rvd()
