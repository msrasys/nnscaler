import pytest
from cube.graph.gener.gen import IRAdapterGener

from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.parser.converter import convert_model
from cube.ir.operator import IRFwOperation
from cube.ir.tensor import IRFullTensor
from cube.ir.adapter import IRWeightReducer

import torch
import tempfile


def make_param(shape, dtype) -> IRFullTensor:
    param = IRFullTensor(shape=shape, dtype=dtype, requires_grad=True)
    param.as_param()
    return param


class ReducerModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float16))
        self.param2 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float16))

    def forward(self, x):
        x = torch.matmul(x, self.param1)
        x = torch.matmul(x, self.param2)
        x = x + self.param1
        x = torch.sum(x)
        return x
    

def build_graph():
    # build graph
    model = ReducerModule()
    with tempfile.TemporaryDirectory() as tempdir:
        graph = convert_model(
            model,
            {'x': torch.randn([128, 128], dtype=torch.float16)},
            attr_savedir=tempdir,
            dynamic_shape=False
        )
    graph.backward(graph.output(0))
    return graph


def test_cross_segment_weight_reducer():

    graph = build_graph()
    [matmul1, matmul2, add, sum] = graph.select(ntype=IRFwOperation)
    graph.group([matmul1, matmul2])
    graph.group([add, sum])

    for idx, segment in enumerate(graph.select(ntype=IRSegment, flatten=False)):
        if not segment.isfw():
            continue
        for node in segment.nodes():
            graph.assign(node, idx)
        
    print(graph.extra_repr())

    # build reducer
    graph = IRAdapterGener.gen_weight(graph)
    print(graph.extra_repr())
    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 1
    assert len(reducers[0].inputs()) == 1
    assert reducers[0].input(0) == matmul1.input(1)
    assert reducers[0].device == (0, 1)


def test_replicate_shared_param():

    graph = build_graph()
    for node in graph.select(ntype=IRFwOperation):
        sn1, sn2 = graph.replicate(node, 2)
        graph.assign(sn1, 0)
        graph.assign(sn2, 1)

    graph = IRAdapterGener.gen_weight(graph)
    print(graph.extra_repr())

    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 0


def test_reducer_partially_shared_part():
    graph = build_graph()
    [matmul1, matmul2, add, sum] = graph.select(ntype=IRFwOperation)

    m1, m2 = graph.partition(matmul1, matmul1.algorithms('dim'), idx=0, dim=1, num=2)
    graph.assign(m1, 0)
    graph.assign(m2, 1)

    add1, add2 = graph.partition(add, add.algorithms('dim'), idx=0, dim=1, num=2)
    graph.assign(add1, 0)
    graph.assign(add2, 1)

    for node in [matmul2, sum]:
        sn1, sn2 = graph.replicate(node, 2)
        graph.assign(sn1, 0)
        graph.assign(sn2, 1)
    
    print(graph.extra_repr())

    with pytest.raises(RuntimeError):
        graph = IRAdapterGener.gen_weight(graph)
    print(graph.extra_repr())
