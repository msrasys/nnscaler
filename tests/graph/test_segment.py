#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import nnscaler
import nnscaler.graph.function.function as F
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph import IRGraph
from nnscaler.ir.adapter import IRAdapter


def _tensor(shape, requires_grad=True):
    return IRFullTensor(shape, requires_grad=requires_grad).tosub()


def test_create_segment_loss_func():
    data = _tensor([256, 256], False)
    w1 = _tensor([256, 256])
    out1 = _tensor([256, 256])
    matmul_1 = F.Linear(data, w1)
    matmul_1.set_output(0, out1)
    w2 = _tensor([256, 256])
    out2 = _tensor([256, 256])
    matmul_2 = F.Linear(out1, w2)
    matmul_2.set_output(0, out2)
    loss = _tensor([1])
    sum = F.Sum(out2)
    sum.set_output(0, loss)
    d = _tensor([1], False)
    get = F.GetAttr(loss, 'data', 'getattr')
    get.set_output(0, d)
    nodes = [matmul_1, matmul_2, sum, get]
    graph = IRGraph(nodes, [data], [loss, d], 'genmodel')
    graph.backward(loss)
    segment = graph.create_segment([matmul_2, sum, get])
    print(segment.extra_repr())
    assert len(segment.outputs()) == 2
    assert segment.output(0) == loss
    assert segment.output(1) == d


def test_create_segment_loss_adapter():
    data = _tensor([256, 256], False)
    w1 = _tensor([256, 256])
    out1 = _tensor([256, 256])
    matmul_1 = F.Linear(data, w1)
    matmul_1.set_output(0, out1)
    w2 = _tensor([256, 256])
    out2 = _tensor([256, 256])
    matmul_2 = F.Linear(out1, w2)
    matmul_2.set_output(0, out2)
    loss = _tensor([1])
    sum = F.Sum(out2)
    sum.set_output(0, loss)
    sum.device = 0
    adapter = IRAdapter([sum.output(0)], [sum.output(0)])
    nodes = [matmul_1, matmul_2, sum, adapter]
    graph = IRGraph(nodes, [data], [loss], 'genmodel')
    graph.backward(loss)
    segment = graph.create_segment([matmul_2, sum, adapter])
    print(segment.extra_repr())
    assert len(segment.outputs()) == 1
    assert segment.output(0) == loss
