import torch
from cube.graph.operator.operator import IRBpOperation, IRDataOperation, IRFwOperation

from cube.schedule.translator import LogicTranslator
from cube.schedule.translator import IRDataLoader
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph, SUGraphGener
from cube.schedule.pool import SchedulePool

from cube.graph.tensor import IRFullTensor, IRSubTensor
from cube.graph.operator.function import Linear
from cube.graph.graph import IRGraph


class FakeDataLoader:
    def __init__(self, batch_size, num=640):
        self.batch_size = batch_size
        self.length = num
        self.pos = 0
    def __iter__(self):
        self.pos = 0
        return self
    def __next__(self):
        self.pos += 1
        if self.pos == self.length:
            raise StopIteration 
        return torch.randn((self.batch_size, 1024))


def construct_graph():

    input = IRFullTensor(shape=[64,1024], name='data')
    weight1 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias1 = IRFullTensor(shape=[1024, 1024], name='bias')
    weight2 = IRFullTensor(shape=[1024, 1024], name='weight')
    weight3 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias3 = IRFullTensor(shape=[1024, 1024], name='bias')

    # linear1
    linear1 = Linear(
        name='linear1',
        signature='torch.nn.functional.linear',
        inputs= [input, weight1, bias1],
    )
    linear1.infer_shape()

    # linear2
    linear2 = Linear(
        name='linear2',
        signature='torch.nn.functional.linear',
        inputs= [linear1.outputs(0), weight2, None],
    )
    linear2.infer_shape()

    # linear3
    linear3 = Linear(
        name='linear3',
        signature='torch.nn.functional.linear',
        inputs= [linear2.outputs(0), weight3, bias3],
    )
    linear3.infer_shape()

    graph = IRGraph(
        nodes=[linear1, linear2, linear3],
        input_tensors=[input],
        output_tensors=linear3.outputs(), 
        module_name="Test"
    )
    return graph


def test_load_dataloader():

    SchedulePool().clear()
    dataloader = IRDataLoader(FakeDataLoader(batch_size=64))

    data1 = next(dataloader)
    assert isinstance(data1, IRSubTensor)
    assert data1.shape == [64, 1024]

    data2 = next(dataloader)
    assert len(SchedulePool().nodes()) == 2
    assert all([isinstance(node, IRDataOperation) for node in SchedulePool().nodes()])

    data3 = LogicTranslator.load_data(dataloader)
    assert isinstance(data1, IRSubTensor)
    assert data1.shape == [64, 1024]
    assert len(SchedulePool().nodes()) == 3
    assert all([isinstance(node, IRDataOperation) for node in SchedulePool().nodes()])


def test_translator_forward():
    SchedulePool().clear()

    graph = construct_graph()
    print(graph)
    data = IRFullTensor(shape=[64,1024], name='data').tosub()
    output = graph(data)

    assert isinstance(output, IRSubTensor)
    assert output.shape == [64, 1024]

    nodes = SchedulePool().nodes()
    assert len(nodes) == 3
    assert isinstance(SchedulePool().get_tape(output), list)
    for node in nodes:
        assert isinstance(node, IRFwOperation)
        assert isinstance(node.mirror, IRBpOperation)
    

def test_translator_backward():
    SchedulePool().clear()

    graph = construct_graph()
    data = IRFullTensor(shape=[64,1024], name='data').tosub()
    output = graph(data)
    output.backward()

    nodes = SchedulePool().nodes()
    assert len(nodes) == 6
    fnodes = nodes[0:3]
    bnodes = nodes[3:]
    for fsu, bsu in zip(fnodes, bnodes[::-1]):
        assert fsu.mirror == bsu
        assert bsu.mirror == fsu


def test_sugraph_gener_gen():
    SchedulePool().clear()

    graph = construct_graph()
    data = IRFullTensor(shape=[64,1024], name='data').tosub()
    output = graph(data)

    # forward adatpers
    nodes = SchedulePool().nodes()
    sugraph = SUGraphGener.gen_sugraph(nodes)
    assert len(sugraph.sus()) == 7
    su1, send_su12, recv_su12, su2, send_su23, recv_su23, su3 = sugraph.sus()
    assert su1.stype == SUType.Forward
    assert su2.stype == SUType.Forward
    assert su3.stype == SUType.Forward
    assert send_su12.stype == SUType.Comm
    assert recv_su12.stype == SUType.Comm
    assert send_su23.stype == SUType.Comm
    assert recv_su23.stype == SUType.Comm

    # backward adapters
    output.backward()
    nodes = SchedulePool().nodes()
    sugraph = SUGraphGener.gen_sugraph(nodes)
    for su in sugraph.sus():
        print(su)
    # note loss will be the input to autograd, therefore
    # have additional adapters
    assert len(sugraph.sus()) == 14
