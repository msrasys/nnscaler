import torch

from cube.schedule.translator import LogicTranslator
from cube.schedule.translator import IRDataLoader
from cube.schedule.su import SUType
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
    assert len(SchedulePool().sus()) == 2
    assert all([su.stype == SUType.Dataloader for su in SchedulePool().sus()])

    data3 = LogicTranslator.load_data(dataloader)
    assert isinstance(data1, IRSubTensor)
    assert data1.shape == [64, 1024]
    assert len(SchedulePool().sus()) == 3
    assert all([su.stype == SUType.Dataloader for su in SchedulePool().sus()])


def test_translator_forward():
    SchedulePool().clear()

    graph = construct_graph()
    print(graph)
    data = IRFullTensor(shape=[64,1024], name='data').tosub()
    output = graph(data)

    assert isinstance(output, IRSubTensor)
    assert output.shape == [64, 1024]
    assert output.trace is not None

    sus = SchedulePool().sus()
    assert len(sus) == 3
    assert output.trace == sus
    for su in sus:
        assert su.stype == SUType.Forward
        assert su.mirror is not None
    

def test_translator_backward():
    SchedulePool().clear()

    graph = construct_graph()
    data = IRFullTensor(shape=[64,1024], name='data').tosub()
    output = graph(data)

    output.backward()

    sus = SchedulePool().sus()
    assert len(sus) == 6
    fsus = sus[0:3]
    bsus = sus[3:]
    for fsu, bsu in zip(fsus, bsus[::-1]):
        assert fsu.mirror == bsu
        assert bsu.mirror == fsu
        assert bsu.stype == SUType.Backward


def test_translator_gen_adapter():
    SchedulePool().clear()

    graph = construct_graph()
    data = IRFullTensor(shape=[64,1024], name='data').tosub()
    output = graph(data)

    # forward adatpers
    sus = SchedulePool().sus()
    sus = LogicTranslator.gen_adapter(sus)
    assert len(sus) == 7
    su1, send_su12, recv_su12, su2, send_su23, recv_su23, su3 = sus
    assert su1.stype == SUType.Forward
    assert su2.stype == SUType.Forward
    assert su3.stype == SUType.Forward
    assert send_su12.stype == SUType.Adapter
    assert recv_su12.stype == SUType.Adapter
    assert send_su23.stype == SUType.Adapter
    assert recv_su23.stype == SUType.Adapter

    # backward adapters
    output.backward()
    sus = SchedulePool().sus()
    sus = LogicTranslator.gen_adapter(sus)
    for su in sus:
        print(su)
    # note loss will be the input to autograd, therefore
    # have additional adapters
    assert len(sus) == 16
