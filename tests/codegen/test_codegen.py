from cube.graph.tensor import IRFullTensor
from cube.graph.operator import IROperation
from cube.graph.graph import IRGraph
from cube.schedule.pool import SchedulePool
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraph
from cube.schedule.translator import IRDataLoader
from cube.schedule.translator import LogicTranslator
from cube.schedule.graphpass import SUGraphPass
import torch

from cube.codegen.codegen import ModelCodeGen, ScheduleCodeGen


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
    weight4 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias4 = IRFullTensor(shape=[1024, 1024], name='bias')

    # linear1
    linear1 = IROperation(
        name='linear1',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )
    linear1.set_input(0, input)
    linear1.set_input(1, weight1)
    linear1.set_input(2, bias1)
    linear1.infer_shape()

    # linear2
    linear2 = IROperation(
        name='linear2',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )
    linear2.set_input(0, linear1.outputs(0))
    linear2.set_input(1, weight2)
    linear2.infer_shape()

    # linear3
    linear3 = IROperation(
        name='linear3',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )
    linear3.set_input(0, linear2.outputs(0))
    linear3.set_input(1, weight3)
    linear3.set_input(2, bias3)
    linear3.infer_shape()

    # linear4
    linear4 = IROperation(
        name='linear4',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )
    linear4.set_input(0, linear3.outputs(0))
    linear4.set_input(1, weight4)
    linear4.set_input(2, bias4)
    linear4.infer_shape()

    graph = IRGraph(
        nodes=[linear1, linear2, linear3, linear4],
        input_tensors=[input],
        output_tensors=linear3.outputs(), 
        module_name="Test"
    )
    return graph


def test_model_gen():

    SchedulePool().clear()

    graph = construct_graph()
    dataloader = IRDataLoader(FakeDataLoader(batch_size=64))
    
    data = next(dataloader)
    output = graph(data)
    output.backward()

    sus = SchedulePool().sus()
    sus = LogicTranslator.gen_adapter(sus)

    sugraph = SUGraph(sus)
    fsus = [su for su in sugraph.sus() if su.stype == SUType.Forward]
    dsus = [su for su in sugraph.sus() if su.stype == SUType.Dataloader]
    for dsu in dsus:
        sugraph.assign(dsu, 0)
    for idx, su in enumerate(fsus):
        if su.stype == SUType.Forward:
            if idx < 2:
                sugraph.assign(su, 0)
                sugraph.assign(su.mirror, 0)
            else:
                sugraph.assign(su, 1)
                sugraph.assign(su.mirror, 1)
    
    sugraph = SUGraphPass.remove_redundant_adapters(sugraph)
    sugraph = SUGraphPass.merge_small_sus(sugraph)

    print(sugraph)

    mgener = ModelCodeGen(sugraph)
    tgener = ScheduleCodeGen(sugraph)

    mcode0 = mgener.gen(device = 0)
    tcode0 = tgener.gen(device = 0)
    print('model code on device 0: ')
    print(mcode0)
    print('schedule code on device 0: ')
    print(tcode0)

    mcode1 = mgener.gen(device = 1)
    tcode1 = tgener.gen(device = 1)
    print('model code on device 1: ')
    print(mcode1)
    print('schedule code on device 1: ')
    print(tcode1)

    assert False
