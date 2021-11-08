import copy

from cube.graph.tensor import IRFullTensor
from cube.graph.operator.function import Linear
from cube.graph.graph import IRGraph

from cube.schedule.su import SUType, ScheduleUnit


def construct_model():

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

    # return [input], [ops], [output]
    return [input], [linear1, linear2, linear3], [linear3.outputs(0)]


def test_su_init():

    inputs, nodes, outputs = construct_model()
    graph = IRGraph(nodes, inputs, outputs, 'Test')
    linear1, linear2, linear3 = nodes

    su1 = ScheduleUnit([linear1], stype=SUType.Forward)
    assert len(su1.inputs()) == 3
    assert len(su1.outputs()) == 1
    assert su1.signature == SUType.Forward.value

    assert su1.mirror is None
    assert su1.stype == SUType.Forward
    assert su1._nodes == [linear1]
    assert len(su1._send_in_adapters) == 3
    assert len(su1._recv_in_adapters) == 3
    assert len(su1._send_out_adapters) == 1
    assert len(su1._recv_out_adapters) == 1
    assert len(su1._ctrl_predecessors) == 0
    assert len(su1._ctrl_successors) == 0

    su2 = ScheduleUnit([linear1, linear2], stype=SUType.Forward)
    print('su2:', su2)
    assert len(su2.inputs()) == 4
    assert len(su2.outputs()) == 1
    assert su2.signature == SUType.Forward.value

    su3 = ScheduleUnit([linear1, linear2, linear3], stype=SUType.Forward)
    print('su3:', su3)
    assert len(su3.inputs()) == 6
    assert len(su3.outputs()) == 1
    assert su3.signature == SUType.Forward.value


def test_su_copy():

    inputs, nodes, outputs = construct_model()
    graph = IRGraph(nodes, inputs, outputs, 'Test')
    linear1, linear2, linear3 = nodes

    su1 = ScheduleUnit([linear1, linear2], stype=SUType.Forward)
    su2 = ScheduleUnit([linear1, linear2, linear3], stype=SUType.Forward)
    su1.mirror = su2

    csu = copy.copy(su1)
    assert csu.inputs() == su1.inputs()
    assert csu.outputs() == su1.outputs()
    
    assert csu.mirror is not None
    mirror = csu.mirror
    assert mirror.inputs() == su2.inputs()
    assert mirror.outputs() == su2.outputs()
