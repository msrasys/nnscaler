from cube.graph.tensor import IRFullTensor
from cube.graph.operator.function import Linear
from cube.graph.graph import IRGraph

from cube.schedule.pool import SchedulePool
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraphGener

from cube.execplan import ExectuionPlan
from cube.execplan.planpass.redundant import RemoveRedundantAdapters


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


def test_remove_adapter():

    SchedulePool().clear()

    graph = construct_graph()
    data = IRFullTensor(shape=[64,1024], name='data').tosub()
    output = graph(data)
    output.backward()

    nodes = SchedulePool().nodes()
    sugraph = SUGraphGener.gen_sugraph(nodes)

    for su in sugraph.sus():
        sugraph.assign(su, 0)

    execplan = ExectuionPlan(sugraph)
    execplan = RemoveRedundantAdapters.apply(execplan)

    for devid in execplan.devices():
        print(f'> device {devid}')
        for su in execplan.sequence(devid):
            print(su)
            assert su.stype != SUType.P2P
    assert len(execplan.sequence(0)) == 6