from cube.graph.tensor import IRFullTensor
from cube.graph.operator.function import Linear
from cube.graph.graph import IRGraph


from cube.schedule.graphpass import SUGraphPass
from cube.schedule.pool import SchedulePool
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraphGener


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
    sugraph = SUGraphPass.remove_redundant_adapters(sugraph)
    for su in sugraph.sus():
        print(su)
    for su in sugraph.sus():
        assert su.stype != SUType.Comm
    assert len(sugraph.sus()) == 6


def test_merge_small_sus():

    SchedulePool().clear()

    graph = construct_graph()
    data = IRFullTensor(shape=[64,1024], name='data').tosub()
    output = graph(data)
    output.backward()

    nodes = SchedulePool().nodes()
    sugraph = SUGraphGener.gen_sugraph(nodes)

    for su in sugraph.sus():
        if su.stype != SUType.Comm:
            sugraph.assign(su, 0)

    print('orignal:')
    print(sugraph)

    sugraph = SUGraphPass.merge_small_sus(sugraph)

    print('merged:')
    print(sugraph)

    assert len(sugraph.sus()) == 2
