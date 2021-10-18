from cube.graph.tensor import IRFullTensor
from cube.graph.operator import IROperation
from cube.graph.graph import IRGraph

from cube.schedule.su import SUType, ScheduleUnit
from cube.schedule.sugraph import SUGraph



def construct_graph():

    input = IRFullTensor(shape=[64,1024], name='data')
    weight1 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias1 = IRFullTensor(shape=[1024, 1024], name='bias')
    weight2 = IRFullTensor(shape=[1024, 1024], name='weight')
    weight3 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias3 = IRFullTensor(shape=[1024, 1024], name='bias')

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

    # linear2
    linear2 = IROperation(
        name='linear2',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )
    linear2.set_input(0, linear1.outputs(0))
    linear2.set_input(1, weight2)

    # linear3
    linear3 = IROperation(
        name='linear2',
        signature='torch.nn.functional.linear',
        input_length=3,
        output_length=1
    )
    linear3.set_input(0, linear2.outputs(0))
    linear3.set_input(1, weight3)
    linear3.set_input(2, bias3)

    graph = IRGraph(
        nodes=[linear1, linear2, linear3],
        input_tensors=[input],
        output_tensors=linear3.outputs(), 
        module_name="Test"
    )
    return graph


def test_graph_init():

    graph = construct_graph()
    sus = [ScheduleUnit([node], SUType.Forward) for node in graph.nodes()]

    sugraph = SUGraph(sus)
    assert len(sugraph.inputs()) == 1
    assert len(sugraph.outputs()) == 1
    assert graph.inputs() == sugraph.inputs()
    assert graph.outputs() == sugraph.outputs()

    assert sugraph.sequence == sus

    # test dependency
    su1, su2, su3 = sus
    assert su2 in su1.successors()
    assert su3 in su2.successors()
    assert su3 not in su1.successors()
    assert su1 in su2.predecessors()
    assert su1 in su2.predecessors(0)
    assert su2 in su3.predecessors()
    assert su1 not in su3.predecessors()


def test_sugraph_happen_before():

    graph = construct_graph()
    sus = [ScheduleUnit([node], SUType.Forward) for node in graph.nodes()]

    sugraph = SUGraph(sus)
    su1, su2, su3 = sugraph.sus()

    assert sugraph.happen_before(su1, su2)
    assert not sugraph.happen_before(su2, su1)
    assert sugraph.happen_before(su1, su3)
    assert not sugraph.happen_before(su3, su1)
    assert sugraph.happen_before(su2, su3)
    assert not sugraph.happen_before(su3, su2)


def test_sugraph_merge():

    graph = construct_graph()
    sus = [ScheduleUnit([node], SUType.Forward) for node in graph.nodes()]

    sugraph = SUGraph(sus)
    su1, su2, su3 = sugraph.sus()

    assert sugraph.merge(su1, su3) is None
    
    su12 = sugraph.merge(su1, su2)
    assert sugraph.nnodes == 2
    assert len(su12.inputs()) == 1
    assert len(su12.outputs()) == 1
    assert len(su12.nodes()) == 2
    assert su12 in sugraph.sus()
    assert su1 not in sugraph.sus()
    assert su2 not in sugraph.sus()
    assert sugraph.happen_before(su12, su3)


def test_sugraph_add_flow():

    graph = construct_graph()
    sus = [ScheduleUnit([node], SUType.Forward) for node in graph.nodes()]
    
    sugraph = SUGraph(sus)
    su1, su2, su3 = sugraph.sus()

    assert su1 not in su3.predecessors()
    assert su3 not in su1.successors()

    assert not sugraph.add_flow(su3, su1)

    assert sugraph.add_flow(su1, su3)
    assert su1 in su3.predecessors()
    assert su3 in su1.successors()
