from cube.graph.tensor import IRFullTensor
from cube.graph.operator.function import Linear
from cube.graph.graph import IRGraph

from cube.schedule.su import SUType, ScheduleUnit
from cube.schedule.sugraph import SUGraph
from cube.schedule.adapter.comm import IRCommunication


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


def test_sugraph_assign():

    graph = construct_graph()
    sus = [ScheduleUnit([node], SUType.Forward) for node in graph.nodes()]

    su1, su2, su3 = sus

    # adapter between su1-su2
    send_op = IRCommunication(
        send_tensors=[su1.outputs(0)],
        send_ranks = [-1]
    )
    recv_op = IRCommunication(
        recv_tensors=[su1.outputs(0)],
        recv_ranks = [-1]
    )
    send_op.pair(recv_op)
    send_su12 = ScheduleUnit([send_op], SUType.Adapter, name='send')
    recv_su12 = ScheduleUnit([recv_op], SUType.Adapter, name='recv')
    su1._add_out_adapter(0, send_su12, recv_su12)
    su2._add_in_adapter(0, send_su12, recv_su12)

    # adapter between su2-su3
    send_op = IRCommunication(
        send_tensors=[su1.outputs(0)],
        send_ranks = [-1]
    )
    recv_op = IRCommunication(
        recv_tensors=[su1.outputs(0)],
        recv_ranks = [-1]
    )
    send_op.pair(recv_op)
    send_su23 = ScheduleUnit([send_op], SUType.Adapter, name='send')
    recv_su23 = ScheduleUnit([recv_op], SUType.Adapter, name='recv')
    su2._add_out_adapter(0, send_su23, recv_su23)
    su3._add_in_adapter(0, send_su23, recv_su23)

    sugraph = SUGraph(
        [su1, send_su12, recv_su12, su2, send_su23, recv_su23, su3]
    )

    assert sugraph.assign(su1, 0)
    assert su1.device == [0]
    assert send_su12.device == [0]
    assert send_su12.nodes(0).send_ranks == [-1]
    assert recv_su12.device == []
    assert recv_su12.nodes(0).recv_ranks == [0]
    
    assert sugraph.assign(su2, 1)
    assert su1.device == [0]
    assert send_su12.device == [0]
    assert send_su12.nodes(0).send_ranks == [1]
    assert recv_su12.device == [1]
    assert recv_su12.nodes(0).recv_ranks == [0]

    assert sugraph.assign(su3, 1)
    assert su3.device == [1]
    assert send_su23.device == [1]
    assert send_su23.nodes(0).send_ranks == [1]
    assert recv_su23.device == [1]
    assert recv_su23.nodes(0).recv_ranks == [1]

    assert not sugraph.assign(send_su12, 3)


def test_sugraph_assign():

    graph = construct_graph()
    sus = [ScheduleUnit([node], SUType.Forward) for node in graph.nodes()]

    su1, su2, su3 = sus

    # adapter between su1-su2
    send_op = IRCommunication(
        send_tensors=[su1.outputs(0)],
        send_ranks = [-1]
    )
    recv_op = IRCommunication(
        recv_tensors=[su1.outputs(0)],
        recv_ranks = [-1]
    )
    send_op.pair(recv_op)
    send_su12 = ScheduleUnit([send_op], SUType.Adapter, name='send')
    recv_su12 = ScheduleUnit([recv_op], SUType.Adapter, name='recv')
    su1._add_out_adapter(0, send_su12, recv_su12)
    su2._add_in_adapter(0, send_su12, recv_su12)

    # adapter between su2-su3
    send_op = IRCommunication(
        send_tensors=[su1.outputs(0)],
        send_ranks = [-1]
    )
    recv_op = IRCommunication(
        recv_tensors=[su1.outputs(0)],
        recv_ranks = [-1]
    )
    send_op.pair(recv_op)
    send_su23 = ScheduleUnit([send_op], SUType.Adapter, name='send')
    recv_su23 = ScheduleUnit([recv_op], SUType.Adapter, name='recv')
    su2._add_out_adapter(0, send_su23, recv_su23)
    su3._add_in_adapter(0, send_su23, recv_su23)

    sugraph = SUGraph(
        [su1, send_su12, recv_su12, su2, send_su23, recv_su23, su3]
    )

    assert not sugraph.set_order(
        [su2, send_su12, recv_su12, su1, send_su23, recv_su23, su3]
    )

    assert sugraph.set_order(
        [su1, send_su12, recv_su12, su2, send_su23, recv_su23, su3]
    )
