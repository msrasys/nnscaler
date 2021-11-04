from cube.graph.graph import IRGraph
from cube.graph.tensor import IRFullTensor, IRSubTensor
from cube.graph.operator.function import Linear
from cube.ir.cten import IRTensor


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


def test_graph_init():

    inputs, ops, outputs = construct_model()
    graph = IRGraph(ops, inputs, outputs, 'MLP')
    print(graph)

    assert len(graph.inputs()) == 1
    assert len(graph.outputs()) == 1
    assert graph.tag == 'forward'
    assert graph.name == 'MLP'

    all_inputs = list()
    all_outputs = list()
    for node in graph.nodes():
        all_inputs += node.inputs()
        all_outputs += node.outputs()

    for input in all_inputs:
        if isinstance(input, IRTensor):
            assert isinstance(input, IRSubTensor)
    for output in all_outputs:
        if isinstance(output, IRTensor):
            assert isinstance(output, IRSubTensor)

    # check inputs
    for full_input, sub_input in zip(inputs, graph.inputs()):
        assert full_input.overlap(sub_input)
        assert full_input.shape == sub_input.shape
        assert sub_input in all_inputs
    for full_output, sub_output in zip(outputs, graph.outputs()):
        assert full_output.overlap(sub_output)
        assert full_output.shape == sub_output.shape
        assert sub_output in all_outputs

    # check dependency
    node1, node2, node3 = graph.nodes()
    assert node2 in node1.successors()
    assert node3 in node2.successors()
    assert node1 in node2.predecessors()
    assert node2 in node3.predecessors()
    # one-hop test
    assert node1 not in node3.predecessors()
    assert node3 not in node1.successors()
    # false test
    assert node1 not in node2.successors()
    assert node3 not in node2.predecessors()

    # weight test
    params = graph.parameters()
    assert len(params) == 5


def test_graph_nodes():
    inputs, ops, outputs = construct_model()
    graph = IRGraph(ops, inputs, outputs, 'MLP')
    assert id(graph.nodes()) != id(graph.nodes())
    assert graph.nodes(1) == ops[1]


def test_graph_copy():
    inputs, ops, outputs = construct_model()
    graph = IRGraph(ops, inputs, outputs, 'MLP')

    cgraph = graph.copy(reverse=False)
    print(cgraph)

    cparam_id = [param._id for param in cgraph.parameters()]
    param_id = [param._id for param in graph.parameters()]
    assert set(cparam_id) == set(param_id)

    for gnode, cnode in zip(graph.nodes(), cgraph.nodes()):
        assert gnode.name == cnode.name
        assert gnode.signature == cnode.signature
        assert len(gnode.inputs()) == len(cnode.inputs())
        assert len(gnode.outputs()) == len(cnode.outputs())
        assert len(gnode.predecessors()) == len(cnode.predecessors())
        assert len(gnode.successors()) == len(cnode.successors())

    rgraph = graph.copy(reverse=True)
    print(rgraph)
    for gnode, cnode in zip(graph.nodes(), rgraph.nodes()[::-1]):
        assert gnode.name == cnode.name
        assert gnode.signature == cnode.signature
        assert len(gnode.outputs()) == len(cnode.inputs())
        assert len(gnode.inputs()) == len(cnode.outputs())
        assert len(gnode.predecessors()) == len(cnode.successors())
        assert len(gnode.successors()) == len(cnode.predecessors())


def test_graph_partition():

    inputs, ops, outputs = construct_model()
    graph = IRGraph(ops, inputs, outputs, 'MLP')

    node1, node2, node3 = graph.nodes()

    algo = node2.algorithms('data')
    sub_nodes = graph.partition(node2, algo, config=dict(chunk_num=4))
    assert sub_nodes is not None
    assert len(graph.nodes()) == 6
    dnode1, dnode2, dnode3, dnode4 = sub_nodes
    assert dnode2 not in dnode1.successors()
    assert dnode3 not in dnode1.successors()
    assert dnode4 not in dnode1.successors()

    algo = node3.algorithms('column')
    sub_nodes = graph.partition(node3, algo, config=dict(chunk_num=4))
    print(graph)

    cnode1, cnode2, cnode3, cnode4 = sub_nodes
    for cnode in sub_nodes:
        assert dnode1 in cnode.predecessors()
        assert dnode2 in cnode.predecessors()
        assert dnode3 in cnode.predecessors()
        assert dnode4 in cnode.predecessors()
    assert len(graph.nodes()) == 9
