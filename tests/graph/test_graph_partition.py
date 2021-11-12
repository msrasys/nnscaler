import enum
from cube.graph.graph import IRGraph
from cube.graph.tensor import IRFullTensor, ValueMap
from cube.graph.operator.function import Linear, ElementWise
from cube.schedule.pool import SchedulePool
from cube.schedule.sugraph import SUGraphGener


def simple_linear():
    input1 = IRFullTensor(shape=[64,1024], name='data1')
    weight1 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias1 = IRFullTensor(shape=[1024, 1024], name='bias')
    weight2 = IRFullTensor(shape=[1024, 1024], name='weight')
    weight3 = IRFullTensor(shape=[1024, 1024], name='weight')
    bias3 = IRFullTensor(shape=[1024, 1024], name='bias')

    # linear1
    linear1 = Linear(
        name='linear1',
        signature='torch.nn.functional.linear',
        inputs= [input1, weight1, bias1],
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
    return [input1], [linear1, linear2, linear3], [linear3.outputs(0)]


def test_linear_dp_partition():

    SchedulePool().clear()

    inputs, ops, outputs = simple_linear()
    linear1, linear2, linear3, = ops
    graph = IRGraph(ops, inputs, outputs, 'MLP')
    print(graph)

    inputs = [inputs[0].tosub()]
    loss = graph(*inputs)
    loss.backward()

    nodes = SchedulePool().nodes()
    fbgraph = IRGraph(nodes, None, None, 'MLPFull')
    print(fbgraph)

    # replace first linear by data parallel
    algo = linear1.algorithms('data')
    subnodes = fbgraph.partition(linear1, algo, config=dict(chunk_num=4))

    algo = linear2.algorithms('data')
    subnodes = fbgraph.partition(linear2, algo, config=dict(chunk_num=4))
    
    algo = linear3.algorithms('data')
    subnodes = fbgraph.partition(linear3, algo, config=dict(chunk_num=4))

    print(fbgraph)
    for node in subnodes:
        print(node)
        print(node.mirror)
    # assert False

def test_linear_hybrid_partition():

    SchedulePool().clear()
    ngpus = 2

    inputs, ops, outputs = simple_linear()
    linear1, linear2, linear3, = ops
    graph = IRGraph(ops, inputs, outputs, 'MLP')
    print(graph)

    inputs = [inputs[0].tosub()]
    loss = graph(*inputs)
    loss.backward()

    nodes = SchedulePool().nodes()
    fbgraph = IRGraph(nodes, None, None, 'MLPFull')
    print(fbgraph)

    # replace first linear by data parallel
    algo = linear1.algorithms('column')
    subnodes1 = fbgraph.partition(linear1, algo, config=dict(chunk_num=ngpus))

    algo = linear2.algorithms('column')
    subnodes2 = fbgraph.partition(linear2, algo, config=dict(chunk_num=ngpus))
    
    algo = linear3.algorithms('column')
    subnodes3 = fbgraph.partition(linear3, algo, config=dict(chunk_num=ngpus))

    print(fbgraph)
    # for node in subnodes:
    #     print(node)
    #     print(node.mirror)

    sugraph = SUGraphGener.gen_sugraph(fbgraph.nodes())
    algosu1 = sugraph.fsus()[:ngpus]
    for idx, su in enumerate(algosu1):
        sugraph.assign(su, idx)
        sugraph.assign(su.mirror, idx)
    algosu2 = sugraph.fsus()[ngpus: ngpus * 2]
    for idx, su in enumerate(algosu2):
        sugraph.assign(su, idx)
        sugraph.assign(su.mirror, idx)
    algosu3 = sugraph.fsus()[ngpus * 2: ngpus * 3]
    for idx, su in enumerate(algosu3):
        sugraph.assign(su, idx)
        sugraph.assign(su.mirror, idx)
    print(sugraph)

    print('===== algo 1 =====')
    for idx, su in enumerate(algosu1):
        print('F:', su)
        print('B:', su.mirror)
        data_grad = su.mirror.outputs(0)
        data_grad_ref = su.inputs(0).get_grad(su.nodes(0))
        print('grad    :', data_grad)
        print('grad ref:', data_grad_ref)
        assert data_grad == data_grad_ref
        assert data_grad.val_map == ValueMap(idx, ngpus)

    print('===== algo 2 =====')
    for idx, su in enumerate(algosu2):
        print('F:', su)
        print('B:', su.mirror)
        data_grad = su.mirror.outputs(0)
        data_grad_ref = su.inputs(0).get_grad(su.nodes(0))
        print('grad    :', data_grad)
        print('grad ref:', data_grad_ref)
        assert data_grad == data_grad_ref
        assert data_grad.val_map == ValueMap(idx, ngpus)

    print('===== algo 3 =====')
    for idx, su in enumerate(algosu3):
        print('F:', su)
        print('B:', su.mirror)
        data_grad = su.mirror.outputs(0)
        data_grad_ref = su.inputs(0).get_grad(su.nodes(0))
        print('grad    :', data_grad)
        print('grad ref:', data_grad_ref)
        assert data_grad == data_grad_ref
        assert data_grad.val_map == ValueMap(idx, ngpus)

    assert False
