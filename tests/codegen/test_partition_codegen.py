from cube.graph.operator.operator import IRDataOperation, IRFwOperation
from cube.graph.tensor import IRFullTensor
from cube.graph.operator.function import Linear
from cube.graph.graph import IRGraph
from cube.schedule.pool import SchedulePool
from cube.schedule.su import SUType
from cube.schedule.sugraph import SUGraphGener
from cube.schedule.translator import IRDataLoader

from cube.execplan import ExectuionPlan
from cube.execplan.planpass.redundant import RemoveRedundantAdapters
from cube.execplan.planpass.merge import MergeComputeSU

from cube.codegen.codegen import ModelCodeGen, ScheduleCodeGen


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


def test_linear_col_codegen():

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

    execplan = ExectuionPlan(sugraph)
    execplan = RemoveRedundantAdapters.apply(execplan)

    execplan = MergeComputeSU.apply(execplan)

    mgener = ModelCodeGen(execplan)
    tgener = ScheduleCodeGen(execplan)

    for devid in range(ngpus):
        mcode0 = mgener.gen(device=devid, outfile=f'test{devid}.py')
        tcode0 = tgener.gen(device=devid, outfile=f'test{devid}.py', attach=True)
        print(f'===> model code on device {devid}: ')
        print(mcode0)
        print(f'===> schedule code on device {devid}: ')
        print(tcode0)

    assert False