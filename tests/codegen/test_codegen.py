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

    # linear4
    linear4 = Linear(
        name='linear4',
        signature='torch.nn.functional.linear',
        inputs= [linear3.outputs(0), weight4, bias4],
    )
    linear4.infer_shape()

    graph = IRGraph(
        nodes=[linear1, linear2, linear3, linear4],
        input_tensors=[input],
        output_tensors=linear4.outputs(), 
        module_name="Test"
    )
    return graph


def test_model_gen():

    SchedulePool().clear()

    grad_accum = 2

    graph = construct_graph()
    dataloader = IRDataLoader(FakeDataLoader(batch_size=64))
    
    data = next(dataloader)
    output = graph(data)
    output.backward()

    nodes = SchedulePool().nodes()
    graph = IRGraph(nodes, None, None, module_name='Test')

    for node in graph.nodes():
        if isinstance(node, IRDataOperation) or isinstance(node, IRFwOperation):
            algo = node.algorithms('data')
            graph.partition(node, algo, config=dict(chunk_num=grad_accum))

    # print(graph)

    sugraph = SUGraphGener.gen_sugraph(graph.nodes())

    # print(sugraph)

    fb_seqs = list()
    for fsu in sugraph.fsus():
        for fb_seq in fb_seqs:
            for ksu in fb_seq[::-1]:
                if sugraph.happen_before(ksu, fsu):
                    fb_seq.append(fsu)
                    break
            else:
                continue
            break
        else:
            fb_seqs.append([fsu])
    
    if len(fb_seqs) != grad_accum:
        for idx, fb_seq in enumerate(fb_seqs):
            print(f'> sequence {idx}:')
            for su in fb_seq:
                print(su)
    assert len(fb_seqs) == grad_accum

    for su in sugraph.sus():
        if su.stype == SUType.Dataloader:
            sugraph.assign(su, 0)

    for fb_seq in fb_seqs:
        for idx, su in enumerate(fb_seq):
            if idx < 2:
                sugraph.assign(su, 0)
                sugraph.assign(su.mirror, 0)
            else:
                sugraph.assign(su, 1)
                sugraph.assign(su.mirror, 1)

    for fb_seq in fb_seqs:
        fb_seq += [fsu.mirror for fsu in fb_seq][::-1]
    
    print('========= after asignment: ==========\n', sugraph)

    seqs = list()
    for fb_seq in fb_seqs:
        seqs += fb_seq
    print('> seqs:')
    for su in seqs:
        print(su)
    sugraph.partial_set_order(seqs)

    # print('========= after reorder: ==========\n', sugraph)

    execplan = ExectuionPlan(sugraph)
    execplan = RemoveRedundantAdapters.apply(execplan)

    # print('========= after remove adapter: ==========\n', execplan)

    execplan = MergeComputeSU.apply(execplan)
    # print('========= after merge small SU: ==========\n', execplan)

    mgener = ModelCodeGen(execplan)
    tgener = ScheduleCodeGen(execplan)

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
