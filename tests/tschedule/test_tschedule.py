from cube.graph.ir_graph import IRGraph
from cube.tschedule.pool import TSchedulePool
from cube.graph.ir_cten import IRTensor
from cube.tschedule.suseq import SUSequence
from cube.sschedule.adapter import Adapter
from torch import nn
import torch

import cube.graph.parser as parser
from cube.codegen.codegen import SScheduleCodeGen, TScheduleCodeGen


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=16, classes=1000):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, data):
        output = self.linear1(data)
        output = self.gelu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.classifier(output)
        loss = torch.sum(output)
        return loss


model = FeedForward(dim=1024)
ir_graph = parser.convert(model, input_shapes=([64,1024],))

# device assignment
for nid, node in enumerate(ir_graph.nodes()):
    if nid < 3:
        node.device = 0 
    else:
        node.device = 1

print('====== Forward Graph =======\n')
print(ir_graph)
ir_graph = Adapter.adapt(ir_graph)
print('====== Forward Graph =======\n')


def test_graph_forward(ir_graph):

    TSchedulePool().clear()
    tensor1 = ir_graph(IRTensor(shape=[64,1024]))
    tensor2 = ir_graph(IRTensor(shape=[64,1024]))
    assert tensor1 != tensor2
    print('====== Forward Test =======')
    seq = SUSequence(TSchedulePool().sus())

    for su in seq.sus():
        print(su)

    print('\n====== Forward Test =======\n')


def test_su_merge(ir_graph):

    TSchedulePool().clear()
    loss = ir_graph(IRTensor(shape=[64,1024]))
    seq = SUSequence(TSchedulePool().sus())

    first_stage = seq.sus()[0:3]
    second_stage = seq.sus()[5:8]

    su1 = seq.sus(0)
    for su in first_stage[1:]:
        su1 = seq.merge(su1, su)
        assert su1 is not None

    su2 = second_stage[0]
    for su in second_stage[1:]:
        su2 = seq.merge(su2, su)

    for su in seq.sus():
        print(su)

    # spatial code
    sgener = SScheduleCodeGen(seq)
    scode = sgener.gen(device=0)
    print(scode)

    # temporal code
    tgener = TScheduleCodeGen(seq)
    tcode = tgener.gen(device=0)
    print(tcode)


def test_graph_backward(ir_graph):

    TSchedulePool().clear()
    micro_bs = 1
    for _ in range(micro_bs):
        loss = ir_graph(IRTensor(shape=[64,1024]))
        loss.backward()
    print('====== Backward Test =======\n')
    print(TSchedulePool())

    seq = SUSequence(TSchedulePool().sus())
    first_stage = seq.sus()[0:3]
    second_stage = seq.sus()[5:8]

    su1 = seq.sus(0)
    for su in first_stage[1:]:
        su1 = seq.merge(su1, su)
        assert su1 is not None

    su2 = second_stage[0]
    for su in second_stage[1:]:
        su2 = seq.merge(su2, su)

    print('===== seq before gen ====')
    print(seq)
    for su in seq.sus():
        print(f'pair: {su} <-> {su.mirror}')

    sgener = SScheduleCodeGen(seq)
    scode = sgener.gen(device=0)
    print(scode)
    scode = sgener.gen(device=1)
    print(scode)

    # temporal code
    tgener = TScheduleCodeGen(seq)
    tcode = tgener.gen(device=0)
    print(tcode)
    tcode = tgener.gen(device=1)
    print(tcode)

    print('\n====== Backward Test =======\n')


def test_su_merge(ir_graph):
    TSchedulePool().clear()
    loss = ir_graph(IRTensor(shape=[64,1024]))
    seq = SUSequence(TSchedulePool().sus())
    sus = seq.sus()[0:4]
    
    su1 = seq.sus(0)
    for su in sus[1:]:
        su1 = seq.merge(su1, su)
        assert su1 is not None
    print(su1)
    for node in su1.nodes():
        print(node)
    print('====')
    for node in ir_graph.nodes():
        print(node)


if __name__ == '__main__':

    test_graph_forward(ir_graph)
    test_su_merge(ir_graph)
    test_graph_backward(ir_graph)
    test_su_merge(ir_graph)
