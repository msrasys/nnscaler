from cube.graph.ir_graph import IRGraph
from cube.tschedule.pool import TSchedulePool
from cube.graph.ir_cten import IRTensor
from cube.graph.ir_seq import IRSequence
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
print('====== Forward Graph =======\n')
print(ir_graph)
print('====== Forward Graph =======\n')

# device assignment
for input in ir_graph.inputs():
    if isinstance(input, IRTensor):
        input.device = [0]
for nid, node in enumerate(ir_graph.nodes()):
    if nid <= 2:
        node.device = 0 
    else:
        node.device = 1


def test_graph_forward(ir_graph):

    TSchedulePool().clear()
    tensor1 = ir_graph(IRTensor(shape=[64,1024]))
    print(tensor1)
    print('====== Forward Test =======')
    for action in TSchedulePool().actions():
        gener = SScheduleCodeGen(action)
        code = gener.gen(device=action.device[0])
        print(code)
    print(TSchedulePool())
    print('\n====== Forward Test =======\n')


def test_graph_backward(ir_graph):

    TSchedulePool().clear()
    input = IRTensor(shape=[64,1024])
    input.device = [0]
    tensor = ir_graph(input)
    tensor.backward()
    input = IRTensor(shape=[64,1024])
    input.device = [0]
    tensor = ir_graph(input)
    tensor.backward()
    print('====== Backward Test =======\n')
    print(TSchedulePool())

    sequence = IRSequence(TSchedulePool().actions())
    from cube.codegen.codegen import TScheduleCodeGen
    gener = TScheduleCodeGen(sequence)
    code = gener.gen(device=0)
    print(code)
    code = gener.gen(device=1)
    print(code)

    print('\n====== Backward Test =======\n')


if __name__ == '__main__':

    test_graph_forward(ir_graph)
    test_graph_backward(ir_graph)
