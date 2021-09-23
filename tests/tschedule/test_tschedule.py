from cube.tschedule.pool import TSchedulePool
from cube.graph.ir_cten import IRTensor
from torch import nn
import torch

import cube.graph.parser as parser
from cube.codegen.codegen import SScheduleCodeGen


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
print(" > Forward IRGraph ========")
print(ir_graph)
print(" < ==============\n")

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
    # print(tschedule.pool.TSchedulePool())
    # tensor2 = ir_graph()
    # print(tensor2)
    for action in TSchedulePool().actions():
        print('\n', action)
        gener = SScheduleCodeGen(action.graph)
        code = gener.gen(device=action.device[0])
        print("> ===== Generated Code =====")
        print(code)
        print("< ===== Generated Code =====")
    print(TSchedulePool())


def test_graph_backward(ir_graph):

    TSchedulePool().clear()
    tensor = ir_graph(IRTensor(shape=[64,1024]))
    tensor.backward()
    #tensor = ir_graph()
    #tensor.backward()
    print('====== Backward Test =======')
    print(TSchedulePool())


if __name__ == '__main__':

    #test_graph_forward(ir_graph)
    test_graph_backward(ir_graph)



"""
loss = cube.runtime.temporal.forward(model, input1, input2, xxx)
grad1, grad2, ... = cube.runtime.temporal.backward(loss, None)
"""

"""
out1, out2 = cube.runtime.temporal.forward(model, input1)
cube.runtime.temporal.backward(out1, out2, out1_grad, out2_grad)
"""