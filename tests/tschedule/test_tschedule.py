from torch import nn

import cube.graph.parser as parser
import cube.tschedule as tschedule


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=16, classes=1000):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, data, label: int = 4):
        output = self.linear1(data)
        output = self.gelu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = output + data
        output = self.classifier(output)
        return output


model = FeedForward(dim=1024)
ir_graph = parser.convert(model, input_shapes=([64,1024],[64,]))

# device assignment
for node in ir_graph.nodes():
    node.device = 0


def test_graph_forward(ir_graph):

    tensor1 = ir_graph()
    print(tensor1)
    print(tschedule.pool.TSchedulePool())
    tensor2 = ir_graph()
    print(tensor2)
    print(tschedule.pool.TSchedulePool())


def test_graph_backward(ir_graph):

    tensor = ir_graph()
    tensor.backward()


if __name__ == '__main__':

    test_graph_forward(ir_graph)
