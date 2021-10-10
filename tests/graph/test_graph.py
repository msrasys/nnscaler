
import cube.graph.parser as parser
from cube.sschedule.adapter import Adapter
from cube.graph.unique import IDGenerator
import copy

import torch
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0., mult=16, classes=1000):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * mult, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim * mult, dim)
        self.classifier = nn.Linear(dim, classes)

    def forward(self, data, x: int = 4):
        output = self.linear1(data)
        output = self.gelu(output)
        output = self.dropout(output)
        output = output + data
        output = self.linear2(output)
        output = self.classifier(output)
        return output

model = FeedForward(dim=1024)
graph = parser.convert(model, input_shapes=([1024,1024],[1,]))


def test_sendrecv_adapter(graph):
    for nid, node in enumerate(graph.nodes()):
        if nid < 3:
            node.device = 0
        else:
            node.device = 1
    print('==== graph (not adapted) ====')
    print(graph)
    graph = Adapter.adapt(graph)
    print('==== graph (after adapter) ====')
    print(graph)


def test_graph_copy(graph):
    graph = graph.copy()
    print('====== Copied Graph =====')
    print(graph)
    print('====== Copied Graph =====')


def test_graph_reverse(graph):
    graph = graph.copy(reverse=True)
    print('====== Reversed Graph =====')
    print(graph)
    print('====== Reversed Graph =====')


if __name__ == '__main__':

    test_sendrecv_adapter(graph)
    test_graph_copy(graph)
    test_graph_reverse(graph)
