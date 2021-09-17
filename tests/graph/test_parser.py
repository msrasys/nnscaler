from cube.graph import parser
import cube.graph.parser as parser
from torch import nn

import cube.graph as cgraph
from cube.graph.parser import ScriptModuleParser

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


def test_flatten(smodule):
    ScriptModuleParser.flatten(smodule)

def test_parse_module(model):
    return parser.convert(model, input_shapes=([1024,1024],[1,]))


if __name__ == '__main__':

    
    # test_flatten(smodule)
    graph = test_parse_module(model)
    print(graph)