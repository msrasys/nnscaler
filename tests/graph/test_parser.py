from torch import nn

import cube.graph.parser as parser
from cube.ir.cten import IRTensor


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


def test_parse_module():

    graph = parser.convert(model, input_shapes=([1024,1024],[1,]))
    print(graph)
    assert len(graph.nodes()) == 6
    assert len(graph.inputs()) == 2
    assert len(graph.outputs()) == 1
    
    node1, node2, node3, node4, node5, node6 = graph.nodes()
    assert node1.signature == 'torch.nn.functional.linear'
    assert node2.signature == 'torch.nn.functional.gelu'
    assert node3.signature == 'torch.nn.functional.dropout'
    assert node4.signature == 'torch.add'
    assert node5.signature == 'torch.nn.functional.linear'
    assert node6.signature == 'torch.nn.functional.linear'

    assert node1.inputs(2) is None
    assert isinstance(node5.inputs(2), IRTensor)

    # dependency
    assert node2.predecessors() == [node1]
    assert node3.predecessors() == [node2]
    assert node4.predecessors() == [node3]
    assert node5.predecessors() == [node4]
    assert node6.predecessors() == [node5]
    assert node1.successors() == [node2]
    assert node2.successors() == [node3]
    assert node3.successors() == [node4]
    assert node4.successors() == [node5]
    assert node5.successors() == [node6]

    assert graph.outputs(0).shape == [1024, 1000]
