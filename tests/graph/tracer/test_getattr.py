import torch

from cube.graph.parser.converter import to_fx_graph


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.nn.functional.dropout(x, 0.1, self.training)

def test_getattr_from_root():
    model = SimpleModel()
    dummy_input = {'x': torch.rand(10)}
    traced_graph = to_fx_graph(model, dummy_input)
    traced_graph(**dummy_input)
