import tempfile
import torch
from cube.graph.parser.converter import to_fx_graph, to_ir_graph


def test_multi_consume():

    class MyModule(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.param1 = torch.nn.Parameter(torch.empty(4, 4))
            self.param2 = torch.nn.Parameter(torch.empty(4, 4))

        def forward(self, x):
            shortcut = x
            x = torch.matmul(x, self.param1)
            x = x + self.param2
            x = x + shortcut
            x = x + self.param1
            return torch.sum(x)

    dummy_input = {'x': torch.randn(4, 4)}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)

    print(fx_graph.graph)

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = to_ir_graph(fx_graph, dummy_input, attr_savedir=tempdir, dynamic_shape=True)
    assert ir_graph is not None
    assert len(ir_graph.attributes()) == 2  # param1 and param2
    assert len(ir_graph.full_tensors()) == 8
