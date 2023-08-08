import tempfile
from pathlib import Path

import torch
import pytest

from cube.graph.parser.converter import to_fx_graph, to_ir_graph
from cube.graph.parser import FxModuleParser
from cube.ir.cten import IRObject, IRTensor


def test_to_graph():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 5)

        def forward(self, x, **kwargs):
            return self.linear(x)
    dummy_input = {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)
    assert fx_graph is not None
    nodes = list(fx_graph.graph.nodes)

    # starts with placeholder, and ends with output
    assert nodes[0].op == 'placeholder'
    assert nodes[0].target == 'x'
    assert nodes[1].op == 'placeholder'
    assert nodes[1].target == '**kwargs'  # should keep the double stars
    assert nodes[-1].op == 'output'
    assert nodes[-1].target == 'output'

    # should have linear.weight, linear.bias, and linear(x)
    assert any(node.op == 'get_attr' and node.target == 'linear.weight' for node in nodes)
    assert any(node.op == 'get_attr' and node.target == 'linear.bias' for node in nodes)
    assert any(node.op == 'call_function' and node.target == torch.nn.functional.linear for node in nodes)

    with tempfile.TemporaryDirectory() as tempdir:
        to_ir_graph(fx_graph, dummy_input, attr_savedir=tempdir, dynamic_shape=True)
        ir_graph = to_ir_graph(fx_graph, dummy_input, attr_savedir=tempdir, dynamic_shape=True)
        assert ir_graph is not None
        assert (Path(tempdir) / FxModuleParser.ATTR_MAP_FILE).exists()
        assert (Path(tempdir) / FxModuleParser.ATTR_CONTENT_FILE).exists()
        assert ir_graph.name == 'MyModule'
        inputs = ir_graph.inputs()
        assert len(inputs) == 2
        assert inputs[0].name == nodes[0].name
        assert isinstance(inputs[0], IRTensor)
        assert inputs[1].name == nodes[1].name
        assert isinstance(inputs[1], IRObject)

        outputs = ir_graph.outputs()
        assert len(outputs) == 1

        nodes = list(ir_graph.nodes())
        assert any(node.signature == 'torch.nn.functional.linear' for node in nodes)


def test_to_ir_graph_args():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 5)

        def forward(self, x, *args):
            return self.linear(x)
    dummy_input = {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    module = MyModule()
    fx_graph = to_fx_graph(module, dummy_input)

    with tempfile.TemporaryDirectory() as tempdir:
        # currently we don't support *args
        with pytest.raises(RuntimeError):
            to_ir_graph(fx_graph, dummy_input, attr_savedir=tempdir, dynamic_shape=True)
