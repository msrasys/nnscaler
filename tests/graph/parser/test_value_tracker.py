#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile

import pytest
import torch

from nnscaler.graph.parser.converter import convert_model
from nnscaler import register_op, mark_dynamic

from ...utils import replace_all_device_with


@replace_all_device_with('cpu')
def test_hidden_dim():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return x.repeat(4, 1)
    dummy_input = {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])}
    module = MyModule()

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(module, dummy_input, attr_savedir=tempdir, constant_folding=False)
        assert ir_graph is not None
        assert len(ir_graph.nodes()) == 1
        node = ir_graph.node(0)
        assert str(node.anno) == 'a^ b -> (4^ a^) b'
        dim0_vi = node.input(0).dim_tracks[0].value_id
        dim1_vi = node.input(0).dim_tracks[1].value_id

        assert node.output(0).dim_tracks[0].value_id != dim0_vi
        assert node.output(0).dim_tracks[0].deps == [dim0_vi]
        assert node.output(0).dim_tracks[1].value_id == dim1_vi
        assert node.output(0).dim_tracks[1].deps == []


@replace_all_device_with('cpu')
def test_equiv_class():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor, y: torch.Tensor):
            x = x + 1
            y = y * 2
            return x@y

    dummy_input = {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 'y': torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
    module = MyModule()

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(module, dummy_input, attr_savedir=tempdir, constant_folding=False)
        assert ir_graph is not None
        assert len(ir_graph.nodes()) == 3
        x_node = ir_graph.node(0)
        y_node = ir_graph.node(1)
        assert x_node.input(0).dim_tracks[0] is x_node.output(0).dim_tracks[0]
        assert x_node.input(0).dim_tracks[1] is x_node.output(0).dim_tracks[1]

        assert y_node.input(0).dim_tracks[0] is y_node.output(0).dim_tracks[0]
        assert y_node.input(0).dim_tracks[1] is y_node.output(0).dim_tracks[1]

        node = ir_graph.node(-1)
        assert str(node.anno) == 'm k+, k+ n -> m n'
        # the `k` dimension of input 1 should be the same as input 0
        # they are in the same equivalence class
        assert node.input(0).dim_tracks[0] is x_node.input(0).dim_tracks[0]
        assert node.input(0).dim_tracks[1] is x_node.input(0).dim_tracks[1]
        assert node.input(1).dim_tracks[0] is node.input(0).dim_tracks[1]
        assert node.input(1).dim_tracks[1] is y_node.input(0).dim_tracks[1]

        assert node.output(0).dim_tracks[0] is node.input(0).dim_tracks[0]
        assert node.output(0).dim_tracks[1] is node.input(1).dim_tracks[1]


@replace_all_device_with('cpu')
@pytest.mark.parametrize('dynamic_dim', [True, False])
def test_size(dynamic_dim):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            x = x + 1
            s = x.size()
            y = torch.randn(s)
            return x + y

    dummy_input = {'x': mark_dynamic(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), [0] if dynamic_dim else [])}
    module = MyModule()

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(module, dummy_input, attr_savedir=tempdir, constant_folding=False)
        assert ir_graph is not None
        assert len(ir_graph.nodes()) == 4
        size_node = ir_graph.node(1)
        randn_node = ir_graph.node(2)

        assert size_node.output(0)[0].value_track is ir_graph.inputs()[0].dim_tracks[0]
        assert size_node.output(0)[0].value_track.is_constant != (dynamic_dim is True)
        assert size_node.output(0)[1].value_track is ir_graph.inputs()[0].dim_tracks[1]
        assert size_node.output(0)[1].value_track.is_constant is True

        # dim tracks of randn node is from equivalence class originally from torch.add
        assert randn_node.output(0).dim_tracks[0] is ir_graph.inputs()[0].dim_tracks[0]
        assert randn_node.output(0).dim_tracks[0].is_constant != (dynamic_dim is True)
        assert randn_node.output(0).dim_tracks[1] is ir_graph.inputs()[0].dim_tracks[1]
        assert randn_node.output(0).dim_tracks[1].is_constant is True


# Note: the custom op here is just for testing purpose
@register_op('l (2 m) n -> n (2 l) m')
def my_op(x: torch.Tensor) -> torch.Tensor:
    return torch.randn(x.size(2), x.size(0) * 2, x.size(1) // 2)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('dynamic_dim', [True, False])
def test_custom_op(dynamic_dim):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            x = my_op(x)
            s = x.size()
            y = torch.randn(s)
            return x + y

    dummy_input = {'x': mark_dynamic(torch.randn(2, 2, 2), [0, 2] if dynamic_dim else [])}
    module = MyModule()

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(module, dummy_input, attr_savedir=tempdir, constant_folding=False)
        assert ir_graph is not None
        assert len(ir_graph.nodes()) == 4
        my_op_node = ir_graph.node(0)
        size_node = ir_graph.node(1)
        randn_node = ir_graph.node(2)

        assert [t.is_constant for t in ir_graph.inputs()[0].dim_tracks] == [not dynamic_dim, True, not dynamic_dim]

        assert my_op_node.output(0).dim_tracks[0] is ir_graph.inputs()[0].dim_tracks[2]
        assert ir_graph.inputs()[0].dim_tracks[0].value_id in my_op_node.output(0).dim_tracks[1].deps
        assert ir_graph.inputs()[0].dim_tracks[1].value_id in my_op_node.output(0).dim_tracks[2].deps

        assert [t.is_constant for t in my_op_node.outputs()[0].dim_tracks] == [not dynamic_dim, not dynamic_dim, True]

        assert size_node.output(0)[0].value_track is my_op_node.output(0).dim_tracks[0]
        assert size_node.output(0)[1].value_track is my_op_node.output(0).dim_tracks[1]
        assert size_node.output(0)[2].value_track is my_op_node.output(0).dim_tracks[2]

        assert [t.value_track.is_constant for t in size_node.output(0)] == [not dynamic_dim, not dynamic_dim, True]

        # dim tracks of randn node is from equivalence class originally from torch.add
        assert randn_node.output(0).dim_tracks[0] is my_op_node.output(0).dim_tracks[0]
        # assert randn_node.output(0).dim_tracks[0].is_constant != (dynamic_dim is True)
        assert randn_node.output(0).dim_tracks[1] is my_op_node.output(0).dim_tracks[1]
        assert randn_node.output(0).dim_tracks[2] is my_op_node.output(0).dim_tracks[2]

        assert [t.is_constant for t in randn_node.outputs()[0].dim_tracks] == [not dynamic_dim, not dynamic_dim, True]


# Note: the custom op here is just for testing purpose
@register_op('l l -> l l')
def my_identity(x: torch.Tensor) -> torch.Tensor:
    return x


@replace_all_device_with('cpu')
def test_custom_op2():
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor):
            return my_identity(x)

    dummy_input = {'x': torch.randn(2, 2)}
    module = MyModule()

    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(module, dummy_input, attr_savedir=tempdir, constant_folding=False)
        assert ir_graph is not None
        assert len(ir_graph.nodes()) == 1
        my_op_node = ir_graph.node(0)

        assert ir_graph.inputs()[0].dim_tracks[1] is ir_graph.inputs()[0].dim_tracks[0]
        assert my_op_node.output(0).dim_tracks[0] is ir_graph.inputs()[0].dim_tracks[0]
        assert my_op_node.output(0).dim_tracks[1] is ir_graph.inputs()[0].dim_tracks[1]
