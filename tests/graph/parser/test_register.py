#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import nnscaler
from nnscaler.graph.parser.converter import convert_model
from nnscaler.profiler.database import get_func
from nnscaler.codegen.emit import FuncEmission
import tempfile
import torch

from ...utils import replace_all_device_with


def mock_add(x: torch.Tensor, y: torch.Tensor):
    return x + y

nnscaler.register_op('*, * -> *')(mock_add)


@nnscaler.register_op('*, * -> *')
def mock_add2(x: torch.Tensor, y: torch.Tensor):
    return x + y


@nnscaler.register_op('(h w^) k^ -> h (w^ k^)')
def mock_view_with_obj(x, h):
    return x.view(h, -1)


class MockAGF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        return x + y

    @staticmethod
    def backward(ctx, grad):
        return grad, grad

nnscaler.register_op('*, * -> *')(MockAGF.apply)


class MockModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x, y):
        x, y = self.fc(x), self.fc(y)
        return mock_add(x, y)

class MockModel2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x, y):
        x, y = self.fc(x), self.fc(y)
        return mock_add2(x, y)

class MockModel3(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x, y):
        x, y = self.fc(x), self.fc(y)
        return MockAGF.apply(x, y)


class MockModelObj(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x, h: int):
        # x: [40, 10]
        x = self.fc(x)
        return mock_view_with_obj(x, h)


# passed test
@replace_all_device_with('cpu')
def test_common_register():
    model = MockModel()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}, tempdir, False)

        # test profiler.database
        for node, p_name in zip(ir_graph.nodes(), ['linear', 'linear', 'mock_add']):
            profile_name = get_func(node)[0].__qualname__
            assert profile_name == p_name, f'{profile_name} should be {p_name}'


@replace_all_device_with('cpu')
def test_common_register2():
    model = MockModel2()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}, tempdir, False)

        # test profiler.database
        for node, p_name in zip(ir_graph.nodes(), ['linear', 'linear', 'mock_add2']):
            profile_name = get_func(node)[0].__qualname__
            assert profile_name == p_name, f'{profile_name} should be {p_name}'


@replace_all_device_with('cpu')
def test_autograd_register():
    model = MockModel3()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}, tempdir, False)

        # test profiler.database
        for node, p_name in zip(ir_graph.nodes(), ['linear', 'linear', 'Function.apply']):
            profile_name = get_func(node)[0].__qualname__
            assert profile_name == p_name, f'{profile_name} should be {p_name}'


@replace_all_device_with('cpu')
def test_autograd_register():
    model = MockModelObj()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(40, 10), 'h': 4}, tempdir, False)

        node = ir_graph.select(name='mock_view_with_obj')[0]
        assert node.kwargs['h'] == 4
        sub_nodes = ir_graph.partition(node, node.algorithms('dim'), idx=0, dim=0, num=2)
        for sub_node in sub_nodes:
            assert sub_node.kwargs['h'] == 2

def customized_add(x, y):
    return x + y

def emit_customized_add(node, args, kwargs, runtime_devid, plan_ndevs, runtime_ndevs):
    kw_pairs = list()
    for key, val in kwargs.items():
        code = f'{key}={val}'
        kw_pairs.append(code)

    args = ", ".join(list(args) + kw_pairs)
    return f"torch.add({args})"

nnscaler.register_op('*, * -> *', emit_fn=emit_customized_add)(customized_add)

class ModelCustomizedAdd(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return customized_add(x, y)

@replace_all_device_with('cpu')
def test_customized_emit():
    model = ModelCustomizedAdd()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}, tempdir, False)
        add_node = ir_graph.nodes()[0]
        code = FuncEmission().emit_fnode(add_node, runtime_devid=0, plan_ndevs=1, runtime_ndevs=1)
        assert 'torch.add' in code[-1]
