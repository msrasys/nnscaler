#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import nnscaler
from nnscaler.graph.parser.converter import convert_model, to_fx_graph
from nnscaler.profiler.database import get_func
from nnscaler.codegen.emit import FuncEmission
from nnscaler.graph.function.dimops import DimopSplit, TransformRule
from nnscaler.graph.parser.mapping import SignFx2Op
from nnscaler.graph.parser.register import CustomizedOps
import tempfile
import torch
import pytest

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
        sub_nodes = ir_graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=2)
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


def mock_transform_rule_add(x: torch.Tensor, y: torch.Tensor, z: int):
    return x + y


def build_mock_transform_rules():
    itransform = [
        DimopSplit.D(0),
        DimopSplit.D(0),
    ]

    otransform = [
        DimopSplit.D(0),
    ]

    def modifier(kwargs, idx, dim, num, subnode_idx):
        updated_kwargs = dict(**kwargs)
        if idx == 0 and dim == 0:
            updated_kwargs['z'] = kwargs['z'] * (subnode_idx + 1)
        else:
            updated_kwargs['z'] = kwargs['z']
        return updated_kwargs

    return (TransformRule(itransform, otransform, modifier),)

nnscaler.register_op('*, * -> *', transform_rules=build_mock_transform_rules())(mock_transform_rule_add)


class MockModelTransformRule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 10, bias=False)

    def forward(self, x, y):
        x1, y1 = self.fc(x), self.fc(y)
        z1 = mock_transform_rule_add(x1, y1, 10)
        x2, y2 = self.fc(x), self.fc(y)
        z2 = mock_transform_rule_add(x2, y2, 10)
        return z1 + z2


@replace_all_device_with('cpu')
def test_transform_rule():
    model = MockModelTransformRule()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}, tempdir, False)
        add_node0 = ir_graph.nodes()[2]
        add_node1 = ir_graph.nodes()[5]
        sub0, sub1 = ir_graph.partition(add_node0, add_node0.algorithm('dim'), idx=0, dim=0, num=2)
        assert sub0.kwargs['z'] == 10
        assert sub1.kwargs['z'] == 20

        sub2, sub3 = ir_graph.partition(add_node1, add_node1.algorithm('dim'), idx=0, dim=1, num=2)
        assert sub2.kwargs['z'] == 10
        assert sub3.kwargs['z'] == 10


def mock_select(x: torch.Tensor, selected_rows: torch.Tensor):
    return x[selected_rows, :]


def input_gen_fn(node):
    inputs = []
    row = None
    for i, t in enumerate(node.inputs()):
        if i == 1:
            inputs.append(torch.randint(low=0, high=row, size=t.shape, dtype=torch.int64, requires_grad=t.requires_grad))
        else:
            row = t.shape[0]
            inputs.append(torch.rand(t.shape, dtype=t.dtype, requires_grad=t.requires_grad))
    return tuple(inputs)

nnscaler.register_op('a^ b^, c^ -> c^ b^', input_gen_fn=input_gen_fn)(mock_select)


class MockModelSelect(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, selected_rows):
        return mock_select(x, selected_rows)


@replace_all_device_with('cpu')
def test_input_gen_fn():
    model = MockModelSelect()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'selected_rows': torch.randint(0, 10, (5,), dtype=torch.int64)}, tempdir, False)
        select_node = ir_graph.nodes()[0]
        fn = CustomizedOps.kOpInputGen[select_node.signature]
        ret = mock_select(*fn(select_node))
        assert True


class MockModelKwargs(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x, y):
        x, y = self.fc(x), self.fc(y)
        return mock_add(x=x, y=y)


# passed test
@replace_all_device_with('cpu')
def test_kw_args():
    model = MockModelKwargs()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}, tempdir, False)

        # test profiler.database
        for node, p_name in zip(ir_graph.nodes(), ['linear', 'linear', 'mock_add']):
            profile_name = get_func(node)[0].__qualname__
            assert profile_name == p_name, f'{profile_name} should be {p_name}'


def fake_add_xy(x: torch.Tensor, y: torch.Tensor):
    return x


@nnscaler.register_op('*, * -> *', fake_fn=fake_add_xy)
def add_xy(x: torch.Tensor, y: torch.Tensor):
    raise NotImplementedError("This function should not be called since it's replaced by fake_add_xy in tracing")


class FakeFnModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x, y):
        x, y = self.fc(x), self.fc(y)
        return add_xy(x, y)


@replace_all_device_with('cpu')
def test_register_fake_fn():
    model = FakeFnModel()
    with tempfile.TemporaryDirectory() as tempdir:
        ir_graph = convert_model(model, {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}, tempdir, False)

        # test profiler.database
        for node, p_name in zip(ir_graph.nodes(), ['linear', 'linear', 'add_xy']):
            profile_name = get_func(node)[0].__qualname__
            assert profile_name == p_name, f'{profile_name} should be {p_name}'


class BuiltinFunctionFakeFnModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)


class BuiltinDescriptorFakeFnModel(torch.nn.Module):
    def forward(self, x, y):
        return torch.Tensor.add(x, y)


class BuiltinMethodFakeFnModel(torch.nn.Module):
    def forward(self, x, y):
        return x.add(y)


def _remove_updated_op(signature):
    for registry in (
        CustomizedOps.kOpMap,
        CustomizedOps.kOpRuntime,
        CustomizedOps.kOpFakeRuntime,
        CustomizedOps.kOpCodeDef,
        CustomizedOps.kOpEmit,
        CustomizedOps.kOpInputGen,
    ):
        registry.pop(signature, None)


def test_update_builtin_function_fake_fn():
    calls = []

    def fake_add(x, y):
        calls.append((x, y))
        return x

    nnscaler.update_op(torch.add, fake_fn=fake_add)
    try:
        op_create_fn = CustomizedOps.kOpMap['torch.add']
        assert op_create_fn.func is SignFx2Op.kOpMap['torch.add']
        assert op_create_fn.keywords == {'signature': 'torch.add'}
        assert CustomizedOps.kOpCodeDef['torch.add'] == ''
        assert CustomizedOps.kOpRuntime['torch.add'] is torch.add
        traced = to_fx_graph(
            BuiltinFunctionFakeFnModel(),
            {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)},
        )
    finally:
        _remove_updated_op('torch.add')

    node = next(node for node in traced.graph.nodes if node.op == 'call_function')
    assert len(calls) == 1
    assert node.target is torch.add


def test_update_builtin_descriptor_fake_fn():
    calls = []

    def fake_add(x, y):
        calls.append((x, y))
        return x

    nnscaler.update_op(torch.Tensor.add, fake_fn=fake_add)
    try:
        traced = to_fx_graph(
            BuiltinDescriptorFakeFnModel(),
            {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)},
        )
    finally:
        _remove_updated_op('torch.add')

    node = next(node for node in traced.graph.nodes if node.op == 'call_method')
    assert len(calls) == 1
    assert node.target == 'add'


def test_update_builtin_method_fake_fn():
    calls = []

    def fake_add(x, y):
        calls.append((x, y))
        return x

    nnscaler.update_op(torch.Tensor.add, fake_fn=fake_add)
    try:
        traced = to_fx_graph(
            BuiltinMethodFakeFnModel(),
            {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)},
        )
    finally:
        _remove_updated_op('torch.add')

    node = next(node for node in traced.graph.nodes if node.op == 'call_method')
    assert len(calls) == 1
    assert node.target == 'add'


def test_update_op_preserves_existing_fields():
    def fake_add(x, y):
        return x

    def emit_add(*args, **kwargs):
        return 'updated_add'

    def generate_inputs(node):
        return []

    def create_add(*args, signature=None, **kwargs):
        return None

    nnscaler.update_op(torch.add, fake_fn=fake_add)
    nnscaler.update_op(
        torch.add,
        op_create_fn=create_add,
        code='def updated_add(): pass',
        emit_fn=emit_add,
        input_gen_fn=generate_inputs,
    )
    try:
        assert CustomizedOps.kOpMap['torch.add'] is create_add
        assert CustomizedOps.kOpRuntime['torch.add'] is torch.add
        assert CustomizedOps.kOpFakeRuntime['torch.add'] is fake_add
        assert CustomizedOps.kOpCodeDef['torch.add'] == 'def updated_add(): pass'
        assert CustomizedOps.kOpEmit['torch.add'] is emit_add
        assert CustomizedOps.kOpInputGen['torch.add'] is generate_inputs
        assert SignFx2Op.map('torch.add').func is create_add
    finally:
        _remove_updated_op('torch.add')


def test_update_op_clears_optional_fields():
    def fake_add(x, y):
        return x

    def emit_add(*args, **kwargs):
        return 'updated_add'

    def generate_inputs(node):
        return []

    nnscaler.update_op(
        torch.add,
        code='def updated_add(): pass',
        fake_fn=fake_add,
        emit_fn=emit_add,
        input_gen_fn=generate_inputs,
    )
    nnscaler.update_op(
        torch.add,
        code=None,
        fake_fn=None,
        emit_fn=None,
        input_gen_fn=None,
    )
    try:
        assert CustomizedOps.kOpCodeDef['torch.add'] == ''
        assert CustomizedOps.kOpFakeRuntime['torch.add'] is None
        assert 'torch.add' not in CustomizedOps.kOpEmit
        assert 'torch.add' not in CustomizedOps.kOpInputGen
    finally:
        _remove_updated_op('torch.add')


def test_update_customized_op_by_runtime_fn():
    signature = 'tests.graph.parser.test_register.add_xy'
    original_fake_fn = CustomizedOps.kOpFakeRuntime[signature]

    def updated_fake_add(x, y):
        return y

    nnscaler.update_op(add_xy, fake_fn=updated_fake_add)
    try:
        assert CustomizedOps.kOpRuntime[signature] is add_xy
        assert CustomizedOps.kOpFakeRuntime[signature] is updated_fake_add
    finally:
        CustomizedOps.kOpFakeRuntime[signature] = original_fake_fn


def unregistered_op(x):
    return x


def test_update_rejects_unregistered_op():
    def create_unregistered_op(*args, signature=None, **kwargs):
        return None

    with pytest.raises(ValueError, match='is not an existing operator'):
        nnscaler.update_op(
            unregistered_op,
            op_create_fn=create_unregistered_op,
            code='def unregistered_op(x): return x',
        )
