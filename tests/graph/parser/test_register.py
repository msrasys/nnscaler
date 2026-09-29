#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import nnscaler
from nnscaler.graph.parser.converter import convert_model, to_fx_graph, to_ir_graph
from nnscaler.profiler.database import get_func, profile
from nnscaler.codegen.emit import FuncEmission
from nnscaler.graph.function.dimops import DimopSplit, TransformRule
from nnscaler.graph.parser.mapping import SignFx2Op
from nnscaler.graph.parser.register import CustomizedOps, _get_torch_op_aliases
from nnscaler.utils import get_full_qualified_name, load_type
import tempfile
import torch
import torch.nn.functional as F
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

nnscaler.register_op('*, * -> *')(MockAGF)


def test_autograd_class_registration_uses_apply_name():
    runtime_fn = CustomizedOps.kOpRuntime[get_full_qualified_name(MockAGF)]
    assert runtime_fn.__self__ is MockAGF
    assert runtime_fn.__name__ == 'apply'


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


class BuiltinInplaceMethodFakeFnModel(torch.nn.Module):
    def forward(self, x, y):
        return x.add_(y)


class BuiltinBatchNormModel(torch.nn.Module):
    def forward(self, x, weight, bias, running_mean, running_var):
        return torch.batch_norm(
            x,
            weight,
            bias,
            running_mean,
            running_var,
            False,
            0.1,
            1e-5,
            False,
        )


def _remove_updated_op(signature):
    for alias in _get_torch_op_aliases(signature):
        for registry in (
            CustomizedOps.kOpMap,
            CustomizedOps.kOpRuntime,
            CustomizedOps.kOpFakeRuntime,
            CustomizedOps.kOpCodeDef,
            CustomizedOps.kOpEmit,
            CustomizedOps.kOpInputGen,
        ):
            registry.pop(alias, None)


@pytest.mark.parametrize(
    ('runtime_fn', 'expected'),
    [
        (
            torch.add,
            {
                'torch.add',
                'torch.Tensor.add',
            },
        ),
        (
            torch.Tensor.add_,
            {
                'torch.Tensor.add_',
            },
        ),
        (
            F.linear,
            {'torch.nn.functional.linear', 'torch._C._nn.linear'},
        ),
        (
            torch.relu,
            {
                'torch.relu',
                'torch.Tensor.relu',
            },
        ),
    ],
)
def test_get_torch_op_aliases_from_runtime(runtime_fn, expected):
    aliases = _get_torch_op_aliases(runtime_fn)
    assert expected.issubset(aliases)
    assert len(aliases) == len(set(aliases))
    for alias in aliases:
        assert callable(load_type(alias))


def test_get_torch_op_aliases_is_symmetric():
    expected = set(_get_torch_op_aliases(torch.add))
    assert 'torch.Tensor.add_' not in expected
    assert not any(alias.startswith('_operator.') for alias in expected)
    for runtime_fn in (torch.add, torch.Tensor.add):
        assert set(_get_torch_op_aliases(runtime_fn)) == expected


def test_get_torch_op_aliases_does_not_match_functional_by_name():
    assert _get_torch_op_aliases(F.batch_norm) == ('torch.nn.functional.batch_norm',)
    assert 'torch.nn.functional.relu' not in _get_torch_op_aliases(torch.relu)


@replace_all_device_with('cpu')
def test_functional_fake_fn_does_not_override_same_name_torch_op():
    calls = []

    def fake_batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
    ):
        calls.append(input)
        return input

    nnscaler.update_op(F.batch_norm, fake_fn=fake_batch_norm)
    try:
        traced = to_fx_graph(
            BuiltinBatchNormModel(),
            {
                'x': torch.randn(2, 3, 4, 4),
                'weight': torch.ones(3),
                'bias': torch.zeros(3),
                'running_mean': torch.zeros(3),
                'running_var': torch.ones(3),
            },
        )
    finally:
        _remove_updated_op(F.batch_norm)

    assert not calls
    node = next(node for node in traced.graph.nodes if node.op == 'call_function')
    assert node.target is torch.batch_norm


def test_get_torch_op_aliases_scans_runtime_namespaces(monkeypatch):
    monkeypatch.setattr(torch, 'nnscaler_test_add_alias', torch.add, raising=False)
    assert 'torch.nnscaler_test_add_alias' in _get_torch_op_aliases(torch.add)


def test_get_torch_op_aliases_keeps_custom_op_independent():
    signature = get_full_qualified_name(mock_add)
    assert _get_torch_op_aliases(mock_add) == (signature,)
    assert _get_torch_op_aliases(signature) == (signature,)


def test_get_torch_op_aliases_does_not_flatten_nested_torch_namespaces():
    assert _get_torch_op_aliases(torch.linalg.norm) == ('torch._C._linalg.linalg_norm',)


def test_update_builtin_function_fake_fn():
    calls = []

    def fake_add(x, y):
        calls.append((x, y))
        return x

    nnscaler.update_op(torch.add, fake_fn=fake_add)
    try:
        assert 'torch.add' not in CustomizedOps.kOpMap
        assert 'torch.add' not in CustomizedOps.kOpCodeDef
        assert CustomizedOps.kOpRuntime['torch.add'] is torch.add
        assert CustomizedOps.kOpFakeRuntime['torch.add'] is fake_add
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
        _remove_updated_op('torch.Tensor.add')

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
        _remove_updated_op('torch.Tensor.add')

    node = next(node for node in traced.graph.nodes if node.op == 'call_method')
    assert len(calls) == 1
    assert node.target == 'add'


def test_update_builtin_inplace_method_fake_fn():
    calls = []

    def fake_add_(x, y):
        calls.append((x, y))
        return x

    nnscaler.update_op(torch.Tensor.add_, fake_fn=fake_add_)
    try:
        dummy_input = {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)}
        original_x = dummy_input['x'].clone()
        traced = to_fx_graph(BuiltinInplaceMethodFakeFnModel(), dummy_input)
        with tempfile.TemporaryDirectory() as tempdir:
            ir_graph = to_ir_graph(traced, dummy_input, tempdir, constant_folding=False)
    finally:
        _remove_updated_op('torch.Tensor.add_')

    node = next(node for node in traced.graph.nodes if node.op == 'call_method')
    assert len(calls) == 1
    assert torch.equal(dummy_input['x'], original_x)
    assert node.target == 'add_'
    assert ir_graph.nodes()[-1].signature == 'torch.Tensor.add_'


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
        aliases = _get_torch_op_aliases('torch.add')
        for alias in aliases:
            assert CustomizedOps.kOpMap[alias] is create_add
            assert callable(CustomizedOps.kOpRuntime[alias])
            assert CustomizedOps.kOpFakeRuntime[alias] is fake_add
            assert CustomizedOps.kOpCodeDef[alias] == 'def updated_add(): pass'
            assert CustomizedOps.kOpEmit[alias] is emit_add
            assert CustomizedOps.kOpInputGen[alias] is generate_inputs
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

    def create_add(*args, signature=None, **kwargs):
        return None

    nnscaler.update_op(
        torch.add,
        op_create_fn=create_add,
        code='def updated_add(): pass',
        fake_fn=fake_add,
        emit_fn=emit_add,
        input_gen_fn=generate_inputs,
    )
    nnscaler.update_op(
        torch.add,
        op_create_fn=None,
        code=None,
        fake_fn=None,
        emit_fn=None,
        input_gen_fn=None,
    )
    try:
        aliases = _get_torch_op_aliases('torch.add')
        assert all(alias not in CustomizedOps.kOpMap for alias in aliases)
        assert all(alias not in CustomizedOps.kOpCodeDef for alias in aliases)
        assert all(alias not in CustomizedOps.kOpFakeRuntime for alias in aliases)
        assert all(alias not in CustomizedOps.kOpEmit for alias in aliases)
        assert all(alias not in CustomizedOps.kOpInputGen for alias in aliases)
    finally:
        _remove_updated_op('torch.add')


def test_update_builtin_input_gen_fn_used_by_profiler():
    class InputGenCalled(Exception):
        pass

    calls = []

    def generate_inputs(node):
        calls.append(node)
        raise InputGenCalled

    nnscaler.update_op(torch.add, input_gen_fn=generate_inputs)
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            ir_graph = convert_model(
                BuiltinFunctionFakeFnModel(),
                {'x': torch.rand(10, 10), 'y': torch.rand(10, 10)},
                tempdir,
                False,
            )
        node = ir_graph.nodes()[0]
        func, shapes, dtypes, requires_grads, values, kwargs = get_func(node)
        with pytest.raises(InputGenCalled):
            profile(node, func, shapes, dtypes, requires_grads, values, **kwargs)
    finally:
        _remove_updated_op('torch.add')

    assert calls == [node]


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


def test_update_unregistered_op_metadata():
    def create_unregistered_op(*args, signature=None, **kwargs):
        return None

    signature = get_full_qualified_name(unregistered_op)
    try:
        nnscaler.update_op(
            unregistered_op,
            op_create_fn=create_unregistered_op,
            code='def unregistered_op(x): return x',
        )
        assert CustomizedOps.kOpMap[signature] is create_unregistered_op
        assert CustomizedOps.kOpCodeDef[signature] == 'def unregistered_op(x): return x'
        assert CustomizedOps.kOpRuntime[signature] is unregistered_op
        assert signature not in CustomizedOps.kOpFakeRuntime
    finally:
        _remove_updated_op(signature)
