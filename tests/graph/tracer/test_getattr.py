#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.


import pytest
import torch

import nnscaler
from nnscaler.flags import CompileFlag
from nnscaler.graph.parser.converter import to_fx_graph
from nnscaler.graph.tracer.concrete_tracer import ConcreteTracer

from ...utils import replace_all_device_with


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.dropout(x, 0.1, self.training)


@replace_all_device_with('cpu')
def test_getattr_from_root():
    model = SimpleModel()
    dummy_input = {'x': torch.rand(10)}
    traced_graph = to_fx_graph(model, dummy_input)
    traced_graph(**dummy_input)


class TensorAttributeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.constant = torch.rand(4)
        self.child = torch.nn.Module()
        self.child.offset = torch.rand(4)

    def forward(self, x):
        return x + self.constant + self.child.offset


@pytest.mark.parametrize('strategy', ['cpu', 'meta', 'cuda', 'cuda_run_cpu_offload', 'reuse_cache'])
def test_tensor_attribute_names(monkeypatch, strategy):
    if strategy not in ('cpu', 'meta') and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    model = TensorAttributeModel()
    dummy_input = {'x': torch.rand(4)}
    expected = model(**dummy_input)
    traced = to_fx_graph(model, dummy_input)
    targets = {node.target for node in traced.graph.nodes if node.op == 'get_attr'}
    assert targets == {'constant', 'child.offset'}
    assert torch.equal(traced(**dummy_input), expected)


class DeviceArgumentModel(torch.nn.Module):
    def forward(self, x):
        pos = torch.arange(x.shape[0], device=x.device)
        z = torch.zeros(2, device='cuda')
        moved = x.cuda(non_blocking=True)          # method form of device='cuda'
        first = z.to(0, torch.float32)             # int positional device of `Tensor.to`
        host = moved.to('cpu').sum() + first.cpu().sum()   # explicit cpu: a move out of meta, no data needed
        return moved + pos.sum() + z.sum() + host


@pytest.mark.parametrize('strategy', ['cpu', 'meta'])
def test_cuda_device_arguments_run_on_strategy_device(monkeypatch, strategy):
    """`<tensor>.device` and hard-coded 'cuda' are executed on the strategy device, but recorded as cuda."""
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    # tracing must not need a GPU
    traced = to_fx_graph(DeviceArgumentModel(), {'x': torch.rand(4)})
    zeros_node = next(n for n in traced.graph.nodes if getattr(n.target, '__name__', None) == 'zeros')
    assert zeros_node.kwargs['device'] == 'cuda'
    arange_node = next(n for n in traced.graph.nodes if getattr(n.target, '__name__', None) == 'arange')
    assert arange_node.kwargs['device'].op == 'call_function'  # the `x.device` getattr node
    assert any(n.op == 'call_method' and n.target == 'cuda' for n in traced.graph.nodes)

    tracer = ConcreteTracer(strategy)
    value, _, kwargs = tracer.strategy.run_target(
        'call_function', torch.zeros, (2,), {'device': torch.device('cuda')})
    assert value.device.type == strategy
    assert kwargs['device'] == torch.device('cuda')
    value, _, _ = tracer.strategy.run_target('call_method', 'cuda', (torch.ones(2),), {'non_blocking': True})
    assert value.device.type == strategy


class NonzeroModel(torch.nn.Module):
    def forward(self, x):
        return x[torch.nonzero(x > 0).flatten()].sum()


class ItemModel(torch.nn.Module):
    def forward(self, x):
        return x * x.sum().item()


class BranchOnValueModel(torch.nn.Module):
    def forward(self, x):
        if (x == 0).any():
            x = x + 1
        return x.sum()


@pytest.mark.parametrize('model_class, operator', [(NonzeroModel, 'nonzero'), (ItemModel, 'item'), (BranchOnValueModel, 'bool')])
def test_meta_raises_on_data_dependent_ops(monkeypatch, model_class, operator):
    """Operators that read tensor values cannot run on meta; the error names the operator and the remedies."""
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'meta')
    with pytest.raises(RuntimeError, match=f"cannot execute `{operator}`") as exc_info:
        to_fx_graph(model_class(), {'x': torch.tensor([0.5, 0.0, -1.0, 2.0])})
    assert 'trace_fake' in str(exc_info.value) and 'fake_fn' in str(exc_info.value)
    assert exc_info.value.__cause__ is not None


def test_trace_strategy_validation():
    from nnscaler.parallel import ComputeConfig
    with pytest.raises(ValueError, match="unknown trace strategy"):
        ConcreteTracer('tpu')
    with pytest.raises(ValueError, match="unknown trace strategy"):
        ComputeConfig(1, 1, trace_strategy='meta:cpu', param_init_strategy='shard')
    with pytest.raises(ValueError, match="requires param_init_strategy='shard'"):
        ComputeConfig(1, 1, trace_strategy='meta')
    ComputeConfig(1, 1, trace_strategy='meta', param_init_strategy='shard')


class DictKeyModel(torch.nn.Module):
    def forward(self, data):
        return data['cuda'] + data.get('cuda', 0)   # container keys that happen to spell device names


class DeviceObjectKeyModel(torch.nn.Module):
    def forward(self, x, table):
        return x + table[torch.device('cuda')]       # a `torch.device` used as a dictionary key


class DeviceFieldModel(torch.nn.Module):
    def forward(self, x, cfg):
        cfg.update(device='cuda')                     # a `device` field of a dictionary, not a device parameter
        return x * (2 if cfg['device'] == 'cuda' else 3)


@pytest.mark.parametrize('strategy', ['cpu', 'meta'])
def test_device_mapping_leaves_container_keys_alone(monkeypatch, strategy):
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    # traces without KeyError: the keys are not rewritten when the lookups execute
    to_fx_graph(DictKeyModel(), {'data': {'cuda': torch.ones(2), 'cpu': torch.ones(2)}})
    to_fx_graph(DeviceObjectKeyModel(), {'x': torch.ones(2), 'table': {torch.device('cuda'): torch.ones(2)}})
    graph = to_fx_graph(DeviceFieldModel(), {'x': torch.ones(2), 'cfg': {'device': 'cpu'}})
    mul, = [n for n in graph.graph.nodes if getattr(n.target, '__name__', '') == 'mul']
    assert mul.args[1] == 2                            # the field kept 'cuda', so the trace took the same branch as runtime


def test_patcher_contexts_nest_in_the_supported_order():
    from nnscaler.graph.tracer.function_patcher import FunctionPatcher
    import builtins
    original = builtins.range
    patcher = FunctionPatcher()
    with patcher:
        patcher.patch_method(builtins, 'range', lambda *a: 'patched', deduplicate=False)
        assert builtins.range is not original
        with patcher.revert():
            assert builtins.range is original
            with patcher.revert():                     # reentrant
                assert builtins.range is original
            with patcher.revert_namespace(builtins):   # a no-op inside revert()
                assert builtins.range is original
            assert builtins.range is original
        assert builtins.range is not original
        with patcher.revert_namespace(builtins):
            assert builtins.range is original
        assert builtins.range is not original
    assert builtins.range is original


class TraceFakeModel(torch.nn.Module):
    """Every data-dependent read is given a call-site fake; none of them is executed."""
    def forward(self, x):
        positive = x > 0
        idx = nnscaler.trace_fake(torch.nonzero, torch.zeros(3, 1, dtype=torch.long), positive)
        # a callable fake sees the arguments as the trace does and derives the result from their shapes
        rows, = nnscaler.trace_fake(torch.where, lambda mask: (torch.zeros(mask.numel() - 1, dtype=torch.long),), positive)
        count = nnscaler.trace_fake(torch.Tensor.item, 3, positive.sum())
        if nnscaler.trace_fake(bool, True, positive.any()):
            x = x * 2
        return x[idx.flatten()].sum() + rows.shape[0] + count


def _nodes(graph, name):
    return [n for n in graph.graph.nodes if (n.target if isinstance(n.target, str) else getattr(n.target, '__name__', '')) == name]


@pytest.mark.parametrize('strategy', ['cpu', 'meta'])
def test_trace_fake_replaces_execution(monkeypatch, strategy):
    """Like `fake_fn`, the fake is the trace value under every strategy; the graph records the real operator."""
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    graph = to_fx_graph(TraceFakeModel(), {'x': torch.tensor([0.5, 0.0, -1.0, 2.0])})
    nonzero, = _nodes(graph, 'nonzero')
    assert nonzero.op == 'call_function' and nonzero.target is torch.nonzero
    assert tuple(nonzero.meta['tensor_meta'].shape) == (3, 1)              # the fake's shape, not the data's (2, 1)
    assert nonzero.args[0].target.__name__ == 'gt'                           # original argument, not the fake
    where, = _nodes(graph, 'where')
    assert where.target is torch.where and tuple(where.meta['tensor_meta'][0].shape) == (3,)
    item, = _nodes(graph, 'item')
    assert item.op == 'call_method' and item.meta['tensor_meta'] == 3        # the fake; real data gives 2
    assert _nodes(graph, 'mul')                                              # the faked `bool` chose the branch
    assert not any(getattr(n.target, '__name__', '') == 'trace_fake' for n in graph.graph.nodes)


def count_positive_rows(x):
    # a registered operator whose body reads values; under meta it has to be given a fake as a whole
    rows = nnscaler.trace_fake(torch.nonzero, torch.zeros(2, 1, dtype=torch.long), x > 0)
    count_positive_rows.seen.append(type(rows).__name__)
    return x * rows.shape[0]


count_positive_rows.seen = []
nnscaler.register_op('n -> n')(count_positive_rows)


class FakeInsideLeafModel(torch.nn.Module):
    def forward(self, x):
        return count_positive_rows(x).sum()


def test_trace_fake_inside_a_leaf_body_runs_fn(monkeypatch):
    """The body of a registered operator is one node: a `trace_fake` inside it records nothing and runs `fn`."""
    count_positive_rows.seen.clear()
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'cpu')
    graph = to_fx_graph(FakeInsideLeafModel(), {'x': torch.tensor([0.5, 0.0, -1.0, 2.0])})
    assert [n.target.__name__ for n in graph.graph.nodes if n.op == 'call_function'] == ['count_positive_rows']
    assert count_positive_rows.seen == ['Tensor']                                   # a real result, not a proxy
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'meta')
    with pytest.raises(RuntimeError, match="cannot execute `count_positive_rows`"):  # the leaf itself needs a fake_fn
        to_fx_graph(FakeInsideLeafModel(), {'x': torch.tensor([0.5, 0.0, -1.0, 2.0])})


class KeepPositive(torch.autograd.Function):
    """Reads values in forward, so under meta the whole apply needs a fake."""
    @staticmethod
    def forward(ctx, x):
        return x[x > 0]

    @staticmethod
    def backward(ctx, grad):
        return grad


class FakedAutogradModel(torch.nn.Module):
    def forward(self, x):
        return nnscaler.trace_fake(KeepPositive.apply, torch.zeros(3), x).sum()


@pytest.mark.parametrize('strategy', ['cpu', 'meta'])
def test_trace_fake_accepts_autograd_apply(monkeypatch, strategy):
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    graph = to_fx_graph(FakedAutogradModel(), {'x': torch.tensor([0.5, 0.0, -1.0, 2.0])})
    apply, = [n for n in graph.graph.nodes if getattr(n.target, '__name__', '') == 'apply']
    assert apply.target.__self__ is KeepPositive
    assert tuple(apply.meta['tensor_meta'].shape) == (3,)                            # the fake's shape; the data gives (2,)


def test_trace_fake_runs_fn_outside_tracing():
    x = torch.tensor([0.5, 0.0, -1.0, 2.0])
    assert torch.equal(nnscaler.trace_fake(torch.nonzero, torch.zeros(3, 1, dtype=torch.long), x > 0), torch.nonzero(x > 0))


def test_trace_fake_rejects_meta_fake(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'meta')

    class Model(torch.nn.Module):
        def forward(self, x):
            return nnscaler.trace_fake(torch.nonzero, torch.zeros(3, 1, dtype=torch.long, device='meta'), x > 0)

    with pytest.raises(ValueError, match='must carry data'):
        to_fx_graph(Model(), {'x': torch.rand(4)})


def _select_positive(mask):
    return torch.nonzero(mask).flatten()


def test_trace_fake_rejects_plain_functions(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'meta')

    class Model(torch.nn.Module):
        def forward(self, x):
            return x[nnscaler.trace_fake(_select_positive, torch.zeros(3, dtype=torch.long), x > 0)].sum()

    with pytest.raises(ValueError, match='not an operator the tracer records'):
        to_fx_graph(Model(), {'x': torch.rand(4)})
