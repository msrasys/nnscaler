#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging

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


class ItemModel(torch.nn.Module):
    def forward(self, x):
        return x * x.sum().item()


class BranchOnValueModel(torch.nn.Module):
    def forward(self, x):
        if (x == 0).any():
            x = x + 1
        return x.sum()


class NonzeroModel(torch.nn.Module):
    def forward(self, x):
        return x[x > 0].sum()


class ToListModel(torch.nn.Module):
    def forward(self, x):
        return x * len(x.tolist())


class InplaceInConeModel(torch.nn.Module):
    def forward(self, x):
        keep = x > 0
        keep[0] = False                 # setitem on the mask
        alias = keep.view(-1)
        alias.logical_not_()            # in-place through an alias: `keep` is mutated too
        return x[keep.nonzero().flatten()].sum()


class KeywordInplaceModel(torch.nn.Module):
    def forward(self, x):
        h = x.clone()
        torch.nn.functional.relu(h, inplace=True)   # mutates `h`; the returned value is unused
        return x[(h == 0).nonzero().flatten()].sum()  # `h == 0` references the clone node


class InputSetitemModel(torch.nn.Module):
    def forward(self, x):
        x[0] = 0.0                                   # setitem on the dummy input itself
        return x[(x == 0).nonzero().flatten()].sum()


class DataDependentInplaceModel(torch.nn.Module):
    def forward(self, x):
        h = x.clone()
        h[(x > 0).nonzero().flatten()] = 0.0         # in-place write at data-dependent positions
        return x[(h == 0).nonzero().flatten()].sum()


class FunctionInplaceModel(torch.nn.Module):
    def forward(self, x):
        h = x.clone()
        torch.relu_(h)                               # function form of an in-place op; result unused
        return x[(h == 0).nonzero().flatten()].sum()


class Double(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, g):
        return g * 2


class KeepPositive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x[x > 0]                              # data dependent inside an autograd function

    @staticmethod
    def backward(ctx, g):
        return None


class NestedAutogradModel(torch.nn.Module):
    def forward(self, x):
        return KeepPositive.apply(Double.apply(x)).sum()


class DirectSetitemModel(torch.nn.Module):
    def forward(self, x):
        x[0] = 0.0
        return torch.nonzero(x)                      # consumes the setitem result directly


class BufferInplaceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('b', torch.tensor([1.0, -1.0, 2.0, 0.5]))

    def forward(self, x):
        self.b.mul_(0.0)                             # mutation through one access of the buffer ...
        return x.sum() + torch.nonzero(self.b).shape[0]   # ... must be seen by the next access


class ParameterInConeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, -1.0, 2.0, 0.5]))

    def forward(self, x):
        return x[(self.weight > 0).nonzero().flatten()].sum()


class RandomInConeModel(torch.nn.Module):
    def forward(self, x):
        r = torch.rand_like(x.new_empty(64))   # random values produced on the strategy device
        count = (r > 0.5).nonzero().shape[0]   # replayed once ...
        total = int((r > 0.5).sum().item())    # ... and again: both replays must see the same r
        return x * (count - total)


def _trace_signature(model, inputs):
    graph = to_fx_graph(model, inputs)
    def meta(n):
        value = n.meta.get('tensor_meta')
        return tuple(value.shape) if hasattr(value, 'shape') else value   # python values are stored as they are
    return [(n.op, str(n.target), str(n.args), meta(n)) for n in graph.graph.nodes]


@pytest.mark.parametrize('strategy', [
    'meta:cpu',
    pytest.param('meta:cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')),
])
@pytest.mark.parametrize('model_class', [
    ItemModel, BranchOnValueModel, NonzeroModel, ToListModel, InplaceInConeModel, KeywordInplaceModel,
    InputSetitemModel, DataDependentInplaceModel, DirectSetitemModel, BufferInplaceModel, ParameterInConeModel,
    FunctionInplaceModel, NestedAutogradModel])
def test_meta_strategy_replays_data_dependent_ops(monkeypatch, model_class, strategy):
    """Data-dependent operators replay their traced inputs on the fallback device, so meta traces the same graph as cpu."""
    inputs = {'x': torch.tensor([0.5, 0.0, -1.0, 2.0])}
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'cpu')
    expected = _trace_signature(model_class(), inputs)
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    assert _trace_signature(model_class(), inputs) == expected


@pytest.mark.parametrize('strategy', ['meta', 'meta:no'])
def test_meta_without_replay_option_raises(monkeypatch, strategy):
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    with pytest.raises(RuntimeError, match="cannot execute") as exc_info:
        to_fx_graph(NonzeroModel(), {'x': torch.rand(4)})
    assert "'meta:cpu' or 'meta:cuda'" in str(exc_info.value) and 'fake_fn' in str(exc_info.value)


def test_trace_strategy_option_validation():
    from nnscaler.parallel import ComputeConfig
    with pytest.raises(ValueError, match="accepts the options"):
        ConcreteTracer('meta:gpu')
    with pytest.raises(ValueError, match="takes no ':option'"):
        ConcreteTracer('cpu:fast')
    with pytest.raises(ValueError, match="unknown trace strategy"):
        ConcreteTracer('tpu')
    with pytest.raises(ValueError, match="accepts the options"):
        ComputeConfig(1, 1, trace_strategy='meta:gpu', param_init_strategy='shard')
    with pytest.raises(ValueError, match="requires param_init_strategy='shard'"):
        ComputeConfig(1, 1, trace_strategy='meta')
    config = ComputeConfig(1, 1, trace_strategy='meta:no', param_init_strategy='shard')
    assert (config.trace_strategy_name, config.trace_strategy_option) == ('meta', 'no')
    assert (ComputeConfig(1, 1).trace_strategy_name, ComputeConfig(1, 1).trace_strategy_option) == ('cuda_run_cpu_offload', '')
    # the config is also loaded by processes without CUDA; only tracing needs the device
    assert ComputeConfig(1, 1, trace_strategy='meta:cuda', param_init_strategy='shard').trace_strategy == 'meta:cuda'


class DictKeyModel(torch.nn.Module):
    def forward(self, data):
        return data['cuda'] + data.get('cuda', 0)   # container keys that happen to spell device names


class ReturnsParameterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0, -1.0, 2.0]))
        self.register_buffer('b', torch.tensor([1.0, 2.0, 3.0]))

    def forward(self, x):
        w = self.w.to(x.dtype)                        # returns the parameter object itself
        self.b.mul_(0.0)
        return x.sum() + w.sum() + torch.nonzero(self.b).shape[0]


def test_meta_trace_does_not_pin_graph_nodes_on_module_tensors(monkeypatch):
    """The same instance can be traced again: parameters never carry a node of a previous graph."""
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'meta:cpu')
    model = ReturnsParameterModel()
    inputs = {'x': torch.rand(3)}
    first = _trace_signature(model, inputs)
    for tensor in list(model.parameters()) + list(model.buffers()):
        assert not hasattr(tensor, '_nnscaler_meta_origin')
    assert _trace_signature(model, inputs) == first


@pytest.mark.parametrize('strategy', ['cpu', 'meta'])
def test_device_mapping_leaves_container_keys_alone(monkeypatch, strategy):
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    # traces without KeyError: the keys are not rewritten when the lookups execute
    to_fx_graph(DictKeyModel(), {'data': {'cuda': torch.ones(2), 'cpu': torch.ones(2)}})


def test_meta_strategy_replays_random_values_consistently(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'meta:cpu')
    inputs = {'x': torch.rand(4)}
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    graph = to_fx_graph(RandomInConeModel(), inputs)
    mul = next(n for n in graph.graph.nodes if n.op == 'call_function' and n.target.__name__ == 'mul')
    assert mul.args[1].meta['tensor_meta'] == 0   # count - total, both read from the same replayed `r`
    # the replay's seeding must not leak into the caller's generators
    assert torch.equal(torch.get_rng_state(), cpu_state)
    assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], cuda_state))


class CpuFactoryRandomModel(torch.nn.Module):
    def forward(self, x):
        r = torch.rand(64)                            # no device: runs on cpu even under the meta strategy
        count = torch.nonzero(r > 0.5).shape[0]       # consumes the placed copy of `r` directly ...
        total = int((r > 0.5).sum().item())           # ... and through a node: both see the same replayed `r`
        return x * (count - total)


def test_meta_strategy_replays_cpu_factory_outputs_consistently(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'meta:cpu')
    graph = to_fx_graph(CpuFactoryRandomModel(), {'x': torch.rand(4)})
    mul = next(n for n in graph.graph.nodes if n.op == 'call_function' and n.target.__name__ == 'mul')
    assert mul.args[1].meta['tensor_meta'] == 0


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


class ParameterItemModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))

    def forward(self, x):
        return x * self.weight.sum().item()


def test_meta_strategy_reports_unreplayable_values(monkeypatch):
    """A meta parameter without captured construction has no values to replay; the error says what to do."""
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'meta:cpu')
    model = ParameterItemModel().to_empty(device='meta')
    with pytest.raises(RuntimeError, match="cannot execute `item`") as exc_info:
        to_fx_graph(model, {'x': torch.rand(4)})
    message = str(exc_info.value)
    assert 'Replaying' in message and 'fake_fn' in message
    assert exc_info.value.__cause__ is not None


class TraceFakeModel(torch.nn.Module):
    """Every data-dependent read is given a call-site fake; nothing is executed or replayed for them."""
    def forward(self, x):
        positive = x > 0
        idx = nnscaler.trace_fake(torch.nonzero, torch.zeros(3, 1, dtype=torch.long), positive)
        # a callable fake sees the arguments as the trace does and derives the result from their shapes
        rows, = nnscaler.trace_fake(torch.where, lambda mask: (torch.zeros(mask.numel() - 1, dtype=torch.long),), positive)
        count = nnscaler.trace_fake(torch.Tensor.item, 3, positive.sum())
        if nnscaler.trace_fake(bool, True, positive.any()):
            x = x * 2
        return x[idx.flatten()].sum() + rows.shape[0] + count


class ReplayUsesFakeModel(torch.nn.Module):
    """A later replay stops at the faked node and takes its value."""
    def forward(self, x):
        idx = nnscaler.trace_fake(torch.nonzero, torch.zeros(3, 1, dtype=torch.long), x > 0)
        return x * int((idx > 0).sum().item())      # .item() replays: `nonzero` returns the fake, `x > 0` is not run


def _nodes(graph, name):
    return [n for n in graph.graph.nodes if (n.target if isinstance(n.target, str) else getattr(n.target, '__name__', '')) == name]


@pytest.mark.parametrize('strategy', ['cpu', 'meta', 'meta:no'])
def test_trace_fake_replaces_execution(monkeypatch, caplog, strategy):
    """Like `fake_fn`, the fake is the trace value under every strategy; the graph records the real operator."""
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    with caplog.at_level(logging.WARNING, logger='nnscaler.graph.tracer.trace_strategy'):
        graph = to_fx_graph(TraceFakeModel(), {'x': torch.tensor([0.5, 0.0, -1.0, 2.0])})
    assert 'needs tensor data' not in caplog.text
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


def test_trace_fake_runs_fn_outside_tracing():
    x = torch.tensor([0.5, 0.0, -1.0, 2.0])
    assert torch.equal(nnscaler.trace_fake(torch.nonzero, torch.zeros(3, 1, dtype=torch.long), x > 0), torch.nonzero(x > 0))


def test_trace_fake_feeds_later_replays(monkeypatch, caplog):
    monkeypatch.setattr(CompileFlag, 'trace_strategy', 'meta:cpu')
    with caplog.at_level(logging.WARNING, logger='nnscaler.graph.tracer.trace_strategy'):
        graph = to_fx_graph(ReplayUsesFakeModel(), {'x': torch.tensor([0.5, 0.0, -1.0, 2.0])})
    assert 'replayed 3 traced nodes' in caplog.text                        # nonzero (fake), gt_1, sum: not x, gt
    int_node, = _nodes(graph, 'int')
    assert int_node.meta['tensor_meta'] == 0                                 # the fake is all zeros; real data gives 1


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
