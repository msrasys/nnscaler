# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from types import SimpleNamespace
import sys

import pytest
import torch

import nnscaler
import nnscaler.runtime.checkpoint_stream
from nnscaler.codegen.emit import FuncEmission
from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.codegen.module.module import ModuleCodeGen
from nnscaler.codegen.schedule.schedule import ScheduleCodeGen
from nnscaler.execplan import ExecutionPlan
from nnscaler.flags import CompileFlag
from nnscaler.graph.graph import IRGraph
from nnscaler.graph.schedule.module_streams import checkpoint_stream
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.runtime import checkpoint_tail


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        middle = x * scale
        ctx.save_for_backward(x, scale, middle)
        return middle.square()

    @staticmethod
    def backward(ctx, dy):
        x, scale, middle = ctx.saved_tensors
        return dy * 2 * middle * scale, dy * 2 * middle * x


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, index, value):
        ctx.save_for_backward(index, None)
        return value[index]

    @staticmethod
    def backward(ctx, dy):
        index, _ = ctx.saved_tensors
        return None, dy[torch.argsort(index)]


def product(x, scale):
    return _Product.apply(x, scale)


def gather(index, value):
    return _Gather.apply(index, value)


def build(monkeypatch, callback, *, extra_output=False, duplicate=False, stream='module_overlap', full=False):
    key = tuple(f'{f.__module__}.{f.__name__}' for f in (product, gather))
    monkeypatch.delitem(checkpoint_tail._tails, key, raising=False)
    nnscaler.register_checkpoint_tail(product, gather, input_index=1, rebuild=callback)
    rule = nnscaler.ModuleStreamRule('demo', stream=stream, recompute_only=True, recompute_saved_state=True)
    monkeypatch.setattr(CompileFlag, 'module_streams', (rule,))
    tensors = [IRFullTensor((3,), name=n, requires_grad=n != 'index',
                            dtype=torch.int64 if n == 'index' else torch.float32).tosub()
               for n in ('x', 'w', 'index', 'square', 'result')]
    x, w, index, middle, result = tensors
    a = IRFwOperation('product', key[0], [x, w], 1)
    b = IRFwOperation('gather', key[1], [middle if duplicate else index, middle], 1)
    for node, output in ((a, middle), (b, result)):
        node.set_output(0, output)
        node.device = (0,)
        node.recompute = 0
        node.module_stack = {'demo': torch.nn.Module}
    nodes, inputs = [a, b], [x, w, index]
    outputs = [result, middle] if extra_output else [result]
    if full:
        segment = IRSegment(nodes, inputs, outputs)
        graph = IRGraph([segment], inputs, outputs, 'test_caller_checkpoint')
        plan = ExecutionPlan(graph, [segment])
        return ModuleCodeGen(plan).gen(0), ScheduleCodeGen(plan).gen(0), plan
    emitter = ModuleCodeGen.__new__(ModuleCodeGen)
    FuncEmission.__init__(emitter)
    emitter.plan_ndevs = emitter.runtime_ndevs = 1
    emitter.checkpoint_tails = checkpoint_tail.registered_checkpoint_tails()
    emitter.execplan = SimpleNamespace(mark_module_streams=lambda: None)
    lines = emitter._emit_recompute(nodes, inputs, outputs, LifeCycle(nodes, inputs, outputs), 0)
    source = '\n'.join(['def run(' + ', '.join(map(emitter.tensor_name, inputs)) + '):',
                        *['    ' + line for line in lines],
                        '    return ' + emitter.tensor_name(result)])
    return source


@pytest.mark.parametrize('stream', ['module_overlap', 'default'])
@pytest.mark.parametrize(('applicable', 'early_stop'), [(True, True), (True, False), (False, True)])
def test_generated_tail_rebuild_preserves_values_and_gradients(monkeypatch, applicable, early_stop, stream):
    observations = []

    def rebuild(args, kwargs, following, other_kwargs):
        x, w = args
        index, omitted = following
        assert omitted is None and not kwargs and not other_kwargs
        observations.append(index.tolist())
        if not applicable:
            return False
        with torch.no_grad():
            middle = x * w
        nnscaler.checkpoint_save_for_backward(x, x, w, middle)
        nnscaler.checkpoint_save_for_backward(x, index, None)
        assert not early_stop, 'native checkpoint should have stopped at the last save'
        return True

    source = build(monkeypatch, rebuild, stream=stream)
    assert 'checkpoint_tail.rebuild' in source
    root = __name__.split('.')[0]
    namespace = {'torch': torch, 'nnscaler': nnscaler, root: sys.modules[root]}
    exec(source, namespace)
    x = torch.tensor([2., -3., 4.], requires_grad=True)
    w = torch.tensor([1., 2., 3.], requires_grad=True)
    index = torch.tensor([2, 0, 1])
    expected = gather(index, product(x, w))
    with torch.utils.checkpoint.set_checkpoint_early_stop(early_stop):
        actual = namespace['run'](x, w, index)
    assert not observations
    torch.testing.assert_close(actual, expected)
    for incoming in (torch.tensor([1., 2., 3.]), torch.tensor([2., 3., 1.])):
        check = torch.autograd.grad(expected, (x, w), incoming, retain_graph=True)
        got = torch.autograd.grad(actual, (x, w), incoming, retain_graph=True)
        torch.testing.assert_close(got, check)
    assert observations == [index.tolist()] * 2


@pytest.mark.parametrize('invalid', ['extra_output', 'duplicate'])
def test_falls_back_when_producer_value_escapes_its_declared_edge(monkeypatch, invalid):
    source = build(monkeypatch, lambda *args: True, **{invalid: True})
    assert 'nnscaler.runtime.checkpoint_tail.' not in source and 'ckpt.checkpoint' in source


def test_caller_rebuild_preserves_single_stream_execution_plan(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'max_pending_backward_tasks', 0)
    module, schedule, plan = build(monkeypatch, lambda *args: False, stream='default', full=True)
    assert 'nnscaler.runtime.checkpoint_tail.run(' in module
    assert 'context_fn=' not in module
    assert not plan.use_multi_streams and not plan.track_stream_inputs
    assert 'module_parameter_streams = {}' in module
    assert 'module_stream_parameters = []' in module
    assert 'nnscaler.runtime.stream.ModuleStreams(' not in module
    assert 'nnscaler.runtime.checkpoint_stream.contexts(' not in module
    assert 'cuda_task_queue(' not in schedule
    assert 'track_stream=False' in schedule


@pytest.mark.parametrize('options', [{}, {'recompute_saved_state': True, 'recompute_ready_on': 'dispatch'},
                                    {'recompute_prefetch_on': 'dispatch', 'recompute_saved_state': True}])
def test_caller_rebuild_rejects_other_stream_actions(options):
    with pytest.raises(ValueError):
        nnscaler.ModuleStreamRule('demo', stream='default', recompute_only=True, **options)


def test_caller_rebuild_is_local_to_its_matching_checkpoint():
    value = IRFullTensor((3,), name='x', requires_grad=True, dtype=torch.float32).tosub()
    node = IRFwOperation('op', 'torch.cos', [value], 1)
    node.module_stack = {'demo': torch.nn.Module}
    caller = nnscaler.ModuleStreamRule('demo', stream='default', recompute_only=True, recompute_saved_state=True)
    other = nnscaler.ModuleStreamRule('other', stream='side', recompute_only=True)
    assert checkpoint_stream([node], [other, caller]) == 'default'
    node.module_stack = {'other': torch.nn.Module}
    assert checkpoint_stream([node], [caller, other]) == 'side'
    prefetch = nnscaler.ModuleStreamRule('demo', stream='default', recompute_only=True,
                                        recompute_prefetch_on='dispatch')
    node.module_stack = {'demo': torch.nn.Module}
    with pytest.raises(ValueError, match='cannot share'):
        checkpoint_stream([node], [prefetch, caller])
