# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import json
import copy
import functools
from dataclasses import asdict
from contextlib import contextmanager

import pytest
import torch

from nnscaler import ComputeConfig, ModuleStreamRule, parallelize
from nnscaler.parallel import _load_parallel_module_class
from ..utils import replace_all_device_with, new_empty


class Branch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(8, 16)
        self.down = torch.nn.Linear(16, 8)

    def forward(self, x):
        return self.down(self.up(x).relu())


class ForkModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.side = Branch()

    def forward(self, x):
        y = torch.sin(x)
        y = torch.cos(y)
        return y + self.side(x)


class EndToEndFork(ForkModel):
    def forward(self, data):
        return super().forward(data).sum()


class ReshapeBranch(Branch):
    def forward(self, x):
        return self.down(torch.nn.functional.relu(self.up(x.reshape(-1, 8))))


class LayoutModel(ForkModel):
    def __init__(self):
        super().__init__()
        self.side = ReshapeBranch()


class TiedBranches(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Linear(8, 8)
        self.side = torch.nn.Linear(8, 8)
        self.side.weight = self.main.weight

    def forward(self, x):
        return self.main(x) + self.side(x)


def layout_policy(graph, config):
    from nnscaler.policies import OpPlan, OpPartition, get_pas_ops
    for node in get_pas_ops(graph):
        partition = OpPartition(0, 0) if node.signature in ('torch.sin', 'torch.nn.functional.linear') else 'auto'
        yield OpPlan(node, partition=partition)


class DependentModel(ForkModel):
    def forward(self, x):
        y = torch.sin(x)
        y = torch.cos(y)
        return y + self.side(y)


RULE = ModuleStreamRule('side', launch_before='torch.sin', split_before='down', resume_before='torch.cos')
AFTER_RULE = ModuleStreamRule('side', launch_after='torch.sin', split_before='down', resume_after='torch.cos')


def test_environment_rules_are_frozen_into_cache_config(monkeypatch):
    baseline = ComputeConfig(1, 1)
    monkeypatch.setenv('NNSCALER_MODULE_STREAMS', json.dumps([dict(module='side', launch_before='torch.sin')]))
    enabled = ComputeConfig(1, 1)
    monkeypatch.delenv('NNSCALER_MODULE_STREAMS')
    assert enabled.module_streams == (ModuleStreamRule('side', launch_before='torch.sin'),)
    assert not ComputeConfig.safe_equals(baseline, enabled)
    assert baseline.graph_config == enabled.graph_config
    assert not ComputeConfig(1, 1).module_streams


@replace_all_device_with('cpu', force=True)
def test_async_copy_operators_without_module_streams(tmp_path, monkeypatch):
    baseline = ComputeConfig(1, 1)
    monkeypatch.setenv('NNSCALER_ASYNC_COPY_OPS', '["torch.sin"]')
    enabled = ComputeConfig(1, 1)
    assert enabled.async_copy_ops == ('torch.sin',)
    assert baseline.graph_config == enabled.graph_config
    assert not ComputeConfig.safe_equals(baseline, enabled)
    model = ForkModel()
    reference = copy.deepcopy(model)
    parallelize(model, {'x': torch.randn(4, 8)}, 'dp', enabled,
                gen_savedir=tmp_path, instance_name='async_only', load_module=False)
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    assert code.count('with nnscaler.runtime.stream.AsyncTensorCopies():') == 1
    assert 'with nnscaler.runtime.stream.ModuleStreams()' not in code
    cls = _load_parallel_module_class(ForkModel, gen_savedir=tmp_path, instance_name='async_only', rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(ref_x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)


def test_stream_codegen_version_invalidates_old_packages():
    current = ComputeConfig(1, 1, module_streams=(RULE,))
    old_rule = copy.deepcopy(RULE)
    object.__setattr__(old_rule, '_codegen_version', 0)
    old = ComputeConfig(1, 1, module_streams=(old_rule,))
    assert not ComputeConfig.safe_equals(old, current)
    assert old.graph_config == current.graph_config
    assert ComputeConfig(1, 1, module_streams=(asdict(RULE),)) == current


def test_saved_stream_codegen_version_survives_normalization(tmp_path, monkeypatch):
    current = ComputeConfig(1, 1, module_streams=(RULE,))
    old = copy.deepcopy(current)
    object.__setattr__(old.module_streams[0], '_codegen_version', 0)
    path = tmp_path / 'compute_config.pt'
    ComputeConfig.safe_dump_to_file(old, path)
    loaded = ComputeConfig.safe_load_from_file(path, return_none_on_error=False)
    assert loaded.module_streams[0]._codegen_version == 0
    assert not ComputeConfig.safe_equals(loaded, current)
    payload = torch.load(path, weights_only=False)
    payload['module_streams'][0].pop('_codegen_version')
    torch.save(payload, path)
    assert not ComputeConfig.safe_equals(ComputeConfig.safe_load_from_file(path), current)
    payload.pop('module_streams')
    payload.pop('async_copy_ops')
    torch.save(payload, path)
    monkeypatch.setenv('NNSCALER_MODULE_STREAMS', json.dumps([dict(module='side')]))
    monkeypatch.setenv('NNSCALER_ASYNC_COPY_OPS', '["*.dispatch"]')
    legacy = ComputeConfig.safe_load_from_file(path, return_none_on_error=False)
    assert not legacy.module_streams and not legacy.async_copy_ops


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
def test_codegen_splits_branch_and_joins_at_consumer(tmp_path, workers):
    parallelize(ForkModel(), {'x': torch.randn(4, 8)}, 'dp',
                ComputeConfig(1, 2, module_streams=(RULE,)),
                gen_savedir=tmp_path, load_module=False, codegen_workers=workers)
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    ast.parse(code)
    assignments = [n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'module_parameter_streams' for t in n.targets)]
    affinities = ast.literal_eval(assignments[0].value)
    assert len(affinities) == 4 and set(affinities.values()) == {'module_overlap'}
    start = code.index('with nnscaler.runtime.stream.ModuleStreams()')
    code = code[start:]
    assert code.index(".fork('module_stream_0'") < code.index('torch.sin(')
    assert code.index('torch.sin(') < code.index(".fork('module_stream_1'") < code.index('torch.cos(')
    assert code.index('torch.cos(') < code.index(".join('module_stream_1')")


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
def test_scheduled_executor_tracks_intra_module_streams(tmp_path, workers):
    parallelize(EndToEndFork(), {'data': torch.randn(4, 8)}, 'data',
                ComputeConfig(1, 2, use_end2end=True, module_streams=(RULE,)),
                gen_savedir=tmp_path, load_module=False, codegen_workers=workers)
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    assert 'use_multi_streams = True' in code
    assert 'track_stream=True' in code


@replace_all_device_with('cpu', force=True)
def test_internal_layout_adapters_follow_their_branch(tmp_path):
    from nnscaler.policies import fn
    parallelize(LayoutModel(), {'x': torch.randn(4, 8)}, functools.partial(fn, policy=layout_policy),
                ComputeConfig(2, 2, module_streams=(RULE,)),
                gen_savedir=tmp_path, load_module=False)
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    ast.parse(code)
    branch = code[code.index(".fork('module_stream_0'"):]
    assert branch.index('split_allgather(') < branch.index('torch.nn.functional.linear(')
    assert branch.index('torch.nn.functional.linear(') < branch.index('torch.sin(')


@replace_all_device_with('cpu', force=True)
def test_shared_parameter_keeps_ordinary_gradient_stream(tmp_path):
    rule = ModuleStreamRule('side', launch_before='torch.nn.functional.linear')
    parallelize(TiedBranches(), {'x': torch.randn(4, 8)}, 'dp', ComputeConfig(1, 1, module_streams=(rule,)),
                gen_savedir=tmp_path, load_module=False)
    tree = ast.parse(next(tmp_path.rglob('gencode0.py')).read_text())
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'module_parameter_streams' for t in n.targets))
    affinities = ast.literal_eval(assignment.value)
    assert len(affinities) == 1
    assert all('bias' in name for name in affinities)


@replace_all_device_with('cpu', force=True)
def test_dependent_branch_cannot_move_ahead_of_producer(tmp_path):
    with pytest.raises(RuntimeError, match='Code generation failed') as error:
        parallelize(DependentModel(), {'x': torch.randn(4, 8)}, 'dp',
                    ComputeConfig(1, 1, module_streams=(RULE,)),
                    gen_savedir=tmp_path, load_module=False)
    assert 'violates dependency' in str(error.value.__cause__)


class RepeatedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ForkModel()

    def forward(self, x):
        return self.block(self.block(x))


class SerialStreams:
    """Execute generated stream code on CPU to check its numerical semantics."""
    def __enter__(self):
        self.results = {}
        return self

    @contextmanager
    def fork(self, *args, **kwargs):
        yield

    def publish(self, *args, **kwargs):
        pass

    def capture(self, *args):
        pass

    def after_backward(self, value, dependencies, keys):
        from nnscaler.runtime.deferred import after_backward
        return after_backward(value, dependencies)

    @contextmanager
    def backward_point(self, keys):
        yield

    def submit(self, key, name, function, args, tracked_inputs):
        self.results[key] = function(*args)

    def resolve(self, key, caller_indices=None):
        return self.results.pop(key)

    def join(self, *args):
        pass

    def __exit__(self, *args):
        pass


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('after', [False, True])
@pytest.mark.parametrize('host_concurrency', [False, True])
def test_repeated_invocations_preserve_outputs_and_gradients(tmp_path, monkeypatch, after, host_concurrency):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = RepeatedBlock()
    reference = copy.deepcopy(model)
    x = torch.randn(4, 8, requires_grad=True)
    rule = (ModuleStreamRule('block.side', launch_after='torch.sin', split_before='down', resume_after='torch.cos', host_concurrency=host_concurrency)
            if after else ModuleStreamRule('block.side', launch_before='torch.sin', split_before='down', resume_before='torch.cos', host_concurrency=host_concurrency))
    instance = f'cpu_{after}_{host_concurrency}'
    parallelize(model, {'x': x.detach()}, 'dp',
                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path,
                instance_name=instance, load_module=False)
    cls = _load_parallel_module_class(RepeatedBlock, gen_savedir=tmp_path, instance_name=instance, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    actual = generated(x)
    expected = reference(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    actual_dx = x.grad.clone()
    x.grad = None
    expected.sum().backward()
    torch.testing.assert_close(actual_dx, x.grad)
    # Merged original names avoid depending on generated parameter identifiers.
    for name, parameter in generated.named_parameters():
        original_name = generated.fullmap[name].orig_name
        torch.testing.assert_close(parameter.grad, reference.get_parameter(original_name).grad)


@pytest.mark.parametrize('kwargs', [dict(stream='default'), dict(split_before='down'),
                                    dict(module=''), dict(resume_before='torch.cos'),
                                    dict(recompute_only=True, launch_before='torch.sin')])
def test_invalid_rules(kwargs):
    with pytest.raises(ValueError):
        ModuleStreamRule(**(dict(module='side') | kwargs))


def recompute_side_policy(graph, config):
    from nnscaler.policies import pas_dp
    graph.recompute([node for node in graph.nodes()
                     if 'side' in (node.module_stack or {})])
    return pas_dp(graph, config)


@replace_all_device_with('cpu', force=True)
def test_split_checkpoint_rejected(tmp_path):
    with pytest.raises(RuntimeError, match='Code generation failed') as error:
        parallelize(ForkModel(), {'x': torch.randn(4, 8)}, recompute_side_policy,
                    ComputeConfig(1, 1, module_streams=(RULE,)),
                    gen_savedir=tmp_path, load_module=False)
    assert 'recompute region' in str(error.value.__cause__)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('host_concurrency', [False, True])
def test_whole_checkpoint_runs_on_selected_stream(tmp_path, monkeypatch, host_concurrency):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = ForkModel()
    ref = copy.deepcopy(model)
    rule = ModuleStreamRule('side', launch_before='torch.sin', host_concurrency=host_concurrency)
    instance = f'checkpoint_{host_concurrency}'
    parallelize(model, {'x': torch.randn(4, 8)}, recompute_side_policy,
                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path,
                instance_name=instance, load_module=False)
    cls = _load_parallel_module_class(ForkModel, gen_savedir=tmp_path, instance_name=instance, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8, requires_grad=True)
    y = generated(x)
    expected = ref(x)
    torch.testing.assert_close(y, expected)
    y.sum().backward()
    expected.sum().backward()
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(parameter.grad, ref.get_parameter(generated.fullmap[name].orig_name).grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('after', [False, True])
@pytest.mark.parametrize('host_concurrency', [False, True])
def test_generated_stream_module_cuda_outputs_and_gradients(tmp_path, after, host_concurrency):
    import torch.distributed as dist
    import nnscaler
    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/process_group', rank=0, world_size=1)
    try:
        nnscaler.init()
        model = RepeatedBlock().cuda()
        reference = copy.deepcopy(model)
        rule = (ModuleStreamRule('block.side', launch_after='torch.sin', split_before='down', resume_after='torch.cos', host_concurrency=host_concurrency)
                if after else ModuleStreamRule('block.side', launch_before='torch.sin', split_before='down', resume_before='torch.cos', host_concurrency=host_concurrency))
        x = torch.randn(4, 8, device='cuda', requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        generated = parallelize(model, {'x': x.detach()}, 'dp',
                                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path,
                                instance_name=f'cuda_{after}_{host_concurrency}')
        actual, expected = generated(x), reference(ref_x)
        actual.square().sum().backward()
        expected.square().sum().backward()
        generated.sync_grad()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, ref_x.grad)
        for name, parameter in generated.named_parameters():
            torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
    finally:
        nnscaler.uninit()
        if dist.is_initialized():
            dist.destroy_process_group()


@replace_all_device_with('cpu', force=True)
def test_after_launch_snapshots_readiness_before_the_anchor(tmp_path):
    parallelize(ForkModel(), {'x': torch.randn(4, 8)}, 'dp',
                ComputeConfig(1, 1, module_streams=(AFTER_RULE,)),
                gen_savedir=tmp_path, load_module=False)
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    ast.parse(code)
    assert code.index(".capture('module_stream_0')") < code.index('torch.sin(') < code.index(".fork('module_stream_0'")
    assert code.index(".capture('module_stream_1')") < code.index('torch.cos(') < code.index(".fork('module_stream_1'")


@replace_all_device_with('cpu', force=True)
def test_async_copy_context_is_scoped_to_anchors(tmp_path, monkeypatch):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    rule = ModuleStreamRule('side', launch_after='torch.sin', split_before='down',
                            resume_after='torch.cos', async_anchor_copies=True)
    model = ForkModel()
    reference = copy.deepcopy(model)
    parallelize(model, {'x': torch.randn(4, 8)}, 'dp', ComputeConfig(1, 1, module_streams=(rule,)),
                gen_savedir=tmp_path, instance_name='async_copies', load_module=False)
    cls = _load_parallel_module_class(ForkModel, gen_savedir=tmp_path, instance_name='async_copies', rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8)
    torch.testing.assert_close(generated(x), reference(x))
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    assert code.count('with nnscaler.runtime.stream.AsyncTensorCopies():') == 2


@replace_all_device_with('cpu', force=True)
def test_recompute_only_rule_keeps_forward_on_caller(tmp_path):
    model = ForkModel()
    reference = copy.deepcopy(model)
    rule = ModuleStreamRule('side', stream='expert_recompute', recompute_only=True)
    parallelize(model, {'x': torch.randn(4, 8)}, recompute_side_policy,
                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path,
                instance_name='recompute_only', load_module=False)
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    assert 'checkpoint_stream.contexts(' in code
    assert '_module_streams.fork(' not in code
    cls = _load_parallel_module_class(ForkModel, gen_savedir=tmp_path, instance_name='recompute_only', rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8, requires_grad=True)
    actual, expected = generated(x), reference(x)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)


@replace_all_device_with('cpu', force=True)
def test_recompute_stream_rejects_part_of_a_checkpoint(tmp_path):
    rule = ModuleStreamRule('side.up', recompute_only=True)
    with pytest.raises(RuntimeError, match='Code generation failed') as error:
        parallelize(ForkModel(), {'x': torch.randn(4, 8)}, recompute_side_policy,
                    ComputeConfig(1, 1, module_streams=(rule,)),
                    gen_savedir=tmp_path, load_module=False)
    assert 'whole checkpoint region' in str(error.value.__cause__)


_anchor_backward_order = []


class DispatchAnchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        from nnscaler import notify_async_launch, capture_async_backward
        ctx.point = capture_async_backward()
        notify_async_launch()
        return value * 2

    @staticmethod
    def backward(ctx, grad):
        _anchor_backward_order.append('dispatch')
        if ctx.point is not None:
            ctx.point.record_ready()
        return grad * 2


class CombineAnchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        from nnscaler import notify_async_launch, capture_async_backward
        ctx.point = capture_async_backward()
        notify_async_launch()
        return value * 3

    @staticmethod
    def backward(ctx, grad):
        _anchor_backward_order.append('combine')
        if ctx.point is not None:
            ctx.point.record_ready()
        return grad * 3


from nnscaler import register_op


@register_op('a b -> a b')
def dispatch_anchor(x):
    from nnscaler import prepare_async_input
    return DispatchAnchor.apply(prepare_async_input(x))


@register_op('a b -> a b')
def combine_anchor(x):
    from nnscaler import prepare_async_input
    return CombineAnchor.apply(prepare_async_input(x))


class AnchorModel(ForkModel):
    def forward(self, x):
        y = dispatch_anchor(x)
        y = torch.tanh(y)
        y = combine_anchor(y)
        return y + self.side(x)


class CPUDeferredStreams(SerialStreams):
    def __enter__(self):
        self.states = {}
        self.native = {}
        self.native_results = {}
        return self

    def capture(self, key):
        pass

    def prepare(self, key, stream, function, parent, inputs, attrs, specs):
        from nnscaler.runtime.deferred import DeferredRegion
        state = DeferredRegion(function, parent, attrs.keys(), specs)
        self.states[key] = state
        return state.prepare(inputs, tuple(attrs.values()))

    def schedule_native(self, key, stream, function, parent, inputs, attrs):
        self.native[key] = (function, parent, inputs)

    def native_result(self, key):
        return self.native_results.pop(key)

    def native_value(self, key, index):
        return self.native_results[key][index]

    @contextmanager
    def launch_point(self, keys, *, native=False, backward_keys=()):
        from nnscaler.runtime.deferred import LaunchPoint, NativeLaunchPoint
        if native:
            def launch(key):
                function, parent, inputs = self.native.pop(key)
                inputs = inputs() if callable(inputs) else inputs
                self.native_results[key] = function(parent, *inputs)
            dependencies = lambda: tuple(value for key in backward_keys for value in self.native_results[key])
            point = NativeLaunchPoint(tuple(lambda key=key: launch(key) for key in keys),
                                      lambda ready: None, dependencies)
        else:
            point = LaunchPoint(tuple(self.states[key].launch for key in keys))
        with point:
            yield

    def publish(self, outputs, **kwargs):
        pass

    def join(self, key):
        if key in self.states:
            assert self.states[key].launched

    def __exit__(self, *exc):
        if exc[0] is None:
            assert all(state.launched for state in self.states.values())


def _anchor_rule(launch, tied=False):
    native = launch.startswith('input_ready')
    point = 'on' if native else launch
    options = {f'launch_{point}': '*.dispatch_anchor'}
    if not tied:
        options.update(split_before='down', **{f'resume_{point}':
                       '*.dispatch_anchor' if launch.endswith('_same') else '*.combine_anchor'})
    return ModuleStreamRule('side', overlap_backward=True,
                            launch_phase='input_ready' if native else 'submitted', **options)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('launch', ['on', 'before', 'input_ready', 'input_ready_same'])
def test_deferred_codegen_preserves_math_and_orders_both_backward_regions(tmp_path, monkeypatch, workers, launch):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', CPUDeferredStreams)
    model = AnchorModel()
    reference = copy.deepcopy(model)
    rule = _anchor_rule(launch)
    parallelize(model, {'x': torch.randn(4, 8, requires_grad=True)}, 'dp',
                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path,
                instance_name=f'launch_points_{launch}_{workers}', load_module=False, codegen_workers=workers)
    cls = _load_parallel_module_class(AnchorModel, gen_savedir=tmp_path,
                                      instance_name=f'launch_points_{launch}_{workers}', rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    for name, parameter in generated.named_parameters():
        original = generated.fullmap[name].orig_name
        if original.endswith('up.weight'):
            parameter.register_hook(lambda grad: _anchor_backward_order.append('up'))
        if original.endswith('down.weight'):
            parameter.register_hook(lambda grad: _anchor_backward_order.append('down'))
    for _ in range(2):
        x = torch.randn(4, 8, requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        actual, expected = generated(x), reference(ref_x)
        torch.testing.assert_close(actual, expected)
        _anchor_backward_order.clear()
        actual.square().sum().backward()
        assert _anchor_backward_order.index('combine') < _anchor_backward_order.index('down')
        assert _anchor_backward_order.index('dispatch') < _anchor_backward_order.index('up')
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
        for name, parameter in generated.named_parameters():
            torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)


class TiedAnchorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Linear(8, 8, bias=False)
        self.side = torch.nn.Linear(8, 8, bias=False)
        self.side.weight = self.main.weight

    def forward(self, x):
        return dispatch_anchor(self.main(x)) + self.side(x)


@replace_all_device_with('cpu', force=True)
def test_deferred_tied_parameters_and_multiple_live_forwards(tmp_path, monkeypatch):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', CPUDeferredStreams)
    model = TiedAnchorModel()
    reference = copy.deepcopy(model)
    rule = ModuleStreamRule('side', launch_on='*.dispatch_anchor', overlap_backward=True)
    parallelize(model, {'x': torch.randn(4, 8, requires_grad=True)}, 'dp',
                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path,
                instance_name='tied_deferred', load_module=False)
    cls = _load_parallel_module_class(TiedAnchorModel, gen_savedir=tmp_path,
                                      instance_name='tied_deferred', rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    values = [torch.randn(4, 8, requires_grad=True) for _ in range(3)]
    references = [x.detach().clone().requires_grad_() for x in values]
    actual = [generated(x) for x in values]
    expected = [reference(x) for x in references]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    sum(x.square().sum() for x in actual).backward()
    sum(x.square().sum() for x in expected).backward()
    for a, b in zip(values, references):
        torch.testing.assert_close(a.grad, b.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('launch', ['on', 'before', 'input_ready'])
def test_generated_deferred_module_cuda_gradients(tmp_path, tied, launch):
    import torch.distributed as dist
    import nnscaler
    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/process_group', rank=0, world_size=1)
    try:
        nnscaler.init()
        model = (TiedAnchorModel() if tied else AnchorModel()).cuda()
        reference = copy.deepcopy(model)
        rule = _anchor_rule(launch, tied)
        generated = parallelize(model, {'x': torch.randn(4, 8, device='cuda', requires_grad=True)}, 'dp',
                                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path,
                                instance_name=f'cuda_{launch}_{tied}')
        for _ in range(2):
            generated.zero_grad()
            # ParallelModule.zero_grad clears reducer buffers. A one-rank
            # module has ordinary parameters without a reducer; an optimizer
            # would clear those separately at the next training step.
            for parameter in generated.parameters():
                parameter.grad = None
            reference.zero_grad()
            x = torch.randn(4, 8, device='cuda', requires_grad=True)
            ref_x = x.detach().clone().requires_grad_()
            actual, expected = generated(x), reference(ref_x)
            actual.square().sum().backward()
            expected.square().sum().backward()
            generated.sync_grad()
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(x.grad, ref_x.grad)
            for name, parameter in generated.named_parameters():
                torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
    finally:
        nnscaler.uninit()
        if dist.is_initialized():dist.destroy_process_group()


class EarlyBranchAnchorModel(ForkModel):
    def forward(self, x):
        shared = self.side(x)
        routed = dispatch_anchor(x)
        return shared + routed


class FinalAnchorModel(ForkModel):
    def forward(self, x):
        shared = self.side(x)
        routed = dispatch_anchor(x)
        return shared, routed


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('model_class', [EarlyBranchAnchorModel, FinalAnchorModel])
def test_explicit_launch_point_can_delay_an_early_independent_module(tmp_path, monkeypatch, model_class):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', CPUDeferredStreams)
    model = model_class()
    reference = copy.deepcopy(model)
    rule = ModuleStreamRule('side', launch_on='*.dispatch_anchor', overlap_backward=True)
    parallelize(model, {'x': torch.randn(4, 8, requires_grad=True)}, 'dp',
                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path,
                instance_name='early_deferred', load_module=False)
    cls = _load_parallel_module_class(model_class, gen_savedir=tmp_path,
                                      instance_name='early_deferred', rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(ref_x)
    torch.testing.assert_close(actual, expected)
    sum(value.sum() for value in (actual if isinstance(actual, tuple) else (actual,))).backward()
    sum(value.sum() for value in (expected if isinstance(expected, tuple) else (expected,))).backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
