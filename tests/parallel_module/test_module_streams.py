# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import json
import copy
import functools
from dataclasses import asdict, replace
from contextlib import contextmanager

import pytest
import torch

from nnscaler import ComputeConfig, ModuleStreamRule, parallelize, register_op
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
    monkeypatch.setenv(
        'NNSCALER_MODULE_STREAMS',
        json.dumps([dict(module='side', launch_before='torch.sin')]),
    )
    enabled = ComputeConfig(1, 1)
    monkeypatch.delenv('NNSCALER_MODULE_STREAMS')
    assert enabled.module_streams == (ModuleStreamRule('side', launch_before='torch.sin'),)
    assert not ComputeConfig.safe_equals(baseline, enabled)
    assert baseline.graph_config == enabled.graph_config
    assert not ComputeConfig(1, 1).module_streams


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
    torch.save(payload, path)
    monkeypatch.setenv('NNSCALER_MODULE_STREAMS', json.dumps([dict(module='side')]))
    legacy = ComputeConfig.safe_load_from_file(path, return_none_on_error=False)
    assert not legacy.module_streams


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
def test_codegen_splits_branch_and_joins_at_consumer(tmp_path, workers):
    parallelize(
        ForkModel(),
        {'x': torch.randn(4, 8)},
        'dp',
        ComputeConfig(1, 2, module_streams=(RULE,)),
        gen_savedir=tmp_path,
        load_module=False,
        codegen_workers=workers,
    )
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    ast.parse(code)
    assignments = [
        n
        for n in ast.walk(ast.parse(code))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'module_parameter_streams' for t in n.targets)
    ]
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
    parallelize(
        EndToEndFork(),
        {'data': torch.randn(4, 8)},
        'data',
        ComputeConfig(1, 2, use_end2end=True, module_streams=(RULE,)),
        gen_savedir=tmp_path,
        load_module=False,
        codegen_workers=workers,
    )
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    assert 'use_multi_streams = True' in code
    assert 'track_stream=True' in code


@replace_all_device_with('cpu', force=True)
def test_internal_layout_adapters_follow_their_branch(tmp_path):
    from nnscaler.policies import fn

    parallelize(
        LayoutModel(),
        {'x': torch.randn(4, 8)},
        functools.partial(fn, policy=layout_policy),
        ComputeConfig(2, 2, module_streams=(RULE,)),
        gen_savedir=tmp_path,
        load_module=False,
    )
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    ast.parse(code)
    branch = code[code.index(".fork('module_stream_0'") :]
    assert branch.index('split_allgather(') < branch.index('torch.nn.functional.linear(')
    assert branch.index('torch.nn.functional.linear(') < branch.index('torch.sin(')


@replace_all_device_with('cpu', force=True)
def test_shared_parameter_keeps_ordinary_gradient_stream(tmp_path):
    rule = ModuleStreamRule('side', launch_before='torch.nn.functional.linear')
    parallelize(
        TiedBranches(),
        {'x': torch.randn(4, 8)},
        'dp',
        ComputeConfig(1, 1, module_streams=(rule,)),
        gen_savedir=tmp_path,
        load_module=False,
    )
    tree = ast.parse(next(tmp_path.rglob('gencode0.py')).read_text())
    assignment = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'module_parameter_streams' for t in n.targets)
    )
    affinities = ast.literal_eval(assignment.value)
    assert len(affinities) == 1
    assert all('bias' in name for name in affinities)


@replace_all_device_with('cpu', force=True)
def test_dependent_branch_cannot_move_ahead_of_producer(tmp_path):
    with pytest.raises(RuntimeError, match='Code generation failed') as error:
        parallelize(
            DependentModel(),
            {'x': torch.randn(4, 8)},
            'dp',
            ComputeConfig(1, 1, module_streams=(RULE,)),
            gen_savedir=tmp_path,
            load_module=False,
        )
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
        from nnscaler.runtime.backward import after_backward

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
def test_repeated_invocations_preserve_outputs_and_gradients(tmp_path, monkeypatch, after):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = RepeatedBlock()
    reference = copy.deepcopy(model)
    x = torch.randn(4, 8, requires_grad=True)
    rule = (
        ModuleStreamRule(
            'block.side',
            launch_after='torch.sin',
            split_before='down',
            resume_after='torch.cos',
        )
        if after
        else ModuleStreamRule(
            'block.side',
            launch_before='torch.sin',
            split_before='down',
            resume_before='torch.cos',
        )
    )
    instance = f'cpu_{after}'
    parallelize(
        model,
        {'x': x.detach()},
        'dp',
        ComputeConfig(1, 1, module_streams=(rule,)),
        gen_savedir=tmp_path,
        instance_name=instance,
        load_module=False,
    )
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


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(stream='default'),
        dict(split_before='down'),
        dict(module=''),
        dict(resume_before='torch.cos'),
        dict(recompute_only=True, launch_before='torch.sin'),
        dict(backward_on='torch.cos'),
        dict(launch_before='torch.sin', overlap_backward=True, backward_on=''),
    ],
)
def test_invalid_rules(kwargs):
    with pytest.raises(ValueError):
        ModuleStreamRule(**(dict(module='side') | kwargs))


def recompute_side_policy(graph, config):
    from nnscaler.policies import pas_dp

    graph.recompute([node for node in graph.nodes() if 'side' in (node.module_stack or {})])
    return pas_dp(graph, config)


@replace_all_device_with('cpu', force=True)
def test_split_checkpoint_rejected(tmp_path):
    with pytest.raises(RuntimeError, match='Code generation failed') as error:
        parallelize(
            ForkModel(),
            {'x': torch.randn(4, 8)},
            recompute_side_policy,
            ComputeConfig(1, 1, module_streams=(RULE,)),
            gen_savedir=tmp_path,
            load_module=False,
        )
    assert 'recompute region' in str(error.value.__cause__)


@replace_all_device_with('cpu', force=True)
def test_whole_checkpoint_runs_on_selected_stream(tmp_path, monkeypatch):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = ForkModel()
    ref = copy.deepcopy(model)
    rule = ModuleStreamRule('side', launch_before='torch.sin')
    instance = f'checkpoint'
    parallelize(
        model,
        {'x': torch.randn(4, 8)},
        recompute_side_policy,
        ComputeConfig(1, 1, module_streams=(rule,)),
        gen_savedir=tmp_path,
        instance_name=instance,
        load_module=False,
    )
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
def test_generated_stream_module_cuda_outputs_and_gradients(tmp_path, after):
    import torch.distributed as dist
    import nnscaler

    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/process_group', rank=0, world_size=1)
    try:
        nnscaler.init()
        model = RepeatedBlock().cuda()
        reference = copy.deepcopy(model)
        rule = (
            ModuleStreamRule(
                'block.side',
                launch_after='torch.sin',
                split_before='down',
                resume_after='torch.cos',
            )
            if after
            else ModuleStreamRule(
                'block.side',
                launch_before='torch.sin',
                split_before='down',
                resume_before='torch.cos',
            )
        )
        x = torch.randn(4, 8, device='cuda', requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        generated = parallelize(
            model,
            {'x': x.detach()},
            'dp',
            ComputeConfig(1, 1, module_streams=(rule,)),
            gen_savedir=tmp_path,
            instance_name=f'cuda_{after}',
        )
        actual, expected = generated(x), reference(ref_x)
        actual.square().sum().backward()
        expected.square().sum().backward()
        generated.sync_grad()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, ref_x.grad)
        for name, parameter in generated.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                reference.get_parameter(generated.fullmap[name].orig_name).grad,
            )
    finally:
        nnscaler.uninit()
        if dist.is_initialized():
            dist.destroy_process_group()


@replace_all_device_with('cpu', force=True)
def test_after_launch_snapshots_readiness_before_the_anchor(tmp_path):
    parallelize(
        ForkModel(),
        {'x': torch.randn(4, 8)},
        'dp',
        ComputeConfig(1, 1, module_streams=(AFTER_RULE,)),
        gen_savedir=tmp_path,
        load_module=False,
    )
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    ast.parse(code)
    assert code.index(".capture('module_stream_0')") < code.index('torch.sin(') < code.index(".fork('module_stream_0'")
    assert code.index(".capture('module_stream_1')") < code.index('torch.cos(') < code.index(".fork('module_stream_1'")


@replace_all_device_with('cpu', force=True)
def test_recompute_only_rule_keeps_forward_on_caller(tmp_path):
    model = ForkModel()
    reference = copy.deepcopy(model)
    rule = ModuleStreamRule('side', stream='expert_recompute', recompute_only=True)
    parallelize(
        model,
        {'x': torch.randn(4, 8)},
        recompute_side_policy,
        ComputeConfig(1, 1, module_streams=(rule,)),
        gen_savedir=tmp_path,
        instance_name='recompute_only',
        load_module=False,
    )
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
        torch.testing.assert_close(
            parameter.grad,
            reference.get_parameter(generated.fullmap[name].orig_name).grad,
        )


@register_op('n h, ? -> n h')
def apply_checkpoint_kwargs(x, options):
    return x * options['scale'] + options['bias']


class KwargsCheckpointBranch(Branch):
    def forward(self, x, metadata):
        options = {'enabled': True}
        options.update(scale=metadata.get('scale'), bias=metadata.get('bias'))
        options['training'] = self.training
        return apply_checkpoint_kwargs(self.down(self.up(x).relu()), options)


class KwargsCheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.side = KwargsCheckpointBranch()

    def forward(self, x, metadata):
        return self.side(x, metadata)


@replace_all_device_with('cpu', force=True)
def test_recompute_stream_with_dynamic_local_kwargs(tmp_path):
    model = KwargsCheckpointModel()
    reference = copy.deepcopy(model)
    rule = ModuleStreamRule('side', stream='kwargs_recompute', recompute_only=True)
    sample = {'x': torch.randn(4, 8), 'metadata': {'scale': 1.25, 'bias': 0.5}}
    parallelize(
        model,
        sample,
        recompute_side_policy,
        ComputeConfig(1, 1, constant_folding=True, module_streams=(rule,)),
        gen_savedir=tmp_path,
        instance_name='local_kwargs',
        load_module=False,
    )
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    assert 'checkpoint_stream.contexts(' in code
    # Fresh traces may fold updates into a literal; the planner tests also
    # cover retained setitem chains from existing graph checkpoints.
    cls = _load_parallel_module_class(
        KwargsCheckpointModel,
        gen_savedir=tmp_path,
        instance_name='local_kwargs',
        rank=0,
    )
    generated = new_empty(cls, device='cpu', init_params=True)
    for metadata in ({'scale': 1.25, 'bias': 0.5}, {'scale': 2.5, 'bias': -0.75}):
        original_metadata = metadata.copy()
        # new_empty has no reducer buckets; clear its ordinary leaf gradients.
        for parameter in generated.parameters():
            parameter.grad = None
        reference.zero_grad()
        x = torch.randn(4, 8, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        actual, expected = generated(x, metadata), reference(rx, metadata)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(x.grad, rx.grad)
        assert metadata == original_metadata
        for name, parameter in generated.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                reference.get_parameter(generated.fullmap[name].orig_name).grad,
            )


@replace_all_device_with('cpu', force=True)
def test_recompute_stream_rejects_part_of_a_checkpoint(tmp_path):
    rule = ModuleStreamRule('side.up', recompute_only=True)
    with pytest.raises(RuntimeError, match='Code generation failed') as error:
        parallelize(
            ForkModel(),
            {'x': torch.randn(4, 8)},
            recompute_side_policy,
            ComputeConfig(1, 1, module_streams=(rule,)),
            gen_savedir=tmp_path,
            load_module=False,
        )
    assert 'whole checkpoint region' in str(error.value.__cause__)


_anchor_backward_order = []


class DispatchAnchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        from nnscaler import capture_async_backward

        ctx.point = capture_async_backward()
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
        from nnscaler import capture_async_backward

        ctx.point = capture_async_backward()
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
    return DispatchAnchor.apply(x)


@register_op('a b -> a b')
def combine_anchor(x):
    return CombineAnchor.apply(x)


class AnchorModel(ForkModel):
    def forward(self, x):
        y = dispatch_anchor(x)
        y = torch.tanh(y)
        y = combine_anchor(y)
        return y + self.side(x)


class CheckpointContainer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.compute = Branch()

    def forward(self, x):
        return self.compute(x)


class ReadyCheckpointModel(ForkModel):
    def __init__(self, nested=False):
        super().__init__()
        if nested:
            self.side = CheckpointContainer()
        self.other = Branch()

    def forward(self, x):
        return combine_anchor(self.side(x)) + self.other(x)


@replace_all_device_with('cpu', force=True)
def test_backward_anchor_cannot_itself_move_to_a_side_stream(tmp_path):
    rules = (
        ModuleStreamRule('block.side', launch_before='*.dispatch_anchor', overlap_backward=True),
        ModuleStreamRule('block', operators='*.dispatch_anchor', stream='transport'),
    )
    with pytest.raises(RuntimeError, match='Code generation failed') as failure:
        parallelize(
            NestedAnchorModel(),
            {'x': torch.randn(4, 8, requires_grad=True)},
            'dp',
            ComputeConfig(1, 1, module_streams=rules),
            gen_savedir=tmp_path,
            load_module=False,
        )
    assert 'Backward anchors must remain on the caller stream' in str(failure.value.__cause__)


class NestedAnchorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = AnchorModel()

    def forward(self, x):
        return self.block(x)


class AmbiguousCheckpointModel(ForkModel):
    def forward(self, x):
        value = self.side(x)
        return combine_anchor(value) + value


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('nested', [False, True])
def test_checkpoint_readiness_codegen_and_gradients(tmp_path, monkeypatch, workers, nested):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    monkeypatch.setattr('nnscaler.runtime.backward.BackwardLaunchPoint.record_ready', lambda self: None)
    model = ReadyCheckpointModel(nested=nested)
    reference = copy.deepcopy(model)
    rules = (
        ModuleStreamRule('side.compute' if nested else 'side', recompute_only=True, recompute_ready_on='*.combine_anchor'),
        ModuleStreamRule('other', launch_before='*.combine_anchor', overlap_backward=True),
    )
    instance = f'checkpoint_ready_{workers}_{nested}'
    parallelize(
        model,
        {'x': torch.randn(4, 8)},
        recompute_side_policy,
        ComputeConfig(1, 1, module_streams=rules),
        gen_savedir=tmp_path,
        instance_name=instance,
        load_module=False,
        codegen_workers=workers,
    )
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    assert 'backward_ready=_checkpoint_ready_0' in code
    assert 'backward_point(' in code and '), _checkpoint_ready_0:' in code
    cls = _load_parallel_module_class(ReadyCheckpointModel, gen_savedir=tmp_path, instance_name=instance, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(ref_x)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, ref_x.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            reference.get_parameter(generated.fullmap[name].orig_name).grad,
        )


@replace_all_device_with('cpu', force=True)
def test_checkpoint_readiness_rejects_another_gradient_consumer(tmp_path):
    rule = ModuleStreamRule('side', recompute_only=True, recompute_ready_on='*.combine_anchor')
    with pytest.raises(RuntimeError, match='Code generation failed') as failure:
        parallelize(
            AmbiguousCheckpointModel(),
            {'x': torch.randn(4, 8)},
            recompute_side_policy,
            ComputeConfig(1, 1, module_streams=(rule,)),
            gen_savedir=tmp_path,
            load_module=False,
        )
    assert 'sole direct gradient consumer' in str(failure.value.__cause__)


def _anchor_rule(launch, tied=False):
    options = dict(launch_before='*.dispatch_anchor')
    if not tied:
        options.update(split_before='down', resume_before='*.combine_anchor')
    if launch == 'independent_backward':
        options['backward_on'] = '*.dispatch_anchor' if tied else '*.combine_anchor'
    return ModuleStreamRule('side', overlap_backward=True, **options)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('launch', ['before', 'independent_backward'])
def test_codegen_preserves_math_and_orders_both_backward_regions(tmp_path, monkeypatch, workers, launch):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = AnchorModel()
    reference = copy.deepcopy(model)
    rule = _anchor_rule(launch)
    parallelize(
        model,
        {'x': torch.randn(4, 8, requires_grad=True)},
        'dp',
        ComputeConfig(1, 1, module_streams=(rule,)),
        gen_savedir=tmp_path,
        instance_name=f'launch_points_{launch}_{workers}',
        load_module=False,
        codegen_workers=workers,
    )
    cls = _load_parallel_module_class(
        AnchorModel,
        gen_savedir=tmp_path,
        instance_name=f'launch_points_{launch}_{workers}',
        rank=0,
    )
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
        if launch == 'independent_backward':
            assert (
                _anchor_backward_order.index('combine')
                < _anchor_backward_order.index('up')
                < _anchor_backward_order.index('dispatch')
            )
        else:
            assert _anchor_backward_order.index('dispatch') < _anchor_backward_order.index('up')
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, ref_x.grad)
        for name, parameter in generated.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                reference.get_parameter(generated.fullmap[name].orig_name).grad,
            )


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize(
    'anchor,reason',
    [
        ('*.dispatch_anchor', 'follow all selected forward chunks'),
        ('*.missing_anchor', 'expected one backward anchor'),
        ('*anchor', 'expected one backward anchor'),
    ],
)
def test_independent_backward_rejects_unavailable_or_ambiguous_anchor(tmp_path, anchor, reason):
    rule = replace(_anchor_rule('independent_backward'), backward_on=anchor)
    with pytest.raises(RuntimeError, match='Code generation failed') as error:
        parallelize(
            AnchorModel(),
            {'x': torch.randn(4, 8, requires_grad=True)},
            'dp',
            ComputeConfig(1, 1, module_streams=(rule,)),
            gen_savedir=tmp_path,
            load_module=False,
        )
    assert reason in str(error.value.__cause__)


class TiedAnchorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Linear(8, 8, bias=False)
        self.side = torch.nn.Linear(8, 8, bias=False)
        self.side.weight = self.main.weight

    def forward(self, x):
        return dispatch_anchor(self.main(x)) + self.side(x)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('independent', [False, True])
def test_tied_parameters_and_multiple_live_forwards(tmp_path, monkeypatch, independent):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = TiedAnchorModel()
    reference = copy.deepcopy(model)
    rule = (
        _anchor_rule('independent_backward', tied=True)
        if independent
        else ModuleStreamRule('side', launch_before='*.dispatch_anchor', overlap_backward=True)
    )
    instance = f'tied_streams_{independent}'
    parallelize(
        model,
        {'x': torch.randn(4, 8, requires_grad=True)},
        'dp',
        ComputeConfig(1, 1, module_streams=(rule,)),
        gen_savedir=tmp_path,
        instance_name=instance,
        load_module=False,
    )
    cls = _load_parallel_module_class(TiedAnchorModel, gen_savedir=tmp_path, instance_name=instance, rank=0)
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
        torch.testing.assert_close(
            parameter.grad,
            reference.get_parameter(generated.fullmap[name].orig_name).grad,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('launch', ['before', 'independent_backward'])
def test_generated_backward_overlap_cuda_gradients(tmp_path, tied, launch):
    import torch.distributed as dist
    import nnscaler

    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/process_group', rank=0, world_size=1)
    try:
        nnscaler.init()
        model = (TiedAnchorModel() if tied else AnchorModel()).cuda()
        reference = copy.deepcopy(model)
        rule = _anchor_rule(launch, tied)
        generated = parallelize(
            model,
            {'x': torch.randn(4, 8, device='cuda', requires_grad=True)},
            'dp',
            ComputeConfig(1, 1, module_streams=(rule,)),
            gen_savedir=tmp_path,
            instance_name=f'cuda_{launch}_{tied}',
        )
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
                torch.testing.assert_close(
                    parameter.grad,
                    reference.get_parameter(generated.fullmap[name].orig_name).grad,
                )
    finally:
        nnscaler.uninit()
        if dist.is_initialized():
            dist.destroy_process_group()


@register_op('a b -> a b')
def auxiliary_statistics(x):
    return x.square()


class OperatorBranchBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(8, 8)

    def forward(self, x):
        route = self.router(x)
        y = combine_anchor(torch.tanh(dispatch_anchor(route)))
        return y + auxiliary_statistics(route)


class RepeatedOperatorBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = OperatorBranchBlock()

    def forward(self, x):
        return self.block(self.block(x))


OPERATOR_RULE = ModuleStreamRule(
    'block',
    operators='*.auxiliary_statistics',
    launch_before='*.dispatch_anchor',
    overlap_backward=True,
    backward_on='*.combine_anchor',
)


def test_operator_selector_normalizes_and_invalidates_cache():
    assert OPERATOR_RULE.operators == ('*.auxiliary_statistics',)
    assert replace(OPERATOR_RULE, operators=['*.auxiliary_statistics']) == OPERATOR_RULE
    current = ComputeConfig(1, 1, module_streams=(OPERATOR_RULE,))
    whole = ComputeConfig(1, 1, module_streams=(replace(OPERATOR_RULE, operators=None),))
    assert current.graph_config == whole.graph_config
    assert not ComputeConfig.safe_equals(current, whole)
    assert ComputeConfig(1, 1, module_streams=(asdict(OPERATOR_RULE),)) == current


@pytest.mark.parametrize(
    'options',
    [
        dict(operators=[]),
        dict(operators=''),
        dict(operators=['']),
        dict(operators=[1]),
        dict(operators=1),
        dict(operators=('*.x',), recompute_only=True),
        dict(operators=('*.x',), split_before='child', resume_before='*.anchor'),
    ],
)
def test_operator_selector_rejects_invalid_combinations(options):
    with pytest.raises(ValueError):
        ModuleStreamRule('block', **options)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
def test_operator_selector_scopes_repeated_invocations(tmp_path, monkeypatch, workers):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = RepeatedOperatorBranch()
    reference = copy.deepcopy(model)
    instance = f'operator_selection_{workers}'
    parallelize(
        model,
        {'x': torch.randn(4, 8)},
        'dp',
        ComputeConfig(1, 1, module_streams=(OPERATOR_RULE,)),
        gen_savedir=tmp_path,
        instance_name=instance,
        load_module=False,
        codegen_workers=workers,
    )
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    # Each call moves only its own auxiliary op, after its producer and before
    # its own dispatch. Parent-scope resolution would see two ambiguous anchors.
    for region in range(2):
        fork = code.index(f".fork('module_stream_{region}'")
        assert code.rfind('torch.nn.functional.linear(', 0, fork) >= 0
        assert fork < code.index('auxiliary_statistics(', fork) < code.index('dispatch_anchor(', fork)
    cls = _load_parallel_module_class(RepeatedOperatorBranch, gen_savedir=tmp_path, instance_name=instance, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(ref_x)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, ref_x.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            reference.get_parameter(generated.fullmap[name].orig_name).grad,
        )


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('failure', ['dependency', 'checkpoint'])
def test_operator_selection_retains_dependency_and_checkpoint_checks(tmp_path, failure):
    from nnscaler.policies import pas_dp

    def checkpoint_block(graph, config):
        graph.recompute([node for node in graph.nodes() if 'block' in (node.module_stack or {})])
        return pas_dp(graph, config)

    rule = (
        replace(OPERATOR_RULE, operators='torch.tanh')
        if failure == 'dependency'
        else replace(OPERATOR_RULE, backward_on=None)
    )
    with pytest.raises(RuntimeError, match='Code generation failed') as error:
        parallelize(
            RepeatedOperatorBranch(),
            {'x': torch.randn(4, 8)},
            'dp' if failure == 'dependency' else checkpoint_block,
            ComputeConfig(1, 1, module_streams=(rule,)),
            gen_savedir=tmp_path,
            load_module=False,
        )
    message = 'violates dependency' if failure == 'dependency' else 'crosses a recompute region'
    assert message in str(error.value.__cause__)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_operator_selector_cuda_repeated_invocations(tmp_path):
    import torch.distributed as dist
    import nnscaler

    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/process_group', rank=0, world_size=1)
    try:
        nnscaler.init()
        model = RepeatedOperatorBranch().cuda()
        reference = copy.deepcopy(model)
        generated = parallelize(
            model,
            {'x': torch.randn(4, 8, device='cuda')},
            'dp',
            ComputeConfig(1, 1, module_streams=(OPERATOR_RULE,)),
            gen_savedir=tmp_path,
            instance_name='cuda_operator_selection',
        )
        # Multiple live forwards must retain separate invocation controls.
        inputs = [torch.randn(4, 8, device='cuda', requires_grad=True) for _ in range(2)]
        refs = [x.detach().clone().requires_grad_() for x in inputs]
        actual = [generated(x) for x in inputs]
        expected = [reference(x) for x in refs]
        for result, ref in zip(actual, expected):
            torch.testing.assert_close(result, ref)
            result.square().sum().backward()
            ref.square().sum().backward()
        generated.sync_grad()
        for x, ref in zip(inputs, refs):
            torch.testing.assert_close(x.grad, ref.grad)
        for name, parameter in generated.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                reference.get_parameter(generated.fullmap[name].orig_name).grad,
            )
    finally:
        nnscaler.uninit()
        if dist.is_initialized():
            dist.destroy_process_group()


@register_op('a b, a b -> a b')
def masked_auxiliary_statistics(x, mask):
    return x.square() * mask


class ContainerOperatorBlock(OperatorBranchBlock):
    def forward(self, x, metadata):
        route = self.router(x)
        y = combine_anchor(torch.tanh(dispatch_anchor(route)))
        return y + masked_auxiliary_statistics(route, metadata['mask'])


class ContainerOperatorBranch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ContainerOperatorBlock()

    def forward(self, x, mask):
        return self.block(x, dict(mask=mask))


@replace_all_device_with('cpu', force=True)
def test_operator_selector_includes_builtin_metadata_reads(tmp_path, monkeypatch):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = ContainerOperatorBranch()
    reference = copy.deepcopy(model)
    x = torch.randn(4, 8, requires_grad=True)
    mask = torch.randint(0, 2, (4, 8))
    rule = replace(OPERATOR_RULE, operators='*.masked_auxiliary_statistics')
    parallelize(
        model,
        {'x': x.detach(), 'mask': mask},
        'dp',
        ComputeConfig(1, 1, module_streams=(rule,)),
        gen_savedir=tmp_path,
        load_module=False,
    )
    cls = _load_parallel_module_class(ContainerOperatorBranch, gen_savedir=tmp_path, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    ref_x = x.detach().clone().requires_grad_()
    actual, expected = generated(x, mask), reference(ref_x, mask)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, ref_x.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            reference.get_parameter(generated.fullmap[name].orig_name).grad,
        )


class InternalStreamOperatorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 8)

    def forward(self, data):
        return auxiliary_statistics(self.linear(data)).sum()


def test_internal_stream_declarations_are_frozen_and_cache_safe(tmp_path, monkeypatch):
    baseline = ComputeConfig(1, 1)
    monkeypatch.setenv('NNSCALER_MULTI_STREAM_OPS', '["*.auxiliary_statistics"]')
    enabled = ComputeConfig(1, 1)
    assert enabled.multi_stream_ops == ('*.auxiliary_statistics',)
    assert enabled.graph_config == baseline.graph_config
    assert not ComputeConfig.safe_equals(enabled, baseline)
    path = tmp_path / 'compute_config.pt'
    ComputeConfig.safe_dump_to_file(enabled, path)
    assert ComputeConfig.safe_load_from_file(path) == enabled
    payload = asdict(baseline)
    payload.pop('multi_stream_ops')
    torch.save(payload, path)
    # Current environment must not retroactively label an old generated plan.
    assert ComputeConfig.safe_load_from_file(path) == baseline
    monkeypatch.delenv('NNSCALER_MULTI_STREAM_OPS')
    assert not ComputeConfig(1, 1).multi_stream_ops


@pytest.mark.parametrize('patterns', ['torch.sin', [None], [''], 1])
def test_internal_stream_declarations_reject_invalid_patterns(patterns):
    with pytest.raises(ValueError, match='multi_stream_ops'):
        ComputeConfig(1, 1, multi_stream_ops=patterns)


def test_internal_stream_declarations_reject_zero3():
    with pytest.raises(ValueError, match='multi_stream_ops currently supports ZeRO 0/1'):
        ComputeConfig(1, 1, use_zero=2, multi_stream_ops=('*.auxiliary_statistics',))


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('checkpointed', [False, True])
@pytest.mark.parametrize('matched', [False, True])
def test_internal_stream_operator_scheduled_lifetimes(tmp_path, workers, checkpointed, matched):
    from nnscaler.policies import pas_dp
    from nnscaler.ir.operator import IRFwOperation

    def policy(graph, config):
        if checkpointed:
            graph.recompute(graph.select(ntype=IRFwOperation))
        return pas_dp(graph, config)

    patterns = ('*.auxiliary_statistics' if matched else '*.not_present',)
    parallelize(
        InternalStreamOperatorModel(),
        {'data': torch.randn(4, 8)},
        policy,
        ComputeConfig(1, 2, use_end2end=True, multi_stream_ops=patterns),
        gen_savedir=tmp_path,
        load_module=False,
        codegen_workers=workers,
        instance_name=f'internal_stream_{workers}_{checkpointed}_{matched}',
    )
    for path in tmp_path.rglob('gencode*.py'):
        code = path.read_text()
        assert f'use_multi_streams = {matched}' in code
        assert f'track_stream={matched}' in code
        assert 'with nnscaler.runtime.stream.ModuleStreams()' not in code
