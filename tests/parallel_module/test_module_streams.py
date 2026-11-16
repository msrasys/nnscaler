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
    def fork(self, *args):
        yield

    def publish(self, *args, **kwargs):
        pass

    def capture(self, *args):
        pass

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
