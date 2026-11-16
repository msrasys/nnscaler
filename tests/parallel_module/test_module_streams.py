# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import json
import copy
import functools
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


class ReshapeBranch(Branch):
    def forward(self, x):
        return self.down(torch.nn.functional.relu(self.up(x.reshape(-1, 8))))


class LayoutModel(ForkModel):
    def __init__(self):
        super().__init__()
        self.side = ReshapeBranch()


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
@pytest.mark.parametrize('workers', [1, 2])
def test_codegen_splits_branch_and_joins_at_consumer(tmp_path, workers):
    parallelize(ForkModel(), {'x': torch.randn(4, 8)}, 'dp',
                ComputeConfig(1, 2, module_streams=(RULE,)),
                gen_savedir=tmp_path, load_module=False, codegen_workers=workers)
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    ast.parse(code)
    start = code.index('with nnscaler.runtime.stream.ModuleStreams()')
    code = code[start:]
    assert code.index(".fork('module_stream_0'") < code.index('torch.sin(')
    assert code.index('torch.sin(') < code.index(".fork('module_stream_1'") < code.index('torch.cos(')
    assert code.index('torch.cos(') < code.index(".join('module_stream_1')")


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
        return self

    @contextmanager
    def fork(self, *args):
        yield

    def publish(self, *args):
        pass

    def join(self, *args):
        pass

    def __exit__(self, *args):
        pass


@replace_all_device_with('cpu', force=True)
def test_repeated_invocations_preserve_outputs_and_gradients(tmp_path, monkeypatch):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = RepeatedBlock()
    reference = copy.deepcopy(model)
    x = torch.randn(4, 8, requires_grad=True)
    rule = ModuleStreamRule('block.side', launch_before='torch.sin', split_before='down', resume_before='torch.cos')
    parallelize(model, {'x': x.detach()}, 'dp',
                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path, load_module=False)
    cls = _load_parallel_module_class(RepeatedBlock, gen_savedir=tmp_path, rank=0)
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
                                    dict(module=''), dict(resume_before='torch.cos')])
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
def test_whole_checkpoint_runs_on_selected_stream(tmp_path, monkeypatch):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = ForkModel()
    ref = copy.deepcopy(model)
    rule = ModuleStreamRule('side', launch_before='torch.sin')
    parallelize(model, {'x': torch.randn(4, 8)}, recompute_side_policy,
                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path, load_module=False)
    cls = _load_parallel_module_class(ForkModel, gen_savedir=tmp_path, rank=0)
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
def test_generated_stream_module_cuda_outputs_and_gradients(tmp_path):
    import torch.distributed as dist
    import nnscaler
    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/process_group', rank=0, world_size=1)
    try:
        nnscaler.init()
        model = RepeatedBlock().cuda()
        reference = copy.deepcopy(model)
        rule = ModuleStreamRule('block.side', launch_before='torch.sin', split_before='down', resume_before='torch.cos')
        x = torch.randn(4, 8, device='cuda', requires_grad=True)
        ref_x = x.detach().clone().requires_grad_()
        generated = parallelize(model, {'x': x.detach()}, 'dp',
                                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path)
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
