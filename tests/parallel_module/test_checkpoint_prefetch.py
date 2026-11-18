import copy
import ast

import pytest
import torch

import nnscaler
from nnscaler import ComputeConfig, ModuleStreamRule
from nnscaler.parallel import _load_parallel_module_class
from nnscaler.runtime.stream import ModuleStreams
from .test_module_streams import Branch, CPUDeferredStreams
from .test_weight_backward import Expert, Streams as WeightStreams, weight_anchor, combine_anchor
from ..utils import replace_all_device_with, new_empty


class Anchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.point = nnscaler.capture_async_backward()
        nnscaler.notify_async_launch()
        return value * 2

    @staticmethod
    def backward(ctx, grad):
        if ctx.point is not None:
            ctx.point.record_ready()
            ctx.point.submitted()
        return grad * 2


@nnscaler.register_op('a b -> a b')
def prefetch_anchor(value):
    return Anchor.apply(nnscaler.prepare_async_input(value))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Branch()
        self.other = Branch()

    def forward(self, x):
        return prefetch_anchor(self.layer(x).sin()) + self.other(x)


class RepeatedBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Branch()

    def forward(self, x):
        return prefetch_anchor(self.layer(x).sin())


class RepeatedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = RepeatedBlock()

    def forward(self, x):
        return self.block(self.block(x))


def policy(graph, config):
    from nnscaler.policies import pas_dp
    graph.recompute([node for node in graph.nodes() if 'layer' in (node.module_stack or {})])
    return pas_dp(graph, config)


class CPUStreams(CPUDeferredStreams):
    created = []
    prefetch_launch_point = ModuleStreams.prefetch_launch_point

    def __enter__(self):
        super().__enter__()
        self.prefetch_points = []
        return self

    def checkpoint_prefetch(self, name, budget_bytes):
        point = ModuleStreams.checkpoint_prefetch(self, name, budget_bytes)
        self.created.append(point)
        return point

    def __exit__(self, *exc):
        for point in self.prefetch_points:
            point.finish_forward()
        return super().__exit__(*exc)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('phase', ['forward', 'backward'])
@pytest.mark.parametrize('other_stream', [False, True])
@pytest.mark.parametrize('workers', [1, 2])
def test_generated_prefetch_composes_with_module_streams(tmp_path, monkeypatch, phase, other_stream, workers):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', CPUStreams)
    monkeypatch.setattr(CPUStreams, 'created', [])
    model = Model()
    reference = copy.deepcopy(model)
    rules = [ModuleStreamRule('layer', stream='recompute', recompute_only=True,
                              recompute_prefetch_on='*.prefetch_anchor', recompute_prefetch_phase=phase)]
    if other_stream:
        rules.append(ModuleStreamRule('other', launch_on='*.prefetch_anchor', launch_phase='input_ready'))
    instance = f'prefetch_{phase}_{other_stream}_{workers}'
    nnscaler.parallelize(model, {'x': torch.randn(4, 8, requires_grad=True)}, policy,
                        ComputeConfig(1, 1, module_streams=rules), gen_savedir=tmp_path,
                        instance_name=instance, load_module=False, codegen_workers=workers)
    source = next(tmp_path.rglob('gencode0.py')).read_text()
    ast.parse(source)
    assert '.checkpoint_prefetch(' in source and '_checkpoint_prefetch_0.contexts(' in source
    cls = _load_parallel_module_class(Model, gen_savedir=tmp_path, instance_name=instance, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(rx)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, rx.grad)
    for name, value in generated.named_parameters():
        torch.testing.assert_close(value.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
    assert len(CPUStreams.created) == 1
    point = CPUStreams.created[0]
    assert point.stats['launched'] == point.stats['reused'] == 1
    assert point.budget.used == 0


@pytest.mark.parametrize('kwargs', [dict(recompute_prefetch_phase='wrong'),
                                   dict(recompute_prefetch_bytes=0),
                                   dict(recompute_prefetch_bytes=True),
                                   dict(recompute_ready_on='*.other')])
def test_invalid_prefetch_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ModuleStreamRule('layer', recompute_only=True, recompute_prefetch_on='*.anchor', **kwargs)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('phase', ['forward', 'backward'])
def test_repeated_module_invocations_have_separate_prefetches(tmp_path, monkeypatch, phase):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', CPUStreams)
    monkeypatch.setattr(CPUStreams, 'created', [])
    model = RepeatedModel()
    reference = copy.deepcopy(model)

    def repeated_policy(graph, config):
        from nnscaler.policies import pas_dp
        calls = {}
        for node in graph.nodes():
            if 'block.layer' in (node.module_stack or {}):
                calls.setdefault(node.module_call_stack['block'], []).append(node)
        assert len(calls) == 2
        for nodes in calls.values():
            graph.recompute(nodes)
        return pas_dp(graph, config)

    rules = [ModuleStreamRule('*.layer', stream='recompute', recompute_only=True,
                              recompute_prefetch_on='*.prefetch_anchor', recompute_prefetch_phase=phase)]
    instance = f'repeated_prefetch_{phase}'
    nnscaler.parallelize(model, {'x': torch.randn(4, 8, requires_grad=True)}, repeated_policy,
                        ComputeConfig(1, 1, module_streams=rules), gen_savedir=tmp_path,
                        instance_name=instance, load_module=False)
    cls = _load_parallel_module_class(RepeatedModel, gen_savedir=tmp_path, instance_name=instance, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(rx)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, rx.grad)
    for name, value in generated.named_parameters():
        torch.testing.assert_close(value.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
    assert len(CPUStreams.created) == 2
    assert all(point.stats['launched'] == point.stats['reused'] == 1 for point in CPUStreams.created)
    assert all(point.budget.used == 0 for point in CPUStreams.created)


class CombinedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU())
        self.expert = Expert()
        self.shared = torch.nn.Linear(4, 6, bias=False)

    def forward(self, x):
        h = self.attention(x)
        return combine_anchor(self.expert(weight_anchor(h))) + self.shared(h)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('phase', ['forward', 'backward'])
def test_prefetch_and_weight_backward_share_one_backend_boundary(tmp_path, monkeypatch, phase):
    class CombinedStreams(WeightStreams):
        created = []
        checkpoint_prefetch = CPUStreams.checkpoint_prefetch
        prefetch_launch_point = ModuleStreams.prefetch_launch_point

        def __enter__(self):
            super().__enter__()
            self.prefetch_points = []
            return self

        def __exit__(self, *exc):
            for point in self.prefetch_points:
                point.finish_forward()
            return super().__exit__(*exc)

    def combined_policy(graph, config):
        from nnscaler.policies import pas_dp
        for path in ('attention', 'expert'):
            graph.recompute([node for node in graph.nodes() if path in (node.module_stack or {})])
        return pas_dp(graph, config)

    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', CombinedStreams)
    model = CombinedModel()
    reference = copy.deepcopy(model)
    rules = [ModuleStreamRule('attention', stream='recompute', recompute_only=True,
                              recompute_prefetch_on='*.weight_anchor', recompute_prefetch_phase=phase),
             ModuleStreamRule('expert', stream='weights', weight_backward_on='*.weight_anchor'),
             ModuleStreamRule('shared', launch_before='*.combine_anchor', overlap_backward=True)]
    instance = f'prefetch_weights_{phase}'
    nnscaler.parallelize(model, {'x': torch.randn(8, 4, requires_grad=True)}, combined_policy,
                        ComputeConfig(1, 1, module_streams=rules), gen_savedir=tmp_path,
                        instance_name=instance, load_module=False)
    cls = _load_parallel_module_class(CombinedModel, gen_savedir=tmp_path, instance_name=instance, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(8, 4, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(rx)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, rx.grad)
    for name, value in generated.named_parameters():
        torch.testing.assert_close(value.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
    assert len(CombinedStreams.created) == 1
    point = CombinedStreams.created[0]
    assert point.stats['launched'] == point.stats['reused'] == 1 and point.budget.used == 0
