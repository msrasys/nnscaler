# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import ast

import pytest
import torch

from nnscaler import AsyncResult, ComputeConfig, ModuleStreamRule, parallelize, register_op, register_async_op
from nnscaler.parallel import _load_parallel_module_class
from nnscaler.runtime.backward import capture_async_backward
from nnscaler.runtime.checkpoint_prefetch import Prefetch
from .test_module_streams import Branch, SerialStreams
from ..utils import replace_all_device_with, new_empty


class Anchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, value):
        ctx.point = capture_async_backward()
        return value

    @staticmethod
    def backward(ctx, grad):
        if ctx.point is not None:
            ctx.point.record_ready()
            ctx.point.submitted()
        return grad * 2, None


@register_op('n h -> n h')
def prefetch_anchor(x):
    return Anchor.apply(x, x.detach() * 2)


def submit(x):
    return AsyncResult(x.detach() * 2, lambda: None)


def complete(value, x):
    return Anchor.apply(x, value)


register_async_op(prefetch_anchor, submit=submit, complete=complete)


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = Branch()

    def forward(self, x):
        return prefetch_anchor(self.layer(x).sin())


class Model(Block):
    def __init__(self):
        super().__init__()
        self.other = Branch()

    def forward(self, x):
        return super().forward(x) + self.other(x)


class Repeated(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = Block()

    def forward(self, x):
        return self.block(self.block(x))


def policy(graph, config):
    from nnscaler.policies import pas_dp
    groups = {}
    for node in graph.nodes():
        for path in node.module_stack or ():
            if path.split('.')[-1] == 'layer':
                groups.setdefault((path, (node.module_call_stack or {}).get(path)), []).append(node)
                break
    for nodes in groups.values():
        graph.recompute(nodes)
    return pas_dp(graph, config)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('phase', ['forward', 'backward'])
@pytest.mark.parametrize('repeated,other_stream,workers', [(False, False, 1), (False, True, 2), (True, False, 2)])
def test_prefetch_preserves_generated_math_and_invocation_lifetimes(tmp_path, monkeypatch, phase, repeated, other_stream, workers):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    points = []
    original_init = Prefetch.__init__
    def initialize(point, *args):
        original_init(point, *args)
        points.append(point)
    monkeypatch.setattr(Prefetch, '__init__', initialize)
    model = Repeated() if repeated else Model()
    reference = copy.deepcopy(model)
    rules = [ModuleStreamRule('*.layer' if repeated else 'layer', stream='default', recompute_only=True,
                               recompute_prefetch_on='*.prefetch_anchor', recompute_prefetch_phase=phase)]
    if other_stream:
        rules.append(ModuleStreamRule('other', launch_after='*.prefetch_anchor'))
    instance = f'prefetch_{phase}_{repeated}_{other_stream}_{workers}'
    parallelize(model, {'x': torch.randn(4, 8, requires_grad=True)}, policy,
                ComputeConfig(1, 1, module_streams=rules), gen_savedir=tmp_path,
                instance_name=instance, codegen_workers=workers, load_module=False)
    source = next(tmp_path.rglob('gencode0.py')).read_text()
    ast.parse(source)
    assert 'checkpoint_prefetch.Prefetch(' in source and '.contexts(' in source
    cls = _load_parallel_module_class(type(model), gen_savedir=tmp_path, instance_name=instance, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    actual, expected = generated(x), reference(rx)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x.grad, rx.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
    assert len(points) == (2 if repeated else 1)
    assert all(point.stats['launched'] == point.stats['reused'] == 1 and point.budget.used == 0 for point in points)
