import copy

import pytest
import torch
import nnscaler
from nnscaler import ComputeConfig, ModuleStreamRule
from nnscaler.parallel import _load_parallel_module_class
from nnscaler.runtime.stream import ModuleStreams
from .test_module_streams import CPUDeferredStreams
from .test_checkpoint_prefetch import prefetch_anchor
from ..utils import replace_all_device_with, new_empty


@nnscaler.register_op('a k, k n -> a n')
def projection(x, w):
    return x @ w


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4, 8))
        self.w_extra = torch.nn.Parameter(torch.randn(8, 4))
    def forward(self, x):
        return projection(x, self.w).sin().square() @ self.w_extra


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = Attention()
        self.other = torch.nn.Linear(4, 4)
        self.side = torch.nn.Linear(4, 4)
    def forward(self, x):
        return self.other(prefetch_anchor(self.attention(x))) + self.side(x)


def policy(graph, cfg):
    from nnscaler.policies import pas_dp
    for name in ('attention', 'other'):
        graph.recompute([n for n in graph.nodes() if name in (n.module_stack or {})])
    return pas_dp(graph, cfg)


class CPUStreams(CPUDeferredStreams):
    prefetch_launch_point = ModuleStreams.prefetch_launch_point
    staged_backward_point = ModuleStreams.staged_backward_point
    checkpoint_prefetch = ModuleStreams.checkpoint_prefetch
    def __enter__(self):
        super().__enter__()
        self.prefetch_points = []
        return self
    def __exit__(self, *exc):
        for point in self.prefetch_points:
            point.finish_forward()
        return super().__exit__(*exc)


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('side', [False, True])
def test_compiled_stages_only_replay_and_compose_with_submitted_module(tmp_path, monkeypatch, workers, side):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', CPUStreams)
    model = Model()
    reference = copy.deepcopy(model)
    rules = [ModuleStreamRule('attention', stream='staged', recompute_only=True,
             recompute_prefetch_on='*.prefetch_anchor', recompute_prefetch_prefix='*.projection')]
    if side:
        rules.append(ModuleStreamRule('side', stream='side', launch_on='*.prefetch_anchor', overlap_backward=True))
    name = f'staged_{workers}_{side}'
    nnscaler.parallelize(model, {'x': torch.randn(4, 4, requires_grad=True)}, policy,
        ComputeConfig(1, 1, module_streams=rules), gen_savedir=tmp_path,
        instance_name=name, load_module=False, codegen_workers=workers)
    source = next(tmp_path.rglob('gencode0.py')).read_text()
    assert source.count('.set_stages(') == 1
    assert source.count('ckpt.checkpoint(') == 1  # Unselected group keeps its normal checkpoint.
    cls = _load_parallel_module_class(Model, gen_savedir=tmp_path, instance_name=name, rank=0)
    generated = new_empty(cls, device='cpu', init_params=True)
    for _ in range(2):
        x = torch.randn(4, 4, requires_grad=True)
        rx = x.detach().clone().requires_grad_()
        for p in (*generated.parameters(), *reference.parameters()):
            p.grad = None
        y, expected = generated(x), reference(rx)
        y.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
        for attr, value in generated.named_parameters():
            torch.testing.assert_close(value.grad, reference.get_parameter(generated.fullmap[attr].orig_name).grad, rtol=0, atol=0)


@pytest.mark.parametrize('prefix', ['', [], [None], 4])
def test_invalid_prefix(prefix):
    with pytest.raises(ValueError):
        ModuleStreamRule('attention', recompute_only=True,
                         recompute_prefetch_on='*.anchor', recompute_prefetch_prefix=prefix)


def test_prefix_requires_forward_anchor():
    with pytest.raises(ValueError):
        ModuleStreamRule('attention', recompute_only=True, recompute_prefetch_prefix='*.projection')
    with pytest.raises(ValueError):
        ModuleStreamRule('attention', recompute_only=True, recompute_prefetch_on='*.anchor',
                         recompute_prefetch_phase='backward', recompute_prefetch_prefix='*.projection')


def test_training_attribute_is_snapshotted_as_explicit_input():
    from nnscaler.codegen.module.staged_checkpoint import emit_staged_checkpoint
    from nnscaler.runtime.staged_checkpoint import StagedCheckpoint
    from types import SimpleNamespace
    codes = [
        'def recompute(x):',
        '    y = projection(x, self.w, self.training)',
        '    return y.sin()',
        'result = ckpt.checkpoint(recompute, x, use_reentrant=False)',
    ]
    lines = emit_staged_checkpoint(codes, 'point', ['self.w'], ['projection'], [True])
    x = torch.randn(4, 3, requires_grad=True)
    w = torch.nn.Parameter(torch.randn(3, 5))
    module = SimpleNamespace(w=w, training=True)
    def projection(x, w, training):
        return x @ w if training else -(x @ w)
    for training in (True, False):
        module.training = training
        point = StagedCheckpoint('training_snapshot', 1 << 20)
        namespace = dict(self=module, point=point, projection=projection, x=x)
        exec('\n'.join(lines), namespace)
        y = namespace['result']
        module.training = not training  # Replay uses the original call's value.
        expected = projection(x, w, training).sin()
        gradients = torch.autograd.grad(y.sum(), (x, w))
        reference = torch.autograd.grad(expected.sum(), (x, w))
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        for a, b in zip(gradients, reference):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
