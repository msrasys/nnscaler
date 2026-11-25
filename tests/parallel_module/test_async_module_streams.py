# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from contextlib import contextmanager

import pytest
import torch

from nnscaler import AsyncResult, ComputeConfig, ModuleStreamRule, parallelize, register_async_op, register_op
from nnscaler.parallel import _load_parallel_module_class
from nnscaler.runtime.async_op import PendingOp
from ..utils import new_empty, replace_all_device_with
from .test_module_streams import Branch, SerialStreams, auxiliary_statistics


class PreparedSin(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, value):
        ctx.save_for_backward(x)
        return value

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * x.cos(), None


@register_op('a b -> a b')
def submitted_sin(x):
    return x.sin()


def submit_sin(x):
    with torch.no_grad():
        value = x.sin()
    return AsyncResult(value, lambda: None)


def complete_sin(value, x):
    return PreparedSin.apply(x, value)


register_async_op(submitted_sin, submit=submit_sin, complete=complete_sin)


class AsyncModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.side = Branch()

    def forward(self, x):
        main = submitted_sin(x).cos()
        return main + self.side(x)


class AsyncNoGradModel(AsyncModel):
    def forward(self, x):
        with torch.no_grad():
            main = submitted_sin(x)
        return main.cos() + self.side(x)


HOOK_ORDER = []


def before_async(module, metadata, inputs, kwargs):
    HOOK_ORDER.append('pre')


def after_async(module, metadata, inputs, kwargs, outputs):
    HOOK_ORDER.append('post')


@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('split', [False, True])
def test_after_submit_binds_backward_after_side_outputs(tmp_path, monkeypatch, workers, split):
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', SerialStreams)
    model = AsyncModel()
    reference = copy.deepcopy(model)
    instance = f'async_after_{workers}_{split}'
    rule = (ModuleStreamRule('side', launch_before='*.submitted_sin', split_before='down',
                             resume_after='*.submitted_sin', overlap_backward=True)
            if split else ModuleStreamRule('side', launch_after='*.submitted_sin', overlap_backward=True))
    parallelize(model, {'x': torch.randn(4, 8, requires_grad=True)}, 'dp',
                ComputeConfig(1, 1, module_streams=(rule,)), gen_savedir=tmp_path,
                codegen_workers=workers, load_module=False, instance_name=instance)
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    assert (code.index('_module_streams.submit(') < code.index('_module_streams.fork(')) != split
    assert code.index('_module_streams.fork(') < code.index('_module_streams.after_backward(')
    assert code.index('_module_streams.after_backward(') < code.index('.complete(')
    assert 'track_stream_inputs = False' in code
    cls = _load_parallel_module_class(AsyncModel, gen_savedir=tmp_path, rank=0, instance_name=instance)
    generated = new_empty(cls, device='cpu', init_params=True)
    inputs = [torch.randn(4, 8, requires_grad=True) for _ in range(3)]
    ref_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    actual = [generated(x) for x in inputs]
    expected = [reference(x) for x in ref_inputs]
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x, y)
    sum(x.square().sum() for x in actual).backward()
    sum(x.square().sum() for x in expected).backward()
    for x, y in zip(inputs, ref_inputs):
        torch.testing.assert_close(x.grad, y.grad)
    for name, parameter in generated.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)


def test_pending_completion_uses_rebound_input_and_is_single_use():
    x = torch.randn(8, requires_grad=True)
    pending = PendingOp(submitted_sin, (x,), {})
    controlled = x.detach().clone().requires_grad_()
    y = pending.complete(controlled)
    y.sum().backward()
    torch.testing.assert_close(y, x.detach().sin())
    assert x.grad is None
    torch.testing.assert_close(controlled.grad, controlled.detach().cos())
    with pytest.raises(RuntimeError, match='only complete once'):
        pending.complete(x)


@replace_all_device_with('cpu', force=True)
def test_async_hooks_and_grad_context_keep_original_semantics(tmp_path, monkeypatch):
    from nnscaler.policies import pas_dp
    from nnscaler.ir.operator import IRFwOperation

    class LoggedStreams(SerialStreams):
        @contextmanager
        def fork(self, *args, **kwargs):
            HOOK_ORDER.append('side')
            yield

    def submit(x):
        assert not torch.is_grad_enabled()
        HOOK_ORDER.append('submit')
        return submit_sin(x)

    def complete(value, x):
        assert not torch.is_grad_enabled()
        HOOK_ORDER.append('bind')
        return complete_sin(value, x)

    def policy(graph, config):
        for node in graph.select(ntype=IRFwOperation):
            if node.signature.endswith('.submitted_sin'):
                node.pre_hook, node.post_hook = before_async, after_async
        return pas_dp(graph, config)

    monkeypatch.setattr(submitted_sin, '_nnscaler_async_op', (submit, complete))
    monkeypatch.setattr('nnscaler.runtime.stream.ModuleStreams', LoggedStreams)
    model = AsyncNoGradModel()
    reference = copy.deepcopy(model)
    parallelize(model, {'x': torch.randn(4, 8, requires_grad=True)}, policy,
                ComputeConfig(1, 1, module_streams=(ModuleStreamRule(
                    'side', launch_after='*.submitted_sin'),)),
                gen_savedir=tmp_path, load_module=False, instance_name='async_hooks')
    cls = _load_parallel_module_class(AsyncNoGradModel, gen_savedir=tmp_path, rank=0, instance_name='async_hooks')
    generated = new_empty(cls, device='cpu', init_params=True)
    x = torch.randn(4, 8, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    HOOK_ORDER.clear()
    result = generated(x)
    assert HOOK_ORDER == ['pre', 'submit', 'side', 'bind', 'post']
    expected = reference(ref_x)
    result.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(x.grad, ref_x.grad)


@replace_all_device_with('cpu', force=True)
def test_unmanaged_streams_still_enable_executor_tracking(tmp_path):
    class Mixed(AsyncModel):
        def forward(self, data):
            return auxiliary_statistics(super().forward(data)).sum()

    parallelize(Mixed(), {'data': torch.randn(4, 8)}, 'data',
                ComputeConfig(1, 1, use_end2end=True,
                              module_streams=(ModuleStreamRule('side', launch_after='*.submitted_sin'),),
                              multi_stream_ops=('*.auxiliary_statistics',)),
                gen_savedir=tmp_path, load_module=False)
    code = next(tmp_path.rglob('gencode0.py')).read_text()
    assert 'track_stream_inputs = True' in code and 'track_stream=True' in code


@replace_all_device_with('cpu', force=True)
def test_backward_after_requires_async_binding_protocol(tmp_path):
    from .test_module_streams import ForkModel

    with pytest.raises(RuntimeError, match='Code generation failed') as error:
        parallelize(ForkModel(), {'x': torch.randn(4, 8)}, 'dp',
                    ComputeConfig(1, 1, module_streams=(ModuleStreamRule(
                        'side', launch_after='torch.sin', overlap_backward=True),)),
                    gen_savedir=tmp_path, load_module=False)
    assert 'registered async operator' in str(error.value.__cause__)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_generated_async_binding_cuda_with_live_forwards(tmp_path):
    import torch.distributed as dist
    import nnscaler

    dist.init_process_group('nccl', init_method=f'file://{tmp_path}/pg', rank=0, world_size=1)
    try:
        nnscaler.init()
        model = AsyncModel().cuda()
        reference = copy.deepcopy(model)
        generated = parallelize(
            model, {'x': torch.randn(4, 8, device='cuda', requires_grad=True)}, 'dp',
            ComputeConfig(1, 1, module_streams=(ModuleStreamRule(
                'side', launch_after='*.submitted_sin', overlap_backward=True),)),
            gen_savedir=tmp_path, instance_name='cuda_async_submit')
        inputs = [torch.randn(4, 8, device='cuda', requires_grad=True) for _ in range(3)]
        references = [x.detach().clone().requires_grad_() for x in inputs]
        actual = [generated(x) for x in inputs]
        expected = [reference(x) for x in references]
        sum(x.square().sum() for x in actual).backward()
        sum(x.square().sum() for x in expected).backward()
        generated.sync_grad()
        for x, y in zip(actual, expected):
            torch.testing.assert_close(x, y)
        for x, y in zip(inputs, references):
            torch.testing.assert_close(x.grad, y.grad)
        for name, parameter in generated.named_parameters():
            torch.testing.assert_close(parameter.grad, reference.get_parameter(generated.fullmap[name].orig_name).grad)
    finally:
        nnscaler.uninit()
        if dist.is_initialized():
            dist.destroy_process_group()
