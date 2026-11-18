import gc

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import nnscaler
from nnscaler.runtime.stream import _get_module_stream, drain_stream_holds
from nnscaler.runtime.checkpoint_prefetch import Prefetch, Budget


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('when', ['forward', 'backward'])
def test_prefetch_runs_during_communication_and_keeps_gradients(when):
    torch.manual_seed(29)
    x = torch.randn(32, 32, device='cuda', requires_grad=True)
    w = torch.randn_like(x, requires_grad=True)
    rx, rw = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    stream = _get_module_stream(x.device, 'prefetch_cuda')
    comm = torch.cuda.Stream()
    for target in (torch.cuda.current_stream(), stream, comm):
        target.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(target), torch.no_grad():
            for _ in range(2):
                z = ((x @ w).sin() @ w.t()).cos()
            torch.cuda._sleep(1)
    torch.cuda.synchronize()
    del z
    budget = Budget(16 << 20)
    point = Prefetch('prefetch_cuda', budget)
    ready, pending = torch.cuda.Event(), torch.cuda.Event()
    calls = []

    def function(value):
        calls.append(True)
        hidden = value @ w
        if len(calls) == 2:
            finished = torch.cuda.Event()
            finished.record()
            finished.synchronize()
            assert ready.query(), 'recompute ran before the selected launch boundary'
            assert not pending.query(), 'recompute waited for communication completion'
        return (hidden.sin() @ w.t()).cos() * 2.0

    def submit():
        torch.cuda._sleep(100_000_000)
        ready.record()
        point.record_ready()
        comm.wait_event(ready)
        with torch.cuda.stream(comm):
            torch.cuda._sleep(800_000_000)
            pending.record()

    class Anchor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.point = nnscaler.capture_async_backward()
            return value.clone()

        @staticmethod
        def backward(ctx, grad):
            submit()
            ctx.point.submitted()
            torch.cuda.current_stream().wait_event(pending)
            return grad

    y = checkpoint(function, x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x, w)))
    if when == 'forward':
        submit()
        assert point.prefetch()
        point.finish_forward()
        torch.cuda.current_stream().wait_event(pending)
        result = y
    else:
        with point:
            result = Anchor.apply(y)
    result.sum().backward()
    expected = ((rx @ rw).sin() @ rw.t()).cos() * 2.0
    expected.sum().backward()
    torch.cuda.synchronize()
    budget.poll()
    drain_stream_holds(wait=True)
    assert budget.used == 0 and len(calls) == 2
    torch.testing.assert_close(y, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, rx.grad, rtol=0, atol=0)
    torch.testing.assert_close(w.grad, rw.grad, rtol=0, atol=0)


def test_abandoned_forward_prefetch_reclaims_budget_after_cuda_completion():
    x = torch.ones(1024, device='cuda', requires_grad=True)
    budget = Budget(16 << 20)
    point = Prefetch('abandoned_prefetch', budget)
    def function(value):
        torch.cuda._sleep(100_000_000)
        return value.sin().square()
    y = checkpoint(function, x, use_reentrant=False,
                   context_fn=lambda: point.contexts((x,)))
    assert point.prefetch()
    point.finish_forward()
    del y, point
    gc.collect()
    budget.poll(wait=True)
    drain_stream_holds(wait=True)
    assert budget.used == 0
