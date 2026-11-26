# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nnscaler.runtime.backward import capture_async_backward
from nnscaler.runtime.checkpoint_prefetch import Prefetch, Budget

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('phase', ['forward', 'backward'])
@pytest.mark.parametrize('stream_name', ['default', 'prefetch_test'])
def test_prefetch_runs_before_backend_result_wait_and_preserves_gradients(phase, stream_name):
    caller, communication = torch.cuda.current_stream(), torch.cuda.Stream()
    done = torch.cuda.Event()
    x = torch.randn(32, 32, device='cuda', requires_grad=True)
    weight = torch.randn(32, 32, device='cuda', requires_grad=True)

    def function(value, parameter):
        torch.cuda._sleep(100_000_000)
        return (value @ parameter).sin()

    for _ in range(3):
        function(x, weight).sum().backward()
        torch.cuda.synchronize()
    x.grad = weight.grad = None
    point = Prefetch(stream_name, Budget(1 << 20))

    def submit():
        communication.wait_stream(caller)
        with torch.cuda.stream(communication):
            torch.cuda._sleep(500_000_000)
            done.record()

    class Anchor(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.point = capture_async_backward()
            return value * 2

        @staticmethod
        def backward(ctx, grad):
            ctx.point.record_ready()
            submit()
            ctx.point.submitted()
            ctx.point.done.synchronize()
            assert not done.query(), 'Recompute waited for communication completion'
            caller.wait_stream(communication)
            return grad * 2

    output = checkpoint(function, x, weight, use_reentrant=False,
                        context_fn=lambda: point.contexts((x, weight)))
    torch.cuda.synchronize()
    if phase == 'forward':
        submit()
        assert point.prefetch()
        point.done.synchronize()
        assert not done.query(), 'Forward prefetch waited for communication completion'
        caller.wait_stream(communication)
        output = output * 2
    else:
        with point:
            output = Anchor.apply(output)
    observed = torch.autograd.grad(output.sum(), (x, weight))
    expected = torch.autograd.grad((function(x, weight) * 2).sum(), (x, weight))
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    point.budget.poll(wait=True)
    assert point.budget.used == 0
