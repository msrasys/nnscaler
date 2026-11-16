import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nnscaler.runtime.checkpoint_stream import contexts
from nnscaler.runtime.stream import _get_module_stream, drain_stream_holds


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def test_recompute_overlaps_pending_gradient_and_preserves_gradients():
    torch.manual_seed(7)
    x = torch.randn(16, 16, device='cuda', requires_grad=True)
    weight = torch.randn(16, 16, device='cuda', requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    stream = _get_module_stream(x.device, 'checkpoint_test')
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(2):
            warm = (x @ weight).sin() @ weight.t()
    del warm
    torch.cuda.synchronize()
    blocked_done = torch.cuda.Event()
    calls = []

    def function(value):
        calls.append(torch.cuda.current_stream().cuda_stream)
        hidden = value @ weight
        if len(calls) > 1:
            finished = torch.cuda.Event()
            finished.record()
            finished.synchronize()
            assert not blocked_done.query(), 'recompute waited for the incoming gradient'
        return hidden.sin() @ weight.t()

    class GradientWait(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            torch.cuda._sleep(500_000_000)
            blocked_done.record()
            return gradient

    y = checkpoint(function, x, use_reentrant=False,
                   context_fn=lambda: contexts('checkpoint_test', (x, weight)))
    expected = (ref_x @ ref_weight).sin() @ ref_weight.t()
    output = GradientWait.apply(y)
    for iteration in range(2):
        output.sum().backward(retain_graph=iteration == 0)
        expected.sum().backward(retain_graph=iteration == 0)
        torch.testing.assert_close(y, expected)
        torch.testing.assert_close(x.grad, ref_x.grad)
        torch.testing.assert_close(weight.grad, ref_weight.grad)
        x.grad = weight.grad = ref_x.grad = ref_weight.grad = None
    assert len(calls) == 3 and calls[0] != calls[1] == calls[2]
    drain_stream_holds(wait=True)
