import gc
import weakref
from contextlib import nullcontext

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nnscaler.runtime.checkpoint_stream import contexts
from nnscaler.runtime.stream import _get_module_stream, drain_stream_holds


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('reverse_consumers', [False, True])
def test_replayed_saved_tensors_are_ready_on_each_consumer_stream(reverse_consumers):
    class SavedValueVJP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return value.view_as(value)

        @staticmethod
        def backward(ctx, gradient):
            value, = ctx.saved_tensors
            return gradient * value

    consumers = [torch.cuda.Stream(), torch.cuda.Stream()]
    if reverse_consumers:
        consumers.reverse()
    x = torch.ones(4096, device='cuda', requires_grad=True)
    replay = _get_module_stream(x.device, 'multiple_checkpoint_consumers')

    def views(value):
        with torch.cuda.stream(consumers[0]):
            first = SavedValueVJP.apply(value)
        with torch.cuda.stream(consumers[1]):
            second = SavedValueVJP.apply(value)
        return first, second

    # A cold consumer-side cudaMalloc synchronizes the device and can conceal
    # this race. Warm the backward kernels and both stream allocator pools.
    warm_x = x.detach().clone().requires_grad_()
    a, b = views(warm_x * 2 + 1)
    (a + b).sum().backward()
    with torch.cuda.stream(replay):
        torch.cuda._sleep(1)
        garbage = [torch.full_like(x, -19.) for _ in range(4)]
    torch.cuda.synchronize()
    del a, b, warm_x, garbage
    calls = 0

    def function(value):
        nonlocal calls
        calls += 1
        if calls > 1:
            torch.cuda._sleep(200_000_000)
        return views(value * 2 + 1)

    a, b = checkpoint(function, x, use_reentrant=False,
                      context_fn=lambda: contexts('multiple_checkpoint_consumers', (x,)))
    loss = (a + b).sum()
    for iteration in range(2):
        loss.backward(retain_graph=iteration == 0)
        torch.cuda.synchronize()
        # This custom VJP uses the saved value: 2 consumers * (2*x+1) * 2.
        torch.testing.assert_close(x.grad, torch.full_like(x, 12.), rtol=0, atol=0)
        x.grad = None
    assert calls == 3
    drain_stream_holds(wait=True)


@pytest.mark.parametrize('gate_inputs', [False, True])
def test_recompute_overlaps_pending_gradient_and_preserves_gradients(gate_inputs):
    from nnscaler.runtime.deferred import BackwardLaunchPoint, capture_async_backward
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
    input_ready = torch.cuda.Event()
    point = BackwardLaunchPoint(()) if gate_inputs else None
    calls = []

    def function(value):
        calls.append(torch.cuda.current_stream().cuda_stream)
        hidden = value @ weight
        if len(calls) > 1:
            finished = torch.cuda.Event()
            finished.record()
            finished.synchronize()
            if gate_inputs:
                assert input_ready.query(), 'recompute ran before the backend inputs were ready'
            assert not blocked_done.query(), 'recompute waited for the incoming gradient'
        return hidden.sin() @ weight.t()

    class GradientWait(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.point = capture_async_backward()
            return value.clone()

        @staticmethod
        def backward(ctx, gradient):
            if ctx.point is not None:
                torch.cuda._sleep(500_000_000)
                input_ready.record()
                ctx.point.record_ready()
            torch.cuda._sleep(500_000_000)
            blocked_done.record()
            return gradient

    y = checkpoint(function, x, use_reentrant=False,
                   context_fn=lambda: contexts('checkpoint_test', (x, weight), backward_ready=point))
    expected = (ref_x @ ref_weight).sin() @ ref_weight.t()
    with point or nullcontext():
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


def test_opaque_checkpoint_input_survives_queued_backward():
    backward_done = torch.cuda.Event()
    class Owner:
        def __init__(self):
            self.tensor = torch.full((1 << 20,), 3., device='cuda')

    class UseOwner(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, owner):
            ctx.owner = owner
            ctx.save_for_backward(x)
            return x * owner.tensor

        @staticmethod
        def backward(ctx, grad):
            x, = ctx.saved_tensors  # Trigger the checkpoint recomputation.
            _get_module_stream(x.device, 'owner_recompute').synchronize()
            torch.cuda._sleep(1_000_000_000)
            result = grad * ctx.owner.tensor
            backward_done.record()
            ctx.owner = None
            return result, None

    owner = Owner()
    reference = weakref.ref(owner)
    consumer = torch.cuda.Stream()
    consumer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(consumer):
        x = torch.ones_like(owner.tensor, requires_grad=True)
        # Load the broadcast multiply kernel before introducing the delay.
        warm = torch.ones((), device='cuda').expand_as(x) * owner.tensor
        consumer.synchronize()
        del warm
        y = checkpoint(UseOwner.apply, x, owner, use_reentrant=False,
                       context_fn=lambda: contexts('owner_recompute', (x, owner)))
        drain_stream_holds(wait=True)
        del owner
        y.sum().backward()
        del y
    gc.collect()
    drain_stream_holds()
    assert not backward_done.query()
    assert reference() is not None
    torch.cuda.synchronize()
    torch.testing.assert_close(x.grad, torch.full_like(x, 3.))
    drain_stream_holds(wait=True)
    gc.collect()
    assert reference() is None
