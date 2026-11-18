"""Protect returned gradient views through delayed CUDA accumulation."""
import weakref

import pytest
import torch

from nnscaler.runtime import stream as lifetime


@pytest.fixture
def cuda_drain():
    torch.cuda.synchronize()
    lifetime.drain_stream_holds(wait=True)
    yield
    torch.cuda.synchronize()
    lifetime.drain_stream_holds(wait=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_weight_gradient_alias_survives_delayed_accumulation(cuda_drain, monkeypatch):
    import nnscaler
    from nnscaler.runtime.weight_backward import WeightBackwardRegion

    class Anchor(torch.autograd.Function):

        @staticmethod
        def forward(ctx, value):
            ctx.point = nnscaler.capture_async_backward()
            return value.view_as(value)

        @staticmethod
        def backward(ctx, grad):
            ctx.point.record_ready()
            ctx.point.submitted()
            return grad

    class AliasedWeight(torch.autograd.Function):

        @staticmethod
        def forward(ctx, value, weight, payload, binding):
            ctx.save_for_backward(weight, payload)
            ctx.binding = binding
            return value * weight

        @staticmethod
        def backward(ctx, grad):
            weight, payload = ctx.saved_tensors
            ctx.binding.defer(lambda: (payload.view_as(weight),), (payload,))
            return (grad * weight, None, None, None)
    count = 1 << 20
    producer = torch.cuda.Stream()
    weight = torch.nn.Parameter(torch.ones(count, device='cuda'))
    region = WeightBackwardRegion('alias_storage_test', (weight,))
    warm = []
    for stream in (producer, region.stream):
        with torch.cuda.stream(stream):
            warm.extend((torch.empty(count, device='cuda') for _ in range(8)))
    torch.cuda.synchronize()
    del warm
    with torch.cuda.stream(region.stream):
        weight.grad = torch.zeros_like(weight)
        # Warm lazy CUDA loading before the deliberate delay.
        weight.grad.add_(torch.ones_like(weight))
        weight.grad.zero_()
    with torch.cuda.stream(producer):
        payload = torch.ones(count, device='cuda')
    reference = weakref.ref(payload)
    torch.cuda.current_stream().wait_stream(producer)
    value = torch.ones(count, device='cuda', requires_grad=True)
    state = {}
    original = WeightBackwardRegion.gradients

    def delayed_publication(region, incoming):
        assert torch.cuda.current_stream() == region.stream
        state['computed'] = region.tasks[torch._C._current_graph_task_id()].done
        torch.cuda._sleep(1000000000)
        return original(region, incoming)
    monkeypatch.setattr(WeightBackwardRegion, 'gradients', delayed_publication)
    with region.anchor():
        anchored = Anchor.apply(nnscaler.prepare_async_input(value))
    with region.compute():
        binding = nnscaler.prepare_weight_backward((weight,))
        output = AliasedWeight.apply(anchored, binding.parameters[0], payload, binding)
    region.close_forward()
    del payload
    loss = output.sum()
    loss.backward()
    del loss, output
    # Free the callback input hold while the later W-stream read is pending.
    # The independent producer does not inherit the caller's autograd wait.
    state['computed'].synchronize()
    lifetime.drain_stream_holds()
    assert reference() is None
    with torch.cuda.stream(producer):
        replacement = [torch.empty(count, device='cuda').fill_(-7) for _ in range(10)]
    torch.cuda.synchronize()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight), rtol=0, atol=0)
    torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)
    del replacement
