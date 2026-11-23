import pytest
import torch
from nnscaler import BackwardValue


def test_backward_priority_preserves_input_graph_boundaries_and_math():
    from nnscaler.runtime.backward import backward_boundaries, prioritize_backward

    x = torch.randn(8, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    common = x.sin()
    metadata = {'nested': [common]}
    metadata['cycle'] = metadata
    boundaries = backward_boundaries(metadata)
    selected, other = common.tanh().square(), common.cos()
    input_sequence = common.grad_fn._sequence_nr()
    other_sequence = other.grad_fn._sequence_nr()
    leaf = common.grad_fn.next_functions[0][0]
    leaf_sequence = leaf._sequence_nr()
    prioritize_backward((selected,), boundaries)
    priority = selected.grad_fn._sequence_nr()
    prioritize_backward((selected,), boundaries)
    assert selected.grad_fn._sequence_nr() == priority > other_sequence
    assert common.grad_fn._sequence_nr() == input_sequence
    assert other.grad_fn._sequence_nr() == other_sequence
    assert leaf._sequence_nr() == leaf_sequence
    (selected + other).sum().backward()
    (rx.sin().tanh().square() + rx.sin().cos()).sum().backward()
    torch.testing.assert_close(x.grad, rx.grad)


def test_backward_priority_rejects_opaque_input_graphs():
    from types import SimpleNamespace
    from nnscaler.runtime.backward import backward_boundaries

    with pytest.raises(TypeError, match='graph boundaries'):
        backward_boundaries(SimpleNamespace(value=torch.ones(1, requires_grad=True)))


def test_nested_backward_regions_share_one_input_event(monkeypatch):
    from nnscaler.runtime.backward import BackwardLaunchPoint, capture_async_backward

    events = []

    class Event:
        def record(self):
            events.append(self)

    monkeypatch.setattr(torch.cuda, 'Event', Event)
    outer, inner = BackwardLaunchPoint(()), BackwardLaunchPoint(())
    with outer, inner:
        captured = capture_async_backward()
    assert outer.captured and inner.captured
    captured.record_ready()
    assert outer.ready is inner.ready is events[0]
    captured.record_ready()
    assert outer.ready is inner.ready is events[1]
    assert capture_async_backward() is None


class _Front(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packet):
        ctx.packet = packet
        return x * 2

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.packet.pop()
        return value * 2, None


class _Tail(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, packet, producer=None, storage=None):
        ctx.packet, ctx.producer, ctx.storage = packet, producer, storage
        return x * 3

    @staticmethod
    def backward(ctx, gradient):
        value = gradient * 3
        if ctx.producer is None:
            ctx.packet.put((value,))
        else:
            caller = torch.cuda.current_stream()
            ctx.producer.wait_stream(caller)
            with torch.cuda.stream(ctx.producer):
                torch.cuda._sleep(20_000_000)
                ctx.storage.copy_(value)
                ctx.packet.put((ctx.storage,))
        return value, None, None, None


def test_backward_values_support_retained_and_pruned_consumers():
    x = torch.randn(16, requires_grad=True)
    packet = BackwardValue(x.device)
    middle = _Front.apply(x, packet)
    y = _Tail.apply(middle, packet, None, None)
    for _ in range(2):
        torch.testing.assert_close(torch.autograd.grad(y.sum(), x, retain_graph=True)[0], torch.full_like(x, 6))
        assert not packet.pending
    torch.testing.assert_close(torch.autograd.grad(y.sum(), middle)[0], torch.full_like(x, 3))
    assert not packet.pending


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_backward_value_waits_for_an_independent_producer():
    producer = torch.cuda.Stream()
    x = torch.ones(128, device='cuda', requires_grad=True)
    storage = torch.zeros_like(x)
    # Warm allocator caches before the delayed producer so allocation cannot
    # accidentally synchronize away the readiness race.
    for _ in range(2):
        packet = BackwardValue(x.device)
        y = _Tail.apply(_Front.apply(x, packet), packet, producer, storage)
        torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0], torch.full_like(x, 6))
        torch.cuda.synchronize()
        storage.zero_()
        assert not packet.pending
