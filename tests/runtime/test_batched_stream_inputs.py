"""First-order callback input ownership, including delayed CUDA consumers."""
import weakref

import pytest
import torch

from nnscaler.runtime import stream as lifetime


@pytest.fixture
def runtime(monkeypatch):
    from nnscaler.runtime import stream as module
    assert not module._pending_holds
    monkeypatch.setattr(module, '_pending_holds', {})
    records, events = [], []

    class Event:
        ready = False

        def __init__(self):
            events.append(self)

        def record(self, stream):
            self.stream = stream

        def query(self):
            return self.ready

        def synchronize(self):
            self.ready = True

    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda: 'consumer')
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda tensor: True))
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda tensor, stream: records.append(id(tensor)))
    return module, records, events


def test_default_forward_protection_still_records_tensor_stream(runtime):
    module, records, events = runtime
    tensor = torch.ones(8)
    identity, ref = id(tensor), weakref.ref(tensor)
    with module.input_stream_lifetime((tensor,), True):
        pass
    del tensor
    assert records == [identity]
    assert not events
    assert ref() is None


@pytest.mark.parametrize('fail', [False, True])
def test_batch_snapshots_mutable_inputs_and_keeps_tensors_until_done(runtime, fail):
    module, records, events = runtime
    tensor = torch.ones(8)
    ref = weakref.ref(tensor)
    nested = {'inputs': [tensor, tensor]}
    try:
        with module.input_stream_lifetime(nested, True, retain_tensors=True) as owners:
            storage = weakref.ref(owners[0].storage)
            if fail:
                raise RuntimeError('callback failure')
    except RuntimeError as error:
        assert fail and str(error) == 'callback failure'
    nested.clear()
    del tensor, owners
    module.drain_stream_holds()
    assert ref() is None
    assert storage() is not None
    assert not records
    assert len(events) == 1
    events[0].ready = True
    module.drain_stream_holds()
    assert storage() is None


def test_returned_gradient_views_keep_separate_record_stream_protection(runtime):
    module, records, events = runtime
    tensor = torch.ones(8)
    with module.input_stream_lifetime((tensor,), True, retain_tensors=True):
        gradient = tensor.view_as(tensor)
        with module.input_stream_lifetime((gradient,), True):
            pass
    assert records == [id(gradient)]
    assert len(events) == 1
    module.drain_stream_holds(wait=True)


def test_alias_holds_storage_without_keeping_gradient_tensor(runtime, monkeypatch):
    module, records, events = runtime
    held = []
    monkeypatch.setattr(module, 'hold_owners_through_backward',
                        lambda owners, streams: held.extend(owners))
    tensor = torch.ones(8)
    with module.input_stream_lifetime((tensor,), True, retain_tensors=True) as owners:
        gradient = tensor.view_as(tensor)
        reference = weakref.ref(gradient)
        module.hold_aliased_inputs_through_backward(owners, (gradient,), 'consumer')
        assert held and all(isinstance(owner, module._StorageOwner) for owner in held)
        del gradient
        assert reference() is None
        held.clear()
        module.hold_aliased_inputs_through_backward(owners, (torch.ones(8),), 'consumer')
        assert not held  # Fresh output storage can follow ordinary stream order.
    module.drain_stream_holds(wait=True)


def test_completed_callback_inputs_release_before_next_forward(runtime):
    module, records, events = runtime
    tensor = torch.ones(8)
    with module.input_stream_lifetime((tensor,), True, retain_tensors=True) as owners:
        reference = weakref.ref(owners[0].storage)
    del tensor, owners
    events[0].ready = True
    with module.ModuleStreams():
        assert reference() is None


@pytest.fixture
def cuda_drain():
    torch.cuda.synchronize()
    lifetime.drain_stream_holds(wait=True)
    yield
    torch.cuda.synchronize()
    lifetime.drain_stream_holds(wait=True)


def streams_and_storage():
    producer, consumer = torch.cuda.current_stream(), torch.cuda.Stream()
    count = 1 << 20
    warm = []
    for stream in (producer, consumer):
        with torch.cuda.stream(stream):
            warm.extend(torch.empty(count, device='cuda') for _ in range(8))
    torch.cuda.synchronize()
    del warm
    return producer, consumer, count


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('fail', [False, True])
def test_callback_inputs_remain_alive_until_gpu_completion(fail, cuda_drain):
    producer, consumer, count = streams_and_storage()
    tensor = torch.ones(count, device='cuda')
    reference = weakref.ref(tensor)
    inputs = {'nested': [tensor, tensor]}
    consumer.wait_stream(producer)
    try:
        with torch.cuda.stream(consumer), lifetime.input_stream_lifetime(
                inputs, True, retain_tensors=True) as owners:
            storage = weakref.ref(owners[0].storage)
            torch.cuda._sleep(100_000_000)
            output = tensor + 1
            if fail:
                raise RuntimeError('callback failure')
    except RuntimeError as error:
        assert fail and str(error) == 'callback failure'
    inputs.clear()
    del tensor, owners
    assert reference() is None
    assert storage() is not None
    replacement = [torch.empty(count, device='cuda').fill_(-7) for _ in range(8)]
    consumer.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 2), rtol=0, atol=0)
    lifetime.drain_stream_holds()
    assert storage() is None
    del replacement, output


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_aliased_gradient_survives_later_publication_after_input_event(cuda_drain):
    producer, consumer, count = streams_and_storage()
    tensor = torch.ones(count, device='cuda')
    reference = weakref.ref(tensor)
    consumer.wait_stream(producer)
    with torch.cuda.stream(consumer), lifetime.input_stream_lifetime(
            (tensor,), True, retain_tensors=True):
        gradient = tensor.view_as(tensor)
        with lifetime.input_stream_lifetime((gradient,), True):
            pass
    # The input completion event precedes the eventual gradient reader. This
    # models an input workspace reused for a gradient, then reducer publication.
    inputs_done = torch.cuda.Event()
    inputs_done.record(consumer)
    with torch.cuda.stream(consumer):
        torch.cuda._sleep(100_000_000)
        output = gradient + 1
    del tensor, gradient
    inputs_done.synchronize()
    lifetime.drain_stream_holds()
    assert reference() is None
    replacement = [torch.empty(count, device='cuda').fill_(-7) for _ in range(8)]
    consumer.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 2), rtol=0, atol=0)
    del replacement, output


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('retain', [False, True], ids=['allocator', 'retained'])
def test_weight_alias_storage_survives_delayed_accumulation(cuda_drain, monkeypatch, retain):
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
            ctx.binding.defer(lambda: (payload.view_as(weight),), (payload,), retain_inputs=retain)
            return grad * weight, None, None, None

    count = 1 << 20
    producer = torch.cuda.Stream()
    weight = torch.nn.Parameter(torch.ones(count, device='cuda'))
    region = WeightBackwardRegion('alias_storage_test', (weight,))
    warm = []
    for stream in (producer, region.stream):
        with torch.cuda.stream(stream):
            warm.extend(torch.empty(count, device='cuda') for _ in range(8))
    torch.cuda.synchronize()
    del warm
    with torch.cuda.stream(region.stream):
        weight.grad = torch.zeros_like(weight)
        # Warm the same accumulation kernel before queuing the deliberate
        # delay; CUDA lazy module loading can otherwise synchronize and hide
        # an unprotected input reuse.
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
        torch.cuda._sleep(1_000_000_000)
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
    # This event precedes the delayed AccumulateGrad read. The producer has no
    # dependency on the caller's final autograd wait, so reuse there is a race
    # unless either allocator tracking or the aliased storage hold protects it.
    state['computed'].synchronize()
    lifetime.drain_stream_holds()
    assert reference() is None
    with torch.cuda.stream(producer):
        replacement = [torch.empty(count, device='cuda').fill_(-7) for _ in range(10)]
    torch.cuda.synchronize()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight), rtol=0, atol=0)
    torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)
    del replacement
