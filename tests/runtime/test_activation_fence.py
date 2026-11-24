"""Batch completion publication, exceptional exits and consumer dependencies."""
from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import threading

import pytest
import torch

from nnscaler.runtime.activation_config import CPUOffloadConfig
from nnscaler.runtime.activation_fence import TransferFence
from nnscaler.runtime.activation_observer import _Observation
from nnscaler.runtime.activation_transfer import ActivationTransferEngine


class Event:
    def __init__(self):
        self.records = self.queries = self.syncs = 0
        self.complete = False

    def record(self, stream):
        self.records += 1

    def query(self):
        self.queries += 1
        return self.complete

    def synchronize(self):
        self.syncs += 1


class Stream:
    def __init__(self, identity=17):
        self.cuda_stream = identity
        self.waits = []
        self.syncs = 0

    def wait_event(self, event):
        assert event.records > 0, 'CUDA must not receive an unrecorded event'
        self.waits.append(event)

    def synchronize(self):
        self.syncs += 1


def engine(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'stream', lambda stream: nullcontext())
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda *args: None)
    value = ActivationTransferEngine(CPUOffloadConfig(reload_on_compute_stream=False),
                                     torch.device('cpu'), Counter(), _Observation())
    value.d2h_stream = Stream()
    value.h2d_stream = Stream(18)
    return value


def test_unpublished_fence_cannot_be_polled_as_complete_or_consumed_by_its_owner():
    event = Event()
    fence = TransferFence(event)
    assert not fence.query() and event.queries == 0
    with pytest.raises(RuntimeError, match='before its batch closes'):
        fence.cuda_event()
    fence.publish(Stream())
    assert fence.cuda_event() is event and not fence.query()
    event.complete = True
    assert fence.query()
    with pytest.raises(RuntimeError, match='cannot be re-recorded'):
        fence.publish(Stream())


def test_another_consumer_waits_for_publication_not_dma_completion():
    event = Event()
    fence = TransferFence(event)
    entered, done = threading.Event(), threading.Event()
    result = []
    def consume():
        entered.set()
        result.append(fence.cuda_event())
        done.set()
    thread = threading.Thread(target=consume)
    thread.start()
    assert entered.wait(1)
    assert not done.is_set()
    fence.publish(Stream())
    assert done.wait(1)
    thread.join(1)
    assert result == [event] and not event.complete and not event.syncs


def test_completion_cache_does_not_regress_after_a_delayed_concurrent_query():
    entered, release = threading.Event(), threading.Event()
    class DelayedEvent(Event):
        def query(self):
            self.queries += 1
            if self.queries == 1:
                entered.set()
                assert release.wait(2)
                return False
            return True
    event = DelayedEvent()
    fence = TransferFence(event)
    fence.publish(Stream())
    result = []
    thread = threading.Thread(target=lambda: result.append(fence.query()))
    thread.start()
    assert entered.wait(1)
    try:
        assert fence.query()
    finally:
        release.set()
        thread.join(2)
    assert result == [True] and fence.query() and event.queries == 2


def test_failed_publication_never_recycles_a_buffer():
    class FailedEvent(Event):
        def record(self, stream):
            raise RuntimeError('record failed')
    fence = TransferFence(FailedEvent())
    with pytest.raises(RuntimeError, match='record failed'):
        fence.publish(Stream())
    assert not fence.query()
    with pytest.raises(RuntimeError, match='publication failed'):
        fence.synchronize()


def test_pool_reaps_shared_fences_once_and_preserves_pending_buffers():
    from nnscaler.runtime.activation_memory import _OffloadPool
    pool = _OffloadPool(CPUOffloadConfig())
    event = Event()
    buffers = [torch.empty(8, dtype=torch.uint8) for _ in range(3)]
    pool.pending = [(buffer, event) for buffer in buffers]
    pool._reap()
    assert event.queries == 1 and len(pool.pending) == 3 and not pool.free
    event.complete = True
    pool._reap()
    assert event.queries == 2 and not pool.pending and len(pool.free[8]) == 3


def test_d2h_batch_shares_one_completion_and_one_poll(monkeypatch):
    value = engine(monkeypatch)
    sources = [torch.arange(n, dtype=torch.uint8) for n in (7, 11, 19)]
    destinations = [torch.empty_like(x) for x in sources]
    with value._stream_batch(value.d2h_stream, value._d2h_batch):
        fences = [value.to_host(src, dst) for src, dst in zip(sources, destinations)]
        assert not fences[0].query()
    assert all(fence is fences[0] for fence in fences)
    assert fences[0].event.records == value.stats['d2h_completion_events'] == 1
    assert all(torch.equal(a, b) for a, b in zip(sources, destinations))
    assert value.inflight_bytes == 37
    fences[0].event.complete = True
    value.poll()
    assert not value.inflight and value.inflight_bytes == 0
    assert fences[0].event.queries == 1


def test_h2d_batch_shares_completion_and_elides_only_identical_consumer_waits(monkeypatch):
    value = engine(monkeypatch)
    dependency = Event()
    dependency.record(value.d2h_stream)
    with value.prefetch_batch():
        first, fence = value.to_device(torch.arange(7, dtype=torch.uint8), dependency)
        second, same = value.to_device(torch.arange(11, dtype=torch.uint8), dependency)
    assert fence is same and fence.event.records == 1
    assert torch.equal(first, torch.arange(7, dtype=torch.uint8))
    assert torch.equal(second, torch.arange(11, dtype=torch.uint8))
    consumer, other = Stream(19), Stream(20)
    value.wait_for_restore(consumer, fence, storage_id=1, nbytes=7)
    value.wait_for_restore(consumer, fence, storage_id=2, nbytes=11)
    value.wait_for_restore(other, fence, storage_id=2, nbytes=11)
    assert len(consumer.waits) == len(other.waits) == 1
    assert value.stats['restore_waits_elided'] == 1


def test_later_allocation_failure_publishes_earlier_copies_without_draining(monkeypatch):
    value = engine(monkeypatch)
    dependency = Event()
    dependency.record(value.d2h_stream)
    source = torch.arange(7, dtype=torch.uint8)
    def fail(*args, **kwargs):
        raise torch.OutOfMemoryError('allocation failed')
    with pytest.raises(torch.OutOfMemoryError, match='allocation failed'):
        with value.prefetch_batch():
            destination, fence = value.to_device(source, dependency)
            monkeypatch.setattr(torch, 'empty', fail)
            value.to_device(source, dependency)
    assert torch.equal(destination, source)
    assert fence.event.records == 1 and value.h2d_stream.syncs == 0


def test_batch_body_error_is_preserved_when_event_publication_also_fails(monkeypatch):
    value = engine(monkeypatch)
    class FailedEvent(Event):
        def record(self, stream):
            raise RuntimeError('record failed')
    monkeypatch.setattr(torch.cuda, 'Event', FailedEvent)
    with pytest.raises(ValueError, match='original error'):
        with value._stream_batch(value.d2h_stream, value._d2h_batch):
            fence = value.to_host(torch.ones(7, dtype=torch.uint8), torch.empty(7, dtype=torch.uint8))
            raise ValueError('original error')
    assert not fence.query()
    with pytest.raises(RuntimeError, match='publication failed'):
        fence.cuda_event()


@pytest.mark.parametrize('pending', [0, 7])
def test_stream_backpressure_requires_a_real_pending_source_budget(pending):
    with pytest.raises(ValueError, match='max_pending_bytes'):
        CPUOffloadConfig(backpressure_mode='stream', max_inflight_bytes=8, max_pending_bytes=pending)


def test_stream_pacing_does_not_uncharge_unfinished_gpu_sources(monkeypatch):
    value = engine(monkeypatch)
    value.config = replace(value.config, backpressure_mode='stream', max_inflight_bytes=8, max_pending_bytes=16)
    consumer = Stream(21)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *args: consumer)
    ready = Event()
    ready.record(consumer)
    def submit():
        source, destination = torch.ones(8, dtype=torch.uint8), torch.empty(8, dtype=torch.uint8)
        record = SimpleNamespace(size=8, producer_stream=consumer, producer_ready_event=ready,
                                 offload=lambda: bool(value.to_host(source, destination)))
        value.throttle()
        value.submit_batch([record])
    submit()
    submit()
    assert value.inflight_bytes == 16 and value.ordered_bytes == 8
    assert len(consumer.waits) == 1 and value.stats['d2h_stream_waits'] == 1
    assert not value.stats['d2h_throttles']
    assert not any(fence.event.syncs for _, fence in value.inflight)
    submit()  # One admitted batch may overshoot the physical threshold.
    assert value.inflight_bytes == 24
    first = value.inflight[0][1]
    value.throttle()
    assert first.event.syncs == 1 and value.inflight_bytes == 16
    assert value.stats['d2h_throttles'] == 1
    value.clear()
    assert value.inflight_bytes == value.ordered_bytes == 0 and not value.inflight


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('backpressure', ['host', 'stream'])
@pytest.mark.parametrize('backend,compute_allocations', [('dma', False), ('dma_native', False), ('dma_native', True)])
def test_cuda_public_store_batches_share_fences_and_preserve_values(backpressure, backend, compute_allocations):
    from nnscaler.runtime.activation_memory import _pools
    from nnscaler.runtime.activation_store import ActivationStore
    _pools.clear()
    config = CPUOffloadConfig(max_cpu_bytes=1024**2, host_reserve_bytes=0,
                             min_tensor_bytes=0, reload_on_compute_stream=compute_allocations,
                             d2h_backend=backend, h2d_backend=backend, copy_chunk_bytes=17,
                             max_inflight_bytes=128, max_pending_bytes=4096,
                             backpressure_mode=backpressure)
    store = ActivationStore(config, device='cuda')
    with store.batch() as batch:
        handles = [batch.save(torch.arange(n, dtype=torch.float32, device='cuda')) for n in (31, 47)]
    assert handles[0]._value.record.d2h_event is handles[1]._value.record.d2h_event
    assert store.prefetch_many(handles) == 2
    assert handles[0]._value.record.h2d_event is handles[1]._value.record.h2d_event
    for handle, n in zip(handles, (31, 47)):
        torch.testing.assert_close(handle.load().cpu(), torch.arange(n, dtype=torch.float32))
        handle.release()
    store.close()
    assert store.pool.stats['d2h_completion_events'] == 1
    assert store.pool.stats['h2d_completion_events'] == 1
    if backend == 'dma_native':
        for direction in ('h2d', 'd2h'):
            assert store.pool.stats[direction + '_native_dma_batches'] == 1
            assert store.pool.stats[direction + '_native_dma_descriptors'] == 2
    assert store.pool.snapshot()['live_cpu_bytes'] == 0
    _pools.clear()


def test_native_batch_submission_failure_does_not_publish_partial_data(monkeypatch):
    value = engine(monkeypatch)
    value.config = replace(value.config, d2h_backend='dma_native')
    def fail(destinations, sources, waits, chunk):
        destinations[0].copy_(sources[0])  # A runtime failure after the first copy.
        raise RuntimeError('partial DMA')
    value._native_dma_submit = fail
    with pytest.raises(RuntimeError, match='partial DMA'):
        with value._stream_batch(value.d2h_stream, value._d2h_batch):
            fence = value.to_host(torch.ones(7, dtype=torch.uint8), torch.empty(7, dtype=torch.uint8))
            value.to_host(torch.ones(11, dtype=torch.uint8), torch.empty(11, dtype=torch.uint8))
    assert value.d2h_stream.syncs == 1
    assert fence.event.records == 0 and not fence.query()
    with pytest.raises(RuntimeError, match='partial DMA'):
        fence.cuda_event()


def test_native_batch_body_error_submits_valid_descriptors_and_keeps_original_error(monkeypatch):
    value = engine(monkeypatch)
    value.config = replace(value.config, d2h_backend='dma_native')
    calls = []
    def copy(destinations, sources, waits, chunk):
        calls.append(len(sources))
        for dst, src in zip(destinations, sources):
            dst.copy_(src)
        return len(sources)
    value._native_dma_submit = copy
    src, dst = torch.ones(7, dtype=torch.uint8), torch.empty(7, dtype=torch.uint8)
    with pytest.raises(ValueError, match='later allocation'):
        with value._stream_batch(value.d2h_stream, value._d2h_batch):
            fence = value.to_host(src, dst)
            raise ValueError('later allocation')
    assert calls == [1] and torch.equal(src, dst)
    assert fence.event.records == 1 and not value.d2h_stream.syncs
