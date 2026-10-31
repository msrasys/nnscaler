"""Bounded DMA and optional dependency timing without acquiring a GPU."""
from collections import Counter
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from nnscaler.runtime.activation_config import CPUOffloadConfig
from nnscaler.runtime.activation_observer import _Observation
from nnscaler.runtime.activation_transfer import ActivationTransferEngine


@pytest.mark.parametrize('chunk', [0, 1, 7, 64])
@pytest.mark.parametrize('direction', ['d2h', 'h2d'])
def test_chunked_copy_preserves_offsets_contents_and_storage(chunk, direction):
    observation = _Observation()
    engine = ActivationTransferEngine(CPUOffloadConfig(copy_chunk_bytes=chunk),
                                      torch.device('cpu'), Counter(), observation)
    source = torch.arange(37, dtype=torch.uint8)
    backing = torch.full((49,), 255, dtype=torch.uint8)
    destination = backing[5:42]
    storage = backing.untyped_storage()
    with observation.trace(8) as trace:
        engine._copy_into(destination, source, direction)
    assert destination.untyped_storage() is storage
    assert torch.equal(destination, source)
    assert torch.equal(backing[:5], torch.full((5,),255,dtype=torch.uint8))
    assert torch.equal(backing[42:], torch.full((7,),255,dtype=torch.uint8))
    count = (37+chunk-1)//chunk if chunk else 1
    assert engine.stats[direction+'_copy_chunks'] == count
    assert trace.events[0].nbytes == 37 and trace.events[0].count == count


class Stream:
    cuda_stream = 17
    def __init__(self):
        self.time = 0
        self.waits = 0
        self.syncs = 0
    def wait_event(self, ready):
        self.time = max(self.time, ready.time)
        self.waits += 1
    def synchronize(self):
        self.syncs += 1


class Event:
    def __init__(self, *, enable_timing):
        assert enable_timing
        self.complete = False
    def record(self, stream):
        self.time = stream.time
    def query(self):
        return self.complete
    def elapsed_time(self, other):
        return other.time - self.time


def test_batch_uses_each_record_producer_and_reports_physical_bytes(monkeypatch):
    observation = _Observation()
    engine = ActivationTransferEngine(CPUOffloadConfig(), torch.device('cpu'), Counter(), observation)
    joined = []
    engine.d2h_stream = SimpleNamespace(wait_event=joined.append, wait_stream=joined.append)
    monkeypatch.setattr(engine, '_stream_batch', lambda *args: nullcontext())
    first, later = object(), object()
    source_stream = SimpleNamespace(cuda_stream=1)
    records = [SimpleNamespace(size=4,producer_stream=source_stream,producer_ready_event=event,
                               offload=lambda: True) for event in (first,later)]
    with observation.trace(4) as trace:
        engine.submit_batch(records, {1:later})
    assert joined == [first,later]
    assert engine.stats['peak_d2h_batch_bytes'] == 8
    assert trace.events[0].action == 'd2h_batch_submit' and trace.events[0].nbytes == 8


def test_wait_timing_is_opt_in_bounded_and_never_synchronizes(monkeypatch):
    created = []
    def event(**kwargs):
        value = Event(**kwargs)
        created.append(value)
        return value
    monkeypatch.setattr(torch.cuda, 'Event', event)
    observation, stream = _Observation(), Stream()
    with observation.trace(4):
        observation.wait_event(stream, SimpleNamespace(time=1), storage_id=0, nbytes=8)
    assert not created and stream.syncs == 0
    with observation.trace(1, cuda_timing=True) as trace:
        observation.wait_event(stream, SimpleNamespace(time=6), storage_id=1, nbytes=8)
        observation.wait_event(stream, SimpleNamespace(time=9), storage_id=2, nbytes=8)
        value = trace.to_dict()
        assert value['cuda_waits'][0]['elapsed_ms'] is None
        assert value['dropped_cuda_waits'] == 1 and len(created) == 2
    for event in created:
        event.complete = True
    value = trace.to_dict()
    assert value['cuda_waits'][0]['elapsed_ms'] == 5
    assert value['cuda_waits'][0]['stream_id'] == 17
    assert value['cuda_waits'][0]['storage_id'] == 1
    observation.wait_event(stream, SimpleNamespace(time=10), storage_id=3, nbytes=8)
    assert len(created) == 2 and stream.syncs == 0 and stream.waits == 4
    assert not observation._timing_traces


def test_nested_traces_share_timing_events_and_release_subscriptions(monkeypatch):
    import weakref
    created = []
    def event(**kwargs):
        value = Event(**kwargs)
        created.append(value)
        return value
    monkeypatch.setattr(torch.cuda, 'Event', event)
    observation, stream = _Observation(), Stream()
    with observation.trace(2, cuda_timing=True) as outer:
        with observation.trace(1, cuda_timing=True) as inner:
            for index in range(2):
                observation.wait_event(stream, SimpleNamespace(time=index+1), storage_id=index, nbytes=8)
    assert len(created) == 4
    assert len(outer.to_dict()['cuda_waits']) == 2
    assert inner.to_dict()['dropped_cuda_waits'] == 1
    observed = weakref.ref(inner)
    del inner
    assert observed() is None


@pytest.mark.parametrize('direction', ['d2h','h2d'])
def test_copy_submission_failure_drains_queued_work_before_buffer_reuse(monkeypatch, direction):
    observation = _Observation()
    engine = ActivationTransferEngine(CPUOffloadConfig(copy_chunk_bytes=4),
                                      torch.device('cpu'), Counter(), observation)
    stream = Stream()
    engine.d2h_stream = engine.h2d_stream = stream
    source, destination = torch.ones(8,dtype=torch.uint8), torch.zeros(8,dtype=torch.uint8)
    monkeypatch.setattr(torch.Tensor, 'record_stream', lambda *args: None)
    def fail_copy(dst, src, direction):
        dst[:4].copy_(src[:4])
        raise RuntimeError('second chunk failed')
    monkeypatch.setattr(engine, '_copy_into', fail_copy)
    with pytest.raises(RuntimeError, match='second chunk failed'):
        if direction == 'd2h':
            token = engine._d2h_batch.set(True)
            try:
                engine._to_host(source, destination)
            finally:
                engine._d2h_batch.reset(token)
        else:
            engine._restore_on_stream(source, SimpleNamespace(time=0), destination)
    assert torch.equal(destination[:4], source[:4]) and stream.syncs == 1
    assert not engine.inflight and engine.inflight_bytes == 0
