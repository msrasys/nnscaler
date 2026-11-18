"""Stream-ordered reclamation of asynchronous input owners."""
import weakref

import pytest


@pytest.fixture
def runtime(monkeypatch):
    from nnscaler.runtime import stream as module
    assert not module._pending_holds
    monkeypatch.setattr(module, '_pending_holds', {})
    events = []

    class Event:
        ready = False
        fail = False
        on_query = None
        queries = 0
        waits = 0

        def __init__(self):
            events.append(self)

        def record(self, stream):
            self.stream = stream

        def query(self):
            self.queries += 1
            if self.on_query:
                self.on_query()
            if self.fail:
                raise RuntimeError('event query failed')
            return self.ready

        def synchronize(self):
            self.waits += 1
            self.ready = True

    monkeypatch.setattr(module.torch.cuda, 'Event', Event)
    return module, events


class Owner:
    pass


def enqueue(module, stream):
    owner = Owner()
    reference = weakref.ref(owner)
    module._hold_stream_owners((owner,), stream)
    return reference


def test_pending_stream_does_not_poll_all_later_events(runtime):
    module, events = runtime
    references = [enqueue(module, 'busy') for _ in range(1000)]
    free = enqueue(module, 'completed')
    events[-1].ready = True
    module.drain_stream_holds()
    assert sum(event.queries for event in events) == 2
    assert free() is None
    assert all(ref() is not None for ref in references)
    for event in events:
        event.ready = True
    module.drain_stream_holds()
    assert all(ref() is None for ref in references)


def test_failed_query_keeps_owners_for_retry(runtime):
    module, events = runtime
    references = [enqueue(module, stream) for stream in ('first', 'second')]
    events[0].fail = True
    with pytest.raises(RuntimeError, match='event query failed'):
        module.drain_stream_holds()
    assert all(ref() is not None for ref in references)
    events[0].fail = False
    module.drain_stream_holds(wait=True)
    assert all(ref() is None for ref in references)
    assert all(event.waits == 1 for event in events)


def test_concurrent_submission_survives_drain(runtime):
    module, events = runtime
    old = enqueue(module, 'stream')
    added = []
    events[0].on_query = lambda: added.append(enqueue(module, 'stream'))
    module.drain_stream_holds()
    events[0].on_query = None
    assert old() is not None and added[0]() is not None
    events[0].ready = True
    module.drain_stream_holds()
    assert old() is None and added[0]() is not None
    module.drain_stream_holds(wait=True)
    assert added[0]() is None


def test_reclaim_finished_prefix_before_pending_event(runtime):
    module, events = runtime
    references = [enqueue(module, 'stream') for _ in range(3)]
    events[0].ready = True
    module.drain_stream_holds()
    assert references[0]() is None
    assert all(ref() is not None for ref in references[1:])
    assert [event.queries for event in events] == [1, 1, 0]
    module.drain_stream_holds(wait=True)
    assert all(ref() is None for ref in references)
