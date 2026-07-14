#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from contextlib import contextmanager

import pytest

import chronotrigger.trace as ct
from nnscaler.runtime.executor import AsyncCommHandler


class FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


@pytest.fixture(autouse=True)
def clear_async_handler():
    AsyncCommHandler().clear()
    yield
    AsyncCommHandler().clear()


@pytest.fixture
def trace_observer(monkeypatch):
    trace_context = ct.TraceContext(
        entity="adapter7:move:0->1:t9",
        peer=1,
        step=12,
        payload_fields={"mb": 5},
    )
    waits = []
    monkeypatch.setattr(ct, "current_context", lambda: trace_context)

    @contextmanager
    def observe_range_from(context, *, kind, **payload):
        waits.append((context, kind, payload))
        yield context

    monkeypatch.setattr(ct, "range_from", observe_range_from)
    return trace_context, waits


def test_async_submit_reuses_exact_trace_context_at_wait(trace_observer):
    trace_context, waits = trace_observer
    tensor = object()
    work = FakeWork()
    handler = AsyncCommHandler()

    handler.submit(tensor, [work])

    assert handler._metas[id(tensor)] is trace_context
    assert handler.wait(tensor) is tensor
    assert work.waited
    assert waits == [(trace_context, ct.Kind.WAIT, {})]


def test_held_send_reuses_exact_trace_context_at_drain(trace_observer):
    trace_context, waits = trace_observer
    work = FakeWork()
    handler = AsyncCommHandler()

    handler.hold_send(object(), work)
    handler.drain_sends()

    assert work.waited
    assert waits == [(trace_context, ct.Kind.WAIT, {})]