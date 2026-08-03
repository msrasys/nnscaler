#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CPU-only unit tests for the channel/sequence/lifecycle-tracked async P2P
bookkeeping added to ``_AsyncCommHandler`` (``issue_recv`` / ``issue_send``,
and the transparently channel-aware ``wait``). See
``CompileFlag.async_recv_channel`` / ``async_recv_max_outstanding`` and
``nnscaler.execplan.planpass.global_schedule`` for how these are driven by
real compiled code.

Style follows ``tests/codegen/test_reschedule.py::test_async_comm_handler_drain``:
a minimal fake ``Work`` (records whether ``.wait()`` was called) and explicit
singleton-state reset before/after each test (the handler is a process-wide
singleton, see ``AsyncCommHandler()``).
"""
import collections

import pytest
import torch

from nnscaler.runtime.executor import AsyncCommHandler, AsyncCommError, _ChannelEntry


class _Work:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True

    def is_completed(self):
        return self.waited


def _reset(handler):
    handler._works.clear()
    handler._callbacks.clear()
    handler._send_holds.clear()
    handler._channel_pending.clear()
    handler._channel_next_seq.clear()
    handler._tensor_channel.clear()


def test_issue_recv_assigns_fifo_sequence_numbers():
    handler = AsyncCommHandler()
    _reset(handler)
    t1, t2 = torch.zeros(2), torch.zeros(2)
    seq1 = handler.issue_recv('chanA', t1, [_Work()], max_outstanding=2)
    seq2 = handler.issue_recv('chanA', t2, [_Work()], max_outstanding=2)
    assert (seq1, seq2) == (0, 1)
    # different channels get their own independent sequence numbering
    t3 = torch.zeros(2)
    seq3 = handler.issue_recv('chanB', t3, [_Work()], max_outstanding=2)
    assert seq3 == 0
    handler.wait(t1)
    handler.wait(t2)
    handler.wait(t3)
    handler.check_clear()
    _reset(handler)


def test_wait_resolves_channel_tracked_tensor():
    handler = AsyncCommHandler()
    _reset(handler)
    t1, w1 = torch.zeros(2), _Work()
    handler.issue_recv('chanA', t1, [w1], max_outstanding=2)
    out = handler.wait(t1)
    assert w1.waited
    assert out is t1
    assert t1 not in handler._tensor_channel
    assert len(handler._channel_pending['chanA']) == 0
    handler.check_clear()
    _reset(handler)


def test_outstanding_cap_violation_raises_clear_error():
    """Illegal-configuration / lifecycle-cap unit test: exceeding
    max_outstanding on a channel raises AsyncCommError, not a silent
    unbounded buffer/handle accumulation."""
    handler = AsyncCommHandler()
    _reset(handler)
    handler.issue_recv('chanB', torch.zeros(2), [_Work()], max_outstanding=1)
    with pytest.raises(AsyncCommError, match='outstanding'):
        handler.issue_recv('chanB', torch.zeros(2), [_Work()], max_outstanding=1)
    _reset(handler)


def test_illegal_max_outstanding_config_raises():
    """max_outstanding must be >= 1: an illegal-configuration unit test."""
    handler = AsyncCommHandler()
    _reset(handler)
    with pytest.raises(AsyncCommError, match='max_outstanding'):
        handler.issue_recv('chanC', torch.zeros(2), [_Work()], max_outstanding=0)
    with pytest.raises(AsyncCommError, match='max_outstanding'):
        handler.issue_send('chanC', torch.zeros(2), _Work(), max_outstanding=-1)
    _reset(handler)


def test_reissue_without_wait_is_rejected():
    """A buffer that was issued must be waited before being reissued --
    catches a lifecycle bug (double-issue) rather than silently overwriting
    bookkeeping."""
    handler = AsyncCommHandler()
    _reset(handler)
    t = torch.zeros(2)
    handler.issue_recv('chanD', t, [_Work()], max_outstanding=5)
    with pytest.raises(AsyncCommError, match='already has an outstanding issue'):
        handler.issue_recv('chanD', t, [_Work()], max_outstanding=5)
    _reset(handler)


def test_wait_sequence_mismatch_raises():
    """Waits on a channel must resolve strictly FIFO; resolving a later
    sequence number while an earlier one is still outstanding is a scheduling
    bug and must raise, not silently resolve the wrong buffer."""
    handler = AsyncCommHandler()
    _reset(handler)
    t1, t2 = torch.zeros(2), torch.zeros(2)
    handler.issue_recv('chanE', t1, [_Work()], max_outstanding=2)
    handler.issue_recv('chanE', t2, [_Work()], max_outstanding=2)
    with pytest.raises(AsyncCommError, match='sequence mismatch'):
        handler._resolve_channel_strict(t2)
    # clean up manually: the strict resolver already popped t2 from
    # _tensor_channel before raising (see its docstring), so only chanE's
    # deque and t1's bookkeeping remain to clear
    handler._channel_pending['chanE'].clear()
    handler._tensor_channel.pop(t1, None)
    handler._works.pop(t1, None)
    handler._callbacks.pop(t1, None)
    handler._works.pop(t2, None)
    handler._callbacks.pop(t2, None)
    _reset(handler)


def test_issue_send_and_drain_sends_releases_channel():
    handler = AsyncCommHandler()
    _reset(handler)
    t, w = torch.zeros(2), _Work()
    handler.issue_send('chanF', t, w, max_outstanding=1)
    assert len(handler._channel_pending['chanF']) == 1
    handler.drain_sends()
    assert w.waited
    assert len(handler._channel_pending['chanF']) == 0
    assert t not in handler._tensor_channel
    handler.check_clear()
    _reset(handler)


def test_drain_sends_completed_releases_channel_for_completed_only():
    handler = AsyncCommHandler()
    _reset(handler)
    t1, w1 = torch.zeros(2), _Work()
    t2, w2 = torch.zeros(2), _Work()
    handler.issue_send('chanG', t1, w1, max_outstanding=2)
    handler.issue_send('chanG', t2, w2, max_outstanding=2)
    w1.waited = True  # simulate w1's transport op having completed already
    handler.drain_sends_completed()
    assert t1 not in handler._tensor_channel
    assert t2 in handler._tensor_channel  # not completed yet, still tracked
    handler.drain_sends()  # cleanup
    _reset(handler)


def test_drain_releases_callback_less_channel_tracked_receive():
    """`drain()` (used before a step returns) must also release channel
    bookkeeping for a callback-less pending receive, matching its docstring
    ("an async-recv whose output is a step output is never explicitly
    waited")."""
    handler = AsyncCommHandler()
    _reset(handler)
    t, w = torch.zeros(2), _Work()
    w.waited = True  # already completed
    handler.issue_recv('chanH', t, [w], max_outstanding=1)
    handler.drain()
    assert t not in handler._tensor_channel
    handler.check_clear()
    _reset(handler)


def test_check_clear_detects_leaked_channel_state():
    handler = AsyncCommHandler()
    _reset(handler)
    handler.check_clear()  # empty: passes

    t, w = torch.zeros(2), _Work()
    handler.issue_recv('chanI', t, [w], max_outstanding=1)
    with pytest.raises(AssertionError, match='not cleared'):
        handler.check_clear()
    handler.wait(t)
    handler.check_clear()
    _reset(handler)

    # a channel-only leak (bookkeeping present with no corresponding
    # tensor-keyed entry) is a distinct, narrower bug this second assertion
    # exists specifically to catch
    handler._channel_pending['chanJ'] = collections.deque([_ChannelEntry(0, t, [w])])
    with pytest.raises(AssertionError, match='channel state not cleared'):
        handler.check_clear()
    _reset(handler)
