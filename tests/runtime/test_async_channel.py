#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CPU-only unit tests for the channel/sequence/lifecycle-tracked async P2P
receive bookkeeping added to ``_AsyncCommHandler`` (``issue_recv``, and the
transparently channel-aware ``wait``), plus its exception-safety cleanup
(``force_clear_after_exception``). See ``CompileFlag.async_recv_channel`` /
``async_recv_max_outstanding`` and ``nnscaler.execplan.planpass.global_schedule``
for how these are driven by real compiled code. There is no channel-tracked
``issue_send``: an earlier attempt at one turned out to be structurally
unreachable from codegen (the only path that would emit one,
``CompileFlag.async_comm``, is unconditionally rejected by
``GlobalCommSchedule``), so it was removed rather than kept as untested,
misleadingly-named dead code.

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
        handler.issue_recv('chanC', torch.zeros(2), [_Work()], max_outstanding=-1)
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
    # atomic on failure: `_resolve_channel_strict` must not have mutated
    # ANY state before raising -- both t1 and t2 remain exactly as they
    # were (still outstanding, still FIFO-ordered), unlike the pre-fix
    # behavior which popped t2 from `_tensor_channel` before validating.
    assert t1 in handler._tensor_channel and t2 in handler._tensor_channel
    assert len(handler._channel_pending['chanE']) == 2
    # resolving in the correct (FIFO) order now succeeds cleanly
    handler.wait(t1)
    handler.wait(t2)
    handler.check_clear()
    _reset(handler)


def test_drain_sends_and_drain_sends_completed_wait_held_sends():
    """`drain_sends()` (unconditional) and `drain_sends_completed()`
    (only already-finished ones) must still work correctly for plain
    (non-channel-tracked) held sends -- regression guard: these two methods'
    channel-release calls were removed when channel-tracked sends
    (`issue_send`) were removed as structurally unreachable dead code (see
    module docstring), so this specifically re-covers their core, still-live
    plain-send behavior."""
    handler = AsyncCommHandler()
    _reset(handler)
    t1, w1 = torch.zeros(2), _Work()
    t2, w2 = torch.zeros(2), _Work()
    handler.hold_send(t1, w1)
    handler.hold_send(t2, w2)
    w1.waited = True  # simulate w1's transport op having completed already
    handler.drain_sends_completed()
    assert w1.waited and len(handler._send_holds) == 1
    handler.drain_sends()
    assert w2.waited and len(handler._send_holds) == 0
    handler.check_clear()
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


def test_force_clear_after_exception_clears_all_state_without_waiting():
    """`force_clear_after_exception` must wipe ALL bookkeeping (tensor-keyed
    AND channel-tracked) so a subsequent, unrelated step does not inherit
    stale state -- and must NOT call `.wait()` on any outstanding work (the
    communicator may be broken after the crash that triggered this)."""
    handler = AsyncCommHandler()
    _reset(handler)

    t_plain, w_plain = torch.zeros(2), _Work()
    handler.submit(t_plain, [w_plain])
    t_send, w_send = torch.zeros(2), _Work()
    handler.hold_send(t_send, w_send)
    t_chan, w_chan = torch.zeros(2), _Work()
    handler.issue_recv('chanK', t_chan, [w_chan], max_outstanding=1)

    handler.force_clear_after_exception()

    assert not w_plain.waited and not w_send.waited and not w_chan.waited, \
        'force_clear_after_exception must not wait on any outstanding work'
    handler.check_clear()  # fully clean, no assertion error
    _reset(handler)


def test_force_clear_after_exception_never_raises():
    """Even if internal state is bizarrely inconsistent, this method must
    never itself raise (it is meant to be called from an except-block that
    is about to re-raise a real exception; it must not mask it)."""
    handler = AsyncCommHandler()
    _reset(handler)
    handler._works = None  # deliberately corrupt internal state
    try:
        handler.force_clear_after_exception()  # must not raise
    finally:
        handler._works = {}
    _reset(handler)


def test_run_step_with_exception_safety_reraises_original_and_clears_state():
    """`RuntimeModule._run_step_with_exception_safety` must re-raise the
    ORIGINAL exception unmodified, after force-clearing any AsyncCommHandler
    state the aborted step left outstanding -- so a subsequent, healthy step
    reusing the same channel does not spuriously fail with a misattributed
    cap/FIFO error (see the docstring / module.py wiring)."""
    from nnscaler.flags import CompileFlag
    from nnscaler.runtime.module import ParallelModule

    handler = AsyncCommHandler()
    _reset(handler)
    saved_async_recv = CompileFlag.async_recv
    CompileFlag.async_recv = True
    try:
        t, w = torch.zeros(2), _Work()
        def _crashing_step():
            handler.issue_recv('chanCrash', t, [w], max_outstanding=1)
            raise ValueError('simulated mid-step failure')

        # bind the method to a bare instance without running __init__
        # (only the plain, non-generated-code method under test is needed)
        module = ParallelModule.__new__(ParallelModule)
        with pytest.raises(ValueError, match='simulated mid-step failure'):
            module._run_step_with_exception_safety(_crashing_step)

        # the ORIGINAL exception propagated (not e.g. an AsyncCommError from
        # leftover state), and the handler is now fully clean
        handler.check_clear()

        # a subsequent, unrelated issue on the SAME channel must succeed
        # (not spuriously hit a stale outstanding-cap/FIFO error)
        t2, w2 = torch.zeros(2), _Work()
        handler.issue_recv('chanCrash', t2, [w2], max_outstanding=1)
        handler.wait(t2)
        handler.check_clear()
    finally:
        CompileFlag.async_recv = saved_async_recv
        _reset(handler)
