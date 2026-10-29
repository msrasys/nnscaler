#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Which offloaded tensors will be needed next, and when to reload them.

The order comes from execution contexts, not from guessing:

* ``Executor.fexecute(name)`` opens an :class:`ExecContext`; the offload groups
  executed inside are recorded in forward order and the finished context is
  queued per segment name (FIFO).
* ``Executor.backward(name)`` pops the oldest pending context of ``name``.
  Within a context groups are consumed in reverse forward order, so the groups
  preceding the demanded one are prefetched.

Both facts hold for 1F1B, interleaved 1F1B and gradient accumulation, so no
warm-up iteration is needed.  Each batch owns one :class:`~.transport.Transfer`
that is committed when its group ends and prefetched as a unit.  The scheduler code generator refines this with
two hints: ``hint_forward(name, distance)`` (how far away the own backward is)
and ``prefetch_next_backward(name)`` (which backward comes next).  Without an
executor (eager use) batches fall back to a creation-order chain.
"""

from __future__ import annotations

import collections
import threading
import weakref
from typing import Deque, Dict, List, Optional, Tuple

import torch

from .config import OffloadConfig
from .hooks import OffloadedTensor
from .stats import GroupProfile, OffloadStats
from .transport import CopyEngine, Transfer


class OffloadBatch:
    """Saved tensors of one execution of one offload group (owns one :class:`Transfer`)."""

    last_batch: Optional[weakref.ReferenceType['OffloadBatch']] = None

    def __init__(self, tracker: 'ExecutionTracker', group_id: Optional[int],
                 context: Optional['ExecContext'], device: torch.device) -> None:
        self.tracker = tracker
        self.group_id = group_id
        self.context = context
        self.index = -1
        self.transfer: Transfer = tracker.engine.new_transfer(group_id, device)
        self.transfer.owner = self
        self.prefetched = False
        self.released = False
        self.previous: Optional[weakref.ReferenceType['OffloadBatch']] = (
            OffloadBatch.last_batch if context is None else None)

    @property
    def nbytes(self) -> int:
        return self.transfer.nbytes

    @property
    def handles(self) -> List[OffloadedTensor]:
        return self.transfer.handles

    def append(self, tensor: torch.Tensor) -> OffloadedTensor:
        self.tracker.stats.packed_tensors += 1
        return self.transfer.add(tensor)

    def done(self) -> None:
        """The group ended: commit the transfer (regions laid out, D2H scheduled)."""
        self.transfer.commit()
        self.tracker.engine.pump()
        profile = self.tracker.groups[self.group_id if self.group_id is not None else -1]
        profile.executions += 1
        profile.tensors += len(self.transfer.handles)
        if self.transfer.regions and self.context is None:
            OffloadBatch.last_batch = weakref.ref(self)

    def prefetch(self) -> None:
        if self.prefetched or self.released:
            return
        self.prefetched = True
        self.transfer.prefetch()

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.transfer.release()


class ExecContext:
    """Offload batches produced by one forward execution of a segment."""

    def __init__(self, name: str, seq: int) -> None:
        self.name = name
        self.seq = seq
        self.batches: List[OffloadBatch] = []
        self.offload_enabled = True

    def add(self, batch: OffloadBatch) -> None:
        batch.index = len(self.batches)
        self.batches.append(batch)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self.batches)

    def release(self) -> None:
        for batch in self.batches:
            batch.release()


class ExecutionTracker:
    """Forward/backward bookkeeping and the prefetch policy.

    The per-segment queue mirrors ``Executor._detach[name]`` exactly: every
    forward with autograd enabled is enqueued (also when it offloaded nothing)
    and every backward pops one, so the two can never drift apart.  A drift
    would prefetch and release the wrong micro-batch, so it is an error.
    """

    def __init__(self, config: OffloadConfig, engine: CopyEngine, stats: OffloadStats,
                 groups: Dict[int, GroupProfile]) -> None:
        self.config = config
        self.engine = engine
        self.stats = stats
        self.groups = groups
        self._lock = threading.Lock()
        self._forward_stack: List[ExecContext] = []
        self._forward_seq: Dict[str, int] = collections.defaultdict(int)
        self._backward_seq: Dict[str, int] = collections.defaultdict(int)
        self._pending: Dict[str, Deque[ExecContext]] = collections.defaultdict(collections.deque)
        self._backward_stack: List[ExecContext] = []
        self._next_forward_distance: Dict[str, int] = {}
        self._announced_backward: Optional[str] = None

    # -- hints from generated code ------------------------------------------
    def hint_forward(self, name: str, backward_distance: int) -> None:
        """The next forward of ``name`` has its backward this many segment executions later."""
        with self._lock:
            self._next_forward_distance[name] = backward_distance

    def forward_skipped(self, name: str) -> None:
        """A forward of ``name`` ran without autograd: it has no backward and consumes no hint."""
        with self._lock:
            self._next_forward_distance.pop(name, None)

    def prefetch_next_backward(self, name: str) -> None:
        """The next backward to start is segment ``name``.

        Called inside a running backward, the head of ``name``'s queue is that
        backward's context and its last groups are reloaded now.  Called before
        the current backward started (the scheduler emits the hint just before
        the backward call), the announcement is acted upon in
        :meth:`begin_backward` once the current context has been popped.
        """
        self.engine.pump()
        with self._lock:
            if not self._backward_stack:
                self._announced_backward = name
                return
            queue = self._pending.get(name)
            ctx = queue[0] if queue else None
        if ctx is not None and ctx.batches:
            self._prefetch_from(ctx, len(ctx.batches) - 1, include_self=True)

    # -- executor callbacks ---------------------------------------------------
    def begin_forward(self, name: str) -> ExecContext:
        with self._lock:
            ctx = ExecContext(name, self._forward_seq[name])
            self._forward_seq[name] += 1
            distance = self._next_forward_distance.pop(name, None)
            if distance is not None and distance < self.config.min_backward_distance:
                ctx.offload_enabled = False
            self._forward_stack.append(ctx)
        return ctx

    def end_forward(self, ctx: ExecContext) -> None:
        self.engine.pump()
        with self._lock:
            assert self._forward_stack and self._forward_stack[-1] is ctx, 'unbalanced begin/end_forward'
            self._forward_stack.pop()
            self._pending[ctx.name].append(ctx)

    def begin_backward(self, name: str) -> Optional[ExecContext]:
        with self._lock:
            queue = self._pending.get(name)
            if not queue:
                raise RuntimeError(
                    f'CPU offloading: backward of segment {name!r} without a pending forward; '
                    'Executor.fexecute/backward calls are unbalanced')
            ctx = queue.popleft()
            expected = self._backward_seq[name]
            self._backward_seq[name] += 1
            if ctx.seq != expected:
                raise RuntimeError(
                    f'CPU offloading: segment {name!r} backward #{expected} got forward #{ctx.seq}; '
                    'the tracker is out of sync with the executor')
            self._backward_stack.append(ctx)
            announced = self._announced_backward
            self._announced_backward = None
            next_queue = self._pending.get(announced) if announced is not None else None
            next_ctx = next_queue[0] if next_queue else None
        if ctx.batches:
            self._prefetch_from(ctx, len(ctx.batches) - 1, include_self=True)
        if next_ctx is not None and next_ctx.batches:
            self._prefetch_from(next_ctx, len(next_ctx.batches) - 1, include_self=True)
        return ctx

    def end_backward(self, ctx: Optional[ExecContext]) -> None:
        self.engine.pump()
        if self.engine.timed:
            # fold finished copy/stall events so nothing accumulates when the trainer never polls
            self.engine.timing.collect(self.stats, self.groups)
        if ctx is None:
            return
        with self._lock:
            if self._backward_stack and self._backward_stack[-1] is ctx:
                self._backward_stack.pop()
            else:
                self._backward_stack = [c for c in self._backward_stack if c is not ctx]
        ctx.release()

    # -- hooks ----------------------------------------------------------------
    def new_batch(self, group_id: Optional[int], device: torch.device) -> Optional[OffloadBatch]:
        with self._lock:
            ctx = self._forward_stack[-1] if self._forward_stack else None
        if ctx is not None and not ctx.offload_enabled:
            return None
        batch = OffloadBatch(self, group_id, ctx, device)
        if ctx is not None:
            ctx.add(batch)
        return batch

    def on_unpack(self, handle: OffloadedTensor) -> None:
        batch = handle.transfer.owner
        if batch is None:
            return
        self.engine.pump()
        ctx = batch.context
        if ctx is not None:
            self._prefetch_from(ctx, batch.index, include_self=True)
        else:
            batch.prefetch()
            previous = batch.previous() if batch.previous is not None else None
            count = self.config.prefetch_groups
            while previous is not None and count > 0:
                previous.prefetch()
                count -= 1
                previous = previous.previous() if previous.previous is not None else None

    # -- prefetch policy -----------------------------------------------------
    def _prefetch_from(self, ctx: ExecContext, index: int, *, include_self: bool) -> None:
        """Reload ``ctx.batches[index]`` (if ``include_self``) and the groups before it.

        The depth is ``prefetch_groups`` groups beyond ``index``, or when
        ``prefetch_bytes`` is set, as many groups as fit into that many bytes
        (at least one).  Groups that saved nothing cost no budget.
        """
        budget_groups = self.config.prefetch_groups
        budget_bytes = self.config.prefetch_bytes
        taken_groups = 0
        taken_bytes = 0
        for i in range(index, -1, -1):
            batch = ctx.batches[i]
            if i == index:
                if include_self:
                    batch.prefetch()
                continue
            if not batch.transfer.regions:
                continue
            if budget_bytes:
                if taken_groups and taken_bytes >= budget_bytes:
                    break
            elif taken_groups >= budget_groups:
                break
            batch.prefetch()
            taken_groups += 1
            taken_bytes += batch.nbytes

    # -- state ----------------------------------------------------------------
    def pending_summary(self) -> Tuple[int, int]:
        with self._lock:
            contexts = sum(len(q) for q in self._pending.values())
            nbytes = sum(c.nbytes for q in self._pending.values() for c in q)
        return contexts, nbytes
