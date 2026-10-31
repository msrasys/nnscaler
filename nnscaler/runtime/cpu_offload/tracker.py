#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Which offloaded tensors belong to which execution, and when their backward comes.

The order comes from execution contexts, not from guessing:

* ``Executor.fexecute(name)`` opens an :class:`ExecContext`; the offload groups
  executed inside are recorded in forward order and the finished context is
  queued per segment name (FIFO).
* ``Executor.backward(name)`` pops the oldest pending context of ``name``.
  Within a context groups are consumed in reverse forward order.

Both facts hold for 1F1B, interleaved 1F1B and gradient accumulation, so no
warm-up iteration is needed.  The scheduler code generator adds two hints from
the static schedule: ``hint_forward(name, distance)`` (how far away the own
backward is) and ``prefetch_next_backward(*names)`` (the backwards that follow
the one about to run).  The tracker keeps these facts and the sequence
invariants; what to reload ahead of demand is the :mod:`.prefetch` policy's
decision.  Without an executor (eager use) batches fall back to a
creation-order chain.

This module is also the single writer of the per-group profile counts
(:class:`~.stats.GroupProfile` executions, tensors, regions, bytes) and of the
selection counters (what was saved, offloaded, shared, skipped); the transport
reports how bytes moved.
"""

from __future__ import annotations

import collections
import threading
import weakref
from typing import Deque, Dict, List, Optional, Tuple

import torch

from .config import OffloadConfig
from .hooks import OffloadedTensor, covered_bytes
from .prefetch import PrefetchPolicy, prefetch_policy_for
from .stats import GroupProfile, OffloadStats
from .transport import CopyEngine, Transfer


class OffloadBatch:
    """Saved tensors of one execution of one offload group.

    Owns one :class:`~.transport.Transfer`, or several when ``commit_bytes``
    flushes the group early.  With ``dedup_scope=context`` a view already
    shipped by the previous group of the same segment execution is served from
    that group's transfer instead of being copied again.
    """

    last_batch: Optional[weakref.ReferenceType['OffloadBatch']] = None

    def __init__(self, tracker: 'ExecutionTracker', group_id: Optional[int],
                 context: Optional['ExecContext'], device: torch.device) -> None:
        self.tracker = tracker
        self.group_id = group_id
        self.context = context
        self.device = device
        self.index = -1
        self.tensors = 0             # saved views handed to this batch (offloaded or shared)
        self.prefetched = False
        self.released = False
        self.transfers: List[Transfer] = []
        self._open()
        self.previous: Optional[weakref.ReferenceType['OffloadBatch']] = (
            OffloadBatch.last_batch if context is None else None)

    def _open(self) -> Transfer:
        transfer = self.tracker.engine.new_transfer(self.group_id, self.device)
        transfer.owner = self
        # sources stay until the next group committed so its saved views can be matched exactly
        transfer.hold_src = self.tracker.config.dedup_scope == 'context' and self.context is not None
        self.transfers.append(transfer)
        return transfer

    @property
    def transfer(self) -> Transfer:
        return self.transfers[-1]

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self.transfers)

    @property
    def handles(self) -> List[OffloadedTensor]:
        return [h for t in self.transfers for h in t.handles]

    @property
    def has_regions(self) -> bool:
        return any(t.regions for t in self.transfers)

    # -- collection -----------------------------------------------------------------
    def append(self, tensor: torch.Tensor) -> OffloadedTensor:
        self.tracker.stats.packed_tensors += 1
        self.tensors += 1
        shared = self._shared(tensor)
        if shared is not None:
            return shared
        current = self.transfer
        commit_bytes = self.tracker.config.commit_bytes
        if commit_bytes and current.handles and current.pending_bytes >= commit_bytes:
            self._commit(current)                # flush: the copy overlaps the rest of the group
            self.tracker.stats.flushes += 1
            current = self._open()
        return current.add(tensor)

    def _shared(self, tensor: torch.Tensor) -> Optional[OffloadedTensor]:
        """A handle served by the previous group's transfer, when it already holds these bytes."""
        if self.tracker.config.dedup_scope != 'context' or self.context is None or self.index <= 0:
            return None
        lo, hi = covered_bytes(tensor)
        if hi == lo:
            return None
        ptr = tensor.untyped_storage().data_ptr()
        for transfer in self.context.batches[self.index - 1].transfers:
            region = transfer.find_region(ptr, lo, hi)
            if region is not None:
                handle = OffloadedTensor(transfer, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), lo, hi)
                transfer.share(handle, region)
                stats = self.tracker.stats
                stats.shared_bytes += hi - lo
                stats.deduplicated_bytes += hi - lo
                return handle
        return None

    def _commit(self, transfer: Transfer) -> None:
        """Commit one transfer and account for what it selected."""
        transfer.commit()
        stats = self.tracker.stats
        stats.deduplicated_bytes += max(transfer.saved_bytes - transfer.nbytes, 0)
        if transfer.regions:
            stats.offloaded_bytes += transfer.nbytes
            stats.offloaded_regions += len(transfer.regions)
            stats.transfers += 1

    def done(self) -> None:
        """The group ended: commit, account the group's profile, release the previous group's hold."""
        self._commit(self.transfer)
        if len(self.transfers) > 1 and not self.transfer.handles:
            self.transfers.pop()
        if self.context is not None and self.index > 0:
            self.context.batches[self.index - 1].release_hold()
        self.tracker.engine.pump()
        profile = self.tracker.groups[self.group_id if self.group_id is not None else -1]
        profile.executions += 1
        profile.tensors += self.tensors
        profile.regions += sum(len(t.regions) for t in self.transfers)
        profile.bytes += self.nbytes
        if self.has_regions and self.context is None:
            OffloadBatch.last_batch = weakref.ref(self)

    def release_hold(self) -> None:
        for transfer in self.transfers:
            transfer.release_hold()

    # -- reload / release ----------------------------------------------------------
    def prefetch(self) -> None:
        if self.prefetched or self.released:
            return
        self.prefetched = True
        for transfer in reversed(self.transfers):     # last saved = first needed
            transfer.prefetch()

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        for transfer in self.transfers:
            transfer.release()


class ExecContext:
    """Offload batches produced by one forward execution of a segment."""

    def __init__(self, name: str, seq: int) -> None:
        self.name = name
        self.seq = seq
        self.batches: List[OffloadBatch] = []
        self.offload_enabled = True
        self.prefetched = False

    def add(self, batch: OffloadBatch) -> None:
        batch.index = len(self.batches)
        self.batches.append(batch)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self.batches)

    def prefetch(self) -> None:
        """Reload the whole execution, the groups needed first going first."""
        self.prefetched = True
        for batch in reversed(self.batches):
            batch.prefetch()

    def release(self) -> None:
        for batch in self.batches:
            batch.release()


class ExecutionTracker:
    """Forward/backward bookkeeping; reload decisions are delegated to the policy.

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
        self.policy: PrefetchPolicy = prefetch_policy_for(config, engine)
        self._lock = threading.Lock()
        self._forward_stack: List[ExecContext] = []
        self._forward_seq: Dict[str, int] = collections.defaultdict(int)
        self._backward_seq: Dict[str, int] = collections.defaultdict(int)
        self._pending: Dict[str, Deque[ExecContext]] = collections.defaultdict(collections.deque)
        self._backward_stack: List[ExecContext] = []
        self._next_forward_distance: Dict[str, int] = {}
        self._announced: List[str] = []       # segment names of the backwards that follow
        self._announced_done = True

    # -- hints from generated code ------------------------------------------
    def hint_forward(self, name: str, backward_distance: int) -> None:
        """The next forward of ``name`` has its backward this many segment executions later."""
        with self._lock:
            self._next_forward_distance[name] = backward_distance

    def forward_skipped(self, name: str) -> None:
        """A forward of ``name`` ran without autograd: it has no backward and consumes no hint."""
        with self._lock:
            self._next_forward_distance.pop(name, None)

    def prefetch_next_backward(self, *names: str) -> None:
        """The backwards that follow the one about to run, in order.

        Called inside a running backward, the heads of the named queues are
        those backwards and the policy may reload them now.  Called before the
        current backward started (the scheduler emits the hint just before the
        backward call), the announcement is acted upon in :meth:`begin_backward`
        once the current context has been popped.
        """
        self.engine.pump()
        with self._lock:
            self._announced = list(names)
            self._announced_done = not names
            inside = bool(self._backward_stack)
        if inside:
            self._reload_announced()

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
        if ctx.batches:
            ctx.batches[-1].release_hold()        # no next group will share them
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
        if ctx.batches:
            self.policy.reload_demanded(ctx)
        self._reload_announced()
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
        if batch.context is None:
            self.policy.on_eager_unpack(batch)
            return
        self.policy.on_unpack(batch)
        if not self._announced_done:
            self._reload_announced()           # the budget may have room now

    # -- announced backwards ----------------------------------------------------
    def _reload_announced(self) -> None:
        """Resolve the announced names to pending contexts and let the policy reload ahead."""
        with self._lock:
            names = list(self._announced)
            if not names:
                self._announced_done = True
                return
            # the i-th announced occurrence of a name is the i-th pending context of that name
            offsets: Dict[str, int] = collections.defaultdict(int)
            contexts: List[ExecContext] = []
            for name in names:
                queue = self._pending.get(name)
                position = offsets[name]
                offsets[name] += 1
                if queue is not None and position < len(queue):
                    contexts.append(queue[position])
        done = self.policy.reload_ahead(contexts)
        with self._lock:
            self._announced_done = done

    # -- state ----------------------------------------------------------------
    def pending_summary(self) -> Tuple[int, int]:
        with self._lock:
            contexts = sum(len(q) for q in self._pending.values())
            nbytes = sum(c.nbytes for q in self._pending.values() for c in q)
        return contexts, nbytes
