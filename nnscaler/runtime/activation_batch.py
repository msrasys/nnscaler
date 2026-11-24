# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Forward submission and backward restore queues, independent of graph layout.

Forward queues own pending sources until submission. Restore queues hold only
weak references to saved values. Neither queue owns a scope or autograd group.
"""
from collections import deque, OrderedDict

import torch


def record_batches(records, target_bytes):
    """Partition in producer order; an indivisible oversized record stands alone."""
    batch, size = [], 0
    for record in records:
        if target_bytes and batch and size + record.size > target_bytes:
            yield batch
            batch, size = [], 0
        batch.append(record)
        size += record.size
        if target_bytes and size >= target_bytes:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


class ActivationBatch:
    """One forward region; weak saved-value metadata and owned pending sources."""
    def __init__(self, store):
        self.store = store
        self.records = []
        self.pending = []
        self.producer_events = None
        self.closed = False
        self.capture_save_events = store.config.use_save_events

    def capture_producer_event(self, stream):
        if self.producer_events is None:
            self.producer_events = {}
        event = self.producer_events.get(stream.cuda_stream)
        if event is None:
            event = torch.cuda.Event()
            self.producer_events[stream.cuda_stream] = event
        # Flush drops this event map. Later saves must not re-record an event
        # that was already used as a dependency by an earlier submission.
        event.record(stream)
        self.store.pool.stats['saved_tensor_event_records'] += 1
        return event

    def discard_dead(self):
        """Drop inputs a checkpoint no longer retains before charging a queue."""
        pending = []
        for record in self.pending:
            if record.live_saves:
                pending.append(record)
            else:
                self.store.pool.stats['discarded_save_bytes'] += record.size
                self.store.pool.stats['discarded_save_storages'] += 1
        self.pending[:] = pending

    def flush(self):
        """Submit live pending values while keeping the region open for saves."""
        if self.closed:
            return
        self.discard_dead()
        try:
            self.store.offload_batch(self.pending, self.producer_events)
        finally:
            self.producer_events = None
            self.pending.clear()

    def commit(self):
        if not self.closed:
            try:
                self.flush()
            finally:
                self.closed = True

    def abort(self):
        self.pending.clear()
        self.producer_events = None
        self.closed = True

    def has_offloaded_records(self):
        return any((record := ref()) is not None and record.d2h_event is not None
                   for ref in self.records)

    def has_pending_loads(self):
        return any((record := ref()) is not None and record.d2h_event is not None
                   and record.device_tensor is None for ref in self.records)

    def prefetch(self, *, from_schedule=False):
        return self.store._prefetch_records(
            (ref() for ref in reversed(self.records)), from_schedule=from_schedule)

    def save(self, tensor):
        if self.closed or self.store._active_batch.get() is not self:
            raise RuntimeError('Activation batch is not active')
        return self.store.save(tensor)


class ForwardCommitQueue:
    """Collect closed regions according to an independent submission policy."""
    def __init__(self, store, *, unit, target_bytes, deferred=None):
        if unit not in ('group', 'scope'):
            raise ValueError('Forward submission unit must be resolved to group or scope')
        self.unit, self.target_bytes = unit, target_bytes
        self.capture_save_events = store.config.use_save_events
        self.batch = ActivationBatch(store) if unit == 'scope' else None
        self.queued_bytes = 0
        if deferred is not None and unit != 'scope':
            raise ValueError('Deferred D2H needs scope collection')
        self.deferred = deferred

    def add(self, group):
        if group.closed:
            return
        if self.batch is None:
            ActivationBatch.commit(group)
            return
        if self.deferred is not None:
            group.discard_dead()
        self.batch.pending.extend(group.pending)
        self.queued_bytes += sum(record.size for record in group.pending)
        stats = self.batch.store.pool.stats
        stats['peak_scope_queued_bytes'] = max(stats['peak_scope_queued_bytes'], self.queued_bytes)
        if group.producer_events:
            if self.batch.producer_events is None:
                self.batch.producer_events = {}
            self.batch.producer_events.update(group.producer_events)
        group.pending.clear()
        group.producer_events = None
        group.closed = True
        # Region boundaries are safe submission points: no later save can
        # re-record this group's producer events. Physical batches are bounded
        # independently by the store, even when one region exceeds the target.
        if self.deferred is not None:
            self.deferred.enforce_budget(self)
        elif self.target_bytes and self.queued_bytes >= self.target_bytes:
            self.submit(final=False)

    def finish(self):
        if self.deferred is not None:
            self.batch.discard_dead()
            self.queued_bytes = sum(record.size for record in self.batch.pending)
            for record in self.batch.pending:
                record.deferred_offload = True
        else:
            self.submit()

    def submit(self, *, final=True):
        if self.batch is not None:
            records = tuple(self.batch.pending)
            try:
                self.batch.commit() if final else self.batch.flush()
            finally:
                for record in records:
                    record.deferred_offload = False
                self.queued_bytes = 0

    def abort(self):
        if self.batch is not None:
            for record in self.batch.pending:
                record.deferred_offload = False
            self.batch.abort()
        self.queued_bytes = 0


class DeferredCommitQueue:
    """Bounded FIFO of completed forward batches, independent of model regions.

    A backward window launches a configured number of scopes. Sources needed
    by the current consumer stay resident; shared records protect their entire
    pending batch from a late round trip. Pressure can force earlier launches.
    """
    def __init__(self, store):
        self.store = store
        self.pending = OrderedDict()
        self.queued_bytes = 0

    def enqueue(self, key, commits):
        if commits.queued_bytes:
            if key in self.pending:
                raise RuntimeError(f'Duplicate deferred activation scope: {key}')
            self.pending[key] = (commits, commits.queued_bytes)
            self.queued_bytes += commits.queued_bytes
            self.enforce_budget()

    def _take(self, key, *, submit):
        entry = self.pending.pop(key, None)
        if entry is None:
            return
        commits, charged = entry
        self.queued_bytes -= charged
        commits.submit() if submit else commits.abort()

    def cancel(self, key):
        self._take(key, submit=False)

    def launch(self, protected=()):
        remaining = self.store.config.d2h_scopes_per_backward
        for key, (commits, _) in tuple(self.pending.items()):
            if not remaining:
                break
            if any(record.uid in protected and record.live_saves for record in commits.batch.pending):
                continue
            self._take(key, submit=True)
            self.store.pool.stats['d2h_backward_commits'] += 1
            remaining -= 1

    def enforce_budget(self, current=None):
        transport, stats = self.store.transport, self.store.pool.stats
        extra = current.queued_bytes if current is not None else 0
        stats['peak_deferred_d2h_bytes'] = max(stats['peak_deferred_d2h_bytes'], self.queued_bytes + extra)
        limit = self.store.config.max_pending_bytes
        if self.queued_bytes + extra + transport.inflight_bytes <= limit:
            return
        transport.poll()
        while self.pending and self.queued_bytes + extra + transport.inflight_bytes > limit:
            self._take(next(iter(self.pending)), submit=True)
            stats['d2h_deferred_budget_flushes'] += 1
            transport.poll()
        # Admission still charges real unfinished copies. Waiting is necessary
        # if the source budget is exhausted, even with a device-paced engine.
        if extra + transport.inflight_bytes > limit:
            transport.throttle(extra)
        if current is not None and extra > limit:
            current.submit(final=False)
            stats['d2h_deferred_budget_flushes'] += 1

    def abort(self):
        for key in tuple(self.pending):
            self.cancel(key)


class ScopeRestoreQueue:
    """Restore a logical backward scope regardless of its D2H batch boundaries."""
    def __init__(self, store):
        self.store = store
        self.pending = deque()
        self.ready = False
        self.consuming = False

    def seal(self, groups):
        seen = set()
        for group in reversed(groups):
            pending = deque()
            for ref in reversed(group.records):
                record = ref()
                if record is not None and record.uid not in seen:
                    seen.add(record.uid)
                    pending.append(ref)
            if pending:
                self.pending.append(pending)
        self.ready = True
        self.store.pool.stats['scope_commits'] += 1

    def abort(self):
        self.pending.clear()

    def prefetch(self, *, from_schedule=False):
        if not self.ready:
            return False
        if not self.pending:
            return True
        # A scope request submits its admitted prefix together. Budget pressure
        # closes that batch before returning; the next request resumes exactly
        # where it stopped. Group transfer mode remains available for finer
        # completion granularity instead of changing the native backend.
        with self.store.transport.prefetch_batch():
            while self.pending:
                group = self.pending[0]
                while group:
                    record = group[0]()
                    if record is not None and getattr(record, 'deferred_offload', False) and not self.consuming:
                        return False
                    if record is None or record.d2h_event is None:
                        group.popleft()
                    elif record.prefetch(from_schedule=from_schedule):
                        group.popleft()
                    elif record.size > self.store.config.max_prefetch_bytes:
                        group.popleft()
                        self.store.pool.stats['scope_prefetch_oversize_bytes'] += record.size
                    else:
                        return False
                self.pending.popleft()
        return True
