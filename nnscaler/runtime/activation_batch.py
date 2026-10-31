# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Forward submission and backward restore queues, independent of graph layout.

Forward queues own pending sources until submission. Restore queues hold only
weak references to saved values. Neither queue owns a scope or autograd group.
"""
from collections import deque

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

    def flush(self):
        """Submit live pending values while keeping the region open for saves."""
        if self.closed:
            return
        pending = []
        for record in self.pending:
            if record.live_saves:
                pending.append(record)
            else:
                self.store.pool.stats['discarded_save_bytes'] += record.size
                self.store.pool.stats['discarded_save_storages'] += 1
        try:
            self.store.offload_batch(pending, self.producer_events)
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
    def __init__(self, store, *, unit, target_bytes):
        if unit not in ('group', 'scope'):
            raise ValueError('Forward submission unit must be resolved to group or scope')
        self.unit, self.target_bytes = unit, target_bytes
        self.capture_save_events = store.config.use_save_events
        self.batch = ActivationBatch(store) if unit == 'scope' else None
        self.queued_bytes = 0

    def add(self, group):
        if group.closed:
            return
        if self.batch is None:
            ActivationBatch.commit(group)
            return
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
        if self.target_bytes and self.queued_bytes >= self.target_bytes:
            try:
                self.batch.flush()
            finally:
                self.queued_bytes = 0

    def finish(self):
        if self.batch is not None:
            try:
                self.batch.commit()
            finally:
                self.queued_bytes = 0

    def abort(self):
        if self.batch is not None:
            self.batch.abort()
        self.queued_bytes = 0


class ScopeRestoreQueue:
    """Restore a logical backward scope regardless of its D2H batch boundaries."""
    def __init__(self, store):
        self.store = store
        self.pending = deque()
        self.ready = False

    def seal(self, groups):
        seen = set()
        for group in reversed(groups):
            for ref in reversed(group.records):
                record = ref()
                if record is not None and record.uid not in seen:
                    seen.add(record.uid)
                    self.pending.append(ref)
        self.ready = True
        self.store.pool.stats['scope_commits'] += 1

    def abort(self):
        self.pending.clear()

    def prefetch(self, *, from_schedule=False):
        if not self.ready:
            return False
        if not self.pending:
            return True
        with self.store.transport.prefetch_batch():
            while self.pending:
                record = self.pending[0]()
                if record is None or record.d2h_event is None:
                    self.pending.popleft()
                elif record.prefetch(from_schedule=from_schedule):
                    self.pending.popleft()
                elif record.size > self.store.config.max_prefetch_bytes:
                    self.pending.popleft()
                    self.store.pool.stats['scope_prefetch_oversize_bytes'] += record.size
                else:
                    return False
        return True
