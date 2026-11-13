# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""CUDA transfer dependencies and bounded submission, independent of autograd."""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import groupby
import time

import torch

from .activation_fence import TransferFence, cuda_event


class ActivationTransferEngine:
    def __init__(self, config, device, stats, observation):
        self.config, self.device, self.stats = config, device, stats
        self.observation = observation
        self.d2h_stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        self.h2d_stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        self.inflight = deque()
        self.inflight_bytes = 0
        self._ordered_batches = deque()
        self.ordered_bytes = 0
        self._d2h_batch = ContextVar('activation_d2h_batch', default=False)
        self._h2d_batch = ContextVar('activation_h2d_batch', default=False)
        self._last_restore_wait = None
        self._sm_copies = {}
        if device.type == 'cuda' and 'sm' in (config.h2d_backend, config.d2h_backend):
            from .activation_sm_copy import copy_d2h, copy_h2d, prepare_sm_copy
            prepare_sm_copy()
            self._sm_copies = {'h2d': copy_h2d, 'd2h': copy_d2h}

    @contextmanager
    def _stream_batch(self, stream, active):
        if active.get():
            yield active.get()
            return
        with torch.cuda.stream(stream):
            fence = TransferFence()
            token = active.set(fence)
            direction = 'd2h' if active is self._d2h_batch else 'h2d'
            def publish():
                if fence.used:
                    fence.publish(stream)
                    self.stats[direction + '_completion_events'] += 1
            try:
                yield fence
            except BaseException:
                # A later allocation can fail after earlier complete DMA calls
                # were enqueued. Publish their fence without draining unrelated
                # queues; individual partial-copy failures drain at the copy site.
                try:
                    publish()
                except BaseException:
                    pass  # Preserve the original error; fence is marked failed.
                raise
            else:
                publish()
            finally:
                active.reset(token)

    @contextmanager
    def prefetch_batch(self):
        # Compute-stream allocations must observe the caller's compute stream,
        # so they retain the per-copy path. Copy-stream allocations can submit
        # an entire restore batch under one stream context.
        if self.config.reload_on_compute_stream:
            yield
        else:
            with self._stream_batch(self.h2d_stream, self._h2d_batch):
                yield

    def poll(self):
        """Reap completed copies without waiting for unfinished GPU work."""
        completed = None
        while self.inflight:
            event = self.inflight[0][1]
            if event is not completed and not event.query():
                break
            nbytes, _ = self.inflight.popleft()
            self.inflight_bytes -= nbytes
            completed = event
        while self._ordered_batches and self._ordered_batches[0][1].query():
            size, _ = self._ordered_batches.popleft()
            self.ordered_bytes -= size

    def throttle(self, size=0):
        started = time.perf_counter_ns() if self.observation.enabled else None
        waited = 0
        self.poll()
        limit = (self.config.max_pending_bytes if self.config.backpressure_mode == 'stream'
                 else self.config.max_inflight_bytes)
        while self.inflight and self.inflight_bytes + size > limit:
            nbytes, event = self.inflight.popleft()
            event.synchronize()
            self.inflight_bytes -= nbytes
            self.stats['d2h_throttles'] += 1
            waited += 1
            self.poll()  # Retire the other storages sharing this batch fence.
        if started is not None:
            self.observation.emit('throttle', count=waited,
                                  cpu_duration_ns=time.perf_counter_ns() - started)

    def submit_batch(self, records, producer_events=None):
        """Submit an admitted batch without allocations or host waits between copies."""
        def readiness(record):
            stream = record.producer_stream
            event = record.producer_ready_event or (producer_events or {}).get(stream.cuda_stream)
            return ('event', id(event)) if event is not None else ('stream', stream.cuda_stream)
        # A completion fence must not make earlier ready values depend on a
        # later producer. Keep producer order, sharing a fence only within one
        # contiguous readiness group (including explicit multi-stream saves).
        for _, group in groupby(records, readiness):
            self._submit_ready_batch(list(group), producer_events)

    def _submit_ready_batch(self, records, producer_events=None):
        waited_events, waited_streams = set(), set()
        copied = copied_bytes = 0
        with self._stream_batch(self.d2h_stream, self._d2h_batch) as completion:
            for record in records:
                stream = record.producer_stream
                event = record.producer_ready_event or (producer_events or {}).get(stream.cuda_stream)
                # A scope may merge several regions or producing streams. A
                # ready copy must not wait for an unrelated later producer.
                if event is not None:
                    if event not in waited_events:
                        self.d2h_stream.wait_event(cuda_event(event))
                        waited_events.add(event)
                elif stream.cuda_stream not in waited_streams:
                    self.d2h_stream.wait_stream(stream)
                    waited_streams.add(stream.cuda_stream)
                if record.offload():
                    copied += 1
                    copied_bytes += record.size
        if copied:
            if self.config.backpressure_mode == 'stream':
                self._pace_compute(copied_bytes, completion)
            self.stats['d2h_committed_groups'] += 1
            self.stats['peak_d2h_batch_bytes'] = max(self.stats['peak_d2h_batch_bytes'], copied_bytes)
            self.stats['peak_queued_d2h_bytes'] = max(
                self.stats['peak_queued_d2h_bytes'], self.inflight_bytes)
            if self.observation.enabled:
                self.observation.emit('d2h_batch_submit', nbytes=copied_bytes, count=copied)

    def _pace_compute(self, nbytes, completion):
        """Order future compute after older batches without blocking its host.

        This is an execution window, not a physical-memory accounting shortcut:
        inflight_bytes continues charging every unfinished source until query()
        reports completion. throttle() enforces the separate hard pending cap.
        At most one indivisible batch can overshoot either admission threshold.
        """
        self._ordered_batches.append((nbytes, completion))
        self.ordered_bytes += nbytes
        stream = torch.cuda.current_stream(self.device)
        while self._ordered_batches and self.ordered_bytes > self.config.max_inflight_bytes:
            size, ready = self._ordered_batches.popleft()
            stream.wait_event(cuda_event(ready))
            self.ordered_bytes -= size
            self.stats['d2h_stream_waits'] += 1
            if self.observation.enabled:
                self.observation.emit('d2h_stream_wait', nbytes=size, count=1)

    def clear(self):
        """Forget completed bookkeeping after the owner has drained streams."""
        self.inflight.clear()
        self.inflight_bytes = 0
        self._ordered_batches.clear()
        self.ordered_bytes = 0
        self._last_restore_wait = None

    def to_host(self, source, destination):
        with self.observation.range('d2h'):
            return self._to_host(source, destination)

    def _to_host(self, source, destination):
        if not self._d2h_batch.get():
            with self._stream_batch(self.d2h_stream, self._d2h_batch):
                return self._to_host(source, destination)
        source.record_stream(self.d2h_stream)
        try:
            self._copy_into(destination, source, 'd2h')
            ready = self._completion(self._d2h_batch, self.d2h_stream, 'd2h')
        except BaseException:
            # A later chunk/event can fail after earlier DMA was enqueued.
            # The caller must not recycle a host buffer without its fence.
            self.d2h_stream.synchronize()
            raise
        self.inflight.append((source.numel() * source.element_size(), ready))
        self.inflight_bytes += source.numel() * source.element_size()
        return ready

    def _completion(self, active, stream, direction):
        fence = active.get()
        if isinstance(fence, TransferFence):
            fence.used = True
            return fence
        # Internal direct-copy calls retain a standalone completion event.
        ready = torch.cuda.Event()
        ready.record(stream)
        self.stats[direction + '_completion_events'] += 1
        return ready

    def to_device(self, source, dependency):
        with self.observation.range('h2d'):
            return self._to_device(source, dependency)

    def _copy_into(self, destination, source, direction):
        """Split transfer submissions without splitting storage ownership or readiness."""
        chunk_bytes = self.config.copy_chunk_bytes
        use_sm = getattr(self.config, direction + '_backend') == 'sm'
        def copy(dst, src):
            if use_sm:
                if direction not in self._sm_copies:
                    raise RuntimeError('SM activation transfer requires a CUDA transfer engine')
                self._sm_copies[direction](dst, src, getattr(self.config, direction + '_sm_blocks'))
            else:
                dst.copy_(src, non_blocking=True)
        if not chunk_bytes or source.numel() * source.element_size() <= chunk_bytes:
            copy(destination, source)
            chunks = int(source.numel() > 0)
        else:
            # The transport operates on flat storage buffers. Never flatten by
            # copying: that would silently allocate a second activation.
            src, dst = source.view(-1), destination.view(-1)
            elements = max(1, chunk_bytes // source.element_size())
            chunks = 0
            for start in range(0, src.numel(), elements):
                copy(dst[start:start+elements], src[start:start+elements])
                chunks += 1
        self.stats[direction + '_copy_chunks'] += chunks
        if use_sm:
            self.stats[direction + '_sm_copy_chunks'] += chunks
        if self.observation.enabled:
            self.observation.emit(direction + '_copy', nbytes=source.numel() * source.element_size(), count=chunks)

    def wait_for_restore(self, stream, ready, *, storage_id, nbytes):
        previous = self._last_restore_wait
        if (isinstance(ready, TransferFence) and previous is not None
                and previous[0] == stream.cuda_stream and previous[1] is ready):
            self.stats['restore_waits_elided'] += 1
            return
        self.observation.wait_event(stream, ready, storage_id=storage_id, nbytes=nbytes)
        self._last_restore_wait = (stream.cuda_stream, ready)

    def _to_device(self, source, dependency):
        stream = self.h2d_stream
        destination = None
        if self.config.reload_on_compute_stream and not self._h2d_batch.get():
            consumer = torch.cuda.current_stream(self.device)
            destination = torch.empty(source.shape, dtype=source.dtype, device=self.device)
            stream.wait_stream(consumer)
        if self._h2d_batch.get():
            return self._restore_on_stream(source, dependency, destination)
        with self._stream_batch(stream, self._h2d_batch):
            return self._restore_on_stream(source, dependency, destination)

    def _restore_on_stream(self, source, dependency, destination):
        stream = self.h2d_stream
        # Allocation failure has not submitted a read of this host buffer.
        # Propagate it immediately: draining unrelated queued restores can wait
        # on peers that also stopped at OOM. Only partial DMA needs draining.
        if destination is None:
            destination = torch.empty(source.shape, dtype=source.dtype, device=self.device)
        destination.record_stream(stream)
        try:
            with self.observation.range('h2d_copy'):
                stream.wait_event(cuda_event(dependency))
                self._copy_into(destination, source, 'h2d')
                ready = self._completion(self._h2d_batch, stream, 'h2d')
        except BaseException:
            stream.synchronize()
            raise
        return destination, ready
