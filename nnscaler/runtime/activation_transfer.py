# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""CUDA transfer dependencies and bounded submission, independent of autograd."""
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
import time

import torch


class ActivationTransferEngine:
    def __init__(self, config, device, stats, observation):
        self.config, self.device, self.stats = config, device, stats
        self.observation = observation
        self.d2h_stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        self.h2d_stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
        self.inflight = deque()
        self.inflight_bytes = 0
        self._d2h_batch = ContextVar('activation_d2h_batch', default=False)
        self._h2d_batch = ContextVar('activation_h2d_batch', default=False)

    @contextmanager
    def _stream_batch(self, stream, active):
        if active.get():
            yield
            return
        with torch.cuda.stream(stream):
            token = active.set(True)
            try:
                yield
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
        while self.inflight and self.inflight[0][1].query():
            nbytes, _ = self.inflight.popleft()
            self.inflight_bytes -= nbytes

    def throttle(self, size=0):
        started = time.perf_counter_ns() if self.observation.enabled else None
        waited = 0
        self.poll()
        while self.inflight and self.inflight_bytes + size > self.config.max_inflight_bytes:
            nbytes, event = self.inflight.popleft()
            event.synchronize()
            self.inflight_bytes -= nbytes
            self.stats['d2h_throttles'] += 1
            waited += 1
        if started is not None:
            self.observation.emit('throttle', count=waited,
                                  cpu_duration_ns=time.perf_counter_ns() - started)

    def submit_batch(self, records, producer_events=None):
        """Submit an admitted batch without allocations or host waits between copies."""
        waited_events, waited_streams = set(), set()
        copied = 0
        with self._stream_batch(self.d2h_stream, self._d2h_batch):
            for record in records:
                stream = record.producer_stream
                event = record.producer_ready_event or (producer_events or {}).get(stream.cuda_stream)
                # A scope may merge several regions or producing streams. A
                # ready copy must not wait for an unrelated later producer.
                if event is not None:
                    if event not in waited_events:
                        self.d2h_stream.wait_event(event)
                        waited_events.add(event)
                elif stream.cuda_stream not in waited_streams:
                    self.d2h_stream.wait_stream(stream)
                    waited_streams.add(stream.cuda_stream)
                copied += bool(record.offload())
        if copied:
            self.stats['d2h_committed_groups'] += 1
            self.stats['peak_queued_d2h_bytes'] = max(
                self.stats['peak_queued_d2h_bytes'], self.inflight_bytes)

    def to_host(self, source, destination):
        with self.observation.range('d2h'):
            return self._to_host(source, destination)

    def _to_host(self, source, destination):
        if not self._d2h_batch.get():
            with self._stream_batch(self.d2h_stream, self._d2h_batch):
                return self._to_host(source, destination)
        destination.copy_(source, non_blocking=True)
        source.record_stream(self.d2h_stream)
        ready = torch.cuda.Event()
        ready.record(self.d2h_stream)
        self.inflight.append((source.numel() * source.element_size(), ready))
        self.inflight_bytes += source.numel() * source.element_size()
        return ready

    def to_device(self, source, dependency):
        with self.observation.range('h2d'):
            return self._to_device(source, dependency)

    def _to_device(self, source, dependency):
        stream = self.h2d_stream
        destination = None
        if self.config.reload_on_compute_stream and not self._h2d_batch.get():
            consumer = torch.cuda.current_stream(self.device)
            destination = torch.empty(source.shape, dtype=source.dtype, device=self.device)
            stream.wait_stream(consumer)
        if self._h2d_batch.get():
            return self._restore_on_stream(source, dependency, destination)
        with torch.cuda.stream(stream):
            return self._restore_on_stream(source, dependency, destination)

    def _restore_on_stream(self, source, dependency, destination):
        stream = self.h2d_stream
        with self.observation.range('h2d_copy'):
            stream.wait_event(dependency)
            if destination is None:
                destination = source.to(self.device, non_blocking=True)
            else:
                destination.copy_(source, non_blocking=True)
                destination.record_stream(stream)
            ready = torch.cuda.Event()
            ready.record(stream)
        return destination, ready
