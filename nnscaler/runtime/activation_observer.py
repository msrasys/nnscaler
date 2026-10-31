# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Bounded metadata observation, with optional CUDA-event wait measurements."""
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from itertools import count
import threading
import time
import torch

_store_ids = count()
_no_range = nullcontext()


@dataclass(frozen=True)
class ActivationEvent:
    action: str
    store_id: int
    time_ns: int
    storage_id: int | None = None
    nbytes: int = 0
    count: int = 0
    cpu_duration_ns: int = 0
    reason: str | None = None


class ActivationTrace:
    def __init__(self, max_events, *, cuda_timing=False):
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
            raise ValueError('max_events must be a positive integer')
        if not isinstance(cuda_timing, bool):
            raise ValueError('cuda_timing must be a bool')
        self.max_events = max_events
        self.cuda_timing = cuda_timing
        self.events = []
        self.dropped_events = 0
        self._cuda_waits = []
        self.dropped_cuda_waits = 0
        self._lock = threading.Lock()

    def _append(self, event):
        with self._lock:
            if len(self.events) < self.max_events:
                self.events.append(event)
            else:
                self.dropped_events += 1

    def _accept_cuda_wait(self):
        # The observation lock serializes reservation and append, including
        # trace exit. CUDA events are bounded independently of metadata events.
        with self._lock:
            if len(self._cuda_waits) < self.max_events:
                return True
            self.dropped_cuda_waits += 1
            return False

    def _append_cuda_wait(self, start, end, metadata):
        with self._lock:
            self._cuda_waits.append((start, end, metadata))

    def to_dict(self):
        """Export without synchronizing; unfinished GPU intervals have ms=None."""
        with self._lock:
            waits = []
            for start, end, metadata in self._cuda_waits:
                complete = end.query()
                waits.append(dict(metadata, complete=complete,
                                  elapsed_ms=start.elapsed_time(end) if complete else None))
            return dict(version=2, events=[asdict(event) for event in self.events],
                        dropped_events=self.dropped_events, cuda_timing=self.cuda_timing,
                        cuda_waits=waits, dropped_cuda_waits=self.dropped_cuda_waits)


class _Observation:
    def __init__(self):
        self.store_id = next(_store_ids)
        self._listeners = ()
        self._timing_traces = ()
        self._lock = threading.RLock()

    @property
    def enabled(self):
        return bool(self._listeners)

    def range(self, action):
        return torch.profiler.record_function('activation_store/' + action) if self.enabled else _no_range

    def emit(self, action, *, storage_id=None, nbytes=0, count=0, cpu_duration_ns=0, reason=None):
        if not self._listeners:
            return
        # Exit waits for any in-progress append. A completed trace therefore
        # cannot acquire late events from another autograd/consumer thread.
        with self._lock:
            listeners = self._listeners
            if not listeners:
                return
            event = ActivationEvent(action, self.store_id, time.perf_counter_ns(),
                                    storage_id, nbytes, count, cpu_duration_ns, reason)
            for callback in listeners:
                callback(event)

    def wait_event(self, stream, ready, *, storage_id, nbytes):
        """Measure the consumer-stream dependency interval, never a host wait.

        This includes event overhead/host submission gaps and may overlap
        intervals on other streams. It is not global critical-path cost.
        Normal execution adds no timing events and performs no event queries.
        """
        if not self._timing_traces:
            stream.wait_event(ready)
            return
        with self._lock:
            traces = [trace for trace in self._timing_traces if trace._accept_cuda_wait()]
            if not traces:
                stream.wait_event(ready)
                return
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            stream.wait_event(ready)
            end.record(stream)
            metadata = dict(action='restore_wait', store_id=self.store_id,
                            storage_id=storage_id, nbytes=nbytes, stream_id=stream.cuda_stream)
            for trace in traces:
                trace._append_cuda_wait(start, end, metadata)

    @contextmanager
    def trace(self, max_events, *, cuda_timing=False):
        trace = ActivationTrace(max_events, cuda_timing=cuda_timing)
        callback = trace._append
        with self._lock:
            self._listeners = (*self._listeners, callback)
            if cuda_timing:
                self._timing_traces = (*self._timing_traces, trace)
        try:
            yield trace
        finally:
            with self._lock:
                self._listeners = tuple(value for value in self._listeners if value is not callback)
                self._timing_traces = tuple(value for value in self._timing_traces if value is not trace)
