# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Bounded activation-store observation containing metadata, never tensors."""
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
    def __init__(self, max_events):
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
            raise ValueError('max_events must be a positive integer')
        self.max_events = max_events
        self.events = []
        self.dropped_events = 0
        self._lock = threading.Lock()

    def _append(self, event):
        with self._lock:
            if len(self.events) < self.max_events:
                self.events.append(event)
            else:
                self.dropped_events += 1

    def to_dict(self):
        with self._lock:
            return dict(version=1, events=[asdict(event) for event in self.events],
                        dropped_events=self.dropped_events)


class _Observation:
    def __init__(self):
        self.store_id = next(_store_ids)
        self._listeners = ()
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

    @contextmanager
    def trace(self, max_events):
        trace = ActivationTrace(max_events)
        callback = trace._append
        with self._lock:
            self._listeners = (*self._listeners, callback)
        try:
            yield trace
        finally:
            with self._lock:
                self._listeners = tuple(value for value in self._listeners if value is not callback)
