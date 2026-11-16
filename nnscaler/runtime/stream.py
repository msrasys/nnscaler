# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Keep scheduled inputs alive until their consuming CUDA stream is done."""

from contextlib import contextmanager
from collections import OrderedDict
from enum import Enum
from threading import Lock

import torch


_pending_holds = []
_holds_lock = Lock()
_SCALARS = (type(None), bool, int, float, complex, str, bytes, torch.dtype, torch.device)
_module_streams = {}
_module_streams_lock = Lock()


class ModuleStreams:
    """Event-scoped module execution used by generated segment code.

    Forks wait only for work submitted before their launch. Joins wait for the
    particular region's completion event, not subsequent work on its stream.
    Autograd records the forward streams and schedules backward on those same
    streams, including its cross-stream dependency synchronization.
    """

    def __enter__(self):
        self.caller = torch.cuda.current_stream()
        self.events = {}
        self.joined = set()
        self.owners = []
        return self

    @contextmanager
    def fork(self, key, name, inputs):
        device = self.caller.device
        with _module_streams_lock:
            cache_key = (device, name)
            if cache_key not in _module_streams:
                _module_streams[cache_key] = torch.cuda.Stream(device=device)
            stream = _module_streams[cache_key]
        stream.wait_stream(self.caller)
        with torch.cuda.stream(stream), input_stream_lifetime(inputs, True) as owners:
            self.owners.extend(owners)
            try:
                yield
            finally:
                event = torch.cuda.Event()
                event.record(stream)
                self.events[key] = event

    def publish(self, outputs):
        # record_stream does not read the data or wait for its producer. It
        # protects storage when a later caller-stream consumer outlives Python.
        with input_stream_lifetime(outputs, True) as owners:
            self.owners.extend(owners)

    def join(self, key):
        if key not in self.joined:
            self.caller.wait_event(self.events[key])
            self.joined.add(key)

    def __exit__(self, exc_type, exc_value, traceback):
        # Also covers unused outputs and exceptions. Never synchronize the CPU.
        for key in self.events:
            self.join(key)
        with input_stream_lifetime(self.owners, True):
            pass
        self.owners.clear()


def drain_stream_holds(*, wait=False):
    """Reclaim opaque input owners; wait only at an explicit schedule drain."""
    with _holds_lock:
        holds = _pending_holds[:]
        _pending_holds.clear()
    pending = []
    for event, owners in holds:
        if wait:
            event.synchronize()
        elif not event.query():
            pending.append((event, owners))
    with _holds_lock:
        _pending_holds.extend(pending)


@contextmanager
def input_stream_lifetime(inputs, enabled):
    """Protect inputs and yield opaque owners that a later backward may need.

    Builtin containers are traversed once by identity, including cycles. Custom
    owners may implement ``record_stream(stream)``; otherwise keep the owner
    until an event after the call completes. A caller retaining dependencies for
    backward must carry the yielded owners through its final backward call too.
    Do not inspect arbitrary Python attributes or consume iterators. Stream
    ordering remains the scheduler's job.
    """
    if not enabled or not torch.cuda.is_initialized():
        yield ()
        return

    drain_stream_holds()
    stream = torch.cuda.current_stream()
    pending, seen, opaque = [inputs], set(), []
    try:
        while pending:
            value = pending.pop()
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            if isinstance(value, torch.Tensor):
                if value.is_cuda:
                    if value.layout == torch.strided and not value.is_nested:
                        try:
                            value.record_stream(stream)
                        except NotImplementedError:
                            # DTensor and other wrapper subclasses may not expose
                            # record_stream. Retain their underlying storage via
                            # the owner instead of leaving it unprotected.
                            opaque.append(value)
                    else:
                        opaque.append(value)
            elif type(value) in (dict, OrderedDict):
                pending.extend(value.keys())
                pending.extend(value.values())
                if type(value) is OrderedDict:
                    pending.extend(vars(value).values())
            elif type(value) in (list, tuple, set, frozenset, torch.Size):
                pending.extend(value)
            elif isinstance(value, tuple) and hasattr(type(value), '_fields') and not hasattr(value, '__dict__'):
                # Namedtuple fields are visible; arbitrary subclass attributes
                # still need the opaque-owner fallback below.
                pending.extend(value)
            elif type(value) is slice:
                pending.extend((value.start, value.stop, value.step))
            elif type(value) is range:
                pass
            elif isinstance(value, Enum):
                # Enum values can themselves contain tensors. Skip the class
                # link, but traverse member values and custom member fields.
                pending.extend(item for name, item in vars(value).items() if name != '__objclass__')
            elif type(value) not in _SCALARS:
                record = getattr(value, 'record_stream', None)
                if callable(record):
                    record(stream)
                else:
                    # Includes custom containers/namedtuples, dataclasses and
                    # extension objects whose tensor ownership is opaque.
                    opaque.append(value)
        yield opaque
    finally:
        if opaque:
            event = torch.cuda.Event()
            event.record(stream)
            with _holds_lock:
                _pending_holds.append((event, opaque))
