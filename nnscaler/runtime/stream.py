# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Keep scheduled inputs alive until their consuming CUDA stream is done."""

from contextlib import contextmanager, nullcontext
from collections import OrderedDict
from enum import Enum
from threading import Lock
import weakref

import torch

_pending_holds = []
_holds_lock = Lock()
_SCALARS = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    torch.dtype,
    torch.device,
)
_module_streams = {}
_module_streams_lock = Lock()
_parameter_stream_names = {}


def has_module_streams():
    """Whether this process has used opt-in intra-module streams."""
    return bool(_module_streams)


def _get_module_stream(device, name):
    with _module_streams_lock:
        key = (device, name)
        if key not in _module_streams:
            _module_streams[key] = torch.cuda.Stream(device=device)
        return _module_streams[key]


def register_parameter_stream(param, name):
    """Set compiler-proven stream affinity without mutating parameter objects."""
    identity = id(param)

    def remove(reference):
        entry = _parameter_stream_names.get(identity)
        if entry is not None and entry[0] is reference:
            _parameter_stream_names.pop(identity, None)

    _parameter_stream_names[identity] = (weakref.ref(param, remove), name)


def parameter_uses_module_stream(param):
    entry = _parameter_stream_names.get(id(param))
    return entry is not None and entry[0]() is param


def parameter_gradient_stream_name(param):
    """Return the exclusive gradient stream proven by the compiler, if any."""
    entry = _parameter_stream_names.get(id(param))
    return entry[1] if entry is not None and entry[0]() is param else None


def hold_owners_through_backward(owners, streams):
    """Keep opaque storage through the graph task and its queued CUDA work."""
    owners, streams = tuple(owners), tuple(streams)
    if not owners:
        return

    def after_backward():
        holds = []
        for stream in streams:
            event = torch.cuda.Event()
            event.record(stream)
            holds.append((event, owners))
        with _holds_lock:
            _pending_holds.extend(holds)

    torch.autograd.Variable._execution_engine.queue_callback(after_backward)


@contextmanager
def parameter_gradient_stream(param):
    entry = _parameter_stream_names.get(id(param))
    if param.device.type != 'cuda' or entry is None or entry[0]() is not param or entry[1] is None:
        yield
    else:
        # Use the compiler-proven gradient stream, including deferred dW.
        # Never reset an existing graph or an accumulator's registered hooks.
        with torch.cuda.stream(_get_module_stream(param.device, entry[1])):
            yield


class ModuleStreams:
    """Run independent branches on CUDA streams and join their completion events.

    PyTorch autograd retains the forward stream of each backward operator.
    Publish outputs before handing them to caller-stream consumers.
    """

    def __enter__(self):
        self.caller = torch.cuda.current_stream()
        self.events, self.ready = {}, {}
        self.joined, self.streams = set(), {self.caller}
        self.owners = []
        self.pending_ops = []
        self.region_streams, self.backward_points = {}, {}
        return self

    def submit(self, operator, *args, **kwargs):
        from .async_op import PendingOp

        pending = PendingOp(operator, args, kwargs)
        self.pending_ops.append(pending)
        return pending

    def stream(self, name):
        """Return a shared named stream on this region's device."""
        if not isinstance(name, str) or not name or name == 'default':
            raise ValueError('Module streams require a distinct nonempty name')
        stream = _get_module_stream(self.caller.device, name)
        self.streams.add(stream)
        return stream

    def handoff(self, name, values):
        """Give tensor backward edges to a stream after their current producer.

        This identity adds no GPU arithmetic. It lets independent branches finish
        backward without importing a later communication wait from that stream.
        """
        from .backward import _StreamBoundary

        stream = self.stream(name)
        stream.wait_stream(torch.cuda.current_stream(self.caller.device))
        with torch.cuda.stream(stream):
            return tuple(_StreamBoundary.apply(value) for value in values)

    def after_backward(self, value, dependencies, keys):
        from .backward import BackwardLaunchPoint, after_backward

        point = BackwardLaunchPoint(self.region_streams[key] for key in keys)
        self.backward_points[tuple(keys)] = point
        return after_backward(value, dependencies, point)

    def backward_point(self, keys):
        return self.backward_points.get(tuple(keys), nullcontext())

    def capture(self, key):
        """Snapshot caller readiness before an asynchronous anchor is launched."""
        event = torch.cuda.Event()
        event.record(self.caller)
        self.ready[key] = event

    @contextmanager
    def fork(self, key, name, inputs, *, preserve_accumulators=False):
        if key in self.region_streams:
            raise ValueError(f'Duplicate module stream region: {key}')
        device = self.caller.device
        # Backward control edges must not move a shared input's AccumulateGrad
        # onto this stream and import the anchor's pending result wait.
        accumulator_holds = [
            value.expand_as(value)
            for value in inputs
            if preserve_accumulators
            and torch.is_grad_enabled()
            and isinstance(value, torch.Tensor)
            and value.is_leaf
            and value.requires_grad
        ]
        stream = _get_module_stream(device, name)
        self.region_streams[key] = stream
        self.streams.add(stream)
        if key in self.ready:
            stream.wait_event(self.ready[key])
        else:
            stream.wait_stream(self.caller)
        with torch.cuda.stream(stream), input_stream_lifetime(inputs, True) as owners:
            self.owners.extend(owners)
            try:
                yield
            finally:
                event = torch.cuda.Event()
                event.record(stream)
                self.events[key] = event
        del accumulator_holds

    def publish(self, outputs, *, stream=None):
        # record_stream does not read the data or wait for its producer. It
        # protects storage when a later caller-stream consumer outlives Python.
        context = (
            torch.cuda.stream(_get_module_stream(self.caller.device, stream)) if stream is not None else nullcontext()
        )
        with context, input_stream_lifetime(outputs, True) as owners:
            self.owners.extend(owners)
        if self.owners:
            # An opaque owner saved by a custom backward can lose its final
            # Python reference before the queued backward kernels finish. Keep
            # it through the graph task, then through completion on our streams.
            # Hooks live on output graph nodes, not persistent parameters.
            owners, streams = tuple(self.owners), tuple(self.streams)

            def on_gradient(gradient):
                hold_owners_through_backward(owners, streams)
                return gradient

            pending, seen = [outputs], set()
            while pending:
                value = pending.pop()
                if id(value) in seen:
                    continue
                seen.add(id(value))
                if isinstance(value, torch.Tensor) and value.requires_grad:
                    value.register_hook(on_gradient)
                elif type(value) in (dict, OrderedDict):
                    pending.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    pending.extend(value)

    def join(self, key):
        if key not in self.joined:
            self.caller.wait_event(self.events[key])
            self.joined.add(key)

    def __exit__(self, exc_type, exc_value, traceback):
        for pending in self.pending_ops:
            pending.close()
        self.pending_ops.clear()
        for key in self.events:
            self.join(key)
        if exc_type is not None:
            for stream in self.streams - {self.caller}:
                self.caller.wait_stream(stream)
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
