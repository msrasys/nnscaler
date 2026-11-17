# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Keep scheduled inputs alive until their consuming CUDA stream is done."""

from contextlib import contextmanager, nullcontext
from collections import OrderedDict
from enum import Enum
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import weakref

import torch
from torch.overrides import TorchFunctionMode


_pending_holds = []
_holds_lock = Lock()
_SCALARS = (type(None), bool, int, float, complex, str, bytes, torch.dtype, torch.device)
_module_streams = {}
_module_streams_lock = Lock()
_module_workers = {}
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
        # Creating AccumulateGrad on the forward stream avoids a return to the
        # default stream for every parameter. Never reset an existing graph.
        with torch.cuda.stream(_get_module_stream(param.device, entry[1])):
            yield


class AsyncTensorCopies(TorchFunctionMode):
    """Make small CPU metadata copies inside an async anchor nonblocking.

    Snapshot pageable sources into pinned storage before enqueuing the copy.
    PyTorch's pinned allocator keeps that storage alive until CUDA consumes it.
    Large transfers and differentiable CPU tensors keep their original behavior.
    """

    def __init__(self, max_bytes=4096):
        super().__init__()
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError('max_bytes must be a positive integer')
        self.max_bytes = max_bytes

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Intercept the Python copy boundary, not every ATen allocation inside
        # a C++ communication call. That distinction matters for launch latency.
        if func in (torch.Tensor.to, torch.Tensor.cuda) and 1 <= len(args) <= 2:
            source = args[0]
            destination = kwargs.get('device', args[1] if len(args) == 2 else None)
            if isinstance(destination, torch.Tensor):
                destination = destination.device
            elif destination is None and func is torch.Tensor.cuda:
                destination = torch.device('cuda')
            if not isinstance(destination, (str, int, torch.device)):
                return func(*args, **kwargs)
            if (source.device.type == 'cpu' and destination is not None
                    and torch.device(destination).type == 'cuda'
                    and source.layout == torch.strided and not source.requires_grad
                    and source.numel() * source.element_size() <= self.max_bytes
                    and not kwargs.get('non_blocking', False)):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('Async metadata snapshots do not support CUDA graph capture')
                # Clone even pinned inputs: the original synchronous API lets
                # callers mutate them as soon as to() returns.
                staged = torch.empty_like(source, device='cpu', pin_memory=True)
                staged.copy_(source)
                return func(staged, *args[1:], **(kwargs | {'non_blocking': True}))
        return func(*args, **kwargs)


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
        self.ready = {}
        self.joined = set()
        self.owners = []
        self.futures = {}
        self.future_streams = {}
        self.deferred = {}
        self.streams = {self.caller}
        return self

    def prepare(self, key, name, function, parent, inputs, attributes, output_specs):
        """Prepare tensor/autograd edges; enqueue their work at an async anchor."""
        from .deferred import DeferredRegion
        if key in self.deferred or key in self.events or key in self.futures:
            raise ValueError(f'Duplicate module stream region: {key}')
        if key not in self.ready:
            self.capture(key)
        state = DeferredRegion(function, parent, attributes.keys(), output_specs)
        stream = _get_module_stream(self.caller.device, name)
        self.streams.add(stream)
        # Preparing a branch earlier must not accidentally create a shared
        # input's AccumulateGrad on the side stream. A later main-path gradient
        # could then enqueue its communication wait onto that same side stream.
        # Existing compiler/reducer affinities are retained by expand_as().
        accumulator_holds = [value.expand_as(value) for value in (*inputs, *attributes.values())
                             if torch.is_grad_enabled() and value.is_leaf and value.requires_grad]
        with torch.cuda.stream(stream):
            outputs = state.prepare(inputs, tuple(attributes.values()))
        del accumulator_holds
        self.deferred[key] = (name, state, (*inputs, *attributes.values()))
        return outputs

    def launch_point(self, keys):
        from .deferred import LaunchPoint

        def launch(key):
            name, state, inputs = self.deferred.pop(key)
            with self.fork(key, name, inputs):
                state.launch()

        return LaunchPoint(tuple(lambda key=key: launch(key) for key in keys))

    def capture(self, key):
        """Snapshot caller readiness before an asynchronous anchor is launched."""
        event = torch.cuda.Event()
        event.record(self.caller)
        self.ready[key] = event

    def submit(self, key, name, function, args, tracked_inputs):
        """Submit a pure region without waiting for its Python/CUDA launches.

        A single worker per named stream preserves submission order. Snapshot
        readiness on the caller now; waiting on its stream later in the worker
        would accidentally include the very communication we want to overlap.
        """
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Host-concurrent module submission does not support CUDA graph capture')
        if key in self.futures or key in self.events:
            raise ValueError(f'Duplicate module stream region: {key}')
        if key not in self.ready:
            self.capture(key)
        grad_enabled = torch.is_grad_enabled()
        inference = torch.is_inference_mode_enabled()
        if torch.__version__ >= (2, 4, 0):
            autocast_enabled = torch.is_autocast_enabled('cuda')
            autocast_dtype = torch.get_autocast_dtype('cuda')
        else:
            autocast_enabled = torch.is_autocast_enabled()
            autocast_dtype = torch.get_autocast_gpu_dtype()
        autocast_cache = torch.is_autocast_cache_enabled()

        def run():
            with torch.cuda.device(self.caller.device), torch.inference_mode(inference), \
                    torch.set_grad_enabled(grad_enabled), torch.autocast(
                        'cuda', enabled=autocast_enabled, dtype=autocast_dtype, cache_enabled=autocast_cache):
                with self.fork(key, name, tracked_inputs):
                    return function(*args)

        with _module_streams_lock:
            cache_key = (self.caller.device, name)
            if cache_key not in _module_workers:
                _module_workers[cache_key] = ThreadPoolExecutor(max_workers=1, thread_name_prefix='nnscaler-module')
            worker = _module_workers[cache_key]
        self.futures[key] = worker.submit(run)
        self.future_streams[key] = name

    def resolve(self, key, caller_indices=None):
        """Wait for host submission, then expose tensors to caller-stream users."""
        result = self.futures.pop(key).result()
        name = self.future_streams.pop(key)
        if caller_indices is None:
            self.publish(result)
        else:
            caller = tuple(value for i, value in enumerate(result) if i in caller_indices)
            producer = tuple(value for i, value in enumerate(result) if i not in caller_indices)
            if caller:
                self.publish(caller)
            if producer:
                self.publish(producer, stream=name)
        return result

    @contextmanager
    def fork(self, key, name, inputs):
        device = self.caller.device
        stream = _get_module_stream(device, name)
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

    def publish(self, outputs, *, stream=None):
        # record_stream does not read the data or wait for its producer. It
        # protects storage when a later caller-stream consumer outlives Python.
        context = (torch.cuda.stream(_get_module_stream(self.caller.device, stream))
                   if stream is not None else nullcontext())
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
        # Drain outstanding host submissions even when an unrelated node fails.
        error = None
        for key in list(self.futures):
            try:
                self.resolve(key)
            except BaseException as failure:
                error = error or failure
        # CUDA completion uses events, never a device/stream synchronize.
        for key in self.events:
            self.join(key)
        unlaunched = tuple(self.deferred)
        for _, state, _ in self.deferred.values():
            state.clear()
        self.deferred.clear()
        with input_stream_lifetime(self.owners, True):
            pass
        self.owners.clear()
        if error is not None and exc_type is None:
            raise error
        if unlaunched and exc_type is None:
            raise RuntimeError(f'Deferred module regions were not launched: {unlaunched}')


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
