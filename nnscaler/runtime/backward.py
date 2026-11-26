# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Autograd ordering and temporary values for cooperating CUDA-stream branches."""

from collections import OrderedDict
from threading import local

import torch

_active = local()


class _StreamBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.set_materialize_grads(False)
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class BackwardValue:
    """Transfer tensors between two backwards in one forward invocation.

    Construct on the consuming operator's forward stream. ``put`` records the
    producer's readiness; ``pop`` must run on that original consumer stream.
    Pruned consumer paths release their temporary tensors at GraphTask completion.
    Values must be a tuple of tensors on the declared device.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.consumer = torch.cuda.current_stream(self.device) if self.device.type == 'cuda' else None
        self.pending = {}

    def put(self, values):
        task = torch._C._current_graph_task_id()
        if task < 0 or torch.is_grad_enabled() or self.pending:
            raise RuntimeError('BackwardValue requires one pending first-order backward')
        if not isinstance(values, tuple) or any(
            not isinstance(t, torch.Tensor) or t.device != self.device for t in values
        ):
            raise ValueError('BackwardValue requires tensors on its declared device')
        ready = None
        if self.consumer is not None:
            producer = torch.cuda.current_stream(self.device)
            if producer != self.consumer:
                ready = torch.cuda.Event()
                ready.record(producer)
        self.pending[task] = (values, ready)
        torch.autograd.Variable._execution_engine.queue_callback(lambda: self.pending.pop(task, None))

    def pop(self):
        if self.consumer is not None and torch.cuda.current_stream(self.device) != self.consumer:
            raise RuntimeError('BackwardValue must be read on its original consumer stream')
        values, ready = self.pending.pop(torch._C._current_graph_task_id())
        if ready is not None:
            self.consumer.wait_event(ready)
            for tensor in values:
                tensor.record_stream(self.consumer)
        return values


def _backward_tensors(values):
    pending, seen, result = [values], set(), []
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            result.append(value)
        elif type(value) in (dict, OrderedDict):
            pending.extend(value.keys())
            pending.extend(value.values())
            if type(value) is OrderedDict:
                pending.extend(vars(value).values())
        elif type(value) in (tuple, list, torch.Size):
            pending.extend(value)
        elif isinstance(value, tuple) and hasattr(type(value), '_fields') and not hasattr(value, '__dict__'):
            pending.extend(value)
        elif value is not None and not isinstance(
            value, (bool, int, float, complex, str, bytes, torch.dtype, torch.device)
        ):
            raise TypeError('backward_on requires tensor/builtin-container graph boundaries')
    return result


def backward_boundaries(inputs):
    """Snapshot graph boundaries before codegen releases local input names."""
    return {value.grad_fn for value in _backward_tensors(inputs) if value.grad_fn is not None}


def prioritize_backward(outputs, boundaries):
    """Prioritize a local autograd subgraph without moving any dependency.

    An early forward prefix otherwise retains a lower ready-queue priority
    than intervening work, even after a later anchor releases its controls.
    Reserve a high sequence-number band, preserving order within that band.
    Input graph nodes and leaf accumulators keep their original priorities.
    No backward is started here and no work is queued for another GraphTask.
    """
    if not torch.is_grad_enabled():
        return
    pending = [value.grad_fn for value in _backward_tensors(outputs) if value.grad_fn is not None]
    seen = set()
    priority_band = 1 << 60
    while pending:
        node = pending.pop()
        if node is None or node in seen or node in boundaries:
            continue
        seen.add(node)
        if not hasattr(node, '_sequence_nr') or not hasattr(node, '_set_sequence_nr'):
            raise RuntimeError('backward_on requires PyTorch autograd node priority support')
        sequence = node._sequence_nr()
        # AccumulateGrad has UINT64_MAX priority; already promoted nodes are
        # idempotent. Dependencies still determine when a node becomes ready.
        if sequence < priority_band:
            node._set_sequence_nr(priority_band + sequence)
        pending.extend(parent for parent, _ in node.next_functions)


class BackwardLaunchPoint:
    """Align side backward with the GPU inputs of an asynchronous operation."""

    def __init__(self, streams):
        self.streams = tuple(set(streams))
        self.captured = False
        self.ready = None

    def __enter__(self):
        self.previous = getattr(_active, 'backward', None)
        _active.backward = self
        return self

    def __exit__(self, *exc):
        _active.backward = self.previous

    def record_ready(self):
        # Record before the backend inserts its result wait. The host can run
        # far ahead of this event; submission order alone cannot align kernels.
        self.ready = torch.cuda.Event()
        self.ready.record()

    def wait(self, stream=None):
        if self.captured and self.ready is None:
            raise RuntimeError('The async backward did not record its input readiness')
        if self.ready is not None:
            for target in self.streams if stream is None else (stream,):
                target.wait_event(self.ready)

    def submitted(self):
        """Backend hook after enqueueing communication, before its result wait."""
        pass


class _BackwardLaunchPoints:
    """One backend event can gate independent side regions and checkpoints."""

    def __init__(self, points):
        self.points = tuple(points)

    def record_ready(self):
        ready = torch.cuda.Event()
        ready.record()
        for point in self.points:
            point.ready = ready

    def submitted(self):
        for point in self.points:
            point.submitted()


def capture_async_backward():
    """Capture during backend forward; record_ready() before backward submission."""
    point = getattr(_active, 'backward', None)
    points = []
    while point is not None:
        if point.captured:
            raise RuntimeError('An async backward launch point must identify one backend operation')
        point.captured = True
        points.append(point)
        point = point.previous
    if not points:
        return None
    return points[0] if len(points) == 1 else _BackwardLaunchPoints(points)


class _BackwardOrder(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, point, *dependencies):
        ctx.count = len(dependencies)
        ctx.point = point
        ctx.set_materialize_grads(False)
        return value

    @staticmethod
    def backward(ctx, gradient):
        # None completes the autograd dependency without producing a CUDA
        # tensor/event on the communication consumer's blocked stream.
        if ctx.point is not None:
            ctx.point.wait()
        return (gradient, None) + (None,) * ctx.count


def after_backward(value, dependencies, point=None):
    """Let dependency backward nodes become ready after value's downstream work.

    Place this identity on an async anchor's input. Its backward runs after
    that anchor's backward has enqueued communication, and returns no gradient
    along the control edges. It adds no GPU arithmetic or data dependency.
    """
    dependencies = tuple(t for t in dependencies if t.requires_grad)
    if not torch.is_grad_enabled() or not dependencies:
        return value
    if not value.requires_grad:
        raise ValueError('A backward anchor must already have a differentiable input')
    return _BackwardOrder.apply(value, point, *dependencies)
