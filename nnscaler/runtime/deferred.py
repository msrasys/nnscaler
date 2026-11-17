# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deferred, differentiable tensor regions and asynchronous launch boundaries.

An operation calls ``notify_async_launch`` after enqueuing independent device
work and before waiting for its result. Compiled regions can prepare their
autograd edges before that operation, then launch computation at that boundary.
No communication backend or model names are part of this protocol.
"""

from threading import local
import types
import weakref

import torch


_active = local()


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


class _BackwardLaunchPoints:
    """One backend event can gate independent side regions and checkpoints."""

    def __init__(self, points):
        self.points = tuple(points)

    def record_ready(self):
        ready = torch.cuda.Event()
        ready.record()
        for point in self.points:
            point.ready = ready


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


class LaunchPoint:
    """Run callbacks once at an operation's explicit asynchronous launch point."""

    def __init__(self, callbacks):
        self.callbacks = tuple(callbacks)
        self.notified = False

    def __enter__(self):
        self.grad = torch.is_grad_enabled()
        self.inference = torch.is_inference_mode_enabled()
        if torch.__version__ >= (2, 4):
            self.autocast = torch.is_autocast_enabled('cuda')
            self.dtype = torch.get_autocast_dtype('cuda')
        else:
            self.autocast = torch.is_autocast_enabled()
            self.dtype = torch.get_autocast_gpu_dtype()
        self.cache = torch.is_autocast_cache_enabled()
        self.previous = getattr(_active, 'point', None)
        _active.point = self
        return self

    def notify(self):
        if self.notified:
            return False
        self.notified = True
        callbacks, self.callbacks = self.callbacks, ()
        # Autograd.Function.forward disables grad mode. Side regions must use
        # the caller's mode so their graphs remain independent and differentiable.
        with torch.inference_mode(self.inference), torch.set_grad_enabled(self.grad), \
                torch.autocast('cuda', enabled=self.autocast, dtype=self.dtype, cache_enabled=self.cache):
            for callback in callbacks:
                callback()
        return True

    def __exit__(self, exc_type, exc_value, traceback):
        _active.point = self.previous
        self.callbacks = ()
        if exc_type is None and not self.notified:
            raise RuntimeError('The async anchor did not call notify_async_launch()')


def notify_async_launch():
    """Signal after device submission, before inserting the result wait.

    With no active compiler launch point this is a no-op, so backend adapters
    can expose the boundary without changing their ordinary execution path.
    """
    point = getattr(_active, 'point', None)
    return point.notify() if point is not None else False


class NativeLaunchPoint(LaunchPoint):
    """Build an ordinary side graph at the backend's prepared-input boundary."""

    def __init__(self, callbacks, ready_callback, dependencies, backward_point=None):
        super().__init__(callbacks)
        self.ready_callback = ready_callback
        self.dependencies = dependencies
        self.backward_point = backward_point

    def prepare_input(self, value):
        if self.notified:
            raise RuntimeError('A native launch point must prepare exactly one backend input')
        ready = None
        if value.device.type == 'cuda':
            ready = torch.cuda.Event()
            ready.record()
        self.ready_callback(ready)
        super().notify()
        return after_backward(value, self.dependencies(), self.backward_point)

    def notify(self):
        if not self.notified:
            raise RuntimeError('The async backend must call prepare_async_input before its autograd function')
        return False


def prepare_async_input(value):
    """Backend hook after input preparation, immediately before Function.apply.

    A native launch point submits independent graphs here and attaches their
    backward controls to this actual backend input. Otherwise return it as-is.
    """
    point = getattr(_active, 'point', None)
    return point.prepare_input(value) if isinstance(point, NativeLaunchPoint) else value


def native_async_launch_pending():
    """Whether a backend can submit work before binding its native side graph."""
    point = getattr(_active, 'point', None)
    return isinstance(point, NativeLaunchPoint) and not point.notified


class _AttributeView:
    def __init__(self, parent, attributes):
        self.parent = parent
        self.attributes = attributes

    def __getattr__(self, name):
        if name in self.attributes:
            return self.attributes[name]
        value = getattr(self.parent, name)
        if isinstance(value, types.MethodType) and value.__self__ is self.parent:
            return types.MethodType(value.__func__, self)
        return value


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


class DeferredRegion:
    """An eager autograd region whose tensor outputs are prepared before launch.

    Inputs/attributes must be ordinary dense tensors. Outputs must be fresh,
    non-aliasing tensors. Higher-order gradients and custom operators that write
    parameter gradients manually are outside this functional protocol.
    """

    def __init__(self, function, parent, attribute_names, output_specs):
        self.function = function
        self.parent = parent
        self.attribute_names = tuple(attribute_names)
        self.output_specs = tuple(output_specs)  # (shape, dtype, requires_grad)
        if not self.output_specs:
            raise ValueError('Deferred regions must return tensors')
        self.prepared = False
        self.launched = False
        self.cleared = False

    def prepare(self, inputs, attributes):
        if self.prepared:
            raise RuntimeError('A deferred region can only be prepared once')
        self.prepared = True
        self.num_inputs = len(inputs)
        values = (*inputs, *attributes)
        if not values or any(type(x) not in (torch.Tensor, torch.nn.Parameter)
                             or x.layout != torch.strided or x.is_nested for x in values):
            raise ValueError('Deferred regions require ordinary dense tensor inputs and attributes')
        if len({x.device for x in values}) != 1:
            raise ValueError('Deferred regions require one device')
        self.device = values[0].device
        self.grad_enabled = torch.is_grad_enabled()
        self.track_grad = (self.grad_enabled and any(x.requires_grad for x in values)
                           and any(spec[2] for spec in self.output_specs))
        if torch.is_inference_mode_enabled():
            raise ValueError('Deferred autograd regions do not support inference_mode')
        if self.device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Deferred eager regions do not support CUDA graph capture')
        if self.track_grad:
            return _Deferred.apply(self, *values)
        return self._prepare_values(values)

    def _prepare_values(self, values):
        # Another deferred region may fill an input at the same launch point.
        # Bind storage only when launching, after its producer has run on the
        # host. The original autograd edges are already captured by apply().
        self.sources = values
        self.values = ()
        outputs = tuple(torch.empty(shape, dtype=dtype, device=self.device)
                        for shape, dtype, _ in self.output_specs)
        self.outputs = tuple(weakref.ref(output) for output in outputs)
        return outputs

    def launch(self):
        if self.launched or self.cleared:
            raise RuntimeError('Deferred region was already launched or released')
        self.launched = True
        self.values = tuple(x.detach().requires_grad_(x.requires_grad) for x in self.sources)
        for original, detached in zip(self.sources, self.values):
            if original.is_leaf and hasattr(original, 'grad_dtype'):
                detached.grad_dtype = original.grad_dtype
        self.sources = ()
        attributes = dict(zip(self.attribute_names, self.values[self.num_inputs:]))
        with torch.set_grad_enabled(self.grad_enabled):
            outputs = self.function(_AttributeView(self.parent, attributes), *self.values[:self.num_inputs])
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.output_specs):
            raise RuntimeError('Deferred region output count changed')
        input_storage = {x.untyped_storage().data_ptr() for x in self.values if x.numel()}
        output_storage = set()
        for output, reference, (shape, dtype, requires_grad) in zip(outputs, self.outputs, self.output_specs):
            destination = reference()
            if destination is None:
                raise RuntimeError('A deferred output was discarded before launch')
            if (not isinstance(output, torch.Tensor) or tuple(output.shape) != tuple(shape)
                    or output.dtype != dtype or output.device != self.device):
                raise RuntimeError('Deferred region output metadata changed')
            address = output.untyped_storage().data_ptr()
            if output.numel() and (address in input_storage or address in output_storage):
                raise ValueError('Deferred region outputs must not alias inputs or one another')
            output_storage.add(address)
            if output.requires_grad != (requires_grad and self.track_grad):
                raise RuntimeError('Deferred region output differentiation changed')
            # Keep the inner graph's saved values independent from mutations of
            # the visible output. The outer autograd edge and shape stay intact.
            with torch.no_grad():
                destination.data = output.detach().clone()
        self.inner_outputs = outputs
        if not self.track_grad:
            self.clear()

    def clear(self):
        self.cleared = True
        self.values = ()
        self.sources = ()
        self.inner_outputs = ()
        self.function = self.parent = None


class _Deferred(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, *values):
        outputs = state._prepare_values(values)
        ctx.state = state
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(*(output for output, spec in zip(outputs, state.output_specs) if not spec[2]))
        # SavedVariable releases this token when a non-retained backward ends,
        # or when an unused graph dies. Merely retaining the output tensor must
        # not keep the inner graph's activations after backward completion.
        token = torch.empty(0, device='cpu')
        weakref.finalize(token, state.clear)
        ctx.save_for_backward(token)
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *gradients):
        token, = ctx.saved_tensors  # Enforce ordinary retain_graph semantics.
        state = ctx.state
        if not state.launched or state.cleared:
            raise RuntimeError('Deferred backward requires a completed host launch')
        active = [(o, g) for o, g in zip(state.inner_outputs, gradients) if g is not None and o.requires_grad]
        targets = tuple(x for x in state.values if x.requires_grad)
        if active and targets:
            with torch.enable_grad():
                grads = torch.autograd.grad(tuple(x[0] for x in active), targets,
                                            tuple(x[1] for x in active), allow_unused=True,
                                            retain_graph=True, create_graph=False)
        else:
            grads = (None,) * len(targets)
        iterator = iter(grads)
        return (None,) + tuple(next(iterator) if x.requires_grad else None for x in state.values)
