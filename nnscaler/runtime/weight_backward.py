# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module-local dWeight work retaining ordinary autograd/reducer edges.

A None control edge through an upstream asynchronous operation delays parameter
backward until communication is submitted. Every task belongs to one forward
invocation and GraphTask; there is no device-wide or cross-microbatch queue.
"""
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from threading import local
import weakref

import torch
from torch.autograd.function import once_differentiable
from .deferred import BackwardLaunchPoint, after_backward
from .stream import (_get_module_stream, input_stream_lifetime,
                     parameter_gradient_stream, parameter_uses_module_stream)

_active = local()


@dataclass
class _Task:
    callbacks: list = field(default_factory=list)
    results: list = field(default_factory=list)
    launched: bool = False
    done: object = None
    ready_events: dict = field(default_factory=dict)


class _RegistrationToken(torch.autograd.Function):
    """Retain an autograd dependency without retaining the anchor's storage."""

    @staticmethod
    def forward(ctx, value):
        ctx.set_materialize_grads(False)
        return torch.empty(0, dtype=value.dtype, device=value.device)

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        return None


class _WeightPoint(BackwardLaunchPoint):
    def __init__(self, region):
        super().__init__(())
        self.region = weakref.ref(region)
        self.cuda = region.device.type == 'cuda'

    def record_ready(self):
        if self.cuda:
            super().record_ready()
        else:
            self.ready = True

    def wait(self, stream=None):
        if not self.captured or self.ready is None:
            raise RuntimeError('Weight backward requires the anchor to record input readiness')
        if self.cuda:
            super().wait(stream)

    def submitted(self):
        region = self.region()
        if region is not None:
            region.launch()


class _ParameterGrads(torch.autograd.Function):
    @staticmethod
    def forward(ctx, region, *parameters):
        ctx.region = region
        ctx.set_materialize_grads(False)
        return tuple(parameter.view_as(parameter) for parameter in parameters)

    @staticmethod
    @once_differentiable
    def backward(ctx, *gradients):
        return (None, *ctx.region.gradients(gradients))


@dataclass
class WeightBackwardBinding:
    region: object
    parameters: tuple
    indices: tuple

    def defer(self, callback, inputs):
        """Callback returns one gradient (or None) per bound parameter."""
        self.region.defer(self.indices, callback, inputs)

    def accumulate(self, callback, inputs):
        """Add gradients into supplied buffers when the region supports it.

        ``callback(outputs)`` returns one gradient per bound parameter. With
        an accumulator, each bound output must be updated additively and
        returned unchanged. Ordinary deferred regions pass ``None`` instead,
        retaining their existing allocation and launch behavior.
        """
        accumulate = getattr(self.region, 'accumulate', None)
        if accumulate is None:
            self.defer(lambda: callback(None), inputs)
        else:
            accumulate(self.indices, callback, inputs)


class WeightBackwardRegion:
    def __init__(self, name, parameters, *, independent=False):
        self.originals = tuple(parameters)
        if not self.originals or any(not p.requires_grad for p in self.originals):
            raise ValueError('Weight backward regions require trainable tensor parameters')
        devices = {p.device for p in self.originals}
        if len(devices) != 1:
            raise ValueError('Weight backward regions require one device')
        self.device = next(iter(devices))
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('Weight backward supports CPU validation and CUDA execution')
        self.enabled = torch.is_grad_enabled()
        if self.enabled and self.device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Weight backward regions require eager execution')
        self.stream = (_get_module_stream(self.device, name)
                       if self.enabled and self.device.type == 'cuda' else None)
        self.point = _WeightPoint(self)
        self.tasks = {}
        self.bound = self.prepared = False
        self.independent = independent
        self.registration_token = None
        self.inputs_ordered = False
        if self.enabled and self.stream is not None:
            # Parameter publication must run on the weight stream too. A
            # main-stream autograd node would enqueue a W wait ahead of the
            # unrelated upstream dInput path, undoing the split.
            holds = []
            for parameter in self.originals:
                # Respect compiler-proven mixed/native parameter affinity and
                # preserve any accumulator that already owns reducer hooks.
                context = (parameter_gradient_stream(parameter)
                           if parameter_uses_module_stream(parameter)
                           else torch.cuda.stream(self.stream))
                with context:
                    holds.append(parameter.expand_as(parameter))
            with torch.cuda.stream(self.stream):
                self.aliases = _ParameterGrads.apply(self, *self.originals)
            del holds
        else:
            self.aliases = _ParameterGrads.apply(self, *self.originals) if self.enabled else self.originals

    @contextmanager
    def compute(self):
        previous = getattr(_active, 'compute', None)
        _active.compute = self
        try:
            yield
        finally:
            _active.compute = previous

    @contextmanager
    def anchor(self):
        previous = getattr(_active, 'anchors', ())
        _active.anchors = (*previous, self)
        try:
            with self.point if self.enabled else nullcontext():
                yield
            if self.enabled and not self.prepared:
                raise RuntimeError('Weight backward anchor did not call prepare_async_input')
        finally:
            _active.anchors = previous

    def control(self, value):
        if not self.enabled:
            return value
        if value.device != self.device:
            raise ValueError('Weight backward anchor and parameters must share a device')
        if self.prepared:
            raise RuntimeError('A weight backward anchor must prepare exactly one input')
        self.prepared = True
        return after_backward(value, self.aliases, self.point)

    def capture_output(self, value):
        if not self.enabled:
            return
        if not self.independent or self.registration_token is not None or self.inputs_ordered:
            raise RuntimeError('Independent weight registration must capture one anchor output')
        if value.device != self.device or not value.requires_grad:
            raise ValueError('Weight registration requires a differentiable output on the parameter device')
        self.registration_token = _RegistrationToken.apply(value)

    def order_inputs(self, values):
        """Make anchor backward wait for this branch to register its dWeights.

        Called on the branch's forward stream. Only None travels toward the
        anchor, so this control cannot import the anchor's pending GPU wait.
        Each callback separately records its real input readiness in defer().
        """
        if not self.enabled:
            return values
        if not self.independent or self.registration_token is None or self.inputs_ordered:
            raise RuntimeError('Independent weight inputs require an unused anchor registration token')
        if not torch.is_grad_enabled():
            raise RuntimeError('Independent weight registration requires eager autograd')
        from torch.utils import _pytree
        flat, spec = _pytree.tree_flatten(values)
        cache = {}
        for index, value in enumerate(flat):
            if torch.is_tensor(value) and value.requires_grad:
                if value.device != self.device:
                    raise ValueError('Independent weight inputs must share the parameter device')
                if id(value) not in cache:
                    cache[id(value)] = after_backward(value, (self.registration_token,))
                flat[index] = cache[id(value)]
        if not cache:
            raise ValueError('Independent weight registration needs differentiable activation inputs')
        self.inputs_ordered = True
        self.registration_token = None
        return _pytree.tree_unflatten(flat, spec)

    def bind(self, parameters):
        if not self.enabled:
            return None
        if not torch.is_grad_enabled():
            raise RuntimeError('Bind weight backward before Function.apply; use non-reentrant checkpointing')
        indices, values = [], []
        recomputing = torch._C._current_graph_task_id() >= 0
        for parameter in parameters:
            index = next((i for i, original in enumerate(self.originals) if original is parameter), None)
            if index is None and parameter.requires_grad:
                raise ValueError('Backend weight does not match the compiled weight region parameters')
            indices.append(index)
            values.append(parameter if index is None or recomputing else self.aliases[index])
        self.bound = True
        return WeightBackwardBinding(self, tuple(values), tuple(indices))

    def close_forward(self, failed=False):
        # Break region -> aliases -> autograd context -> region cycles even
        # when automatic Python GC is disabled during training.
        self.aliases = ()
        self.registration_token = None
        if self.enabled and self.independent and not self.inputs_ordered and not failed:
            raise RuntimeError('Independent weight region did not order its activation inputs')
        if self.enabled and not self.bound and not failed:
            raise RuntimeError('Selected module did not bind the weight backward backend protocol')

    def defer(self, indices, callback, inputs):
        task_id = torch._C._current_graph_task_id()
        if task_id < 0:
            raise RuntimeError('Weight work can only be deferred inside autograd backward')
        if torch.is_grad_enabled():
            raise RuntimeError('Deferred weight backward supports first-order gradients only')
        if task_id not in self.tasks:
            if self.tasks:
                raise RuntimeError('Concurrent backwards of one weight region are not supported')
            self.tasks[task_id] = _Task()

            def cleanup(region=self):
                task = region.tasks.pop(task_id, None)
                # Keep the region alive until the GraphTask callback, even if
                # autograd.grad prunes its parameter edges and drops the graph.
                if task is not None:
                    region._join(task)

            torch.autograd.Variable._execution_engine.queue_callback(cleanup)
        task = self.tasks[task_id]
        if task.launched:
            raise RuntimeError('Weight work registered after its anchor was submitted')
        autocast = (torch.is_autocast_enabled(self.device.type),
                    torch.get_autocast_dtype(self.device.type), torch.is_autocast_cache_enabled())
        if self.independent and self.stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            task.ready_events[len(task.callbacks)] = ready
        task.callbacks.append((indices, callback, inputs, autocast))

    @torch.no_grad()
    def launch(self):
        task = self.tasks.get(torch._C._current_graph_task_id())
        if task is None or task.launched:
            return
        self.point.wait(self.stream)
        task.launched = True
        failed = True
        try:
            with torch.cuda.stream(self.stream) if self.stream is not None else nullcontext():
                for ordinal, (indices, callback, inputs, (amp, dtype, cache)) in enumerate(task.callbacks):
                    ready = task.ready_events.get(ordinal)
                    if ready is not None:
                        self.stream.wait_event(ready)
                    current_amp = (torch.is_autocast_enabled(self.device.type),
                                   torch.get_autocast_dtype(self.device.type), torch.is_autocast_cache_enabled())
                    amp_context = (torch.autocast(self.device.type, enabled=amp, dtype=dtype, cache_enabled=cache)
                                   if (amp, dtype, cache) != current_amp else nullcontext())
                    with amp_context, input_stream_lifetime(inputs, self.stream is not None):
                        gradients = tuple(callback())
                    if len(gradients) != len(indices):
                        raise RuntimeError('Deferred weight callback returned an incorrect gradient count')
                    task.results.extend((index, grad) for index, grad in zip(indices, gradients)
                                        if index is not None and grad is not None)
            failed = False
        finally:
            if self.stream is not None:
                task.done = torch.cuda.Event()
                task.done.record(self.stream)
            task.callbacks.clear()
            task.ready_events.clear()
            if failed:
                self._join(task)
                self.tasks.pop(torch._C._current_graph_task_id(), None)

    def _join(self, task):
        if task.done is not None:
            caller = torch.cuda.current_stream(self.device)
            if caller == self.stream:
                return  # Stream order already covers W and its publication.
            caller.wait_event(task.done)
            for _, gradient in task.results:
                gradient.record_stream(caller)

    def gradients(self, incoming):
        task_id = torch._C._current_graph_task_id()
        task = self.tasks.get(task_id)
        if task is None:
            if all(gradient is None for gradient in incoming):
                raise RuntimeError('No parameter gradients were registered by the weight backward backend')
            return incoming  # A backend may retain its ordinary path for empty inputs.
        try:
            self.launch()  # Fallback if no post-submission notification exists.
            self._join(task)
            result = list(incoming)
            for index, gradient in task.results:
                parameter = self.originals[index]
                if gradient.shape != parameter.shape or gradient.device != parameter.device:
                    raise RuntimeError('Deferred parameter gradient shape/device mismatch')
                gradient = gradient.to(getattr(parameter, 'grad_dtype', parameter.dtype))
                result[index] = gradient if result[index] is None else result[index] + gradient
            return tuple(result)
        finally:
            self.tasks.pop(task_id, None)


def prepare_weight_backward(parameters):
    """Backend forward: bind parameter edges in the selected module scope."""
    region = getattr(_active, 'compute', None)
    return region.bind(parameters) if region is not None else None


def prepare_weight_anchor(value):
    for region in getattr(_active, 'anchors', ()):
        value = region.control(value)
    return value
