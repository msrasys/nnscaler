# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stream-local gradient accumulation, scoped to one forward and autograd task."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from threading import local

import torch
from torch.autograd.function import once_differentiable

from .stream import (
    _get_module_stream,
    input_stream_lifetime,
    parameter_gradient_stream,
    parameter_gradient_stream_name,
    parameter_uses_module_stream,
)
from .utils import get_grad_dtype

_active = local()


@contextmanager
def weight_backward_scopes():
    """Close aliases even if an earlier anchor fails before its compute call."""
    regions = []
    try:
        yield regions
    finally:
        for region in regions:
            region.close_forward(failed=True)


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
        """Add one contribution to each supplied parameter-gradient buffer."""
        self.region.accumulate(self.indices, callback, inputs)


@dataclass
class _AccumulationTask:
    buffers: dict = field(default_factory=dict)
    pending: list = field(default_factory=list)
    done: object = None
    direct: set = field(default_factory=set)
    hooks: list = field(default_factory=list)


class WeightGradientAccumulator:
    """Combine fragment dWeights on a stream, preserving autograd publication.

    Use as a context manager around cooperative backend calls. The backend
    binds parameters with ``prepare_weight_backward`` and calls
    ``binding.accumulate(callback, inputs)`` from its first-order backward.
    The callback receives zero-initialized private gradient buffers on first use and
    must ADD each contribution, returning those same buffers. Buffer dtypes
    follow each parameter's ``grad_dtype`` (or its dtype when unspecified).

    Existing ``binding.defer`` callbacks also work, with an extra copy into
    the accumulation buffers. Frozen parameters may be passed to the backend,
    but only trainable parameters belong in this region. A callback can receive
    ``None`` for an unbound/frozen parameter; its corresponding return is ignored.

    ``direct_parameter_grads`` permits adding to an existing native leaf grad
    during a full backward. CUDA parameters require compiler-proven exclusive
    stream affinity. Hooks, reducer parameters and autograd.grad use private
    buffers. Backward tasks must be serialized; custom AccumulateGrad node hooks
    are not supported in this opt-in mode.
    """

    def __init__(self, stream, parameters, *, defer_until_backward=False, direct_parameter_grads=False):
        if type(defer_until_backward) is not bool:
            raise TypeError('defer_until_backward must be a boolean')
        if type(direct_parameter_grads) is not bool:
            raise TypeError('direct_parameter_grads must be a boolean')
        if not isinstance(stream, str) or not stream or stream == 'default':
            raise ValueError('Weight accumulation requires a distinct named stream')
        parameters = tuple(parameters)
        if any(
            type(p) not in (torch.Tensor, torch.nn.Parameter) or p.layout != torch.strided or p.is_nested
            for p in parameters
        ):
            raise ValueError('Weight accumulation requires ordinary dense parameters')
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
        self.deferred = defer_until_backward
        self.direct_parameter_grads = direct_parameter_grads
        self.stream_name = stream
        self.backward_ready = None
        self.anchored = False
        if self.enabled and self.device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Weight backward regions require eager execution')
        self.stream = _get_module_stream(self.device, stream) if self.enabled and self.device.type == 'cuda' else None
        self.tasks = {}
        if self.enabled and self.stream is not None:
            # Parameter publication must run on the weight stream too. A
            # main-stream autograd node would enqueue a W wait ahead of the
            # unrelated upstream dInput path, undoing the split.
            holds = []
            for parameter in self.originals:
                # Respect compiler-proven mixed/native parameter affinity and
                # preserve any accumulator that already owns reducer hooks.
                context = (
                    parameter_gradient_stream(parameter)
                    if parameter_uses_module_stream(parameter)
                    else torch.cuda.stream(self.stream)
                )
                with context:
                    holds.append(parameter.expand_as(parameter))
            with torch.cuda.stream(self.stream):
                self.aliases = _ParameterGrads.apply(self, *self.originals)
            del holds
        else:
            self.aliases = _ParameterGrads.apply(self, *self.originals) if self.enabled else self.originals
        self.bound_indices = set()
        self.failed = False
        self.forward_closed = False
        self._scope = None

    def after_backward(self, value):
        """Delay dW publication until an earlier async operator's backward.

        Call before that operator's forward, then enter ``backward_point()``
        around it. The operator must use the async-backward readiness protocol.
        Parameter aliases carry only control edges; no communication-result
        wait is imported onto the dW stream.
        """
        from .backward import (
            BackwardLaunchPoint, after_backward, backward_boundaries, prioritize_backward,
        )

        if not self.deferred or self.anchored or self.forward_closed:
            raise RuntimeError('A deferred weight region requires exactly one earlier backward anchor')
        self.anchored = True
        if not self.enabled:
            return value
        if self.stream is not None:
            self.backward_ready = BackwardLaunchPoint((self.stream,))
        boundaries = backward_boundaries((value, self.aliases))
        controlled = after_backward(value, self.aliases, self.backward_ready)
        prioritize_backward((controlled,), boundaries)
        prioritize_backward(self.aliases, backward_boundaries(self.originals))
        return controlled

    def backward_point(self):
        return self.backward_ready or nullcontext()

    @contextmanager
    def compute(self):
        previous = getattr(_active, 'compute', None)
        _active.compute = self
        try:
            yield
        finally:
            _active.compute = previous

    @contextmanager
    def operator(self):
        """One cooperative forward call, reused by non-reentrant replay."""
        if not self.forward_closed:
            with self:
                yield
        else:
            if torch._C._current_graph_task_id() < 0:
                raise RuntimeError('A weight accumulator owns one forward invocation')
            with self.compute():
                yield

    def bind(self, parameters):
        if self.forward_closed and torch._C._current_graph_task_id() < 0:
            raise RuntimeError('A closed weight accumulator cannot bind another forward')
        if not self.enabled:
            return None
        if not torch.is_grad_enabled():
            raise RuntimeError('Bind weight backward before Function.apply; use non-reentrant checkpointing')
        indices, values = [], []
        recomputing = torch._C._current_graph_task_id() >= 0
        for parameter in parameters:
            index = next(
                (i for i, original in enumerate(self.originals) if original is parameter),
                None,
            )
            if index is None and parameter.requires_grad:
                raise ValueError('Backend weight does not match the compiled weight region parameters')
            indices.append(index)
            values.append(parameter if index is None or recomputing else self.aliases[index])
        self.bound_indices.update(i for i in indices if i is not None)
        return WeightBackwardBinding(self, tuple(values), tuple(indices))

    def close_forward(self, failed=False):
        self.aliases = ()  # Break the autograd cycle when automatic GC is disabled.
        self.forward_closed = True
        if self.enabled and not failed and self.deferred and not self.anchored:
            raise RuntimeError('Deferred weight accumulation requires an earlier backward anchor')
        if self.enabled and not failed and len(self.bound_indices) != len(self.originals):
            raise RuntimeError('Every accumulator parameter must be bound by a cooperative backend')

    def __enter__(self):
        if self._scope is not None or self.forward_closed:
            raise RuntimeError('A weight accumulator owns one forward invocation')
        self._scope = self.compute()
        self._scope.__enter__()
        return self

    def __exit__(self, *exc):
        try:
            self._scope.__exit__(*exc)
        finally:
            self._scope = None
            self.close_forward(failed=exc[0] is not None)

    def _task(self):
        gid = torch._C._current_graph_task_id()
        if gid < 0 or torch.is_grad_enabled():
            raise RuntimeError('Weight accumulation requires first-order autograd backward')
        if self.failed:
            raise RuntimeError('Cannot reuse a failed weight accumulator')
        if gid not in self.tasks:
            if self.tasks:
                raise RuntimeError('Concurrent backwards of one accumulator are unsupported')
            task = self.tasks[gid] = _AccumulationTask()

            def cleanup(region=self, task=task):
                pending = region.tasks.pop(gid, None)
                if pending is not None or task.direct:
                    # Direct grads are not autograd edge tensors. Protect them
                    # on the backward caller before it reads/resets .grad.
                    region._join_accumulation(task)
                for hook in task.hooks:
                    hook.remove()
                task.hooks.clear()

            torch.autograd.Variable._execution_engine.queue_callback(cleanup)
        return gid, self.tasks[gid]

    def _parameter_buffer(self, index, task):
        """Use a native leaf's accumulation buffer only for a full backward.

        Tensor hooks and reducer-owned parameters keep ordinary publication.
        The opt-in mode requires serial backward tasks and no custom node hooks.
        """
        from .adapter.reducer import has_reducer_grad_accumulator

        parameter = self.originals[index]
        if (not self.direct_parameter_grads or not isinstance(parameter, torch.nn.Parameter)
                or not parameter.is_leaf or has_reducer_grad_accumulator(parameter)
                or (self.stream is not None and parameter_gradient_stream_name(parameter) != self.stream_name)
                or getattr(parameter, '_backward_hooks', None)
                or getattr(parameter, '_post_accumulate_grad_hooks', None)):
            return None
        node = torch.autograd.graph.get_gradient_edge(parameter).node
        try:
            if not torch._C._will_engine_execute_node(node):
                return None
        except RuntimeError:
            # autograd.grad captures leaf gradients without executing their
            # AccumulateGrad nodes. Preserve its returned-gradient semantics.
            return None
        value = parameter.grad
        if value is not None and (not value.is_contiguous() or value.requires_grad
                or value.dtype != get_grad_dtype(parameter) or value.shape != parameter.shape):
            return None
        if value is None:
            value = torch.zeros_like(parameter, dtype=get_grad_dtype(parameter))
            parameter.grad = value
        task.direct.add(index)
        producer = self.stream

        def wait_for_writes(gradients, task=task, producer=producer):
            if task.done is not None:
                caller = torch.cuda.current_stream(value.device)
                if caller != producer:
                    caller.wait_event(task.done)
                    value.record_stream(caller)
            return gradients

        task.hooks.append(node.register_prehook(wait_for_writes))
        return value

    def accumulate(self, indices, callback, inputs):
        gid, task = self._task()
        ready = None
        if self.stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        if self.deferred:
            task.pending.append((indices, callback, inputs, ready))
        else:
            self._accumulate(gid, task, indices, callback, inputs, ready)

    def _accumulate(self, gid, task, indices, callback, inputs, ready):
        try:
            with (
                torch.cuda.stream(self.stream) if self.stream is not None else nullcontext(),
                input_stream_lifetime(inputs, self.stream is not None),
            ):
                for index in indices:
                    if index is not None and index not in task.buffers:
                        parameter = self.originals[index]
                        buffer = self._parameter_buffer(index, task)
                        task.buffers[index] = (buffer if buffer is not None else
                                               torch.zeros_like(parameter, dtype=get_grad_dtype(parameter)))
                # Zeroing private outputs depends only on parameter metadata.
                # It can precede the producer event; callback reads cannot.
                if ready is not None:
                    self.stream.wait_event(ready)
                outputs = tuple(task.buffers.get(index) for index in indices)
                layouts = tuple(
                    ((value.data_ptr(), value.stride(), value.storage_offset()) if value is not None else None)
                    for value in outputs
                )
                returned = tuple(callback(outputs))
                if len(returned) != len(indices):
                    raise RuntimeError('Accumulation callback returned an incorrect gradient count')
                for index, value, expected, layout in zip(indices, returned, outputs, layouts):
                    if index is None:
                        continue
                    parameter = self.originals[index]
                    if (
                        value is not expected
                        or (value.data_ptr(), value.stride(), value.storage_offset()) != layout
                        or value.shape != parameter.shape
                        or value.device != parameter.device
                        or value.dtype != get_grad_dtype(parameter)
                        or value.requires_grad
                    ):
                        raise RuntimeError('Accumulation callback must preserve its supplied gradient buffers')
        except BaseException:
            self.failed = True
            for index in task.direct:
                self.originals[index].grad = None
            for hook in task.hooks:
                hook.remove()
            task.hooks.clear()
            raise
        finally:
            if self.stream is not None:
                task.done = torch.cuda.Event()
                task.done.record(self.stream)
            if self.failed:
                self._join_accumulation(task)
                self.tasks.pop(gid, None)

    def defer(self, indices, callback, inputs):
        def add(outputs):
            gradients = tuple(callback())
            if len(gradients) != len(indices):
                raise RuntimeError('Deferred weight callback returned an incorrect gradient count')
            for index, output, gradient in zip(indices, outputs, gradients):
                if index is None:
                    continue
                if gradient is None or gradient.shape != output.shape or gradient.device != output.device:
                    raise RuntimeError('Deferred weight callback must return each bound gradient')
                output.add_(gradient)
            return outputs

        self.accumulate(indices, add, inputs)

    def _join_accumulation(self, task):
        if task.done is not None:
            caller = torch.cuda.current_stream(self.device)
            if caller != self.stream:
                caller.wait_event(task.done)
                for value in task.buffers.values():
                    value.record_stream(caller)

    def gradients(self, incoming):
        gid = torch._C._current_graph_task_id()
        task = self.tasks.get(gid)
        if task is None:
            return incoming  # E.g. an empty backend retaining ordinary zero grads.
        if task.pending:
            if self.backward_ready is not None and not self.backward_ready.captured:
                raise RuntimeError('The weight backward anchor did not capture async input readiness')
            pending, task.pending = task.pending, []
            for indices, callback, inputs, ready in pending:
                self._accumulate(gid, task, indices, callback, inputs, ready)
        self.tasks.pop(gid, None)
        self._join_accumulation(task)
        result = list(incoming)
        for index, value in task.buffers.items():
            if index not in task.direct:
                result[index] = value if result[index] is None else result[index] + value
        return tuple(result)


def prepare_weight_backward(parameters):
    """Bind cooperative backend parameters inside an accumulation scope."""
    region = getattr(_active, 'compute', None)
    return region.bind(parameters) if region is not None else None
