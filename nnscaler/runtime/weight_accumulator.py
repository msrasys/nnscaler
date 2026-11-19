# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Accumulate cooperative dWeights without one full gradient per fragment.

Buffers belong to one forward invocation and autograd GraphTask. Contributions
run on the named weight stream after their real inputs become ready. Ordinary
parameter edges publish the result after all contributors finish registering.
No work is retained for another microbatch or optimizer step.
"""

from contextlib import nullcontext
from dataclasses import dataclass, field

import torch

from .stream import input_stream_lifetime
from .utils import get_grad_dtype
from .weight_backward import WeightBackwardRegion


@dataclass
class _AccumulationTask:
    buffers: dict = field(default_factory=dict)
    done: object = None


class WeightGradientAccumulator(WeightBackwardRegion):
    """Combine fragment dWeights on a stream, preserving autograd publication.

    Use as a context manager around cooperative backend calls. The backend
    binds parameters with ``prepare_weight_backward`` and calls
    ``binding.accumulate(callback, inputs)`` from its first-order backward.
    The callback receives zero-initialized gradient buffers on first use and
    must ADD each contribution, returning those same buffers. Buffer dtypes
    follow each parameter's ``grad_dtype`` (or its dtype when unspecified).

    Existing ``binding.defer`` callbacks also work, with an extra copy into
    the accumulation buffers. Frozen parameters may be passed to the backend,
    but only trainable parameters belong in this region. A callback can receive
    ``None`` for an unbound/frozen parameter; its corresponding return is ignored.
    """

    def __init__(self, stream, parameters):
        if not isinstance(stream, str) or not stream or stream == 'default':
            raise ValueError('Weight accumulation requires a distinct named stream')
        parameters = tuple(parameters)
        if any(type(p) not in (torch.Tensor, torch.nn.Parameter)
               or p.layout != torch.strided or p.is_nested for p in parameters):
            raise ValueError('Weight accumulation requires ordinary dense parameters')
        super().__init__(stream, parameters)
        self.bound_indices = set()
        self.failed = False
        self.forward_closed = False
        self._scope = None

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

    def bind(self, parameters):
        if self.forward_closed and torch._C._current_graph_task_id() < 0:
            raise RuntimeError('A closed weight accumulator cannot bind another forward')
        binding = super().bind(parameters)
        if binding is not None:
            self.bound_indices.update(i for i in binding.indices if i is not None)
        return binding

    def anchor(self):
        self.close_forward(failed=True)
        raise RuntimeError('Ready weight accumulation does not use a communication anchor')

    def close_forward(self, failed=False):
        incomplete = self.enabled and not failed and len(self.bound_indices) != len(self.originals)
        super().close_forward(failed=failed or incomplete)
        self.forward_closed = True
        if incomplete:
            raise RuntimeError('Every accumulator parameter must be bound by a cooperative backend')

    def _task(self):
        gid = torch._C._current_graph_task_id()
        if gid < 0 or torch.is_grad_enabled():
            raise RuntimeError('Weight accumulation requires first-order autograd backward')
        if self.failed:
            raise RuntimeError('Cannot reuse a failed weight accumulator')
        if gid not in self.tasks:
            if self.tasks:
                raise RuntimeError('Concurrent backwards of one accumulator are unsupported')
            self.tasks[gid] = _AccumulationTask()

            def cleanup(region=self):
                task = region.tasks.pop(gid, None)
                if task is not None:
                    region._join_accumulation(task)

            torch.autograd.Variable._execution_engine.queue_callback(cleanup)
        return gid, self.tasks[gid]

    def accumulate(self, indices, callback, inputs):
        gid, task = self._task()
        ready = None
        if self.stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
        try:
            with torch.cuda.stream(self.stream) if self.stream is not None else nullcontext(), \
                    input_stream_lifetime(inputs, self.stream is not None):
                for index in indices:
                    if index is not None and index not in task.buffers:
                        parameter = self.originals[index]
                        task.buffers[index] = torch.zeros_like(parameter, dtype=get_grad_dtype(parameter))
                # Zeroing private outputs depends only on parameter metadata.
                # It can precede the producer event; callback reads cannot.
                if ready is not None:
                    self.stream.wait_event(ready)
                outputs = tuple(task.buffers.get(index) for index in indices)
                layouts = tuple((value.data_ptr(), value.stride(), value.storage_offset())
                                if value is not None else None for value in outputs)
                returned = tuple(callback(outputs))
                if len(returned) != len(indices):
                    raise RuntimeError('Accumulation callback returned an incorrect gradient count')
                for index, value, expected, layout in zip(indices, returned, outputs, layouts):
                    if index is None:
                        continue
                    parameter = self.originals[index]
                    if (value is not expected
                            or (value.data_ptr(), value.stride(), value.storage_offset()) != layout
                            or value.shape != parameter.shape or value.device != parameter.device
                            or value.dtype != get_grad_dtype(parameter) or value.requires_grad):
                        raise RuntimeError('Accumulation callback must preserve its supplied gradient buffers')
        except BaseException:
            self.failed = True
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
                if (gradient is None or gradient.shape != output.shape
                        or gradient.device != output.device):
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
        task = self.tasks.pop(gid, None)
        if task is None:
            return incoming  # E.g. an empty backend retaining ordinary zero grads.
        self._join_accumulation(task)
        result = list(incoming)
        for index, value in task.buffers.items():
            result[index] = value if result[index] is None else result[index] + value
        return tuple(result)
