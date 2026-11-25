# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CUDA replay for pure module regions with several live forwards.

Calibration observes the first optimizer step. Capture runs only at a subsequent
gradient reset, after those backwards finish. Forward state belongs to a slot;
native forward/backward graphs keep each slot's captured storage alive.
"""

from collections import deque
from contextlib import ExitStack
from functools import wraps
from threading import RLock
from weakref import WeakKeyDictionary, WeakSet

import torch

from .stream import input_stream_lifetime
from .utils import get_grad_dtype, set_grad_dtype

_profiles = WeakKeyDictionary()
_prepared = WeakSet()
_contexts = []
_capture_lock = RLock()
_registry_lock = RLock()


def _capture_guard(function):
    @wraps(function)
    def guarded(*args, **kwargs):
        with _capture_lock:
            return function(*args, **kwargs)
    return guarded


def register_capture_context(factory):
    """Register a backend context that makes warmup/capture self-contained.

    A cache of derived weights, for example, must not omit weight-update work
    from a recorded graph. The factory must restore its state on exceptions.
    """
    if not callable(factory):
        raise TypeError('A capture context factory must be callable')
    if factory not in _contexts:
        _contexts.append(factory)


def _tuple(outputs):
    values = (outputs,) if isinstance(outputs, torch.Tensor) else outputs
    if not isinstance(values, tuple) or not values or any(not isinstance(v, torch.Tensor) for v in values):
        raise ValueError('Module graphs require a tensor or nonempty flat tuple of tensors')
    return values, isinstance(outputs, torch.Tensor)


def _signature(inputs, parameters):
    aliases, result = {}, []
    for i, value in enumerate(inputs):
        if type(value) not in (torch.Tensor, torch.nn.Parameter) or not value.is_cuda or not value.is_contiguous():
            raise ValueError('Module graphs require ordinary contiguous CUDA tensor inputs')
        if i in parameters and not value.is_leaf:
            raise ValueError('Module graph parameters must be direct leaf tensors')
        identity = (value.data_ptr(), tuple(value.shape), tuple(value.stride()))
        alias = aliases.setdefault(identity, len(aliases))
        result.append((tuple(value.shape), value.dtype, value.device, value.requires_grad,
                       alias, identity if i in parameters else None, get_grad_dtype(value) if i in parameters else None))
    if len({v.device for v in inputs}) != 1:
        raise ValueError('A module graph uses exactly one CUDA device')
    return tuple(result)


class _Ticket:
    def __init__(self, owner, slot=None):
        self.owner, self.slot = owner, slot
        self.released = False
        self.used = False

    def release(self):
        if not self.released:
            self.released = True
            self.owner.release(self.slot)

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass  # Interpreter/CUDA teardown cannot propagate destructor errors.


class _Calibrate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ticket, *outputs):
        ctx.ticket = ticket
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(*(v for v in outputs if not v.requires_grad))
        return tuple(v.clone() for v in outputs)

    @staticmethod
    def backward(ctx, *grads):
        if ctx.ticket.used or torch.is_grad_enabled():
            raise RuntimeError('Module graphs require one first-order backward per forward')
        ctx.ticket.used = True
        torch.autograd.Variable._execution_engine.queue_callback(ctx.ticket.release)
        return (None, *grads)


def _clone_gradients(gradients, _outputs):
    # Native graphed callables return their static gradient buffers. Clone before
    # the outer engine hands them to consumers or another forward reuses a slot.
    return tuple(None if value is None else value.clone() for value in gradients)


class _Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ticket, *outputs):
        ctx.ticket = ticket
        ctx.save_for_backward(*ticket.parameters)
        ctx.set_materialize_grads(False)
        values = tuple(value.clone() for value in outputs)
        ctx.mark_non_differentiable(*(v for v, grad in zip(values, ticket.owner.output_requires_grad) if not grad))
        return values

    @staticmethod
    def backward(ctx, *gradients):
        if ctx.ticket.used or torch.is_grad_enabled():
            raise RuntimeError('Module graphs require one first-order backward per forward')
        ctx.ticket.used = True
        try:
            ctx.saved_tensors  # Check the model's parameter versions.
            return (None, *gradients)
        finally:
            torch.autograd.Variable._execution_engine.queue_callback(ctx.ticket.release)


class ModuleGraph:
    """A fixed-capacity replay pool; construction must happen with no live calls."""

    def __init__(self, function, sample_inputs, parameters=(), *, capacity=1, stream=None):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Module graph capacity must be positive')
        self.parameters = frozenset(parameters)
        if any(type(i) is not int or i < 0 or i >= len(sample_inputs) for i in self.parameters):
            raise ValueError('Module graph parameter indices must refer to inputs')
        self.signature = _signature(sample_inputs, self.parameters)
        self.stream = stream or torch.cuda.current_stream(sample_inputs[0].device)
        self.lock = RLock()
        self.free = deque(range(capacity))
        self.slots = []
        capture_stream = (torch.cuda.Stream(device=sample_inputs[0].device)
                          if self.stream == torch.cuda.default_stream(sample_inputs[0].device) else self.stream)
        with _capture_lock, torch.cuda.stream(capture_stream), torch.enable_grad(), ExitStack() as stack:
            torch.cuda.synchronize(sample_inputs[0].device)
            for factory in _contexts:
                stack.enter_context(factory())
            stack.enter_context(torch.random.fork_rng(devices=[sample_inputs[0].device.index]))
            for _ in range(capacity):
                buffers, inputs = {}, []
                for i, original in enumerate(sample_inputs):
                    if i in self.parameters:
                        value = torch.nn.Parameter(original.detach(), requires_grad=original.requires_grad)
                        set_grad_dtype(value, get_grad_dtype(original))
                    else:
                        key = self.signature[i][4]
                        if key not in buffers:
                            buffers[key] = torch.zeros_like(original)
                        value = buffers[key].detach().requires_grad_(original.requires_grad)
                    inputs.append(value)
                inputs = tuple(inputs)
                outputs, self.single = _tuple(function(*inputs))
                self.output_requires_grad = tuple(v.requires_grad for v in outputs)
                if not any(self.output_requires_grad):
                    raise ValueError('Module graphs require a differentiable output')
                del outputs
                native = torch.cuda.make_graphed_callables(function, inputs, allow_unused_input=True)
                self.slots.append(dict(inputs=inputs, native=native))
            torch.cuda.synchronize(sample_inputs[0].device)

    @_capture_guard
    def release(self, index):
        with self.lock:
            self.free.append(index)
        with torch.cuda.stream(self.stream), input_stream_lifetime((self,), True):
            pass

    @_capture_guard
    def __call__(self, *inputs):
        if torch.cuda.current_stream(inputs[0].device) != self.stream:
            raise RuntimeError('Module graph execution requires its original stream')
        if _signature(inputs, self.parameters) != self.signature:
            raise ValueError('Module graph input shapes, aliases, gradient flags or parameter storage changed')
        if not torch.is_grad_enabled():
            raise RuntimeError('Use eager execution for no-grad module calls')
        with self.lock:
            if not self.free:
                raise RuntimeError('Module graph live forwards exceeded the calibrated capacity')
            ticket = _Ticket(self, self.free.popleft())
        try:
            with self.lock:
                outputs, _ = _tuple(self.slots[ticket.slot]['native'](*inputs))
                nodes = {value.grad_fn for value in outputs if value.grad_fn is not None}
                for node in nodes:
                    node.register_hook(_clone_gradients)
                ticket.parameters = tuple(inputs[i] for i in sorted(self.parameters))
                outputs = _Replay.apply(ticket, *outputs)
        except BaseException:
            ticket.release()
            raise
        return outputs[0] if self.single else outputs


class _Profile:
    def __init__(self, function, inputs, parameters):
        self.function, self.parameters = function, tuple(parameters)
        self.signature = _signature(inputs, self.parameters)
        self.stream = torch.cuda.current_stream(inputs[0].device)
        self.templates = tuple(value if i in parameters else None for i, value in enumerate(inputs))
        self.live = self.peak = 0
        self.replays = 0
        self.pool = None
        self.lock = RLock()

    def release(self, _):
        with self.lock:self.live -= 1

    def run(self, inputs):
        if self.pool is not None:
            result = self.pool(*inputs)
            self.replays += 1
            return result
        if _signature(inputs, self.parameters) != self.signature or torch.cuda.current_stream(inputs[0].device) != self.stream:
            raise ValueError('Module graph calibration requires fixed input metadata, aliases and stream')
        with self.lock:
            self.live += 1
            self.peak = max(self.peak, self.live)
            ticket = _Ticket(self)
        try:
            outputs, single = _tuple(self.function(*inputs))
            outputs = _Calibrate.apply(ticket, *outputs)
        except BaseException:
            ticket.release()
            raise
        return outputs[0] if single else outputs

    def prepare(self):
        if self.pool is not None or self.live:
            return
        buffers, samples = {}, []
        with torch.cuda.stream(self.stream):
            for spec, template in zip(self.signature, self.templates):
                shape, dtype, device, requires_grad, alias, _, _ = spec
                if template is None:
                    if alias not in buffers:buffers[alias] = torch.zeros(shape, dtype=dtype, device=device)
                    template = buffers[alias].detach().requires_grad_(requires_grad)
                samples.append(template)
            self.pool = ModuleGraph(self.function, tuple(samples), self.parameters, capacity=self.peak, stream=self.stream)


@_capture_guard
def run_module_graph(owner, key, function, inputs, parameters):
    """Run or calibrate one compiler-selected pure region."""
    if not torch.is_grad_enabled() or not any(v.requires_grad for v in inputs):
        return function(*inputs)
    if torch._C._current_graph_task_id() >= 0:
        raise RuntimeError('Module graph regions cannot also be checkpointed')
    with _registry_lock:
        profiles = _profiles.setdefault(owner, {})
        if key not in profiles:
            if owner in _prepared:
                raise RuntimeError('A module graph region was first used after calibration; calibrate all training paths')
            profiles[key] = _Profile(function, inputs, parameters)
    return profiles[key].run(inputs)


@_capture_guard
def prepare_module_graphs(module, *, synchronize=False):
    """Capture after gradient reset, with no live forwards on any pipeline rank.

    Distributed optimizer integration uses one collective before capture and a
    barrier afterwards, so no peer can enter its next transport while another
    rank is still capturing. Ordinary training steps need neither collective.
    """
    with _registry_lock:
        profiles = [p for child in module.modules() for p in _profiles.get(child, {}).values()]
    pending = any(p.pool is None for p in profiles)
    busy = any(p.live or (p.pool is not None and len(p.pool.free) != len(p.pool.slots)) for p in profiles)
    distributed = synchronize and torch.distributed.is_initialized()
    if distributed:
        flags = torch.tensor([pending, busy], dtype=torch.int32, device=torch.cuda.current_device())
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
        pending, busy = flags.tolist()
    if busy or not pending:
        return bool(profiles) and not pending and not busy
    with _capture_lock:
        for profile in profiles:profile.prepare()
    if distributed:
        torch.distributed.barrier()
    with _registry_lock:
        _prepared.update(module.modules())
    return True


def module_graph_stats(module):
    """Return calibration and replay counts without synchronizing the GPU."""
    with _registry_lock:
        return [dict(module=name, region=key, captured=p.pool is not None,
                     capacity=p.peak, replays=p.replays,
                     live=p.live if p.pool is None else len(p.pool.slots)-len(p.pool.free))
                for name, child in module.named_modules()
                for key, p in _profiles.get(child, {}).items()]
