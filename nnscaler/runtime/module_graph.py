# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CUDA replay for pure module regions with several live forwards.

Calibration observes the first optimizer step. Capture runs only at a subsequent
gradient reset, after those backwards finish. Forward state belongs to a slot;
forward graphs keep each slot's saved storage alive. Backward scratch is shared
on the execution stream and copied before another replay can overwrite it.
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


class _GraphReplay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ticket, *inputs):
        pool = ticket.owner
        ctx.ticket = ticket
        ctx.save_for_backward(*(inputs[i] for i in sorted(pool.parameters)))
        ctx.set_materialize_grads(False)
        slot = pool.slots[ticket.slot]
        prepared = getattr(ticket, 'outputs', None)
        if prepared is not None:
            del ticket.outputs  # Returned tensors must not form a ticket/autograd cycle.
        if prepared is None:
            _copy_inputs(pool, slot, inputs)
        slot['forward'].replay()
        values = (tuple(value.detach().clone() for value in slot['outputs']) if prepared is None else
                  tuple(target.copy_(value) for target, value in zip(prepared, slot['outputs'])))
        ctx.mark_non_differentiable(*(v for v, grad in zip(values, pool.output_requires_grad) if not grad))
        return values

    @staticmethod
    def backward(ctx, *gradients):
        ticket, pool = ctx.ticket, ctx.ticket.owner
        if ticket.used or torch.is_grad_enabled():
            raise RuntimeError('Module graphs require one first-order backward per forward')
        ticket.used = True
        try:
            ctx.saved_tensors
            slot = pool.slots[ticket.slot]
            if pool.capture_backward:
                with _capture_lock, pool.lock:
                    if torch.cuda.current_stream(pool.stream.device) != pool.stream:
                        raise RuntimeError('Module graph backward requires its forward stream')
                    for target, gradient in zip(pool.backward_inputs, gradients):
                        if target is not None:
                            target.zero_() if gradient is None else target.copy_(gradient)
                    slot['backward'].replay()
                    result = tuple(None if value is None else value.clone() for value in pool.backward_outputs)
                return (None, *result)
            required = tuple(x for x in slot['inputs'] if x.requires_grad)
            outputs = tuple(y for y in slot['outputs'] if y.requires_grad)
            incoming = tuple(torch.zeros_like(y) if g is None else g
                             for y, g in zip(slot['outputs'], gradients) if y.requires_grad)
            values = iter(torch.autograd.grad(outputs, required, incoming,
                                              allow_unused=True, retain_graph=True))
            result = []
            for value in slot['inputs']:
                grad = next(values) if value.requires_grad else None
                result.append(None if grad is None else grad.clone())
            return (None, *result)
        finally:
            torch.autograd.Variable._execution_engine.queue_callback(ticket.release)


def _copy_inputs(pool, slot, inputs):
    copied = set()
    for i, (target, value) in enumerate(zip(slot['inputs'], inputs)):
        if i not in pool.parameters and target.data_ptr() not in copied:
            target.copy_(value)
            copied.add(target.data_ptr())


class _PreparedForward:
    def __init__(self, pool, inputs):
        self.inputs = tuple(inputs)
        self.versions = tuple(v._version for v in inputs)
        self.ticket = pool._reserve(inputs)
        try:
            slot = pool.slots[self.ticket.slot]
            self.ticket.outputs = tuple(torch.empty_like(v) for v in slot['outputs'])
            with torch.no_grad():
                _copy_inputs(pool, slot, inputs)
        except BaseException:
            self.ticket.release()
            raise

    def run(self, inputs):
        if (self.ticket is None or len(inputs) != len(self.inputs)
                or any(a is not b for a, b in zip(inputs, self.inputs))):
            raise ValueError('A prepared module graph must consume its original inputs exactly once')
        ticket, self.ticket = self.ticket, None
        try:
            if not torch.is_grad_enabled() or tuple(v._version for v in inputs) != self.versions:
                raise RuntimeError('Prepared module graph inputs or grad mode changed before execution')
            if torch.cuda.current_stream(inputs[0].device) != ticket.owner.stream:
                raise RuntimeError('Module graph execution requires its original stream')
            with ticket.owner.lock:
                outputs = _GraphReplay.apply(ticket, *inputs)
            return outputs[0] if ticket.owner.single else outputs
        except BaseException:
            ticket.release()
            raise
        finally:
            self.inputs = ()


class ModuleGraph:
    """A fixed-capacity replay pool; construction must happen with no live calls."""

    def __init__(self, function, sample_inputs, parameters=(), *, capacity=1, stream=None, capture_backward=True):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Module graph capacity must be positive')
        if type(capture_backward) is not bool:
            raise TypeError('capture_backward must be a boolean')
        self.parameters = frozenset(parameters)
        self.capture_backward = capture_backward
        if any(type(i) is not int or i < 0 or i >= len(sample_inputs) for i in self.parameters):
            raise ValueError('Module graph parameter indices must refer to inputs')
        self.signature = _signature(sample_inputs, self.parameters)
        self.stream = stream or torch.cuda.current_stream(sample_inputs[0].device)
        self.lock = RLock()
        self.free = deque(range(capacity))
        self.slots = []
        self.backward_pool = torch.cuda.graph_pool_handle() if capture_backward else None
        self.backward_inputs = self.backward_outputs = None
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
                # Each live forward owns its saved tensors. Backwards share
                # transient storage because replays and gradient copies are
                # serialized on this pool's stream.
                for _ in range(3):
                    warm, _ = _tuple(function(*inputs))
                    needed = tuple(v for v in inputs if v.requires_grad)
                    gradients = iter(torch.autograd.grad(
                        tuple(v for v in warm if v.requires_grad), needed,
                        tuple(torch.zeros_like(v) for v in warm if v.requires_grad),
                        allow_unused=True))
                    warm_gradients = tuple(next(gradients) if v.requires_grad else None for v in inputs)
                    del gradients
                del warm
                graph = torch.cuda.CUDAGraph()
                saved = []

                def pack(value, saved=saved):
                    value = value.detach()
                    saved.append(value)
                    return value

                with (torch.autograd.graph.saved_tensors_hooks(pack, lambda v: v),
                      torch.cuda.graph(graph, stream=capture_stream)):
                    outputs, _ = _tuple(function(*inputs))
                slot = dict(inputs=inputs, forward=graph, outputs=outputs, saved=saved)
                if capture_backward:
                    self._capture_backward(slot, capture_stream, warm_gradients)
                del warm_gradients
                self.slots.append(slot)
            torch.cuda.synchronize(sample_inputs[0].device)

    def _capture_backward(self, slot, stream, warm_gradients):
        inputs, outputs = slot['inputs'], slot['outputs']
        if self.backward_inputs is None:
            self.backward_inputs = tuple(torch.zeros_like(y) if y.requires_grad else None for y in outputs)
        needed = tuple(x for x in inputs if x.requires_grad)
        differentiable = tuple(y for y in outputs if y.requires_grad)
        incoming = tuple(g for g in self.backward_inputs if g is not None)

        def gradients():
            values = iter(torch.autograd.grad(differentiable, needed, incoming,
                                              allow_unused=True))
            return tuple(next(values) if x.requires_grad else None for x in inputs)

        if self.backward_outputs is None:
            self.backward_outputs = tuple(None if v is None else torch.empty_like(v) for v in warm_gradients)
        backward = torch.cuda.CUDAGraph()
        # Compiled backwards may donate saved buffers. Capture one backward per
        # forward; the slot keeps its detached saved tensors alive for replay.
        with torch.cuda.graph(backward, stream=stream, pool=self.backward_pool):
            values = gradients()
            for target, value in zip(self.backward_outputs, values):
                if (target is None) != (value is None):
                    raise RuntimeError('Module graph gradient structure changed between slots')
                if target is not None:
                    target.copy_(value)
            del values
        slot['backward'] = backward

    @_capture_guard
    def release(self, index):
        with self.lock:
            self.free.append(index)
        with torch.cuda.stream(self.stream), input_stream_lifetime((self,), True):
            pass

    def _reserve(self, inputs):
        if torch.cuda.current_stream(inputs[0].device) != self.stream:
            raise RuntimeError('Module graph execution requires its original stream')
        if _signature(inputs, self.parameters) != self.signature:
            raise ValueError('Module graph input shapes, aliases, gradient flags or parameter storage changed')
        if not torch.is_grad_enabled():
            raise RuntimeError('Use eager execution for no-grad module calls')
        with self.lock:
            if not self.free:
                raise RuntimeError('Module graph live forwards exceeded the calibrated capacity')
            return _Ticket(self, self.free.popleft())

    @_capture_guard
    def __call__(self, *inputs):
        ticket = self._reserve(inputs)
        try:
            with self.lock:
                outputs = _GraphReplay.apply(ticket, *inputs)
        except BaseException:
            ticket.release()
            raise
        return outputs[0] if self.single else outputs


class _Profile:
    def __init__(self, function, inputs, parameters, capture_backward):
        self.function, self.parameters = function, tuple(parameters)
        self.signature = _signature(inputs, self.parameters)
        self.stream = torch.cuda.current_stream(inputs[0].device)
        self.capture_backward = capture_backward
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
            self.pool = ModuleGraph(self.function, tuple(samples), self.parameters, capacity=self.peak,
                                    stream=self.stream, capture_backward=self.capture_backward)


@_capture_guard
def prepare_module_graph(owner, key, inputs):
    """Copy inputs and reserve outputs before an independent async submission.

    Calibration keeps its ordinary path.
    Preparation creates no autograd edges and launches no model computation.
    """
    if not torch.is_grad_enabled():
        return None
    with _registry_lock:
        profile = _profiles.get(owner, {}).get(key)
    if profile is None or profile.pool is None:
        return None
    return _PreparedForward(profile.pool, inputs)


@_capture_guard
def run_module_graph(owner, key, function, inputs, parameters, *, capture_backward=True, prepared=None):
    """Run or calibrate one compiler-selected pure region."""
    if prepared is not None and not torch.is_grad_enabled():
        raise RuntimeError('Prepared module graph grad mode changed before execution')
    if not torch.is_grad_enabled() or not any(v.requires_grad for v in inputs):
        return function(*inputs)
    if torch._C._current_graph_task_id() >= 0:
        raise RuntimeError('Module graph regions cannot also be checkpointed')
    with _registry_lock:
        profiles = _profiles.setdefault(owner, {})
        if key not in profiles:
            if owner in _prepared:
                raise RuntimeError('A module graph region was first used after calibration; calibrate all training paths')
            profiles[key] = _Profile(function, inputs, parameters, capture_backward)
    if prepared is not None:
        if prepared.ticket is None or prepared.ticket.owner is not profiles[key].pool:
            raise ValueError('Prepared forward belongs to a different module graph')
        result = prepared.run(inputs)
        profiles[key].replays += 1
        return result
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
                     capacity=p.peak, replays=p.replays, capture_backward=p.capture_backward,
                     live=p.live if p.pool is None else len(p.pool.slots)-len(p.pool.free))
                for name, child in module.named_modules()
                for key, p in _profiles.get(child, {}).items()]
