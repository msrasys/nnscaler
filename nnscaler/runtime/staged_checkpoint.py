# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Checkpoint replay split across forward and backward asynchronous boundaries.

The original forward runs whole under no_grad. A shape-stable replay prefix
may be pre-enqueued behind a progress gate; its remaining graph is built after
the backward anchor's inputs are ready. Each state belongs to one invocation.
All attributes must be explicit inputs, including trainable parameters.
"""
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from threading import Lock, RLock
import weakref

import torch
from torch.autograd.function import once_differentiable
from torch.utils import _pytree

from .checkpoint_prefetch import device_budget, frame_from_hook
from .cuda_gate import Gate
from .deferred import BackwardLaunchPoint
from .stream import _get_module_stream, input_stream_lifetime, side_compute_context

_estimates = OrderedDict()
_estimates_lock = RLock()
_MAX_ESTIMATES = 4096


def _shape_key(value):
    leaves, spec = _pytree.tree_flatten(value)
    result = []
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            if leaf.layout != torch.strided or leaf.is_nested:
                raise ValueError('Staged replay requires ordinary strided tensor inputs')
            result.append((tuple(leaf.shape), tuple(leaf.stride()), leaf.storage_offset(),
                           leaf.dtype, leaf.device, leaf.requires_grad,
                           isinstance(leaf, torch.nn.Parameter)))
        elif leaf is None or type(leaf) in (str, bool, int, float, complex, torch.dtype, torch.device):
            result.append((type(leaf), leaf))
        else:
            raise TypeError('Replay prefix inputs must be tensors or immutable builtin metadata')
    return repr(spec), tuple(result)


class ReplayReady(BackwardLaunchPoint):
    def __init__(self, cuda):
        super().__init__(())
        self.cuda = cuda

    def record_ready(self):
        if self.cuda:
            super().record_ready()
        else:
            self.ready = True

    def wait(self, stream=None):
        if not self.captured or self.ready is None:
            raise RuntimeError('Staged replay anchor did not record backward input readiness')
        if self.cuda:
            super().wait(stream)


class StagedCheckpoint:
    def __init__(self, name, budget_bytes):
        self.name, self.budget_bytes = name, budget_bytes
        self.gate = self.done = self.lease = None
        self.inputs = self.replayed = self.prefix_values = None
        self.backward_point = None
        self.function = None
        self.backward_lock = Lock()
        self.stats = dict(launched=0, reused=0, cold_skips=0, budget_skips=0,
                          saved_bytes=0, gate_watchdog_fallback=0)

    def set_stages(self, prefix, tail, key_inputs=None, output_grad_flags=None):
        if self.function is not None:
            raise RuntimeError('Cannot change stages after checkpoint execution')
        self.prefix, self.tail, self.key_inputs = prefix, tail, key_inputs
        self.output_grad_flags = output_grad_flags

    def checkpoint(self, function, *args):
        if self.function is not None:
            raise RuntimeError('A staged checkpoint belongs to one forward invocation')
        if not torch.is_grad_enabled():
            return function(*args)
        hooks = torch._C._autograd._top_saved_tensors_default_hooks(False)
        if hooks is not None:
            try:
                frame_from_hook(hooks[0])
            except RuntimeError:
                pass
            else:
                raise RuntimeError('Staged replay does not support nested checkpoints')
        self.function = function
        self.flat, self.spec = _pytree.tree_flatten(args)
        self.positions = [i for i, t in enumerate(self.flat) if isinstance(t, torch.Tensor)]
        values = tuple(self.flat[i] for i in self.positions)
        if any(type(t) not in (torch.Tensor, torch.nn.Parameter) or t.layout != torch.strided or t.is_nested for t in values):
            raise ValueError('Staged replay requires ordinary strided tensor inputs')
        if any(t.device.type not in ('cpu', 'cuda') for t in values):
            raise ValueError('Staged replay supports CPU validation and CUDA execution')
        devices = {t.device for t in values if t.is_cuda}
        if len(devices) > 1:
            raise ValueError('Staged replay requires one CUDA device')
        self.device = next(iter(devices)) if devices else torch.device('cpu')
        self.cuda = self.device.type == 'cuda'
        if self.cuda:
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('Staged replay requires eager CUDA execution')
        self.stream = _get_module_stream(self.device, self.name) if self.cuda else None
        self.parameter_inputs = tuple(isinstance(t, torch.nn.Parameter) for t in values)
        self.input_dtypes = tuple(getattr(t, 'grad_dtype', None) if t.is_leaf else None for t in values)
        self.requires_grad = tuple(t.requires_grad for t in values)
        self.autocast = (torch.is_autocast_enabled(self.device.type), torch.get_autocast_dtype(self.device.type))
        key_args = args if self.key_inputs is None else tuple(args[i] for i in self.key_inputs)
        self.key = (self.prefix.__code__, self.device, _shape_key(key_args), self.autocast,
                    torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled())
        self.budget = device_budget(self.device, self.budget_bytes)
        self.cpu_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state(self.device) if self.cuda else None
        # Original edges live in save_for_backward, not in a Python cycle.
        for i in self.positions:
            self.flat[i] = None
        result = _Checkpoint.apply(self, *values)
        flat = list(self.output_constants)
        for i, value in zip(self.output_positions, result):
            flat[i] = value
        return _pytree.tree_unflatten(flat, self.output_spec)

    def arguments(self, values):
        flat = list(self.flat)
        for i, value in zip(self.positions, values):
            flat[i] = value
        return _pytree.tree_unflatten(flat, self.spec)

    @contextmanager
    def rng(self, cpu, cuda):
        with torch.random.fork_rng(devices=[self.device] if self.cuda else []), torch.enable_grad(), \
                torch.autocast(self.device.type, enabled=self.autocast[0], dtype=self.autocast[1]):
            torch.set_rng_state(cpu)
            if self.cuda:
                torch.cuda.set_rng_state(cuda, self.device)
            yield

    def rebuild_prefix(self, values):
        inputs = [torch.nn.Parameter(t.detach(), requires_grad=req) if parameter
                  else t.detach().requires_grad_(req)
                  for t, req, parameter in zip(values, self.requires_grad, self.parameter_inputs)]
        for tensor, dtype in zip(inputs, self.input_dtypes):
            if dtype is not None:
                tensor.grad_dtype = dtype
        input_storage = {t.untyped_storage()._cdata for t in inputs}
        saved = {}
        device = self.device  # Saved-tensor hooks must not retain this state.

        def account(tensor):
            storage = tensor.untyped_storage()
            if tensor.device == device and storage._cdata not in input_storage:
                saved[storage._cdata] = storage.nbytes()

        def pack(tensor):
            account(tensor)
            return tensor.detach()

        with self.rng(self.cpu_rng, self.cuda_rng), \
                torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            self.prefix_values = self.prefix(*self.arguments(inputs))
            self.tail_cpu_rng = torch.get_rng_state()
            self.tail_cuda_rng = torch.cuda.get_rng_state(self.device) if self.cuda else None
        for tensor in _pytree.tree_leaves(self.prefix_values):
            if torch.is_tensor(tensor):
                account(tensor)
        self.inputs, self.replayed = inputs, ()
        size = sum(saved.values())
        self.stats['saved_bytes'] = size
        with _estimates_lock:
            _estimates[self.key] = size
            _estimates.move_to_end(self.key)
            while len(_estimates) > _MAX_ESTIMATES:
                _estimates.popitem(last=False)
        if self.lease is not None and size > self.lease.size:
            raise RuntimeError('Staged prefix storage grew beyond its learned bound; data-dependent allocation is unsupported')

    def rebuild_tail(self):
        with self.rng(self.tail_cpu_rng, self.tail_cuda_rng):
            outputs = self.tail(self.prefix_values)
        flat, spec = _pytree.tree_flatten(outputs)
        if spec != self.output_spec or any(not torch.is_tensor(flat[i]) for i in self.output_positions):
            raise RuntimeError('Staged checkpoint output structure changed during replay')
        # Keep graph edges; outputs that no backward operator saves need not
        # retain their storage merely to initiate autograd.grad later.
        self.replayed = tuple(torch.autograd.graph.get_gradient_edge(flat[i])
                              if flat[i].requires_grad else None for i in self.output_positions)

    def record_done(self):
        if self.cuda:
            self.done = torch.cuda.Event()
            self.done.record(torch.cuda.current_stream(self.device))
            if self.lease is not None:
                self.lease.event = self.done

    def prequeue(self, values):
        with _estimates_lock:
            estimate = _estimates.get(self.key)
        if estimate is None:
            self.stats['cold_skips'] += 1
            return
        self.lease = self.budget.reserve(estimate)
        if self.lease is None:
            self.stats['budget_skips'] += 1
            return
        try:
            if self.cuda:
                self.stream.wait_stream(torch.cuda.current_stream(self.device))
                self.gate = Gate(self.device)
            with torch.cuda.stream(self.stream) if self.cuda else nullcontext(), \
                    side_compute_context(), input_stream_lifetime((values, self.flat), self.cuda):
                try:
                    if self.gate:
                        self.gate.wait(self.stream)
                    self.rebuild_prefix(values)
                finally:
                    self.record_done()
            self.stats['launched'] += 1
        except BaseException:
            if self.gate:
                self.gate.open()
            self.clear()
            raise

    def prefetch(self):
        if self.gate:
            self.gate.open(torch.cuda.current_stream(self.device))
        return self.replayed is not None

    def finish_forward(self):
        if self.done is not None:
            if self.gate and not self.gate.queued:
                self.gate.open()
            torch.cuda.current_stream(self.device).wait_event(self.done)
        if self.gate:
            self.gate.finish()

    def clear(self):
        self.inputs = self.replayed = self.prefix_values = None
        if self.lease is not None:
            self.lease.release(self.done)
            self.lease = None

    def __del__(self):
        try:
            if self.gate:
                self.gate.open()
        finally:
            self.clear()


def _release_abandoned(reference):
    state = reference()
    if state is not None:
        try:
            if state.gate:
                state.gate.open()
                state.gate.finish()
        finally:
            state.clear()


class _Checkpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, *values):
        ctx.state = state
        finalizer = weakref.finalize(ctx, _release_abandoned, weakref.ref(state))
        finalizer.atexit = False
        ctx.save_for_backward(*values)
        ctx.set_materialize_grads(False)
        result = state.function(*state.arguments(values))
        if state.output_grad_flags is not None:
            values_out = (result,) if len(state.output_grad_flags) == 1 else result
            nondifferentiable = [t for value, differentiable in zip(values_out, state.output_grad_flags)
                                 if not differentiable for t in _pytree.tree_leaves(value) if torch.is_tensor(t)]
            ctx.mark_non_differentiable(*nondifferentiable)
        flat, state.output_spec = _pytree.tree_flatten(result)
        state.output_positions = [i for i, t in enumerate(flat) if torch.is_tensor(t)]
        result = tuple(flat[i] for i in state.output_positions)
        state.output_constants = [None if torch.is_tensor(t) else t for t in flat]
        if state.cuda:
            state.forward_done = torch.cuda.Event()
            state.forward_done.record(torch.cuda.current_stream(state.device))
        state.prequeue(values)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, *grads):
        state = ctx.state
        if not state.backward_lock.acquire(blocking=False):
            raise RuntimeError('Concurrent backward of one staged checkpoint is unsupported')
        try:
            values = ctx.saved_tensors  # Check versions even for a prefetched graph.
            prefetched = state.replayed is not None
            with torch.cuda.stream(state.stream) if state.cuda else nullcontext(), \
                    side_compute_context(), input_stream_lifetime((values, state.flat), state.cuda):
                try:
                    if state.backward_point is not None:
                        state.backward_point.wait(state.stream)
                    if not prefetched:
                        if state.cuda:
                            state.stream.wait_event(state.forward_done)
                        state.rebuild_prefix(values)
                    else:
                        state.stats['reused'] += 1
                    state.rebuild_tail()
                finally:
                    state.record_done()
            if state.cuda:
                torch.cuda.current_stream(state.device).wait_event(state.done)
            selected = [(t, g) for t, g in zip(state.replayed, grads) if t is not None and g is not None]
            inputs = [t for t in state.inputs if t.requires_grad]
            if selected and inputs:
                outputs, gradients = zip(*selected)
                results = torch.autograd.grad(outputs, inputs, gradients, allow_unused=True)
            else:
                results = (None,) * len(inputs)
            it = iter(results)
            result = tuple(next(it) if req else None for req in state.requires_grad)
            state.record_done()
            state.stats['gate_watchdog_fallback'] = int(state.gate.fallback()) if state.gate else 0
            return (None, *result)
        finally:
            state.clear()
            state.backward_lock.release()
