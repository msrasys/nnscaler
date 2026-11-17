# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded CUDA graph replay for pure backend kernel sequences.

Backends call run_module_kernel_graph inside their own autograd forward. Misses
execute eagerly and retain detached input snapshots. Capture is explicit at a
completed training-step boundary, never while an async anchor is in flight.
Every replay refreshes all inputs and returns fresh output storage.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock, local

import torch


_scope = local()
_lock = RLock()
_pending = OrderedDict()
_graphs = {}
_max_entries = 4
_max_bytes = 512 * 1024**2  # logical input/output storage per device
_stats = dict(queued=0, captures=0, replays=0, capacity_fallbacks=0)


@contextmanager
def kernel_graph_scope(enabled):
    previous = getattr(_scope, 'enabled', False)
    _scope.enabled = enabled
    try:
        yield
    finally:
        _scope.enabled = previous


@contextmanager
def _backend_modes(precision, deterministic, warn_only):
    previous = (torch.get_float32_matmul_precision(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    torch.set_float32_matmul_precision(precision)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])


def _outputs(value):
    values = value if isinstance(value, tuple) else (value,)
    if not values or any(type(t) is not torch.Tensor or t.device.type != 'cuda' for t in values):
        raise ValueError('Kernel graphs require tensor or flat tensor-tuple outputs')
    return values, isinstance(value, tuple)


def _constant(value):
    if isinstance(value, tuple):
        return tuple(_constant(v) for v in value)
    if type(value) not in (str, bool, int, float, type(None), torch.dtype, torch.device):
        raise ValueError('Kernel graph keyword arguments must be immutable non-tensor values')
    return value


@dataclass
class _Pending:
    function: object
    inputs: tuple
    specs: tuple
    kwargs: dict
    stream: object
    output_tuple: bool
    storage_bytes: int
    autocast: bool
    autocast_dtype: object


class _Graph:
    def __init__(self, pending):
        self.stream = pending.stream
        self.lock = RLock()
        self.storage_bytes = pending.storage_bytes
        self.output_tuple = pending.output_tuple
        self.inputs = tuple(torch.empty_strided(shape, stride, dtype=dtype, device=self.stream.device,
                                               requires_grad=requires_grad)
                            for shape, stride, dtype, requires_grad in pending.specs)
        for destination, source in zip(self.inputs, pending.inputs):
            destination.copy_(source)
        # Warm library handles/JIT before capture. The caller has already
        # drained the complete training step, including pipeline transfers.
        pending.function(*self.inputs, **pending.kwargs)
        self.graph = torch.cuda.CUDAGraph()
        self.graph.capture_begin(capture_error_mode='relaxed')
        try:
            self.outputs, _ = _outputs(pending.function(*self.inputs, **pending.kwargs))
        finally:
            self.graph.capture_end()

    def replay(self, inputs):
        # Multiple host callers may share a named stream. Submit each complete
        # copy/replay/clone sequence atomically so their staging cannot mix.
        with self.lock:
            for destination, source in zip(self.inputs, inputs):
                destination.copy_(source)
            self.graph.replay()
            values = tuple(t.clone() for t in self.outputs)
        return values if self.output_tuple else values[0]


def run_module_kernel_graph(function, inputs, **kwargs):
    """Replay a pure no-grad CUDA backend function in an enabled module scope.

    The enclosing autograd function remains responsible for gradients. All
    tensor state, including quantized weights, must be explicit inputs; keyword
    arguments must be immutable. No RNG, host reads, mutations or observable
    aliases are supported. Changes to other backend configuration require
    clearing the cache at a completed-step boundary.
    """
    if not getattr(_scope, 'enabled', False):
        return function(*inputs, **kwargs)
    if torch.is_grad_enabled():
        raise RuntimeError('Kernel graph replay belongs inside a backend autograd forward or no_grad')
    if (not inputs or any(type(t) not in (torch.Tensor, torch.nn.Parameter)
                          or t.device.type != 'cuda' or t.layout != torch.strided or t.is_nested
                          for t in inputs) or len({t.device for t in inputs}) != 1):
        raise ValueError('Kernel graphs require ordinary CUDA tensor inputs on one device')
    if torch.cuda.is_current_stream_capturing():
        return function(*inputs, **kwargs)
    stream = torch.cuda.current_stream(inputs[0].device)
    for value in inputs:
        value.record_stream(stream)
    specs = tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.requires_grad) for t in inputs)
    constants = tuple(sorted((name, _constant(value)) for name, value in kwargs.items()))
    autocast, dtype = torch.is_autocast_enabled('cuda'), torch.get_autocast_dtype('cuda')
    key = (inputs[0].device, stream.cuda_stream, function, specs, constants, autocast, dtype,
           torch.is_inference_mode_enabled(), torch.get_float32_matmul_precision(),
           torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled())
    with _lock:
        graph = _graphs.get(key)
    if graph is not None:
        _stats['replays'] += 1
        return graph.replay(inputs)
    result = function(*inputs, **kwargs)
    with _lock:
        if key in _pending:
            return result
        outputs, output_tuple = _outputs(result)
        if any(t.device != inputs[0].device or t.layout != torch.strided or t.is_nested for t in outputs):
            raise ValueError('Kernel graph outputs must be dense tensors on the input device')
        size = sum(t.numel() * t.element_size() for t in (*inputs, *outputs))
        residents = [v for k, v in (*_graphs.items(), *_pending.items()) if k[0] == key[0]]
        if len(residents) >= _max_entries or size + sum(v.storage_bytes for v in residents) > _max_bytes:
            _stats['capacity_fallbacks'] += 1
            return result
        # Copies protect warmup from subsequent in-place updates, buffer reuse,
        # or pipeline pseudo-free. They retain no original autograd graph.
        snapshots = tuple(t.detach().clone() for t in inputs)
        _pending[key] = _Pending(function, snapshots, specs, dict(kwargs), stream, output_tuple,
                                 size, autocast, dtype)
        _stats['queued'] += 1
    return result


def prepare_module_kernel_graphs():
    """Capture pending kernels after a complete training/optimizer step.

    Trainer calls this automatically. Standalone callers must use this boundary
    explicitly, after all required GPU communication has been submitted.
    """
    with _lock:
        if not _pending:
            return 0
        pending = list(_pending.items())
        for device in {key[0] for key, _ in pending}:
            torch.cuda.synchronize(device)
        for key, value in pending:
            with torch.cuda.device(key[0]), torch.cuda.stream(value.stream), torch.inference_mode(key[7]), \
                    torch.no_grad(), kernel_graph_scope(False), \
                    torch.autocast('cuda', enabled=value.autocast, dtype=value.autocast_dtype), \
                    _backend_modes(*key[8:]):
                _graphs[key] = _Graph(value)
            del _pending[key]
            _stats['captures'] += 1
        return len(pending)


def module_kernel_graph_stats():
    with _lock:
        return dict(_stats, graphs=len(_graphs), pending=len(_pending),
                    logical_storage_bytes=sum(v.storage_bytes for v in _graphs.values()))


def clear_module_kernel_graphs():
    """Clear at a completed-step boundary; never call during an in-flight replay."""
    with _lock:
        for device in {key[0] for key in (*_graphs, *_pending)}:
            torch.cuda.synchronize(device)
        _graphs.clear()
        _pending.clear()
        for key in _stats:
            _stats[key] = 0
