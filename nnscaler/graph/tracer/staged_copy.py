#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Pinned-memory staged copies between pageable host memory and CUDA.

Tracing with ``cuda_run_cpu_offload``/``reuse_cache`` copies every executed op's
inputs to CUDA and its results and inputs back to pageable CPU memory. On some
hosts the driver's pageable path is far slower than a pinned transfer plus a host
memcpy, so large dense tensors are copied through two reusable pinned buffers.

Host-side copies use numpy on a small thread pool: filling fresh pageable pages is
limited by page faults on one thread, and ``Tensor.copy_`` wakes the whole OpenMP
pool even for small chunks, which stalls when other processes share the CPUs.

The result matches ``tensor.to(device)``: same values, shape, contiguous strides,
dtype and autograd relation (a non-leaf when the source requires grad in grad
mode). Anything else falls back to ``tensor.to(device)``.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch

from nnscaler.flags import CompileFlag


_PARALLEL_HOST_COPY_BYTES = 4 << 20
_MIN_BYTES_PER_COPY_THREAD = 1 << 20


class _Stager:
    def __init__(self, chunk_bytes: int, copy_threads: int):
        self.chunk_bytes = chunk_bytes
        self.copy_threads = max(1, copy_threads)
        self.lock = threading.Lock()
        self.buffers = None
        self.events = None
        self.pool = None

    def _host_copy(self, dst: torch.Tensor, src: torch.Tensor):
        dst, src = dst.numpy(), src.numpy()
        nbytes = src.shape[0]
        parts = min(self.copy_threads, nbytes // _MIN_BYTES_PER_COPY_THREAD)
        if nbytes < _PARALLEL_HOST_COPY_BYTES or parts < 2:
            np.copyto(dst, src)
            return
        if self.pool is None:
            self.pool = ThreadPoolExecutor(self.copy_threads, thread_name_prefix='nnscaler-staged-copy')
        step = -(-nbytes // parts)
        # numpy releases the GIL while copying.
        for future in [self.pool.submit(np.copyto, dst[o:o + step], src[o:o + step])
                       for o in range(0, nbytes, step)]:
            future.result()

    def _ensure(self):
        if self.buffers is None:
            self.buffers = [torch.empty(self.chunk_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
            self.events = [torch.cuda.Event() for _ in range(2)]

    def to_host(self, src: torch.Tensor) -> torch.Tensor:
        out = torch.empty(src.shape, dtype=src.dtype)
        src_bytes = src.detach().reshape(-1).view(torch.uint8)
        dst_bytes = out.reshape(-1).view(torch.uint8)
        total = src_bytes.numel()
        with self.lock:
            self._ensure()
            stream = torch.cuda.current_stream(src.device)
            pending = [None, None]

            def drain(k):
                self.events[k].synchronize()
                offset, length = pending[k]
                self._host_copy(dst_bytes[offset:offset + length], self.buffers[k][:length])
                pending[k] = None

            for i, offset in enumerate(range(0, total, self.chunk_bytes)):
                k = i & 1
                if pending[k] is not None:
                    drain(k)
                length = min(self.chunk_bytes, total - offset)
                with torch.cuda.stream(stream):
                    self.buffers[k][:length].copy_(src_bytes[offset:offset + length], non_blocking=True)
                self.events[k].record(stream)
                pending[k] = (offset, length)
            for k in sorted((k for k in range(2) if pending[k] is not None), key=lambda k: pending[k][0]):
                drain(k)
        return out

    def to_device(self, src: torch.Tensor, device: torch.device) -> torch.Tensor:
        out = torch.empty(src.shape, dtype=src.dtype, device=device)
        src_bytes = src.detach().reshape(-1).view(torch.uint8)
        dst_bytes = out.reshape(-1).view(torch.uint8)
        total = src_bytes.numel()
        with self.lock:
            self._ensure()
            stream = torch.cuda.current_stream(device)
            for i, offset in enumerate(range(0, total, self.chunk_bytes)):
                k = i & 1
                if i >= 2:
                    self.events[k].synchronize()
                length = min(self.chunk_bytes, total - offset)
                self._host_copy(self.buffers[k][:length], src_bytes[offset:offset + length])
                with torch.cuda.stream(stream):
                    dst_bytes[offset:offset + length].copy_(self.buffers[k][:length], non_blocking=True)
                self.events[k].record(stream)
            # The pageable `.to('cuda')` path returns after the copy; keep that contract.
            stream.synchronize()
        return out


_stager: Optional[_Stager] = None


def _get_stager() -> _Stager:
    global _stager
    if _stager is None:
        _stager = _Stager(max(1, CompileFlag.trace_staged_copy_chunk_mb) << 20,
                          CompileFlag.trace_staged_copy_threads)
    return _stager


class _StagedCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, device: torch.device):
        ctx.src_device = src.device
        if device.type == 'cpu':
            return _get_stager().to_host(src)
        return _get_stager().to_device(src, device)

    @staticmethod
    def backward(ctx, grad):
        return grad.to(ctx.src_device), None


def _target_device(device: str) -> Optional[torch.device]:
    if device == 'cpu':
        return torch.device('cpu')
    if device == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda', torch.cuda.current_device())
    return None


def _can_stage(tensor: torch.Tensor, target: Optional[torch.device]) -> bool:
    if target is None or not CompileFlag.trace_staged_copy:
        return False
    if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
        return False
    if tensor.layout != torch.strided or tensor.is_quantized or not tensor.is_contiguous():
        return False
    if tensor.is_conj() or tensor.is_neg() or torch.is_inference(tensor) or torch.is_inference_mode_enabled():
        return False
    if tensor.untyped_storage().nbytes() == 0 or tensor.numel() * tensor.element_size() < (
            CompileFlag.trace_staged_copy_min_mb << 20):
        return False
    if target.type == 'cpu':
        return tensor.device.type == 'cuda'
    return tensor.device.type == 'cpu' and not tensor.is_pinned()


def to_device(tensor: torch.Tensor, device: str) -> torch.Tensor:
    """``tensor.to(device)`` for tracing, staged through pinned memory when large."""
    target = _target_device(device)
    if not _can_stage(tensor, target):
        return tensor.to(device)
    return _StagedCopy.apply(tensor, target)
