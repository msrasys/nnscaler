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

import ctypes
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np
import torch

from nnscaler.flags import CompileFlag


_PARALLEL_HOST_COPY_BYTES = 4 << 20
_MIN_BYTES_PER_COPY_THREAD = 1 << 20
_MADV_HUGEPAGE = 14
_HUGE_PAGE_BYTES = 2 << 20
_madvise = None


def _advise_huge_pages(tensor: torch.Tensor) -> None:
    """Best effort: back a fresh host buffer with transparent huge pages.

    First-touch faults dominate filling fresh pageable memory; 2 MiB pages cut
    them ~512x (about 2x copy throughput with THP in `madvise` mode). Memory
    use is unchanged: only whole aligned 2 MiB ranges inside the mapping qualify.
    """
    global _madvise
    if not CompileFlag.trace_staged_copy_huge_pages or not sys.platform.startswith('linux'):
        return
    if _madvise is None:
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
            _madvise = libc.madvise
        except (OSError, AttributeError):
            _madvise = False
    if not _madvise:
        return
    start = (tensor.data_ptr() + _HUGE_PAGE_BYTES - 1) & ~(_HUGE_PAGE_BYTES - 1)
    end = (tensor.data_ptr() + tensor.untyped_storage().nbytes()) & ~(_HUGE_PAGE_BYTES - 1)
    if end > start:
        _madvise(start, end - start, _MADV_HUGEPAGE)  # failure only loses the speedup


_copy_pool: Optional[ThreadPoolExecutor] = None
_copy_pool_lock = threading.Lock()


def _host_copy(dst: torch.Tensor, src: torch.Tensor, threads: int):
    dst, src = dst.numpy(), src.numpy()
    nbytes = src.shape[0]
    parts = min(max(1, threads), nbytes // _MIN_BYTES_PER_COPY_THREAD)
    if nbytes < _PARALLEL_HOST_COPY_BYTES or parts < 2:
        np.copyto(dst, src)
        return
    global _copy_pool
    with _copy_pool_lock:
        if _copy_pool is None:
            _copy_pool = ThreadPoolExecutor(max(1, threads), thread_name_prefix='nnscaler-staged-copy')
    step = -(-nbytes // parts)
    # numpy releases the GIL while copying.
    for future in [_copy_pool.submit(np.copyto, dst[o:o + step], src[o:o + step])
                   for o in range(0, nbytes, step)]:
        future.result()


def _byte_view(tensor: torch.Tensor) -> torch.Tensor:
    # Contiguous tensors (any strides on size-1 dims) store their elements in
    # one dense span in logical order.
    return tensor.detach().reshape(-1).view(torch.uint8)


class _Stager:
    """Two pinned buffers and their events for transfers with one CUDA device."""

    def __init__(self, device: torch.device, chunk_bytes: int, copy_threads: int):
        self.device = device
        self.chunk_bytes = chunk_bytes
        self.copy_threads = copy_threads
        self.lock = threading.Lock()
        # Explicit device: a `torch.device('cuda')` context must not redirect host buffers.
        self.buffers = [torch.empty(chunk_bytes, dtype=torch.uint8, device='cpu', pin_memory=True)
                        for _ in range(2)]
        with torch.cuda.device(device):
            self.events = [torch.cuda.Event() for _ in range(2)]

    def to_host(self, src: torch.Tensor) -> torch.Tensor:
        # Same strides as `Tensor.to` keeps for dense tensors (size-1 dims included).
        out = torch.empty_strided(src.size(), src.stride(), dtype=src.dtype, device='cpu')
        _advise_huge_pages(out)
        src_bytes, dst_bytes = _byte_view(src), _byte_view(out)
        total = src_bytes.numel()
        with self.lock:
            stream = torch.cuda.current_stream(self.device)
            pending = [None, None]

            def drain(k):
                self.events[k].synchronize()
                offset, length = pending[k]
                _host_copy(dst_bytes[offset:offset + length], self.buffers[k][:length], self.copy_threads)
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

    def to_device(self, src: torch.Tensor) -> torch.Tensor:
        out = torch.empty_strided(src.size(), src.stride(), dtype=src.dtype, device=self.device)
        src_bytes, dst_bytes = _byte_view(src), _byte_view(out)
        total = src_bytes.numel()
        with self.lock:
            stream = torch.cuda.current_stream(self.device)
            for i, offset in enumerate(range(0, total, self.chunk_bytes)):
                k = i & 1
                if i >= 2:
                    self.events[k].synchronize()
                length = min(self.chunk_bytes, total - offset)
                _host_copy(self.buffers[k][:length], src_bytes[offset:offset + length], self.copy_threads)
                with torch.cuda.stream(stream):
                    dst_bytes[offset:offset + length].copy_(self.buffers[k][:length], non_blocking=True)
                self.events[k].record(stream)
            # The pageable `.to('cuda')` path returns after the copy; keep that contract.
            stream.synchronize()
        return out


# One stager per CUDA device: CUDA events cannot be recorded on another device's stream.
_stagers: Dict[int, _Stager] = {}
_stagers_lock = threading.Lock()


def _get_stager(device: torch.device) -> _Stager:
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _stagers_lock:
        stager = _stagers.get(index)
        if stager is None:
            stager = _stagers[index] = _Stager(
                torch.device('cuda', index),
                max(1, CompileFlag.trace_staged_copy_chunk_mb) << 20,
                CompileFlag.trace_staged_copy_threads,
            )
    return stager


class _StagedCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, src: torch.Tensor, device: torch.device):
        ctx.src_device = src.device
        if device.type == 'cpu':
            return _get_stager(src.device).to_host(src)
        return _get_stager(device).to_device(src)

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
