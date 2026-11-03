#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Serve large CPU tensor allocations from huge-page-aligned anonymous mmap.

Background
----------
Merging or converting checkpoints materializes many large CPU tensors. With
the default allocator each one costs a malloc, a page-fault storm on first
touch, and for torch.zeros an explicit memset; the pages are also only
16-byte aligned, so they rarely qualify for transparent huge pages.

DirectTensorAllocator is a TorchDispatchMode that intercepts the CPU factory
ops torch.empty / torch.zeros / torch.empty_strided for tensors of at least
one huge page (2 MiB on x86-64) and backs them with a fresh
MAP_PRIVATE|MAP_ANONYMOUS mapping instead:

    - Fresh anonymous pages are zero-filled lazily by the kernel, so zeros()
      costs nothing until the pages are touched.
    - The data pointer is aligned to the huge page size and the region is
      marked MADV_HUGEPAGE, letting transparent huge pages cut page faults
      and TLB misses.
    - Every mapping is returned to the kernel as soon as the tensor dies,
      instead of lingering in the allocator's free lists.

Only direct calls of those three ops are intercepted. Tensors produced by
other ops (ones, clone, cat, arithmetic, ...) allocate their outputs inside
the C++ kernel below the mode and keep using the native allocator.

Usage
-----
    allocator = DirectTensorAllocator()
    with allocator:
        merged = torch.zeros(shape, dtype=dtype)   # aligned, lazily zeroed
        ...
    print(allocator.ram_stats)

The huge page size is read from the kernel at construction. Where transparent
huge pages are unavailable (non-Linux, compiled out, or policy "never") the
mode passes every allocation through to the native allocator, so the same code
runs unchanged, just without the benefits.
"""

import ctypes
import mmap
import sys
import threading
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = ["DirectTensorAllocator"]

_THP_SYSFS = "/sys/kernel/mm/transparent_hugepage"


def _detect_hugepage_bytes():
    """Return the transparent huge page size, or None if THP cannot be used.

    None means MADV_HUGEPAGE would be ignored: not Linux, THP compiled out (no
    sysfs directory), or the system policy set to "never". The size is the
    PMD page size, 2 MiB on x86-64 and 4 KiB-granule arm64 but e.g. 512 MiB
    on 64 KiB-granule arm64; kernels older than 4.11 lack hpage_pmd_size, so
    fall back to 2 MiB there.
    """
    if sys.platform != "linux":
        return None
    try:
        with open(f"{_THP_SYSFS}/enabled") as f:
            if "[never]" in f.read():
                return None
    except OSError:
        return None
    try:
        with open(f"{_THP_SYSFS}/hpage_pmd_size") as f:
            return int(f.read())
    except (OSError, ValueError):
        return 2 << 20


_INTERCEPTED_OPS = (
    torch.ops.aten.empty.memory_format,
    torch.ops.aten.zeros.default,
    torch.ops.aten.empty_strided.default,
)


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


class DirectTensorAllocator(TorchDispatchMode):
    """Back large CPU tensor allocations with aligned, lazily zeroed anonymous mmap.

    Public attributes:
        ram_stats: Byte counters for mappings created by this allocator.
                   live_bytes / peak_bytes track current and maximum resident
                   size; allocated_bytes / released_bytes are cumulative.
                   Each mapping counts its page-rounded usable window, an
                   upper bound on its physical memory: the alignment padding
                   around it is never touched and costs no RAM.
        hugepage_bytes: Huge page size used for alignment and as the
                   interception threshold (smaller tensors cannot use a huge
                   page and would waste up to one of padding). None when THP
                   is unavailable and the allocator is inactive.
    """

    def __init__(self):
        super().__init__()
        self.hugepage_bytes = _detect_hugepage_bytes()
        self.ram_stats = {"live_bytes": 0, "peak_bytes": 0,
                          "allocated_bytes": 0, "released_bytes": 0}
        # Tensors may be allocated and freed from any thread (DataLoader
        # workers, finalizers running wherever the last reference drops).
        self._ram_lock = threading.Lock()

    def _allocate(self, layout):
        """Return a CPU tensor with layout's shape/stride/dtype on a fresh mapping.

        layout is a meta tensor; its storage size already accounts for strides.
        """
        nbytes = layout.untyped_storage().nbytes()
        size = _round_up(nbytes, mmap.PAGESIZE)
        # mmap returns a page-aligned address but we need a huge-page-aligned
        # one, so map more than `size` and use the first aligned window inside.
        # `base` is some multiple of PAGESIZE; the distance from it to the next
        # huge page boundary is a multiple of PAGESIZE below hugepage_bytes, so
        # at most hugepage_bytes - PAGESIZE. Mapping that much extra always
        # leaves room for the whole `size`-byte window after the boundary.
        huge = self.hugepage_bytes
        mapping = mmap.mmap(-1, size + huge - mmap.PAGESIZE,
                            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        try:
            # c_ubyte.from_buffer doesn't allocate the underlying memory;
            # it just creates a view.
            # Allocation happens when the memory is actually touched (read/write).
            base = ctypes.addressof(ctypes.c_ubyte.from_buffer(mapping))
            # Bytes from `base` up to the next huge page boundary (0 if aligned).
            offset = (-base) % huge  # = (huge - base % huge) % huge
            # base + offset is the huge-page-aligned address within the mapping.
            mapping.madvise(mmap.MADV_HUGEPAGE, offset, size)
            # ctypes.c_ubyte * size is a new type: a ctypes array of size `size` bytes
            buffer = (ctypes.c_ubyte * size).from_buffer(mapping, offset)
        except BaseException:
            mapping.close()
            raise

        with self._ram_lock:
            self.ram_stats["live_bytes"] += size
            self.ram_stats["allocated_bytes"] += size
            self.ram_stats["peak_bytes"] = max(self.ram_stats["peak_bytes"], self.ram_stats["live_bytes"])

        def release():
            with self._ram_lock:
                self.ram_stats["live_bytes"] -= size
                self.ram_stats["released_bytes"] += size

        # Ownership chain: the tensor's storage holds `buffer`, and `buffer`
        # holds `mapping` through from_buffer's export. When the last tensor
        # view dies, `buffer` dies, the export is released, and `mapping` is
        # unmapped by its own deallocation. The finalizer only keeps the stats
        # in step with that; it must not reference `mapping` or `buffer`.
        weakref.finalize(buffer, release)
        return torch.frombuffer(
            buffer, dtype=layout.dtype,
            count=nbytes // layout.element_size()
        ).as_strided(layout.shape, layout.stride())

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Without THP the mode is inactive. Anything that is not a plain dense
        # CPU allocation (other devices, pinned memory, sparse layouts,
        # channels_last) also stays on the native path.
        if (self.hugepage_bytes is None
                or func not in _INTERCEPTED_OPS
                or torch.device(kwargs.get("device") or "cpu").type != "cpu"
                or kwargs.get("layout", torch.strided) != torch.strided
                or kwargs.get("pin_memory", False)
                or kwargs.get("memory_format") not in (None, torch.contiguous_format)):
            return func(*args, **kwargs)

        dtype = kwargs.get("dtype") or torch.get_default_dtype()
        # Build a meta tensor to get storage size, shape, and stride without
        # re-implementing the stride rules of each op.
        layout = (
            torch.empty_strided(args[0], args[1], dtype=dtype, device="meta")
            if func is torch.ops.aten.empty_strided.default
            else torch.empty(args[0], dtype=dtype, device="meta")
        )
        if layout.untyped_storage().nbytes() < self.hugepage_bytes:
            return func(*args, **kwargs)
        return self._allocate(layout)
