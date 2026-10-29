#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Drop consumed file-backed input tensors from RSS without unmapping them.

Background
----------
torch.load(mmap=True) maps a checkpoint file MAP_PRIVATE and hands out
tensors as views into that mapping. Pages are faulted in from the file as the
tensors are read, and they stay resident until the whole state dict is
released. When many shards are merged one tensor at a time, the pages of
tensors already consumed keep accumulating and can exhaust RAM long before
the merge finishes.

madvise(MADV_DONTNEED) on a consumed tensor's range tells the kernel to drop
those pages now. The mapping stays valid; a later access simply refaults the
page from the file. Nothing is lost as long as the page is a clean file page,
which is why eviction is restricted to storages registered up front:

    - Anonymous memory (any tensor the program allocated itself) would read
      back as zeros.
    - A MAP_PRIVATE file page that was modified in place is a private copy;
      dropping it reverts to the file contents.

register() therefore records only dense CPU tensors that live in a file-backed
mapping and silently skips everything else (tensors in malloc'd memory, on
other devices, sparse, or empty). MADV_DONTNEED on malloc'd memory would
succeed yet zero the tensor and whatever allocator metadata shares its
boundary pages, corrupting the heap. The upshot is that both register() and
evict() may be called on anything: unregistered tensors are left untouched.
In-place modification of a registered tensor is not supported: a dropped
page may be shared with its neighbours in the same storage, so their
modifications would be lost too.

Usage
-----
    shards = [torch.load(p, mmap=True, weights_only=True) for p in paths]
    deallocator = MappedTensorDeallocator()
    deallocator.register(shards)
    for name in names:
        merged[name].copy_(shards[i][name])
        deallocator.evict(shards[i][name])  # pages go back to the kernel

On systems without madvise semantics we rely on (non-Linux), evict() is a
no-op so the same code runs unchanged.
"""

import bisect
import ctypes
import mmap
import os
import sys

import torch
from torch.multiprocessing.reductions import StorageWeakRef

__all__ = ["MappedTensorDeallocator"]


def _file_backed_ranges():
    """Return the process's file-backed mappings as sorted (start, end) pairs.

    None where /proc/self/maps is unavailable, meaning the check cannot be done
    and nothing may be registered. Read once per register() call rather than
    per tensor: a state dict holds one storage per tensor, and the file can be
    hundreds of KiB in a process with many shared libraries.
    """
    try:
        with open("/proc/self/maps") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    ranges = []
    for line in lines:
        # address perms offset dev inode [pathname]. Anonymous mappings have no
        # pathname or a bracketed pseudo-name such as [heap] or [stack]. Device
        # files are excluded too: a private /dev/zero mapping behaves like
        # anonymous memory, and /dev/shm or /dev/nvidia* are not checkpoints.
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/") and not fields[5].startswith("/dev/"):
            start, end = (int(x, 16) for x in fields[0].split("-"))
            ranges.append((start, end))
    return ranges  # /proc/self/maps is already sorted by address.


def _is_file_backed(ranges, address):
    i = bisect.bisect_right(ranges, (address, float("inf"))) - 1
    return i >= 0 and ranges[i][0] <= address < ranges[i][1]


class MappedTensorDeallocator:
    """Registry of file-backed input storages whose pages may be dropped."""

    def __init__(self):
        # {storage data_ptr: StorageWeakRef}. Keyed by storage address so that
        # any view of a registered tensor resolves to it.
        self._storages = {}
        # mmap.mmap.madvise only works on mappings Python created. Evicting
        # torch's own file mappings needs the raw libc call.
        self._libc = None
        if sys.platform == "linux":
            self._libc = ctypes.CDLL(None, use_errno=True)
            self._libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
            self._libc.madvise.restype = ctypes.c_int

    def register(self, obj):
        """Record the storages of all dense CPU tensors nested in obj.

        obj may be a tensor or any nesting of dict / list / tuple of tensors,
        e.g. a loaded state dict. Only registered storages are eligible for
        evict(). Tensors that cannot be evicted safely (not CPU, not dense,
        empty, or not in a file-backed mapping) are skipped silently; see the
        module docstring for why.
        """
        self._register(obj, _file_backed_ranges())

    def _register(self, obj, ranges):
        if isinstance(obj, torch.Tensor):
            if obj.device.type != "cpu" or obj.layout != torch.strided:
                return
            storage = obj.untyped_storage()
            if storage.nbytes() == 0:
                return                  # Nothing to evict, and data_ptr may be null.
            # Fail closed: without the mapping table we cannot prove the storage
            # is file-backed, and evicting anything else destroys data.
            if ranges is None or not _is_file_backed(ranges, storage.data_ptr()):
                return
            self._storages[storage.data_ptr()] = StorageWeakRef(storage)
        elif isinstance(obj, dict):
            for value in obj.values():
                self._register(value, ranges)
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                self._register(value, ranges)

    def evict(self, tensor):
        """Drop the pages spanned by tensor from RSS.

        Silently ignored when tensor's storage was not registered (see the
        module docstring for why) or on platforms without madvise.
        """
        # Same guards as register(): sparse tensors have no untyped_storage().
        if (self._libc is None or tensor.device.type != "cpu"
                or tensor.layout != torch.strided or tensor.numel() == 0):
            return
        storage = tensor.untyped_storage()
        source = self._storages.get(storage.data_ptr())
        # The address may have been reused by a different storage after the
        # registered one died, hence the cdata identity check.
        if source is None or source.expired() or source.cdata != storage._cdata:
            return
        # Byte span of the view, valid for non-contiguous strides as well:
        # In pytorch, stride always non-negative.
        # element_ptr = data_ptr + (i0 * stride0 + i1 * stride1 + ...) * element_size
        # so the last element sits at sum((dim - 1) * stride) elements from data_ptr.
        last = sum((dim - 1) * stride for dim, stride in zip(tensor.shape, tensor.stride()))
        address = tensor.data_ptr()
        end = address + (last + 1) * tensor.element_size()
        # madvise needs a page-aligned start; widen the range to whole pages.
        start = address // mmap.PAGESIZE * mmap.PAGESIZE
        length = (end + mmap.PAGESIZE - 1) // mmap.PAGESIZE * mmap.PAGESIZE - start
        if self._libc.madvise(start, length, mmap.MADV_DONTNEED):
            error = ctypes.get_errno()
            raise OSError(error, f"madvise: {os.strerror(error)}")
