#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Pinned host memory for offloaded tensors: NUMA placement and a buffer pool."""

from __future__ import annotations

import bisect
import logging
import os
import threading
from typing import List, Optional, Set, Tuple

import torch


_logger = logging.getLogger(__name__)

_MPOL_DEFAULT = 0
_MPOL_BIND = 2
_SYS_SET_MEMPOLICY = 238  # x86_64 and aarch64 share this syscall number


def gpu_numa_info(device_index: int) -> Tuple[Optional[int], Set[int]]:
    """``(numa_node, local_cpus)`` of a CUDA device from sysfs; ``(None, set())`` if unknown."""
    try:
        props = torch.cuda.get_device_properties(device_index)
        bus = f'{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0'
        base = f'/sys/bus/pci/devices/{bus}'
        with open(f'{base}/numa_node') as f:
            node = int(f.read().strip())
        with open(f'{base}/local_cpulist') as f:
            spec = f.read().strip()
        cpus: Set[int] = set()
        for part in spec.split(','):
            if part:
                lo, _, hi = part.partition('-')
                cpus.update(range(int(lo), int(hi or lo) + 1))
        return (node if node >= 0 else None), cpus
    except (OSError, AttributeError, ValueError):
        return None, set()


class MemPolicy:
    """Bind page allocation of the calling thread to one NUMA node while active.

    Pinned pages land on the node of the thread that first touches them.  With
    every GPU of a node streaming through pages on the remote socket the copies
    share the inter-socket link; ``set_mempolicy(MPOL_BIND)`` puts them on the
    GPU's node regardless of the process' CPU affinity (launchers often confine
    all ranks to one socket).  ``node=None`` makes the context a no-op.
    """

    def __init__(self, node: Optional[int]) -> None:
        self.node = node
        self.failed = False
        self._libc = None

    def _call(self, mode: int, node: Optional[int]) -> None:
        import ctypes
        if self._libc is None:
            self._libc = ctypes.CDLL(None, use_errno=True)
        if node is None:
            rc = self._libc.syscall(_SYS_SET_MEMPOLICY, mode, None, 0)
        else:
            words = node // 64 + 1
            mask = (ctypes.c_ulong * words)()
            mask[node // 64] = 1 << (node % 64)
            rc = self._libc.syscall(_SYS_SET_MEMPOLICY, mode, mask, words * 64 + 1)
        if rc != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))

    def __enter__(self) -> 'MemPolicy':
        if self.node is not None and not self.failed:
            try:
                self._call(_MPOL_BIND, self.node)
            except OSError as error:
                self.failed = True
                _logger.warning(f'[cpu_offload] set_mempolicy(MPOL_BIND, node {self.node}) failed: {error}')
        return self

    def __exit__(self, *exc) -> None:
        if self.node is not None and not self.failed:
            try:
                self._call(_MPOL_DEFAULT, None)
            except OSError:  # pragma: no cover
                pass


def bind_to_gpu_numa(device_index: int, *, membind: bool, affinity: bool) -> MemPolicy:
    """Return the :class:`MemPolicy` for the GPU's node and optionally pin the thread.

    Logs one line describing the decision so logs of a run show per rank which
    node the pinned memory went to.
    """
    node, cpus = gpu_numa_info(device_index)
    if node is None:
        _logger.info(f'[cpu_offload] GPU {device_index}: NUMA node unknown, pinned memory not bound')
        return MemPolicy(None)
    detail = ''
    if affinity and cpus:
        try:
            allowed = os.sched_getaffinity(0)
            target = cpus & allowed
            if target and target != allowed:
                os.sched_setaffinity(0, target)
                detail = f', thread affinity -> {len(target)} local CPUs'
            elif not target:
                detail = ', local CPUs not in the allowed set (affinity unchanged)'
        except OSError as error:  # pragma: no cover
            detail = f', affinity failed: {error}'
    _logger.info(
        f'[cpu_offload] GPU {device_index}: NUMA node {node}, '
        f'pinned memory membind={"on" if membind else "off"}{detail}')
    return MemPolicy(node if membind else None)


class PinnedPool:
    """Best-fit free list of pinned ``uint8`` slabs.

    One slab holds everything one offload group execution saves, so a step
    needs tens of slabs, not thousands of buffers.  ``cudaHostAlloc`` costs
    ~0.3 ms per MiB and never returns to a fixed-size pattern with MoE routing
    (the saved bytes change every microbatch), hence:

    * sizes are rounded up to a quantum between 1/32 and 1/16 of their
      magnitude (at most 6.25% slack) so slightly different requests share
      slabs;
    * ``acquire`` takes the smallest free slab that fits and whose last reader
      (the H2D copy recorded at ``release``) has finished;
    * ``max_bytes`` bounds the pinned memory the pool owns: at the bound the
      acquiring stream waits for the best fitting busy slab instead of
      allocating a new one (counted in ``waits``).
    """

    MIN_SLAB = 2 << 20
    ALIGN = 512

    def __init__(self, max_bytes: int = 0, mempolicy: Optional[MemPolicy] = None) -> None:
        # sorted by size: (nbytes, buf, event-of-last-reader)
        self._free: List[Tuple[int, torch.Tensor, Optional[torch.cuda.Event]]] = []
        self._sizes: List[int] = []
        self._lock = threading.Lock()
        self.max_bytes = max_bytes
        self.mempolicy = mempolicy or MemPolicy(None)
        self.allocated_bytes = 0   # owned by the pool: free + in use
        self.free_bytes = 0
        self.allocations = 0
        self.waits = 0

    @classmethod
    def slab_size(cls, nbytes: int) -> int:
        """Round ``nbytes`` up to the pool's quantum (1/32..1/16 of its magnitude, >= 2 MiB)."""
        nbytes = max(nbytes, cls.MIN_SLAB)
        quantum = max(cls.MIN_SLAB, 1 << (nbytes.bit_length() - 5))
        return (nbytes + quantum - 1) // quantum * quantum

    def _take(self, index: int) -> torch.Tensor:
        size, buf, _ = self._free.pop(index)
        del self._sizes[index]
        self.free_bytes -= size
        return buf

    def acquire(self, nbytes: int, stream: torch.cuda.Stream) -> torch.Tensor:
        """A pinned slab of at least ``nbytes`` bytes, safe to write on ``stream``."""
        need = self.slab_size(nbytes)
        with self._lock:
            start = bisect.bisect_left(self._sizes, need)
            for index in range(start, len(self._free)):
                event = self._free[index][2]
                if event is None or event.query():
                    return self._take(index)
            if self.max_bytes and self.allocated_bytes + need > self.max_bytes and start < len(self._free):
                event = self._free[start][2]
                if event is not None:
                    stream.wait_event(event)
                self.waits += 1
                return self._take(start)
        with self.mempolicy:
            buf = torch.empty(need, dtype=torch.uint8, device='cpu', pin_memory=True)
        with self._lock:
            self.allocated_bytes += need
            self.allocations += 1
        return buf

    def release(self, buf: torch.Tensor, event: Optional[torch.cuda.Event]) -> None:
        """Return ``buf``; ``event`` marks the completion of its last reader."""
        size = buf.numel()
        with self._lock:
            index = bisect.bisect_right(self._sizes, size)
            self._sizes.insert(index, size)
            self._free.insert(index, (size, buf, event))
            self.free_bytes += size

    def clear(self) -> None:
        with self._lock:
            self._free.clear()
            self._sizes.clear()
            self.allocated_bytes = self.free_bytes = self.allocations = 0
