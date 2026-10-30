# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Process-wide pinned activation buffers and host-memory accounting."""
from collections import Counter
from bisect import bisect_left, insort
import os
from pathlib import Path
import threading
import torch


def _cgroup_headroom(path):
    """Estimate allocatable bytes without treating clean inactive cache as live.

    Only inactive file pages are considered reclaimable, conservatively minus
    all dirty/writeback file pages. Active cache, anonymous and pinned memory
    remain charged. No cache eviction is requested by this function.
    """
    limit = (path / 'memory.max').read_text().strip()
    if limit == 'max':
        return None
    used = int((path / 'memory.current').read_text())
    reclaimable = 0
    try:
        stats = {key: int(value) for key, value in
                 (line.split() for line in (path / 'memory.stat').read_text().splitlines())}
        reclaimable = min(used, max(0, stats.get('inactive_file', 0)
                                   - stats.get('file_dirty', 0) - stats.get('file_writeback', 0)))
    except (OSError, ValueError):
        pass  # Older/unavailable stats retain the conservative raw-usage check.
    return max(0, int(limit) - used + reclaimable)


def available_host_bytes():
    """Host availability constrained by every visible cgroup v2 ancestor."""
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'MemTotal'):
            values[key] = int(value.split()[0]) * 1024
    available = values['MemAvailable']
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if not line.startswith('0::'):
            continue
        root = Path('/sys/fs/cgroup')
        path = root / line.split('::', 1)[1].lstrip('/')
        # Namespaced mounts expose the current cgroup at the mount root.
        if not path.exists():
            path = root
        while True:
            try:
                headroom = _cgroup_headroom(path)
                if headroom is not None:
                    available = min(available, headroom)
            except (OSError, ValueError):
                pass
            if path == root:
                break
            path = path.parent
    return available

class _OffloadPool:
    """Reusable tensors sized to the host allocator's charged block capacity."""
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.free = {}
        self._free_sizes = []
        self.pending = []
        self.allocated = 0
        self.allocated_pinned = 0
        self.live = 0
        self.live_pinned = 0
        self.prefetched = 0
        self.stats = Counter()

    @staticmethod
    def allocation_charge(size):
        # ATen CachingHostAllocator rounds to PowerOf2Ceil before CUDA host
        # allocation (also reported by torch.cuda.memory.host_memory_stats).
        # Make the whole charged capacity available for subsequent leases.
        return 1 << (size - 1).bit_length() if size else 0

    def _reap(self):
        pending = []
        for buffer, event in self.pending:
            if event is None or event.query():
                size = buffer.numel()
                if size not in self.free:
                    self.free[size] = []
                    insort(self._free_sizes, size)
                self.free[size].append(buffer)
            else:
                pending.append((buffer, event))
        self.pending = pending

    def acquire(self, size):
        with self.lock:
            self._reap()
            index = bisect_left(self._free_sizes, size)
            if index < len(self._free_sizes):
                capacity = self._free_sizes[index]
                bucket = self.free[capacity]
                buffer = bucket.pop()
                if not bucket:
                    del self.free[capacity]
                    self._free_sizes.pop(index)
                self.stats['pool_reuses'] += 1
            else:
                charge = self.allocation_charge(size)
                if self.allocated_pinned + charge > self.config.max_cpu_bytes:
                    self.stats['budget_fallback_bytes'] += size
                    return None
                # This check is intentionally only on allocations, not every save.
                if available_host_bytes() < charge + self.config.host_reserve_bytes:
                    self.stats['headroom_fallback_bytes'] += size
                    return None
                # ATen already reserves this rounded capacity. Exposing it in
                # the tensor lets a later, slightly larger request reuse the
                # block instead of paying another cudaHostAlloc in the same
                # size class. Transfers still use only record.size bytes.
                buffer = torch.empty(charge, dtype=torch.uint8, device='cpu', pin_memory=True)
                self.allocated += charge
                self.allocated_pinned += charge
                capacity = charge
            self.live += capacity
            self.live_pinned += self.allocation_charge(capacity)
            self.stats['peak_live_cpu_bytes'] = max(self.stats['peak_live_cpu_bytes'], self.live)
            self.stats['peak_live_pinned_bytes'] = max(self.stats['peak_live_pinned_bytes'], self.live_pinned)
            return buffer

    def release(self, buffer, event):
        with self.lock:
            self.live -= buffer.numel()
            self.live_pinned -= self.allocation_charge(buffer.numel())
            self.pending.append((buffer, event))

    def snapshot(self):
        with self.lock:
            self._reap()
            counters = dict.fromkeys(('d2h_bytes', 'h2d_bytes', 'resident_reused_bytes',
                                      'offloaded_storages', 'prefetch_loads', 'demand_loads',
                                      'schedule_prefetch_calls', 'schedule_prefetch_loads', 'budget_fallback_bytes',
                                      'headroom_fallback_bytes'), 0)
            counters.update(self.stats)
            return dict(counters, allocated_cpu_bytes=self.allocated,
                        allocated_pinned_bytes=self.allocated_pinned,
                        live_pinned_bytes=self.live_pinned,
                        live_cpu_bytes=self.live, prefetched_gpu_bytes=self.prefetched,
                        pending_cpu_bytes=sum(b.numel() for b, _ in self.pending))

    def close(self):
        """Release idle buffers; never invalidate an active saved tensor."""
        with self.lock:
            if self.live:
                raise RuntimeError('Cannot close activation offload pool with live saved tensors')
            for _, event in self.pending:
                if event is not None:
                    event.synchronize()
            self.pending.clear()
            self.free.clear()
            self._free_sizes.clear()
            self.allocated = 0
            self.allocated_pinned = 0


# Keep the process pool alive between module instances. Otherwise destroying a
# module could drop our accounting while PyTorch still caches its pinned pages,
# and the next module would receive a fresh, uncharged allocation allowance.
_pools = {}
_pools_lock = threading.Lock()


def _shared_pool(config):
    with _pools_lock:
        pool = _pools.get(os.getpid())
        if pool is None:
            if config.pool_slab_bytes:
                from nnscaler.runtime.activation_pool import SlabOffloadPool
                pool = SlabOffloadPool(config)
            else:
                pool = _OffloadPool(config)
            _pools[os.getpid()] = pool
        elif pool.config.max_cpu_bytes != config.max_cpu_bytes:
            raise ValueError('All CPU-offloading modules in a process must use the same max_cpu_bytes')
        elif pool.config.pool_slab_bytes != config.pool_slab_bytes:
            raise ValueError('All CPU-offloading modules in a process must use the same pool_slab_bytes')
        return pool
