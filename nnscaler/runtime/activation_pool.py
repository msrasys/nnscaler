"""Event-safe suballocation of persistent pinned activation buffers."""

from dataclasses import dataclass

import torch

from nnscaler.runtime.cpu_offloading import _OffloadPool, available_host_bytes


@dataclass(eq=False)
class _Slab:
    tensor: torch.Tensor
    free: list[tuple[int, int]]
    live: int = 0


class SlabOffloadPool(_OffloadPool):
    """Pack independent saved storages into budgeted, power-of-two slabs.

    Each lease keeps its exact logical length. Released slots become reusable
    only after their last DMA event; coalescing happens within the same slab.
    Slabs stay owned by the process pool, including all idle capacity.
    """

    alignment = 256

    def __init__(self, config):
        super().__init__(config)
        self.slabs = []
        self.leases = {}
        self.pending_slots = {}

    def _reap(self):
        waiting = []
        for buffer, event in self.pending:
            if event is not None and not event.query():
                waiting.append((buffer, event))
                continue
            slab, offset, length = self.pending_slots.pop(id(buffer))
            slab.free.append((offset, length))
            merged = []
            for start, size in sorted(slab.free):
                if merged and merged[-1][0] + merged[-1][1] == start:
                    before, span = merged.pop()
                    merged.append((before, span + size))
                else:
                    merged.append((start, size))
            slab.free = merged
        self.pending = waiting

    def acquire(self, size):
        if size <= 0:
            raise ValueError('Pinned slab leases must have positive size')
        length = (size + self.alignment - 1) // self.alignment * self.alignment
        with self.lock:
            self._reap()
            candidates = [
                (span, i, j)
                for i, slab in enumerate(self.slabs)
                for j, (_, span) in enumerate(slab.free)
                if span >= length
            ]
            if candidates:
                _, i, j = min(candidates)
                slab = self.slabs[i]
                self.stats['pool_reuses'] += 1
            else:
                minimum = self.allocation_charge(length)
                remaining = self.config.max_cpu_bytes - self.allocated_pinned
                if remaining < minimum:
                    self.stats['budget_fallback_bytes'] += size
                    return None
                desired = max(minimum, self.allocation_charge(self.config.pool_slab_bytes))
                capacity = min(desired, 1 << (remaining.bit_length() - 1))
                if available_host_bytes() < capacity + self.config.host_reserve_bytes:
                    self.stats['headroom_fallback_bytes'] += size
                    return None
                tensor = torch.empty(capacity, dtype=torch.uint8, device='cpu', pin_memory=True)
                slab = _Slab(tensor, [(0, capacity)])
                self.slabs.append(slab)
                self.allocated += capacity
                self.allocated_pinned += capacity
                j = 0
            offset, span = slab.free.pop(j)
            if span > length:
                slab.free.append((offset + length, span - length))
            buffer = slab.tensor.narrow(0, offset, size)
            self.leases[id(buffer)] = (slab, offset, length)
            if slab.live == 0:
                self.live_pinned += slab.tensor.numel()
            slab.live += 1
            self.live += size
            self.stats['peak_live_cpu_bytes'] = max(self.stats['peak_live_cpu_bytes'], self.live)
            self.stats['peak_live_pinned_bytes'] = max(
                self.stats['peak_live_pinned_bytes'], self.live_pinned
            )
            return buffer

    def release(self, buffer, event):
        with self.lock:
            slot = self.leases.pop(id(buffer))
            slab, _, _ = slot
            slab.live -= 1
            if slab.live == 0:
                self.live_pinned -= slab.tensor.numel()
            self.live -= buffer.numel()
            self.pending_slots[id(buffer)] = slot
            self.pending.append((buffer, event))

    def snapshot(self):
        with self.lock:
            result = super().snapshot()
            result['pool_slabs'] = len(self.slabs)
            result['free_slab_bytes'] = sum(span for slab in self.slabs for _, span in slab.free)
            return result

    def close(self):
        with self.lock:
            if self.live:
                raise RuntimeError('Cannot close activation offload pool with live saved tensors')
            for _, event in self.pending:
                if event is not None:
                    event.synchronize()
            self._reap()
            self.slabs.clear()
            super().close()
