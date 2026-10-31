#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Counters, GPU-side timings and per-group profiles of the offloader."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class GroupProfile:
    """Per offload group (planning id) accumulation, for the summary table.

    The counts are written by the tracker's batch when a group ends; the
    durations by :class:`TimingCollector` when the copy events complete.
    """
    executions: int = 0
    tensors: int = 0
    regions: int = 0
    bytes: int = 0
    d2h_ms: float = 0.0
    h2d_ms: float = 0.0
    stall_ms: float = 0.0


class SizeHistogram:
    """Saved tensors by size (powers of two), including those below ``min_bytes``.

    The histogram is what ``min_bytes`` is tuned against: it shows how many
    tensors (host-side cost) carry how many bytes (device memory saved).
    """

    def __init__(self) -> None:
        self.counts: Dict[int, int] = {}
        self.bytes: Dict[int, int] = {}

    def add(self, nbytes: int) -> None:
        bucket = nbytes.bit_length()
        self.counts[bucket] = self.counts.get(bucket, 0) + 1
        self.bytes[bucket] = self.bytes.get(bucket, 0) + nbytes

    def merge(self, other: 'SizeHistogram') -> None:
        for bucket, count in other.counts.items():
            self.counts[bucket] = self.counts.get(bucket, 0) + count
            self.bytes[bucket] = self.bytes.get(bucket, 0) + other.bytes[bucket]

    def clear(self) -> None:
        self.counts.clear()
        self.bytes.clear()

    def format(self, steps: int = 1, min_bytes: int = 0) -> str:
        if not self.counts:
            return 'saved tensor sizes: none'
        steps = max(steps, 1)
        parts = []
        marked = False
        for bucket in sorted(self.counts):
            hi = 1 << bucket
            mark = '*' if min_bytes and hi <= min_bytes else ''
            marked = marked or bool(mark)
            parts.append(f'<{_human(hi)}{mark}: {self.counts[bucket] / steps:.0f} '
                         f'({self.bytes[bucket] / steps / 2**30:.2f} GiB)')
        note = ' (* below min_bytes, kept on device)' if marked else ''
        return 'saved tensor sizes per step: ' + ', '.join(parts) + note


def _human(nbytes: int) -> str:
    for unit, scale in (('G', 2**30), ('M', 2**20), ('K', 2**10)):
        if nbytes >= scale:
            return f'{nbytes // scale}{unit}'
    return f'{nbytes}B'


@dataclass
class OffloadStats:
    """Counters since the last reset (one training step when the trainer polls).

    Selection counters are written by the tracker (what was saved, offloaded,
    shared or skipped); reload/transport counters and times by the transport.
    """

    # selection (tracker)
    packed_tensors: int = 0
    offloaded_regions: int = 0   # contiguous byte ranges actually copied
    offloaded_bytes: int = 0
    deduplicated_bytes: int = 0  # bytes of saved views that overlapped another saved view
    skipped_small_bytes: int = 0
    skipped_late_bytes: int = 0  # saved after a backward inside the group shipped its transfer
    shared_bytes: int = 0        # served by the previous group's transfer (dedup_scope=context)
    transfers: int = 0           # transfers that moved bytes (groups, or flushed slices of groups)
    flushes: int = 0             # early commits caused by commit_bytes
    # reload (transport)
    prefetched_regions: int = 0
    prefetched_bytes: int = 0
    stalls: int = 0              # reloads whose copy was still running when demanded (CPU view)
    forced_h2d: int = 0          # reloads issued on demand because no prefetch covered them
    # transport policies
    deferred_d2h: int = 0
    deferred_h2d: int = 0
    comm_windows: int = 0
    pool_waits: int = 0
    # CPU time (seconds); pack/commit/unpack are hook bookkeeping only
    pack_seconds: float = 0.0
    commit_seconds: float = 0.0
    unpack_seconds: float = 0.0
    acquire_seconds: float = 0.0     # pinned allocation (grows the pool)
    d2h_issue_seconds: float = 0.0   # issuing copies
    h2d_issue_seconds: float = 0.0
    # GPU time from CUDA events (milliseconds)
    stall_ms: float = 0.0        # compute stream waiting for reloads
    d2h_ms: float = 0.0          # summed D2H copy durations
    h2d_ms: float = 0.0          # summed H2D copy durations
    pool_trimmed_bytes: int = 0  # pinned slabs released by pool_trim_steps this step
    # gauges sampled at snapshot time (not reset, not summed into profiles)
    d2h_backlog_bytes: int = 0   # bytes whose D2H had not finished when sampled
    reloaded_bytes: int = 0      # reloaded and not yet consumed (prefetch_bytes gauge)
    pool_allocated_bytes: int = 0
    pool_allocations: int = 0
    pending_contexts: int = 0
    pending_bytes: int = 0

    _NOT_RESET = ('d2h_backlog_bytes', 'reloaded_bytes', 'pool_allocated_bytes', 'pool_allocations',
                  'pending_contexts', 'pending_bytes')

    def reset(self) -> None:
        for f in fields(self):
            if f.name in self._NOT_RESET or f.name.startswith('_'):
                continue
            setattr(self, f.name, type(getattr(self, f.name))())

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith('_')}


class TimingCollector:
    """Folds completed CUDA event pairs into the stats; keeps unfinished ones.

    Transfers register ``(start, end)`` event pairs per direction; stalls are
    registered by the fetch that waited.  Events are only queried, never
    synchronized, so collection is safe at any point of a step.
    """

    def __init__(self) -> None:
        self._d2h: List[Tuple[Any, int]] = []     # (transfer, group_id)
        self._h2d: List[Tuple[Any, int]] = []
        self._stalls: List[Tuple[torch.cuda.Event, torch.cuda.Event, int]] = []

    def track_d2h(self, transfer, group_id: int) -> None:
        self._d2h.append((transfer, group_id))

    def track_h2d(self, transfer, group_id: int) -> None:
        self._h2d.append((transfer, group_id))

    def track_stall(self, before: torch.cuda.Event, after: torch.cuda.Event, group_id: int) -> None:
        self._stalls.append((before, after, group_id))

    def collect(self, stats: OffloadStats, groups: Dict[int, GroupProfile]) -> int:
        """Return the bytes of D2H copies still in flight."""
        backlog = 0
        keep = []
        for transfer, gid in self._d2h:
            if transfer.d2h_event is not None and transfer.d2h_event.query():
                if transfer.d2h_start is not None:
                    ms = transfer.d2h_start.elapsed_time(transfer.d2h_event)
                    stats.d2h_ms += ms
                    groups[gid].d2h_ms += ms
            else:
                backlog += transfer.nbytes
                keep.append((transfer, gid))
        self._d2h = keep
        keep = []
        for transfer, gid in self._h2d:
            event = transfer.h2d_event
            if event is not None and event.query():
                if transfer.h2d_start is not None:
                    ms = transfer.h2d_start.elapsed_time(event)
                    stats.h2d_ms += ms
                    groups[gid].h2d_ms += ms
            else:
                keep.append((transfer, gid))
        self._h2d = keep
        keep = []
        for before, after, gid in self._stalls:
            if after.query():
                ms = before.elapsed_time(after)
                stats.stall_ms += ms
                groups[gid].stall_ms += ms
            else:
                keep.append((before, after, gid))
        self._stalls = keep
        return backlog


def format_summary(groups: Dict[int, GroupProfile], steps: int, stats_total: OffloadStats,
                   sizes: Optional[SizeHistogram] = None, min_bytes: int = 0) -> str:
    """Human readable per-group table plus totals, averaged over ``steps``."""
    steps = max(steps, 1)
    rows = sorted(groups.items())
    lines = [
        f'[cpu_offload] profile over {steps} step(s): '
        f'{stats_total.offloaded_bytes / steps / 2**30:.2f} GiB/step offloaded in '
        f'{stats_total.transfers / steps:.0f} transfers / {stats_total.offloaded_regions / steps:.0f} regions '
        f'({stats_total.packed_tensors / steps:.0f} saved tensors, '
        f'{stats_total.deduplicated_bytes / steps / 2**30:.2f} GiB shared), '
        f'D2H {stats_total.d2h_ms / steps / 1000:.2f} s/step '
        f'({_rate(stats_total.offloaded_bytes, stats_total.d2h_ms)} GB/s), '
        f'H2D {stats_total.h2d_ms / steps / 1000:.2f} s/step '
        f'({_rate(stats_total.prefetched_bytes, stats_total.h2d_ms)} GB/s), '
        f'compute waited {stats_total.stall_ms / steps / 1000:.3f} s/step in {stats_total.stalls / steps:.0f} stalls, '
        f'host CPU/step: hooks {(stats_total.pack_seconds + stats_total.commit_seconds + stats_total.unpack_seconds) / steps:.3f} s, '
        f'copy issue {(stats_total.d2h_issue_seconds + stats_total.h2d_issue_seconds) / steps:.3f} s, '
        f'pinned alloc {stats_total.acquire_seconds / steps:.3f} s; '
        f'pinned pool {stats_total.pool_allocated_bytes / 2**30:.1f} GiB',
    ]
    if sizes is not None:
        lines.append('[cpu_offload] ' + sizes.format(steps, min_bytes))
    lines.append(
        f'{"group":>8} {"exec/step":>9} {"tensors":>8} {"regions":>8} {"MiB/exec":>9} {"GiB/step":>9} '
        f'{"d2h ms":>8} {"h2d ms":>8} {"stall ms":>9}')
    for gid, g in rows:
        execs = g.executions / steps
        per_exec = g.bytes / max(g.executions, 1) / 2**20
        lines.append(
            f'{gid:>8} {execs:>9.1f} {g.tensors / max(g.executions, 1):>8.1f} '
            f'{g.regions / max(g.executions, 1):>8.1f} {per_exec:>9.1f} '
            f'{g.bytes / steps / 2**30:>9.2f} {g.d2h_ms / steps:>8.0f} {g.h2d_ms / steps:>8.0f} '
            f'{g.stall_ms / steps:>9.1f}')
    return '\n'.join(lines)


def _rate(nbytes: int, ms: float) -> str:
    return f'{nbytes / (ms / 1000) / 1e9:.1f}' if ms > 0 else 'n/a'


def all_rank_max(values: Dict[str, float], device: Optional[torch.device] = None) -> Dict[str, Tuple[float, int]]:
    """``{name: (max over ranks, rank attaining it)}`` for a dict of numbers.

    The pipeline waits for its slowest rank, so rank-0 figures alone can hide
    the bottleneck.  No-op (returns rank 0) without an initialized process group.
    """
    import torch.distributed as dist
    names = sorted(values)
    if not dist.is_available() or not dist.is_initialized():
        return {n: (float(values[n]), 0) for n in names}
    device = device or torch.device('cuda', torch.cuda.current_device())
    local = torch.tensor([float(values[n]) for n in names] + [float(dist.get_rank())],
                         dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    rows = torch.stack(gathered).cpu()
    out = {}
    for i, n in enumerate(names):
        col = rows[:, i]
        r = int(col.argmax())
        out[n] = (col[r].item(), int(rows[r, -1].item()))
    return out
