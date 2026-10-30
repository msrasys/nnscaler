#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Process-wide composition root of the offloader."""

from __future__ import annotations

import collections
import logging
import threading
from typing import Any, Dict, Optional

import torch

from .config import OffloadConfig
from .host_memory import PinnedPool, bind_to_gpu_numa
from .stats import GroupProfile, OffloadStats, SizeHistogram, TimingCollector, format_summary
from .tracker import ExecutionTracker, OffloadBatch
from .transport import CopyEngine

_logger = logging.getLogger(__name__)


class OffloadManager:
    """Owns configuration, host memory, transport, tracking and statistics.

    One instance per process (:meth:`get`).  :func:`configure` replaces it
    before the first use; afterwards changing the configuration requires
    :meth:`reset`.
    """

    _instance: Optional['OffloadManager'] = None
    _instance_lock = threading.Lock()

    def __init__(self, config: Optional[OffloadConfig] = None) -> None:
        self.config = (config or OffloadConfig.from_env()).validate()
        self.stats = OffloadStats()
        self.groups: Dict[int, GroupProfile] = collections.defaultdict(GroupProfile)
        self.timing = TimingCollector()
        self.sizes = SizeHistogram()
        self.pool = PinnedPool(max_bytes=self.config.pool_max_bytes)
        self.engine = CopyEngine(self.config, self.pool, self.stats, self.timing, self.groups)
        self.tracker = ExecutionTracker(self.config, self.engine, self.stats, self.groups)
        self._debugged_groups: set = set()
        self._pool_waits_seen = 0
        self._steps = 0
        self._profile_total = OffloadStats()
        self._profile_sizes = SizeHistogram()
        if torch.cuda.is_available() and (self.config.numa_membind or self.config.numa_affinity):
            self.pool.mempolicy = bind_to_gpu_numa(
                torch.cuda.current_device(),
                membind=self.config.numa_membind, affinity=self.config.numa_affinity)

    # -- lifecycle -------------------------------------------------------------
    @classmethod
    def get(cls) -> 'OffloadManager':
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = OffloadManager()
        return cls._instance

    @classmethod
    def configure(cls, config: OffloadConfig) -> 'OffloadManager':
        """Install ``config``; must happen before the first offload context."""
        with cls._instance_lock:
            cls._instance = OffloadManager(config)
            OffloadBatch.last_batch = None
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._instance_lock:
            cls._instance = None
            OffloadBatch.last_batch = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    # -- executor / schedule entry points --------------------------------------
    def begin_forward(self, name: str):
        return self.tracker.begin_forward(name)

    def end_forward(self, ctx) -> None:
        self.tracker.end_forward(ctx)

    def begin_backward(self, name: str):
        return self.tracker.begin_backward(name)

    def end_backward(self, ctx) -> None:
        self.tracker.end_backward(ctx)

    def hint_forward(self, name: str, backward_distance: int) -> None:
        self.tracker.hint_forward(name, backward_distance)

    def prefetch_next_backward(self, *names: str) -> None:
        self.tracker.prefetch_next_backward(*names)

    def forward_skipped(self, name: str) -> None:
        self.tracker.forward_skipped(name)

    def comm_begin(self) -> None:
        self.engine.comm_begin()

    def comm_end(self) -> None:
        self.engine.comm_end()

    # -- diagnostics -------------------------------------------------------------
    def debug_tensor(self, group_id: Optional[int], tensor: torch.Tensor, nbytes: int) -> None:
        if self.config.debug and group_id not in self._debugged_groups:
            _logger.info(
                f'[cpu_offload] group {group_id}: shape={tuple(tensor.shape)} dtype={tensor.dtype} '
                f'bytes={nbytes} contiguous={tensor.is_contiguous()}')

    def debug_group_done(self, group_id: Optional[int], batch: OffloadBatch) -> None:
        if self.config.debug and group_id not in self._debugged_groups:
            self._debugged_groups.add(group_id)
            _logger.info(
                f'[cpu_offload] group {group_id}: {batch.tensors} saved tensors, '
                f'{sum(len(t.regions) for t in batch.transfers)} regions in {len(batch.transfers)} transfer(s), '
                f'{batch.nbytes / 2**20:.1f} MiB offloaded')

    def snapshot(self, reset: bool = True) -> Dict[str, Any]:
        """Counters since the last snapshot (trainers call this once per step)."""
        if self.config.timing:
            backlog = self.timing.collect(self.stats, self.groups)
            self.stats.d2h_backlog_bytes = backlog + self.engine.pending_d2h_bytes()
        self.stats.pool_trimmed_bytes = self.pool.trim(self.config.pool_trim_steps)
        self.stats.pool_allocated_bytes = self.pool.allocated_bytes
        self.stats.pool_allocations = self.pool.allocations
        self.stats.pool_waits = self.pool.waits - self._pool_waits_seen
        self._pool_waits_seen = self.pool.waits
        self.stats.reloaded_bytes = self.engine.reloaded_bytes
        self.stats.pending_contexts, self.stats.pending_bytes = self.tracker.pending_summary()
        data = self.stats.as_dict()
        if reset:
            self._accumulate_profile()
            self._profile_sizes.merge(self.sizes)
            self.sizes.clear()
            self.stats.reset()
            self._steps += 1
            if self.config.profile_every and self._steps % self.config.profile_every == 0:
                self.report(log=True)
        return data

    def active(self) -> bool:
        """Whether anything was offloaded since the last snapshot."""
        return bool(self.stats.packed_tensors or self.stats.transfers)

    def _accumulate_profile(self) -> None:
        for name, value in self.stats.as_dict().items():
            if name in OffloadStats._NOT_RESET:
                setattr(self._profile_total, name, value)
            elif isinstance(value, (int, float)):
                setattr(self._profile_total, name, getattr(self._profile_total, name) + value)

    def report(self, log: bool = False) -> str:
        """Per-group summary table over the steps since the last report.

        ``snapshot`` is expected once per training step; activity that was never
        snapshotted counts as one step so the table is never printed over zeros.
        """
        if self._steps == 0 and self.active():
            self.snapshot(reset=True)
        text = format_summary(self.groups, self._steps, self._profile_total,
                              self._profile_sizes, self.config.min_bytes)
        if log:
            for line in text.splitlines():
                _logger.info(line)
        self.groups.clear()
        self._profile_total = OffloadStats()
        self._profile_sizes = SizeHistogram()
        self._steps = 0
        return text
