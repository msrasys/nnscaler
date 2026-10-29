#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CPU offloading of tensors saved for backward.

Three layers, only the first of which depends on how a model is parallelized:

1. **Planning** (``nnscaler.policies``): consecutive forward operators share an
   ``offload_id``; everything autograd saves while they run is offloaded.  An
   operator may also carry a ``recompute_id``: the group is recomputed and the
   inputs its checkpoint keeps are offloaded, so nothing of it stays resident.
2. **Code generation**: each group runs inside ``with
   module.cpu_offloading_hooks(group_id):``; scheduled pipelines also emit
   :func:`hint_forward` / :func:`prefetch_next_backward` from the static
   schedule.
3. **Runtime** (this package): :mod:`.config` knobs, :mod:`.host_memory`
   (NUMA-placed best-fit pinned slab pool), :mod:`.transport` (one transfer per
   group execution: merged regions, one slab, one event pair per direction;
   copy streams and policies), :mod:`.tracker` (execution contexts and
   prefetch order), :mod:`.hooks` (the saved-tensor hooks) and :mod:`.stats`
   (counters, GPU timings, size histogram, per-group profiles), composed by
   :class:`.manager.OffloadManager`.

See ``docs/source/cpu_offloading.md`` for the design and the tuning workflow.
"""

from typing import Any, Dict, Optional

from .config import OffloadConfig, FAKE_MODES, COMM_EXCLUSIVE_MODES
from .hooks import CPUOffloadContext, OffloadedTensor, ModuleTensorRef, covered_bytes, element_size
from .host_memory import PinnedPool, MemPolicy, gpu_numa_info, bind_to_gpu_numa
from .manager import OffloadManager
from .stats import OffloadStats, GroupProfile, SizeHistogram, all_rank_max, format_summary
from .tracker import ExecutionTracker, ExecContext, OffloadBatch
from .transport import CopyEngine, Transfer, Region, D2H_STREAM_NAME, H2D_STREAM_NAME

__all__ = [
    'OffloadConfig', 'OffloadManager', 'CPUOffloadContext', 'comm_exclusive',
    'configure', 'hint_forward', 'prefetch_next_backward', 'offload_stats', 'offload_report',
    'all_rank_max', 'OffloadStats', 'GroupProfile',
]


def configure(config: Optional[OffloadConfig] = None, **overrides) -> OffloadManager:
    """Install an :class:`OffloadConfig` (fields may be overridden by keyword).

    Call before the first training step.  Without it the runtime reads the
    ``CPU_OFFLOADING_*`` environment variables on first use.
    """
    config = config or OffloadConfig.from_env()
    for name, value in overrides.items():
        if not hasattr(config, name):
            raise TypeError(f'unknown OffloadConfig field {name!r}')
        setattr(config, name, value)
    return OffloadManager.configure(config.validate())


def _active_manager() -> Optional[OffloadManager]:
    manager = OffloadManager._instance
    return manager if manager is not None and manager.enabled else None


def hint_forward(name: str, backward_distance: int) -> None:
    """Generated schedule code: the next forward of ``name`` has its backward this many segment executions later."""
    manager = OffloadManager.get()
    if manager.enabled:
        manager.hint_forward(name, backward_distance)


def prefetch_next_backward(name: str) -> None:
    """Generated schedule code: the next backward will be segment ``name``."""
    manager = _active_manager()
    if manager is not None:
        manager.prefetch_next_backward(name)


class comm_exclusive:
    """Mark a communication-heavy region (all-to-all, pipeline P2P) on the current stream.

    PCIe offload copies and NVLink traffic contend inside the GPU's I/O hub;
    with ``OffloadConfig.comm_exclusive`` set the copy engine keeps copies out
    of the region.  No-op when offloading is inactive or the mode is ``off``.
    """

    def __enter__(self):
        self._manager = _active_manager()
        if self._manager is not None:
            self._manager.comm_begin()
        return self

    def __exit__(self, *exc):
        if self._manager is not None:
            self._manager.comm_end()
        return None


def offload_stats(reset: bool = True) -> Optional[Dict[str, Any]]:
    """Per-step counters (see :class:`OffloadStats`); ``None`` before first use."""
    manager = OffloadManager._instance
    return None if manager is None else manager.snapshot(reset=reset)


def offload_report(log: bool = False) -> Optional[str]:
    """Per-group summary table since the last report; ``None`` before first use."""
    manager = OffloadManager._instance
    return None if manager is None else manager.report(log=log)
