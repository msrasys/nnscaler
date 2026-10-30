#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Configuration of CPU activation offloading.

Every knob has a programmatic field and an environment variable of the same
meaning (``CPU_OFFLOADING_<NAME>``).  :func:`OffloadConfig.from_env` is what
the runtime uses when nothing was configured explicitly; trainers can instead
call :func:`nnscaler.runtime.cpu_offload.configure` before the first step.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional


FAKE_MODES = ('off', 'no_copy', 'd2h_only', 'both_keep')
COMM_EXCLUSIVE_MODES = ('off', 'strict', 'soft')
PREFETCH_UNITS = ('backward', 'group')
DEDUP_SCOPES = ('group', 'context')


@dataclass
class OffloadConfig:
    """All tunables of the offloader, grouped by concern.

    Selection (what is offloaded):
        enabled: master switch; ``False`` turns every offload context into a no-op.
        min_bytes: tensors smaller than this stay on the device.  Small tensors
            cost more host time to move than they save.
        min_backward_distance: a forward whose own backward is fewer than this
            many segment executions away is not offloaded (the scheduler reports
            the distance; the tensors would come back right away).  ``0``
            disables the heuristic.

    Prefetch (when tensors come back):
        prefetch_unit: ``backward`` reloads whole segment executions (the F/B of
            a virtual stage is the unit, as the pipeline schedule sees it);
            ``group`` reloads offload groups one by one inside the running
            segment execution (also the behaviour without an executor).
        prefetch_backwards: with ``prefetch_unit=backward``, how many upcoming
            backward executions (announced by the generated schedule) are
            reloaded ahead of the running one.  ``1`` hides one segment's copy
            time behind one segment's compute; a GPU that computes faster than
            the host link copies wants ``2`` or ``3``.
        prefetch_groups: with ``prefetch_unit=group``, groups reloaded ahead of
            the demanded one inside a segment execution.
        prefetch_bytes: device-memory budget for ahead-of-demand reloads: no
            further reload is issued while this many reloaded bytes are still
            unconsumed (``0`` = unbounded; demanded tensors always load).

    Transport (how bytes move):
        commit_bytes: flush a group's saved tensors to the host once this many
            bytes accumulated instead of at the group's end (``0`` = at end), so
            the copy of a large group overlaps the group's own compute.
        copy_chunk_bytes: reload in chunks of about this size, each with its
            own device buffer and event (``0`` = one chunk per transfer).
        reload_on_copy_stream: allocate reload buffers on the H2D stream so a
            reload never waits for the compute already queued on the consumer
            stream; ``False`` allocates on the consumer stream instead.
        dedup_scope: ``group`` copies a saved view once per offload group;
            ``context`` also serves views already shipped by an earlier group of
            the same segment execution (a block output saved again by the next
            block) from that copy, holding the earlier group's device memory
            until the next group ends.
        max_inflight_copies: transfers queued per direction (``0`` = unbounded).
        comm_exclusive: ``off`` | ``strict`` | ``soft``; behaviour of
            :func:`comm_exclusive` windows around all-to-all kernels (see docs).

    Host memory:
        numa_membind: allocate pinned pages on the GPU's NUMA node.
        numa_affinity: additionally pin the calling thread to the GPU's CPUs.
        pool_max_bytes: bound of pinned memory owned by the pool (``0`` = unbounded).
        pool_trim_steps: release pinned slabs unused for this many steps
            (``0`` = keep them).

    Diagnostics:
        timing: CUDA-event timings of copies and stalls (cheap, on by default).
        debug: log the saved tensors of each group once.
        profile_every: log a per-group summary table every N steps (``0`` = off).
        fake_mode: ``off`` | ``no_copy`` | ``d2h_only`` | ``both_keep``; A/B
            modes that keep tensors on the device while exercising the rest.
    """

    enabled: bool = True
    min_bytes: int = 1 << 20
    min_backward_distance: int = 1

    prefetch_unit: str = 'backward'
    prefetch_backwards: int = 1
    prefetch_groups: int = 2
    prefetch_bytes: int = 0

    commit_bytes: int = 0
    copy_chunk_bytes: int = 0
    reload_on_copy_stream: bool = True
    dedup_scope: str = 'group'
    max_inflight_copies: int = 0
    comm_exclusive: str = 'off'

    numa_membind: bool = True
    numa_affinity: bool = False
    pool_max_bytes: int = 0
    pool_trim_steps: int = 0

    timing: bool = True
    debug: bool = False
    profile_every: int = 0
    fake_mode: str = 'off'

    # ------------------------------------------------------------------
    ENV_PREFIX = 'CPU_OFFLOADING_'
    # env name -> (field, parser); kept explicit so renames stay compatible
    _ENV: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def env_table(cls) -> Dict[str, Any]:
        def gb(v): return int(float(v) * (1 << 30))
        def mb(v): return int(float(v) * (1 << 20))
        def flag(v): return v.strip() not in ('0', '', 'false', 'False')
        def fake(v):
            v = v.strip()
            if v.isdigit():
                return FAKE_MODES[int(v)] if int(v) < len(FAKE_MODES) else 'off'
            return v
        def comm(v):
            v = v.strip()
            if v.isdigit():
                return COMM_EXCLUSIVE_MODES[int(v)] if int(v) < len(COMM_EXCLUSIVE_MODES) else 'off'
            return v
        return {
            'DISABLE': ('enabled', lambda v: not flag(v)),
            'MIN_BYTES': ('min_bytes', int),
            'MIN_BACKWARD_DISTANCE': ('min_backward_distance', int),
            'PREFETCH_UNIT': ('prefetch_unit', lambda v: v.strip()),
            'PREFETCH_BACKWARDS': ('prefetch_backwards', int),
            'PREFETCH_GROUPS': ('prefetch_groups', int),
            'PREFETCH_MB': ('prefetch_bytes', mb),
            'COMMIT_MB': ('commit_bytes', mb),
            'COPY_CHUNK_MB': ('copy_chunk_bytes', mb),
            'RELOAD_ON_COPY_STREAM': ('reload_on_copy_stream', flag),
            'DEDUP_SCOPE': ('dedup_scope', lambda v: v.strip()),
            'MAX_INFLIGHT_COPIES': ('max_inflight_copies', int),
            'COMM_EXCLUSIVE': ('comm_exclusive', comm),
            'NUMA_MEMBIND': ('numa_membind', flag),
            'NUMA_AFFINITY': ('numa_affinity', flag),
            'POOL_MAX_GB': ('pool_max_bytes', gb),
            'POOL_TRIM_STEPS': ('pool_trim_steps', int),
            'TIMING': ('timing', flag),
            'DEBUG': ('debug', flag),
            'PROFILE_EVERY': ('profile_every', int),
            'FAKE_COPY': ('fake_mode', fake),
        }

    @classmethod
    def from_env(cls, environ: Optional[Dict[str, str]] = None) -> 'OffloadConfig':
        environ = os.environ if environ is None else environ
        config = cls()
        for suffix, (name, parse) in cls.env_table().items():
            raw = environ.get(cls.ENV_PREFIX + suffix)
            if raw is None:
                continue
            try:
                setattr(config, name, parse(raw))
            except ValueError as error:
                raise ValueError(f'{cls.ENV_PREFIX}{suffix}={raw!r} is not valid for {name}') from error
        config.validate()
        return config

    def validate(self) -> 'OffloadConfig':
        if self.fake_mode not in FAKE_MODES:
            raise ValueError(f'fake_mode must be one of {FAKE_MODES}, got {self.fake_mode!r}')
        if self.comm_exclusive not in COMM_EXCLUSIVE_MODES:
            raise ValueError(f'comm_exclusive must be one of {COMM_EXCLUSIVE_MODES}, got {self.comm_exclusive!r}')
        if self.prefetch_unit not in PREFETCH_UNITS:
            raise ValueError(f'prefetch_unit must be one of {PREFETCH_UNITS}, got {self.prefetch_unit!r}')
        if self.dedup_scope not in DEDUP_SCOPES:
            raise ValueError(f'dedup_scope must be one of {DEDUP_SCOPES}, got {self.dedup_scope!r}')
        for name in ('min_bytes', 'min_backward_distance', 'prefetch_backwards', 'prefetch_groups',
                     'prefetch_bytes', 'commit_bytes', 'max_inflight_copies', 'copy_chunk_bytes',
                     'pool_max_bytes', 'pool_trim_steps', 'profile_every'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be non-negative')
        return self

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith('_')}

    @property
    def moves_bytes(self) -> bool:
        """Whether real copies happen (fake ``no_copy`` mode does not)."""
        return self.fake_mode != 'no_copy'

    @property
    def keeps_device_tensor(self) -> bool:
        """Whether the original device tensor stays alive (fake modes)."""
        return self.fake_mode != 'off'
