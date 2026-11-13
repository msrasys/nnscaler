# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Configuration shared by activation storage and its scheduling adapters."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CPUOffloadConfig:
    # Rounded pinned block capacity, shared across this process's modules.
    max_cpu_bytes: int = 8 * 1024**3
    min_tensor_bytes: int = 1024**2
    max_inflight_bytes: int = 512 * 1024**2
    max_prefetch_bytes: int = 512 * 1024**2
    prefetch_groups: int = 1
    scope_prefetch_groups: int = 1
    reload_on_compute_stream: bool = True
    skip_retained_inputs: bool = True
    pool_slab_bytes: int = 0
    min_offload_gap: int = 0
    # Minimum backing-storage bytes * scheduled local F/B action distance.
    # Zero disables this size-aware lifetime filter; unknown distances retain
    # the existing behavior (ordinary non-pipeline autograd has no schedule).
    min_offload_size_gap_bytes: int = 0
    # Snapshot producer readiness at saved-tensor capture. Group commit can
    # then overlap D2H with later compute that only reads immutable inputs.
    # Custom asynchronous producers must join the saving stream before save;
    # this option cannot infer producer completion from a tensor's allocation.
    use_save_events: bool = False
    # Deprecated constructor/serialized-config alias. ComputeConfig migrates
    # this to its independent release_pipeline_inputs field and clears it here.
    # The activation store never uses this flag to select regions or transfers.
    release_pipeline_inputs: bool = False
    # Leave headroom for the dataloader, optimizer, compilation and other ranks.
    host_reserve_bytes: int = 32 * 1024**3
    # Keep disproportionate views resident; 0 permits copying their full storage.
    max_storage_to_tensor_ratio: int = 8
    # Restore coverage: auto uses scheduled F/B scopes, groups otherwise.
    transfer_unit: str = 'auto'
    # Forward submission is independent of restore coverage. auto preserves
    # the historical transfer_unit choice; group starts D2H inside a forward.
    # checkpoint additionally submits checkpoint inputs before its compute.
    forward_commit_unit: str = 'auto'
    # Target bytes per physical D2H batch. Scope submissions also flush at a
    # closed region once accumulated bytes reach this target. Zero disables
    # subdivision/early flush. One indivisible storage may exceed the target.
    commit_batch_bytes: int = 0
    # Maximum bytes per transfer call/kernel (0: one call per storage). Chunking preserves
    # whole-storage ownership/aliasing; it does not cap a tensor's allocation.
    copy_chunk_bytes: int = 0
    # Local virtual-stage F/B actions before the consuming backward. Zero
    # disables cross-action lookahead; group/current-backward refill remains.
    pipeline_prefetch_steps: int = 1
    # Generated activation API: 1 added F/B action scheduling; 2 routes
    # checkpoints through their saved-tensor adapter. Missing legacy ABI is 0.
    _schedule_version: int = 2
    # Appended to retain the original positional constructor field order.
    # host bounds CPU admission; stream separates GPU execution pacing from
    # the actual unfinished submitted-source limit in max_pending_bytes.
    backpressure_mode: str = 'host'
    max_pending_bytes: int = 0
    # Which pipeline actions may issue cross-scope H2D prefetch requests.
    # Backward-only avoids launching them during forward CPU rendezvous; the
    # first backward still loads its own scope correctly when no lead exists.
    prefetch_trigger: str = 'all'
    # Pipeline D2H may use backward compute windows instead of forward exits.
    # Explicit scope collection and a pending-source budget are required.
    d2h_launch_phase: str = 'forward'
    d2h_scopes_per_backward: int = 1
    # Optional CUDA transfer implementations, selected independently by direction.
    h2d_backend: str = 'dma'
    h2d_sm_blocks: int = 16
    d2h_backend: str = 'dma'
    d2h_sm_blocks: int = 4

    def __post_init__(self):
        for name, value in vars(self).items():
            if name in ('h2d_backend', 'd2h_backend'):
                if value not in ('dma', 'sm'):
                    raise ValueError(f'{name} must be dma or sm')
                continue
            if name == 'backpressure_mode':
                if value not in ('host', 'stream'):
                    raise ValueError('backpressure_mode must be host or stream')
                continue
            if name == 'prefetch_trigger':
                if value not in ('all', 'backward'):
                    raise ValueError('prefetch_trigger must be all or backward')
                continue
            if name == 'd2h_launch_phase':
                if value not in ('forward', 'backward'):
                    raise ValueError('d2h_launch_phase must be forward or backward')
                continue
            if name in ('transfer_unit', 'forward_commit_unit'):
                choices = ('auto', 'scope', 'group', 'checkpoint') if name == 'forward_commit_unit' else ('auto', 'scope', 'group')
                if value not in choices:
                    raise ValueError(f'{name} must be one of {choices}')
                continue
            if name in ('reload_on_compute_stream', 'skip_retained_inputs', 'use_save_events', 'release_pipeline_inputs'):
                if not isinstance(value, bool):
                    raise ValueError(f'{name} must be a bool')
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer, got {value!r}')
        if self.backpressure_mode == 'stream' and (
                self.max_pending_bytes == 0 or self.max_pending_bytes < self.max_inflight_bytes):
            raise ValueError('stream backpressure requires max_pending_bytes >= max_inflight_bytes and > 0')
        for name in ('h2d_sm_blocks', 'd2h_sm_blocks'):
            if getattr(self, name) == 0:
                raise ValueError(f'{name} must be positive')
        if self.d2h_launch_phase == 'backward' and (
                self.forward_commit_unit != 'scope' or not self.max_pending_bytes
                or not self.d2h_scopes_per_backward):
            raise ValueError('backward D2H launch requires scope forward commits, '
                             'positive max_pending_bytes and d2h_scopes_per_backward')

    def restore_unit(self, *, pipeline):
        return ('scope' if pipeline else 'group') if self.transfer_unit == 'auto' else self.transfer_unit

    def commit_unit(self, *, pipeline):
        if self.forward_commit_unit == 'checkpoint':
            return 'group'
        return self.restore_unit(pipeline=pipeline) if self.forward_commit_unit == 'auto' else self.forward_commit_unit
