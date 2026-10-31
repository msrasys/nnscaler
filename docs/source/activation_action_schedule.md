# Activation scheduling in local F/B units

In pipeline mode, one scheduling unit is one local execution of a virtual
stage's forward or backward for a particular microbatch. Communication adapters
do not increment this counter. It is not a physical GPU index, Transformer layer
number, optimizer step, or a guessed reverse-forward order.

## Independent configuration axes

`CPUOffloadConfig` exposes:

| Setting | Meaning | Default |
| --- | --- | --- |
| `transfer_unit` | Restore coverage: `auto` uses F/B scopes in pipeline and regions otherwise; `scope` restores a scope, `group` uses region lookahead | `auto` |
| `forward_commit_unit` | D2H submission: `group` submits at region boundaries; `scope` accumulates regions; `auto` follows `transfer_unit` for compatibility | `auto` |
| `commit_batch_bytes` | Physical D2H batch target; also enables early scope flush at closed region boundaries; one indivisible storage may exceed it | `0` (disabled) |
| `copy_chunk_bytes` | Maximum DMA-call size; does not split tensor ownership or bound total GPU allocations | `0` (one copy per storage) |
| `pipeline_prefetch_steps` | Start up to N local F/B actions before the consuming backward, once its forward scope has closed | `1` |
| `max_prefetch_bytes` | Shared speculative reload-byte window; independent of lookahead distance | 512 MiB |
| `max_inflight_bytes` | D2H queue threshold before admitting the next complete transfer batch | 512 MiB |
| `scope_prefetch_groups` | Backward-entry group window when using group mode | `1` |
| `prefetch_groups` | Intra-backward group lookahead in group mode; `0` disables all speculative prefetch | `1` |

`pipeline_prefetch_steps=0` disables cross-action lookahead. Backward may still
refill its current scope, or use ordinary group hooks. Use `prefetch_groups=0`
for completely demand-driven restores. Group-count settings do not turn whole
F/B scopes into a group-count window in scope mode.

The application exposes `--offload_transfer_unit`, `--offload_forward_commit_unit`,
`--offload_commit_batch_mb`, `--offload_copy_chunk_mb` and
`--offload_pipeline_prefetch_steps`, alongside the existing byte-window options.

## Scheduling and producer readiness

The generated program records `(scope key, forward action, backward action)` for
each eligible local scope. Keys include the virtual segment and microbatch. A
small runtime cursor applies the configured distance to this actual order.

For a backward at position B with producer F, the desired trigger is before
`max(0, B-N)`. If that point precedes forward completion, it is clipped to after
F. Here completion means closing the CPU-side forward scope and submitting its
offloads; CUDA events still enforce actual GPU data readiness. The scheduler
retries before/after subsequent actions while the target is
within the window. It never requests values from an unfinished forward scope.
Hints before compute precede that action's receive waits; after-action retries
follow scheduled data-reference release.

For example, consider local actions `F(a0), F(b0), F(a1), B(a0), F(b1), B(b0),
B(a1), B(b1)`. With N=1, `a0` first prefetches before `F(a1)`. With N=4, its first
possible prefetch is after `F(a0)`, not before its producer. `b0` is requested
before `a1` because its backward occurs first, regardless of forward nesting.

Nearer backward deadlines have priority when the byte window is full. A blocked
request is retried rather than forgotten; admitted scopes are not rescanned at
every action. Values larger than the entire speculative window remain demand
loads, without permanently blocking all other requests.

The pure `ActivationSchedulePlan.describe(N)` exposes first eligible triggers
for inspection. Public store traces emit `schedule_prefetch` events; their
`count` is the actual remaining local-action lead at that request, which can be
smaller than N after producer clipping or a budget retry.

## Transfer ownership and granularity

With the legacy defaults, scope restore also accumulates a forward-scope batch
and commits D2H when F exits. Submission and restoration are now independent:
`transfer_unit=scope, forward_commit_unit=group` starts D2H inside F while keeping
one scope restore queue and F/B action lookahead. A positive `commit_batch_bytes`
flushes accumulated scope saves at a closed region once its queued bytes reach
the target. The store splits submissions in storage order, with an oversized
storage submitted alone. Discarded saves are pruned and batches rejected by the
CPU budget do not wait for unrelated DMA. Each record keeps its actual
producer-ready event; the transport waits before the relevant copy, instead of
gating earlier ready data on every later producer. Without saved events it uses
the corresponding producer stream dependency. Checkpoint intermediates are still
recomputed, and parameters,
buffers and unselected values are still excluded. This does not mean every
tensor produced by a stage is offloaded.

The consuming backward uses one scope restore queue, ordered from its entry.
The queue advances as values are staged and refills when consuming saved values
releases speculative capacity. It retains weak record references and does not
repeatedly scan all previously restored values. A forward immediately followed
by its own backward keeps the whole scope resident, avoiding a round trip with
no intervening compute action.

Group restore mode retains backward group lookahead.
Its entry window counts groups introducing distinct pending records; aliases
must not consume depth twice. Repeated schedule hints refill that same entry
window rather than silently expanding it to further groups.

Whole-scope D2H can retain more source data until F finishes. Group mode can
release data and overlap D2H earlier inside a long forward. This is an explicit
granularity tradeoff, so neither choice is declared universally faster or more
memory efficient. The D2H threshold permits one whole batch beyond its limit;
speculative byte accounting is not a total physical GPU-memory limit.

`copy_chunk_bytes` limits individual D2H/H2D calls. One final event fences all
chunks of the storage, preserving views, versions and host-buffer reuse. The
destination remains a whole storage: splitting DMA calls alone cannot reduce
the full allocation needed by one tensor. Reclaiming different storages earlier
comes from submission/restore scheduling. A failed partial submission drains
its copy stream before the caller can recycle an unfenced host buffer.

The storage-independent queues live in `activation_batch.py`: `ForwardCommitQueue`
owns pending sources, `ScopeRestoreQueue` owns weak records, and `ActivationBatch`
provides explicit flush/commit boundaries. `ActivationTransferEngine` handles
streams and DMA; the F/B plan has no knowledge of DMA chunks or region types.

## Optional dependency timing

`store.trace(cuda_timing=True)` brackets each restored-value consumer dependency
with CUDA events on that consumer's stream. The normal path records no timing
events and queries no event readiness. Timing samples are bounded separately
from metadata by `max_events`; exports report dropped and pending samples.
`to_dict()` never synchronizes: pending intervals have `elapsed_ms=None`.
Samples retain CUDA events but never tensors or saved records.

These intervals include event overhead/host submission gaps and can overlap across consumer streams;
their sum is not global critical-path cost. The application profiler format 3
reports the sum/max, pending and dropped counts after its existing end-of-profile
synchronization. Ordinary training remains uninstrumented. The diagnostic replay
also exports these intervals separately from its broader backward read ranges.

## Lifetime and compatibility

Scopes and queues do not form Python reference cycles. The schedule context
removes scope metadata on normal return and exceptions; saved tokens still own
their data until actual consumers release them. The generic store remains
responsible for versions, streams, pinned budgets and alias-safe reclamation.

Generated schedules must be regenerated once for this API. Old serialized
offload configs are marked with schedule version 0, making generated-code reuse
fail while leaving the raw trace reusable. Direct use of such a legacy compiled
offload config reports that regeneration is required. Existing config imports
and non-pipeline group behavior remain supported.

Once code is generated, the plan itself does not bake in N. Change a store's
configuration between completed training iterations to tune distance, unit and
byte windows; do not change ownership/producer policy during an active scope.
The ordinary compilation entry point still checks the supplied config for reuse.
The new submission/chunk fields default to the previous behavior. Schedule ABI 1
and its action keys are unchanged; missing fields in older configs use these defaults.

## Validation

`benchmark/benchmark_activation_schedule.py` replays the same local interleaved
F/B order using the real adapter. It checks full-byte transfers, checksums and
empty pools after every iteration. `--sync-actions` optionally waits only for
each compute action to finish, leaving independent transfers in flight; this
makes the available action window explicit. These are controlled synthetic
actions, not full-model throughput. Timing, GPU peaks and trace evidence belong
to the accompanying validation results, not to an assumption that more lookahead
always helps.

Correctness, lookahead/byte-window measurements and resource records are in the
[F/B action validation report](activation_action_validation_20261030.md).
The independent submission/chunk/timing changes were added afterwards; the
October 30 performance measurements do not measure those new opt-in modes.
Their CPU results and pending CUDA checks are listed in the
[submission validation record](activation_submission_validation_20261031.md).
