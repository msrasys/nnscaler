# Combined 1F1B D: performance and topology findings

This note records the measurements and safety boundaries behind
`yomia/combined-1f1b-d-performance-topology`.  It intentionally contains text
summaries only; CUPTI traces and generated temporary modules are not committed.

## C-phase performance

### Method

`tests/parallel_module/test_phase_moe_perf_benchmark.py` runs fresh 2-GPU
jobs for each measurement, uses two model scales, discards two warmups, places
CUDA synchronizations around the train and optimizer portions, alternates the
serial/default order (AB/BA), prints every timed sample, and reports a
per-round median plus MAD.  The default phase configuration has no synthetic
CUDA stream context; `phase-dedicated-stream` is a retained ablation.

The 2026-08-05 `srgws-17` run (RTX PRO 6000 Blackwell, CUDA 12.8 / NCCL
2.27.5) produced these **train-step** median-of-round values:

| scale | serial | phase default (no extra stream) | dedicated dispatch stream | serial-default paired median / MAD |
|---|---:|---:|---:|---:|
| small | 39.883 ms | 66.181 ms | 75.222 ms | -23.593 / 3.358 ms |
| large | 94.338 ms | 127.228 ms | 150.484 ms | -28.939 / 5.203 ms |

The previous C baseline's large-scale phase median was 154.247 ms (six
rounds, synthetic dispatch and expert stream contexts).  The safe no-stream
path therefore reduced the observed large-scale phase time by roughly 17.5%,
but it does not erase the phase-vs-serial regression.  The benchmark honestly
reports this rather than asserting a gain.

### Profiling evidence

`tests/parallel_module/test_phase_moe_profile.py` is opt-in
(`NN_SCALER_RUN_PHASE_PROFILE=1`) and wraps `fexecute`/`backward` with both
`torch.profiler.record_function` and NVTX ranges.  The profile is CUPTI-backed
for CUDA activity and also collects a `cProfile` sample.

On the profile scale, serial used four `fexecute` and four autograd-backward
boundaries; phase mode used 32 of each.  The phase profile showed 674 CUDA
kernel launches, 160 `aten::empty` calls, and 32 `c10d::alltoall_base_` calls;
the all-to-all count and allocator count are not unique phase costs.  The
large extra cost is the eightfold executor/autograd segmentation: per-segment
backward self CPU times dominate the profile, while the final CUDA synchronize
also grows (serial ~9.6 ms in this capture; no-stream phase ~24.6 ms).

### Implemented, safe reductions

* `DeviceGroup.get_stream('default')` now returns the real current default
  stream instead of allocating a named non-default stream.
* Schedule codegen emits no `with torch.cuda.stream(...)` wrapper for
  `None`/`'default'`, omits self-waits, and records inputs only when a real
  named consumer stream exists.
* PhaseMoE's default avoids a synthetic dispatch/expert stream context;
  explicit `Work.wait()` remains the readiness and lifetime edge.
* Executor FIFO state uses `deque.popleft()` rather than O(n) `list.pop(0)`.
* `sync_tensors` skips its completed-work scan only when no work exists; it
  retains the exact drain/wait behavior when any async work is pending.

An attempted phase-wide `sync_tensors` bypass failed lifecycle and
exception-cleanup tests, so it was rejected.  Fusing phase methods would also
remove independently schedulable backward islands; it is intentionally not
claimed as a safe optimization yet.

## PP2 x EP2 independent replica activations

RVD's `R` axis means **equal replicas**.  The phase MoE policy has
replicated attention/gate operations but may feed rank-distinct local batches
and rank-distinct expert shards.  At a PP boundary the pre-existing inter-RVD
path can convert `D -> R` with destination `all_gather`, silently turning
rank lane values into a shared value.  Symmetric inputs hide it because both
source lanes are equal and phase-vs-serial comparison shares the same faulty
adapter.

`IRFullTensor.mark_independent_replica_lanes()` now carries an explicit
semantic marker (including `like()`/gradient copies).  An opt-in policy
`make_pas(..., independent_pp_replica_lanes=True)` marks PP stage outputs.
`ConcurrentGener` validates the pure replica layouts and then **fails closed**
before the generic RVD path can gather them.  The diagnostic tells users that
an explicit bijective redistribution plus a global lane issue/wait schedule is
required.

A direct `src[i] -> dst[i]` P2P prototype and a world-group collective
prototype were both investigated but not shipped: different PP stages can
reach forward and reverse lane operations in different microbatch orders,
which violates the default NCCL process group's global untagged P2P order;
a world collective likewise requires a globally identical call order.  This
is a scheduler/runtime redesign, not a safe local adapter rewrite.  The
fail-fast has unit and CPU-gencode coverage; the existing symmetric PP2xEP2
regression remains available for legacy RVD semantics.

The required implementation plan is:

1. model forward/reverse transfers as explicit `RecvIssue`/`RecvWait` and
   `SendIssue`/`SendWait` nodes, with a stable lane identity;
2. construct a single global cross-rank schedule over those nodes;
3. prove each lane transfer's global ordinal and target layout, then lower to
   paired P2P or an explicit collective redistribution; and
4. add the asymmetric 4-GPU reference only after the global ordinal is part
   of the executable IR (not merely a local policy annotation).

## All-stage phase interleaving

`PhaseAwareSched.sched_1f1b_global_phase_aware` is an opt-in deterministic
list scheduler (`make_pas(..., global_phase_interleave=True)`).  It builds
unit-span tasks for every `(stage, microbatch, phase)` and adds explicit
canonical edges: phase-forward order, mirrored backward order, local F-to-B,
upstream activation, and downstream gradient.  It chooses ready B/F work in
a deterministic alternating preference and finally calls `SchedulePlan`'s
validation.

This differs from simply moving F(m+1) earlier or B(m) later: non-first F is
not ready until the upstream activation completes, and non-last B is not ready
until the downstream gradient completes.  CPU sweeps cover PP2/PP4 and
multiple microbatch counts; real 2-GPU PP2 and 4-GPU PP2/PP4 runs each use
three deadlock-guarded repeats.  PP2 and PP4 compare per-rank weights/Adam
state against the standard phase schedule.
Some intermediate PP4 stages have no ready F/B alternation because both
cross-stage dependencies are on their critical path; the scheduler does not
invent an illegal window.

## Multi-peer A scheduling

The default world NCCL process group has a single untagged P2P ordering domain.
The existing `GlobalCommSchedule` therefore remains the safe production
choice: it builds one global topological order, projects it to ranks, and
when two or more peer pairs exist it fails closed to the single per-device
communication chain.  A per-peer process-group experiment was already
pre-created in a globally consistent order and still deadlocked; that rules
out lazy-group creation as the explanation but does not make separate groups
a verified solution.  `batch_isend_irecv` is useful only for jointly issued
matching send/receive pairs; batching unrelated peer-pair calls separately
does not establish a global cross-pair order.

The 2026-08-05 4-GPU run passed the 4-stage/3-peer-pair repeated
no-deadlock, numeric-equivalence, generated channel/cap, and cap/property
coverage in
`test_combined_1f1b_multistage_e2e.py` and `test_global_schedule.py`.  The
D branch preserves this fail-closed behavior rather than replacing it with an
unverified pair-local optimization.
