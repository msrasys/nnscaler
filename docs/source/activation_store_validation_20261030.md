# Activation store: architecture and correctness validation

The reusable store and its NNScaler/PyTorch adapters have passed CPU and CUDA
correctness checks. Independent transport performance probes are prepared and
queued; there is no new full-model throughput claim and no claim of optimal DMA
efficiency. The October 29 model numbers describe the earlier implementation.

## Implemented contracts

* An explicit forward handle preserves its autograd connection separately from
  the payload. Dense, sliced and expanded values preserve first/second gradients.
* `saved_tensors(exclude=...)` and NNScaler's annotated regions use the same data
  records, pools, transfer engine and hook cleanup. Parameters/buffers can remain
  resident without duplicating an activation store implementation.
* `prefetch/load/evict/release` handles asynchronous copies and repeated forward
  reuse. A surviving known GPU view is reused with its existing version counter.
  Unknown external owners are never overwritten or treated as freed.
* Each physical copy retains weak allocation/view identities and version
  metadata. Evicted copies remain checked while aliases survive. The storage
  lookup index is separate from the live-record registry, so allocator identity
  reuse cannot hide saved values from teardown checks.
* GPU-resident and budget-refused strided values still join producer readiness
  and record consumer streams. This includes loads before batch commit. Changing
  the residency policy must not change stream correctness.
* Shared pinned budgets include live, reusable and in-flight blocks. A closed
  handle rejects further access; closing one store does not close another store's
  pool or invalidate tensors already returned to consumers.

This is explicit activation ownership, not transparent GPU virtual memory.
External Python caches must adopt handles. Leaf-gradient targets, unhandled
autograd saves, and external tensor aliases can still retain GPU allocations.
Opaque writes must maintain version counters. Explicit CUDA handles currently
require strided tensors.

## Checks completed

| Scope | Result |
| --- | --- |
| Runtime, forward handles, aliases, versions, delayed streams, profiler and ledger integration | 189 passed with CUDA enabled |
| Pipeline offload/input-release/output-release combinations | All eight combinations passed |
| Real Attention/MoE PP2/VPP4 policy comparison | Five updates; full gradient and parameter checks passed |
| Core configuration/ownership/parser CPU checks | 153 passed; CUDA-only cases skipped in that run |
| Application configuration, ledger, policy, memory summary and public profiler CPU checks | 115 passed; CUDA-only cases skipped |

These groups overlap and are not added together. The initial policy invocation
used three updates and was rejected by its argument parser before execution;
the corrected five-update invocation passed. During the policy retry another
GPU was compiling, so those timings are not performance evidence.

The CUDA checks used frozen source v3. Two subsequent edits removed trailing
blank lines only; Python AST equivalence with runtime commit `40f0f0d3` and
application commit `bf8ff5b2` is recorded. Source hashes, compressed logs and
resource audit records are in [the evidence directory](../activation-store-results/20261030/validation.json).
Validation GPU leases were released, and their guards report no owned processes
remaining. The largest sampled validation process-tree RSS was about 12.5 GiB;
single-GPU used memory remained below 3 GiB.

## Profiling and configuration

`store.trace()` is bounded, metadata-only, thread-safe at scope exit, and does
not patch runtime methods. With the PyTorch profiler active, it supplies
`activation_store/batch_submit`, `d2h`, `h2d`, and `load` ranges. The application
profiler now uses this public interface and native CUDA activities; it no longer
imports a test model to obtain instrumentation.

The independent probe fixes the payload at 256 MiB each direction and compares
direct copies, a store round trip, and equal-compute serial/overlapped schedules.
Prepared variants change tensor count, transfer-batch size, D2H inflight budget,
or prefetch budget. They bind to the selected GPU's NUMA CPUs, run only under an
exclusive GPU reservation, exclude warmup/source construction/validation from
timing, and verify actual D2H/H2D byte counts and zero residual pool use.

The queued runner is `/tmp/activation-store-20261030/run_probes_v4.py`. It writes
individual results and `probe-summary-v4.json` after every case succeeds. Its
resource guard preserves 256 GiB host/cgroup headroom, 300 GiB disk free and
16 GiB GPU free, with a time limit. No model checkpoint or large generated model
is required. Probe results are pending; CPU event counts and zero demand loads
alone do not establish that GPU transfer cost is hidden.
