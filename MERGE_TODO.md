# `yileiyang/yoco_pp` Main Integration and PR Split TODO

This document records the plan for integrating `yileiyang/yoco_pp` with
`origin/main` and then upstreaming the result as reviewable PRs. The existing
`yileiyang/yoco_pp` branch is the known-good production reference and must not
be rewritten while integration and parity validation are in progress.

## Baseline

- Known-good feature commit: `40e5e197d05ddb0efe2572e0edd2405e7879ec75`
- Main commit used for this integration: `5847e7960d2669522f9dff4de4a058a7955a3566`
- Integration branch: `yileiyang/yoco_pp_main_integration`
- Divergence before merge: 34 feature-only commits and 4 main-only commits.
- Net feature diff before merge: 53 files, 4609 insertions, 1332 deletions.
- Both sides changed 13 files. The pre-merge simulation found 21 conflict
  hunks in 5 files.

Before doing future work, fetch `origin/main` again and update the commit and
conflict counts above if main has advanced.

## Integration Rules

- [x] Preserve `yileiyang/yoco_pp` as the pre-merge reference.
- [x] Perform conflict resolution only on the integration branch.
- [x] Merge, do not rebase, the historical feature branch. Rebasing 31
  non-merge commits would repeatedly surface conflicts from superseded code.
- [x] Preserve current main APIs and data formats unless the feature requires a
  deliberate extension.
- [x] Never resolve a scheduler, reducer, or metadata conflict by blindly
  selecting `ours` or `theirs`.
- [x] Keep the integration merge local until correctness and performance gates
  pass.
- [ ] Use fresh branches from the latest main for real PRs. The integration
  branch is a behavior reference, not the base branch for all PRs.

## Expected Conflict Resolution

### `nnscaler/codegen/schedule/schedule.py`

Main added multi-scheduler support and refactored schedule code around
`self.execplan`. The feature branch added asynchronous pipeline sends/receives,
send-bundle placement, reducer scheduling, communication draining, and
pseudo-free handling.

- [x] Keep main's multi-scheduler and execution-plan-local interfaces.
- [x] Port send-bundle range/event collection onto the main interfaces.
- [x] Preserve async receive tracking and ensure receives are waited before use.
- [x] Preserve send draining and output lifetime/pseudo-free guarantees.
- [x] Preserve scheduled reducer enable/skip transitions.
- [x] Apply equivalent logic to both training and inference generation.

### `nnscaler/parallel.py` and `nnscaler/runtime/module.py`

Main compacted per-rank attribute metadata. The feature branch independently
added merged metadata, multi-process code generation, selective full-model
chunk loading, and mmap loading.

- [x] Use the feature's versioned compact metadata format as the canonical format.
- [x] Keep main merged and legacy per-rank metadata fallbacks for backward
  intended.
- [x] Make multi-process workers emit data that compacts into the canonical format.
- [x] Retain selective parameter-chunk loading.
- [x] Retain mmap loading without keeping unnecessary mappings alive.
- [x] Check reuse detection for fresh compact, main merged, and legacy code.
- [x] Check deep worker payload serialization and worker failure cleanup.

### `nnscaler/runtime/adapter/reducer.py`

Main added per-bucket reducer options, scale-unit DP/EP behavior, and unused
bucket/None-gradient support. The feature branch added scheduled-pipeline async
reducer behavior and manual gradient accumulation/readiness tracking.

- [x] Preserve main's per-bucket configuration as the source of truth.
- [x] Preserve None-gradient and unused-bucket behavior.
- [x] Port manual gradient accumulation and readiness tracking.
- [x] Reset all async/manual state consistently between steps.
- [x] Preserve parameter-level sharding semantics per bucket.
- [x] Verify sparse optimizer-state handling.
- [x] Combine both test suites rather than choosing one side.

## Proposed PR Stack

The historical commits are not suitable PR boundaries: later commits repair or
replace earlier implementations. Each PR below should be reconstructed from
the final integration tree and must pass independently.

### PR 1: Ignore non-tensor autodist cost outputs

- Scope: `nnscaler/autodist/cost_database.py`
- Original intent: `bbc6889a`
- Risk: low
- Dependency: none
- Gate: focused cost-database tests.

### PR 2: Split oversized gradients before Apex L2 norm

- Scope: `nnscaler/runtime/gnorm.py`, `tests/runtime/test_gnorm.py`
- Original intent: `a2235d5e`
- Risk: low
- Dependency: none
- Gate: gnorm unit tests, including large zero-copy gradient views.

### PR 3: Handle sparse optimizer state in pipeline Muon runs

- Scope: `runtime/f16_optimizer.py`, `runtime/hybrid_optimizer.py`, and the
  smallest required reducer changes.
- Original intent: `8035d6e1`
- Risk: low to medium
- Dependency: latest main reducer APIs.
- Gate: sparse-state optimizer tests and one pipeline optimizer step.

### PR 4: Load only required full-model parameter chunks and use mmap

- Scope: parser attribute-content indexing and runtime parameter loading.
- Original intent: `021b1f45`, `bafcc122`
- Risk: medium
- Dependency: main compact metadata format.
- Gate: fresh generation, legacy loading if supported, selective chunk count,
  mmap behavior, and multi-rank load.

### PR 5: Multi-process per-rank code generation

- Scope: `codegen/worker.py`, `codegen/serialization.py`, `parallel.py`, CLI
  configuration, metadata compaction integration, and tests.
- Original intent: `4395a258`, `40e5e197`
- Risk: medium to high
- Dependency: PR 4 only if shared metadata/loading helpers are retained there.
- Gate: workers 1/2/8, deep payloads, worker crash cleanup, deterministic files,
  reuse, and 48-rank generation.

### PR 6: Ring-attention sequence-group and rank-order correctness

- Scope: customized ring-attention operators and focused tests.
- Original intent: `232eecf3`, `23f3d6f5` ring-attention portions.
- Risk: medium
- Dependency: none where possible.
- Gate: return-LSE, variable-length, sequence groups, and reordered ranks.

### PR 7: Narrow pipeline segment boundaries and preserve alias layouts

- Scope: segment expansion, IR tensor aliases, inter-RVD generation, execution
  plan reuse, and boundary tests.
- Original intent: `55030118`, `c86d3dfe`, `3d82460f`, `7209e211`, `c53d12a7`
  boundary-layout portions.
- Risk: high
- Dependency: main segment/execplan APIs.
- Gate: forward/backward parity, alias components, object boundaries, narrowed
  layouts, and cross-stage resharding.

### PR 8: Async pipeline P2P and safe output lifetime

- Scope: runtime communication handlers, dedicated process groups, generated
  send/receive placement, drain semantics, and output pseudo-free.
- Original intent: `0e842b84`, `115f1e83`, `68832cf6`, `1480d8c3`, `9a340d07`,
  `59e7ac12` and relevant later fixes.
- Risk: high
- Dependency: PR 7 if narrowed boundary IDs are used by generated adapters.
- Gate: sync/async numerical parity, multiple sends per output, backward order,
  no use-after-free, no deadlock, and memory regression tests.

### PR 9: Scheduled-pipeline reducer integration

- Scope: scheduled reducer state, async/manual gradient accumulation, codegen
  reducer placement, sparse optimizer interaction, and tests.
- Original intent: `ad1176ae`, `969533c2` and later reducer fixes.
- Risk: high
- Dependency: PR 8 and latest main per-bucket reducer implementation.
- Gate: async reducer on/off, None gradients, unused buckets, per-bucket config,
  parameter sharding, and optimizer-step parity.

### PR 10: VPP colocated shared parameters and stage-local partition policy

- Scope: VPP shared-parameter multiref decisions, stage-local TP/EP dry-run,
  fixed-policy hooks needed by applications, and focused pipeline tests.
- Original intent: `cfc79cf4`, `97449e5a`, `24b66212` and final policy changes.
- Risk: high
- Dependency: PRs 7-9.
- Gate: PP and interleaved VPP parity, shared weights reused on one physical
  stage, shared weights spanning physical stages, TP/EP stage-local partition,
  and application smoke tests.

## Merge Validation Gates

### A. Structural audit

- [x] Merge commit has both the known-good feature commit and latest main as
  ancestors.
- [x] No unresolved conflict markers or unmerged paths.
- [x] `git diff --check` passes.
- [x] Review `old-yoco -> integration` and `main -> integration` diffs for
  accidental deletion or duplicated implementations.
- [x] Confirm all conflict resolutions are covered by tests.

### B. Focused unit tests

- [x] `tests/runtime/test_reducer.py`
- [x] `tests/runtime/test_executor.py`
- [x] `tests/runtime/test_gnorm.py`
- [x] `tests/codegen/test_emit.py`
- [x] `tests/parallel_module/test_codegen_workers.py`
- [x] `tests/parallel_module/test_gencode_pipeline.py`
- [x] `tests/parallel_module/test_shared_param_pipeline.py`
- [x] `tests/graph/parser/test_attr_content.py`
- [x] `tests/graph/test_segment.py`
- [x] `tests/ir/test_adapter_prim.py`
- [x] `tests/ir/test_cten.py`
- [x] `tests/test_execplan.py`
- [x] `tests/test_policies.py`
- [ ] Main multi-scheduler and scale-unit DP/EP tests.

### C. Pre/post numerical parity

Run the same deterministic small model on the pre-merge commit and integration
commit and compare:

- [ ] Forward outputs and loss.
- [ ] Input and parameter gradients.
- [ ] Parameters after one optimizer step.
- [ ] Non-pipeline versus PP versus interleaved VPP results.
- [ ] Sync communication versus async communication.
- [ ] Sync reducer versus async reducer.

Use exact equality where deterministic CPU execution permits it. For GPU/BF16,
define tolerances before looking at the result and also compare loss/gradient
trajectories over multiple steps.

### D. Multi-GPU smoke matrix

- [ ] 2/4-GPU ordinary pipeline.
- [ ] 4/8-GPU interleaved VPP.
- [ ] TP/EP within pipeline stages.
- [ ] Shared parameters colocated and cross-stage.
- [ ] Narrowed tensor and object segment boundaries.
- [ ] Multi-scheduler with changing update frequency.
- [ ] Scale-unit DP/EP with per-bucket reducer options.
- [ ] `codegen_workers=1` and multi-process codegen.
- [ ] Generated-code reuse after process restart.

### E. Production regression

- [ ] Run the known YOCO text PP/VPP configuration before and after merge on
  equivalent hardware.
- [ ] Compare loss, gradient norm, peak memory, step wall time, and TFLOPS.
- [ ] Run enough warm steps to exclude compilation/autotuning time.
- [ ] Confirm no NCCL hang, collective ordering error, delayed reducer work, or
  output lifetime failure.
- [ ] Regenerate code unless old-generated-code compatibility is explicitly
  part of the test contract.

## Completion Criteria

The integration branch is considered usable only when:

1. The focused main and feature tests pass.
2. Small-model PP/VPP numerical parity passes.
3. At least one multi-GPU YOCO PP/VPP smoke run completes.
4. No unexplained memory or steady-state throughput regression remains.
5. Every manual conflict resolution has an identified test or explicit review
   note.

## Current Integration Validation (2026-09-01)

The first integration merge was validated on one node with 8 NVIDIA B200 GPUs.
The checks below are evidence for the integration branch; they do not replace
the production YOCO regression gate above.

### Passed

- 239 distinct tests passed across reducer, executor, gradient norm, codegen,
  compact/merged/legacy metadata, policies, IR boundaries, shared parameters,
  scale-unit DP/EP, CP+EP, mixed modules, PP, interleaved VPP, FBW, asynchronous
  P2P/reducer, and multi-scheduler paths.
- All 4 tests in `tests/cli/test_trainer_pipeline.py` passed on 4 GPUs,
  including async communication/reducer/FBW and multi-scheduler execution.
- `tests/parallel_module/test_async.py::test_pp2` and
  `test_interleaved_pp[2]` passed, comparing synchronous and asynchronous
  reducer results through multiple optimizer steps.
- The 8-GPU `CP4+EP2` runtime test passed.
- Fresh single- and multi-process codegen, generated-code reuse, compact
  metadata, main's merged metadata compatibility, and legacy per-rank metadata
  compatibility passed.
- Conflict markers, Python compilation, and `git diff --check` passed after
  resolution.

The test image did not have the declared development dependency
`mosaicml-streaming`. Pipeline and mixed-module tests that only use
`SimpleDataset` were run with a temporary `/tmp` import stub; no stub or test
environment workaround is committed to the branch.

### Baseline failures, not integration regressions

- Three simulated-DP comparisons fail at `1e-6` tolerance on B200. The same
  three tests fail with the same values on untouched `origin/main`; full- and
  split-batch GEMMs differ by roughly `1e-4` due to accumulation order.
- The narrowed-boundary, shared-parameter pipeline, and 8-GPU PP+TP tests fail
  their historical numerical tolerances on B200. Each failure was reproduced
  on untouched `yileiyang/yoco_pp` with the same displayed tensors and, where
  reported, the same maximum error. All distributed executions completed; no
  hang, crash, or collective-order error occurred.

### Still required before upstream PRs or production adoption

- Run the real YOCO PP/VPP training smoke and compare loss, gradient norm,
  memory, wall time, and TFLOPS against commit `40e5e197`.
- Decide separately whether B200 test tolerances should be hardware-aware;
  this integration intentionally does not hide those pre-existing failures.
- Run the complete CI environment with all development dependencies installed.
