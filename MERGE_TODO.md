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
  compatibility.
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

The earlier 10-PR list was still too coarse. In particular, it combined
selective loading with mmap, compact metadata with multi-process codegen,
multiple segment-layout features, and several distinct VPP policy features.
The target is now 18 candidate PRs. This number is not a quota: adjacent PRs
may be combined only if extraction proves that neither side is independently
correct and testable.

Historical commits are evidence of intent, not PR boundaries. Later commits
repair or replace earlier implementations. Every PR must be reconstructed from
the final integration tree on a fresh branch and must be correct by itself.

### Independent runtime fixes

#### PR 1: Ignore non-tensor autodist cost outputs

- Scope: `nnscaler/autodist/cost_database.py`.
- Original intent: `bbc6889a`.
- Risk/dependency: low; none.
- Gate: focused cost-database tests.

#### PR 2: Split oversized gradients before Apex L2 norm

- Scope: `nnscaler/runtime/gnorm.py` and focused tests.
- Original intent: `a2235d5e`.
- Risk/dependency: low; none.
- Gate: gnorm tests including oversized zero-copy gradient views.

#### PR 3: Handle sparse optimizer state in pipeline Muon runs

- Scope: the smallest changes in `f16_optimizer.py`, `hybrid_optimizer.py`, and
  reducer state merging.
- Original intent: `8035d6e1`.
- Risk/dependency: low to medium; current main reducer API.
- Gate: sparse optimizer-state unit tests and one pipeline optimizer step.

### Full-model loading and code generation

#### PR 4: Load only required full-model parameter chunks

- Scope: parser attribute-content index and selective runtime chunk loading.
- Original intent: `021b1f45` without the mmap follow-up.
- Risk/dependency: medium; none beyond current main loading APIs.
- Gate: indexed chunk selection, missing/corrupt index behavior, and multi-rank
  loading.

#### PR 5: Use mmap for full-model parameter chunks

- Scope: mmap lifetime and tensor loading only.
- Original intent: `bafcc122`.
- Risk/dependency: low to medium; PR 4.
- Gate: parity with normal loading and no retained unnecessary mappings.

#### PR 6: Version and compact per-rank attribute metadata

- Scope: compact metadata format, deduplication, reuse detection, and fallback
  readers for main merged and legacy per-rank formats.
- Original intent: metadata portions of `4395a258` plus the integration
  conflict resolution.
- Risk/dependency: medium; none.
- Gate: compact, main-merged, and legacy load/reuse tests.

#### PR 7: Generate ranks with multiple codegen workers

- Scope: worker process orchestration, staging/promotion, failure cleanup, and
  robust serialization of deep payloads.
- Original intent: worker portions of `4395a258` and `40e5e197`.
- Risk/dependency: medium to high; PR 6.
- Gate: 1/2/8 workers, deep payloads, deterministic outputs, worker failure,
  reuse, and a 48-rank generation smoke test.

Deep-payload serialization stays in this PR because a multi-process codegen
implementation that fails on real deep graphs is not independently complete.

### Operator and collective correctness

#### PR 8: Support ring-attention sequence groups

- Scope: variable-length ring attention, reordered sequence groups, return-LSE,
  and focused operator tests.
- Original intent: `232eecf3` and ring-attention portions of `969533c2`.
- Risk/dependency: medium; none.
- Gate: variable-length, return-LSE, reordered-rank, and sequence-group parity.

#### PR 9: Correct collective device and rank mapping

- Scope: object collective device selection, MovePrim rank ordering, and runtime
  collective mapping.
- Original intent: `c86d3dfe` and non-ring portions of `23f3d6f5`.
- Risk/dependency: medium; keep independent of narrowed boundaries where
  possible.
- Gate: reordered ranks, object collectives, and multi-group runtime tests.

### IR layout and segment boundaries

#### PR 10: Preserve tensor alias layout information

- Scope: IR tensor alias/layout representation and the smallest execution-plan
  support required to preserve it.
- Original intent: alias foundation from `55030118` and `c53d12a7`.
- Risk/dependency: high; current main IR APIs.
- Gate: alias components, slicers, reuse, and serialization tests.

#### PR 11: Narrow pipeline segment boundaries

- Scope: segment expansion, narrowed input/output layouts, boundary generation,
  and cross-stage adapter construction.
- Original intent: remaining final behavior from `55030118`, `3d82460f`,
  `7209e211`, and `c53d12a7`.
- Risk/dependency: high; PRs 9 and 10.
- Gate: forward/backward parity, tensor/object boundaries, narrowed layouts,
  resharding, and peak-memory checks.

### Interleaved VPP

#### PR 12: Add interleaved VPP core execution

- Scope: schedule-codegen correctness, auxiliary/non-tensor outputs, context
  data alignment, cross-stage cache lifetime, and basic interleaved execution.
- Original intent: final behavior derived from `e0558a72`, `def6ff29`,
  `0598ec28`, and `25f9827e`.
- Risk/dependency: high; current main pipeline APIs.
- Gate: PP and interleaved VPP forward/backward/optimizer parity without
  application-specific fixed policy.

#### PR 13: Add fixed-policy interleaved VPP stage mapping

- Scope: generic fixed-policy hooks and logical-to-physical stage mapping.
- Original intent: policy portions of `97449e5a`.
- Risk/dependency: medium to high; PR 12.
- Gate: several PP/VPP stage counts, uneven chunks, and invalid-policy checks.

#### PR 14: Keep colocated VPP shared parameters local

- Scope: shared-parameter multiref decisions when logical stages map to the
  same physical stage.
- Original intent: `cfc79cf4` and final shared-parameter fixes.
- Risk/dependency: medium; PR 12.
- Gate: colocated and cross-physical-stage shared-weight tests.

#### PR 15: Use stage-local TP/EP size during partition dry-run

- Scope: stage-local partition degree selection only.
- Original intent: `24b66212`.
- Risk/dependency: medium; PR 12, and PR 13 if its policy interface is used.
- Gate: heterogeneous stage-local TP/EP dry-run and generated partition checks.

### Pipeline communication and reducer overlap

#### PR 16: Add correct asynchronous pipeline P2P

- Scope: async send/receive runtime support, dedicated process groups, backward
  gradient ordering, drain semantics, and safe output pseudo-free/lifetime.
- Original intent: correctness portions of `0e842b84`, `1480d8c3`,
  `9a340d07`, and `59e7ac12`.
- Risk/dependency: high; PRs 11 and 12.
- Gate: sync/async numerical parity, repeated sends, backward ordering, no
  use-after-free, no hang, and failure cleanup.

Correctness and lifetime safety must stay together; an async P2P PR that can
free a live send buffer is not independently mergeable.

#### PR 17: Overlap pipeline P2P and reduce communication memory

- Scope: early irecv posting, isend placement after compute, send bundles,
  deferred waits, and remaining communication-memory optimizations.
- Original intent: `115f1e83`, `68832cf6`, final parts of `3d82460f`, and
  `7209e211`.
- Risk/dependency: high; PR 16.
- Gate: numerical parity, timeline/overlap evidence, peak memory, throughput,
  and no new scheduling bubbles.

#### PR 18: Integrate async reducer with scheduled pipelines

- Scope: generated expected-contribution counts, manual gradient
  accumulation/readiness, scheduled reducer state, and per-bucket main APIs.
- Original intent: `ad1176ae`, reducer portions of `969533c2`, and later fixes.
- Risk/dependency: high; PR 12 and current main reducer. Submit after PR 17 to
  avoid overlapping scheduler reviews even though async P2P is not a strict
  semantic requirement.
- Gate: sync/async reducer parity, None/unused gradients, per-bucket options,
  ZeRO parameter sharding, sparse optimizer state, and multi-step optimizer
  parity.

## PR Preparation, Review, and Submission Workflow

No PR, remote branch, GitHub comment, or review reply will be created or posted
without explicit user approval for that exact external action.

For each PR, the assistant prepares a local review packet containing:

1. A fresh local branch based on the latest `origin/main`, or on the preceding
   local branch for a dependent stacked PR.
2. A minimal reconstructed diff containing only that PR's feature and tests.
3. Test commands and complete results, including known baseline failures.
4. A draft PR title and body covering motivation, behavior, implementation,
   tests, risk, compatibility, dependency, and rollback notes.
5. Draft responses for any anticipated reviewer questions.

The user then reviews both the code diff and PR text. Only after the user says
to submit a specific PR should either party push the branch or create the PR.
The default handoff is that the user performs the final push/submission. If the
user explicitly asks the assistant to submit, the assistant must show the final
title/body and target/base branches again before doing so.

The same rule applies after submission: reviewer comments are analyzed locally,
and proposed replies or code changes are shown to the user first. The assistant
does not post a GitHub reply merely because it drafted one.

### Parallel and stacked submission strategy

- Independent small PRs may be opened in parallel in waves of roughly 4-6 so
  different reviewers can make progress concurrently.
- Dependent PRs use stacked branches. A child PR targets its parent branch, so
  its GitHub diff contains only the child's incremental feature.
- Dependent children may be prepared and opened as Draft PRs before the parent
  merges. Do not mark a long stack ready for review all at once.
- Keep an active stack no deeper than about 3-4 PRs. Deeper future work remains
  local until earlier layers make review progress.
- After a parent merges, rebase the child onto the latest `origin/main`, retarget
  it to `main`, rerun its gates, and ask the user to review the refreshed diff.
- Small reconstructed PR branches should normally be rebased onto current main
  before submission. The historical 34-commit feature branch is merged rather
  than rebased only because replaying its superseded history is unsafe.
- The integration branch remains the complete behavioral reference and may
  periodically merge main for end-to-end regression testing. It is not the base
  branch for the reconstructed PRs.

Suggested initial waves:

- Wave 1, independent ready PRs: 1, 2, 3, 4, 8, and 9, subject to reviewer
  capacity.
- Stack A: PR 4 -> PR 5.
- Stack B: PR 6 -> PR 7.
- Stack C: PR 10 -> PR 11.
- Stack D: PR 12 as the VPP root, with PRs 13-15 prepared as small dependent
  branches or sibling Draft PRs.
- Stack E: PR 16 -> PR 17 -> PR 18.

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

## Real YOCO VPP Parity (2026-09-01)

Compared `yileiyang/yoco_pp` at `40e5e197` with the integration merge at
`168b33f5` on one 8xB200 node. Both sides used the same llm-train commit
`774d5e64`, independently generated code, seed 1, BF16 model precision,
Attention/MoEExpert recompute, MTP1, Muon, and 8K sequence length.

### Production-like PP2/EP4 run

The production-like case used real phase1 mixture data, MXFP8, DeepEP, 8 VPP
stages, and update frequency 8.

| Run | First-step gnorm | First-step train loss |
| --- | ---: | ---: |
| yoco_pp run 1 | 1.602508902549744 | 17.47869037712033 |
| yoco_pp run 2 | 1.602356791496277 | 17.47857553063220 |
| integration run 1 | 1.602711677551270 | 17.47894038776614 |

- Cross-branch gnorm relative difference: `1.2652e-4`.
- Same-branch yoco_pp repeat gnorm relative difference: `9.4921e-5`.
- Cross-branch loss relative difference: `1.4304e-5`.
- Same-branch yoco_pp repeat loss relative difference: `6.5707e-6`.

Therefore `1e-5` is below the observed run-to-run noise floor for the complete
MXFP8+DeepEP production path and cannot be used by itself as a merge gate.

### Deterministic VPP parity

The strict parity cases retained the real 13.55B YOCO model and VPP execution,
but used the fixed local debug dataset, BFloat16 quant mode, standard all-to-all
instead of DeepEP, and deterministic algorithms. CUDA `bincount` was allowed in
warn-only mode because it only produces per-source logging metrics and is not
part of loss or gradient computation.

| Configuration | yoco_pp gnorm | integration gnorm | gnorm relative diff | train loss diff |
| --- | ---: | ---: | ---: | ---: |
| PP2/EP4, 8 VPP stages, u8 | 1.779542922973633 | 1.779543161392212 | `1.3398e-7` | exactly zero |
| PP4/EP2, 16 VPP stages, u16 | 1.775534987449646 | 1.775534987449646 | exactly zero | exactly zero |

Both configurations pass the requested first-step gnorm threshold of `1e-5`.
PP4 also exercises a different physical-stage mapping and twice as many
micro-batches. These results support mathematical consistency of the merged
scheduler/reducer paths; they are not a steady-state performance benchmark.
