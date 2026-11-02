# Reference KL memory and codegen validation (2026-11-02 UTC)

Base revisions: llm-train main `7234c269adcda0b1cd5424570bbd291d5b9bb4f0`
and NNScaler yileiyang/vl_pp `349449e342e4cd6b5b53bd1a8cebecc8241b4737`.
The final compiler changes are on `yileiyang/fast_ref_kl_gencode`.
The independent training memory changes are on `yileiyang/refine_ref_kl`
in llm-train (`4e37bb7`) and NNScaler (`80ca017`). The compiler branch
alone does not include that executor memory patch; combine the two NNScaler
changes to obtain the combined memory result below.

## What changed

- `IRSegment.exist()` indexes direct children and recurses through child
  segments. Counts preserve duplicate entries; mutation, shallow/deep copies
  and restored graphs are covered. Derived indexes are not pickled.
- Execution-plan construction memoizes fixed cell placement locally and only
  dispatches on a cache miss. Previously `dict.setdefault(key, dispatch())`
  eagerly executed the default even on a hit, repeatedly scanning segments.
- Unused bound tensor methods in FX metadata no longer pin trace activations
  until DCE. Metadata is restored if the method is subsequently used as a value;
  the original node order and scoped context are retained.
- CUDA tracing reuses unchanged original CPU arguments instead of copying them
  back. Version counters, grad flags, types, leafness, strides and offsets are
  checked. In-place changes and inference tensors keep the previous copy path.
- `npbuffer.pt` indexes the existing attribute shards instead of serializing
  frozen reference weights a second time. Old standalone archives remain readable.

**Package compatibility:** newly indexed packages require the new runtime and
all referenced `fullmodel.pt.*` shards, including for checkpoint resume.
Do not remove those shards from a deployable package. The targeted production
YAML stages the entire gencode directory. No source production YAML was edited.

## Training correctness and memory

The real VL training entry point ran on six B200s with the 15B model,
PP3/VPP4 `333|333|333|342`, CP2/EP2/DP1, MBS1, 16K global / 8K local tokens,
12 microbatches, BF16/MXFP8, Muon, MTP1, Block recomputation, frozen first-stage
Vision with trainable projector, ref_kl coefficient 10, subset masks and
cross-file packing. A fixed real text/VL mixture supplied 413 images in 240
microbatches. There were 20 optimizer updates including 5 warmup updates.

The baseline already has ref_kl enabled. Compared runs have exactly equal
per-rank losses, KL, MTP, gradient norms and learning rates at every step,
identical batches and initial RNG, and identical full model/optimizer content
fingerprints at updates 0, 1 and 20. Reference buffers remain unchanged.
KL is nonzero after the first update (rank 0: step 2 = 0.0159967235,
step 20 = 0.0421214462), so this is not only a zero-KL initialization check.
Full per-element gradient dumps were not collected.

| Configuration | Maximum allocated GiB | Rank 0 median seconds, steps 6–20 |
| --- | ---: | ---: |
| Unchanged ref_kl baseline | 68.5020 | 8.3113 |
| llm-train reference-forward memory patch only | 66.5039 | 8.4716 |
| NNScaler boundary-input memory patch only | 67.2364 | 8.3442 |
| Combined memory patches, first measurement | 65.2383 | 8.3540 |
| Combined memory patches with new tracer, repeat | 65.2383 | 8.3089 |

Combined reduction: 3.264 GiB / 4.76%. The timings do not establish a training
speedup. Forward-only reference layouts are quantized again on subsequent calls;
student layouts still use the normal training cache. Original reference weights,
required hidden/K/V and router targets needed by student backward remain live.

The final compiler-only 20-step run, including the execution-plan cache fix,
is recorded in `fast-plan-final-20-content.json` and the accompanying compressed
audit. All checks passed; its allocated peak is 68.5020 GiB and median step time is 8.2990 s. This compiler-only run does not include the separate training memory patches.

This surrogate does not reproduce 150B expert sizes, pretrained routing,
CP32/EP32 communication or DP16. It uses deterministic attention and smaller
prefetch queues. Its memory savings are not a guarantee that the production job
will fit. `VL_PP_PRESET=ref_kl` in llm-train's `scripts/vl_pp_debug.sh` exposes
these major local topology/kernel settings; provide readable data and Vision
assets, and a 15B-compatible reference checkpoint if desired.

## Actual 150B / 256K compilation

The CLI was extracted from
`stage-1.2.256k.pp3ep32cp32dp16.mbs1.192nodes.no-cross.vpp4-342.yaml`.
Model, 256K sequence, MBS1, PP3/CP32/EP32, 12 virtual stages, 12 microbatches,
plan 96, runtime 1536, ref_kl and fixed reference checkpoint were preserved.
Only compilation mode and task-owned output locations were used. One local GPU
launched compilation; 32 CPU workers emitted all 3072 language/Vision rank files.

MoE metadata validation passed for 1536 language ranks, including 36,864
preprocess and 36,864 expert FFN calls. The final planner was also run on the
same saved full-size IR and regenerated every rank. Training on 1536 GPUs was
not run.

### Tracing and serialization

These are single local measurements with full cProfile enabled in both tracing
runs. The first column already includes the dead-method-metadata fix; it is not
the unchanged branch, which could not finish tracing within the memory guard.

| Language phase | Before input-copy reuse | After input-copy reuse |
| --- | ---: | ---: |
| Tracing | 920.397 s | 654.765 s |

Observed tracing reduction: 28.9%. Small correctness tests briefly used idle
GPUs 6–7 while the first compile used GPU 0; this is not a repeated statistical
performance study. Tensor placement also includes synchronization waits, so
residence in `Tensor.to()` is not pure memcpy time.

The successful run wrote 599,193,471,232 logical attribute bytes in 172.842 s.
Its non-persistent-buffer index took 0.030 s and avoided another
299,290,651,648 bytes (278.736 GiB) of writes. Primary attribute shards are still
written. Newer initialization contracts would be needed to omit them safely.

### Same-IR graph processing and code generation

Both runs reuse byte-identical pre-partition IR, excluding tracing and weight
serialization. Whole-program cProfile was disabled for both; phase timers and
an outermost-call timer for `IRSegment.exist()` remained enabled. The last column
includes the final execution-plan cache fix. Do not compare these numbers to
the fully profiled fresh compile's total time or to training startup latency.

| Scope | Baseline | Final compiler |
| --- | ---: | ---: |
| Language graph processing before rank workers | 582.344 s | 213.542 s |
| Language worker codegen, including payload handling | 176.185 s | 163.856 s |
| Language total | 758.530 s | 377.399 s |
| Vision total | 48.197 s | 46.617 s |
| Both modules total | 806.727 s | 424.016 s |
| Outermost `exist()` time, main process | 346.135 s | 0.927 s |
| `exist()` calls including nested calls | 1,044,046 | 1,770,824 |

Both-module IR replay is 47.4% shorter; the language graph-processing part is
63.3% shorter. Counts differ with child traversal order; child codegen processes
are not instrumented by the `exist()` timer. This is the same workload, not an
assertion that both versions perform identical numbers of lookups.

All 3072 emitted rank files have identical remaining source after normalizing
NNScaler checkout prefixes and sorting only the audited import-only custom-op
registry block. That existing generator iterates a Python set, and the standalone
replays did not fix PYTHONHASHSEED. The import multisets match; no computational
statement, identifier, or source-line comment was normalized. Every rank's
attribute metadata also matches, and both input IR files are byte-identical.
See `production-package-comparison.json.gz` for hashes and comparison rules.

The 15B final training comparisons use PYTHONHASHSEED=0. The earlier combined
memory run additionally shifts `arch/linear.py` source-line comments by one;
its generated executable ASTs match the baseline.

## User log and failed attempts

The provided 2026-10-30 log uses 64K / MBS2 / CP16 / 1584 ranks, unlike the
current 256K / MBS1 / CP32 / 1536-rank job. Its 497-second interval after
attribute saving includes the unlogged standalone `npbuffer.pt` save. It
cannot all be assigned to `IRSegment.exist()` from timestamps alone.

The unchanged 256K baseline hit the 128 GiB container-headroom guard during
tracing, before graph transformations or attribute saving. The first patched
attempt completed tracing, then an overly strict monitor treated a transient
`du` ENOENT during atomic file renaming as failure. Its 32-second save-finally
record is NOT a successful serialization time. The incomplete 95.44 GiB of
weights were removed. The monitor was corrected to tolerate disappearing paths
while retaining other I/O failures and all memory/disk limits; a concurrent
1,000-rename stress check passed. The subsequent full compile and both replays
completed successfully.

## Checks, artifacts and reproduction

Focused checks passed: 11 quant-cache/launcher tests; 4 real GPU frozen-expert
cases (BF16/MXFP8/MXFP4 and frozen weights with input backward); 26 executor
boundary tests; 58 graph/buffer/attribute tests; 12 tracer context/lifetime tests;
15 GPU transfer/cache tests; 2 GPU paired-pipeline tests including restored
non-persistent buffers; and 3 execution-plan tests.

Phase JSONL, resource logs, plans, summaries and compressed per-rank audits are
included here. `harness/` preserves task-specific commands and observers rather
than a portable general launcher. Their original root was
`/data/tmp/ref-kl-opt-20261103`; remap source data/output/checkouts if rerunning.
Always source llm-train's `scripts/k8s_env.sh` before tests or local GPU work.
No cluster jobs, shared checkpoints or the user's existing offload worktrees
were changed. Task-owned production weight files and their replay symlinks were removed after validation (primary weight files: 558.86 GiB). Retained generated source, IR, hashes and logs are validation artifacts, not a deployable complete weight package.
