# Actual 150B ref_kl compilation validation — 2026-11-03 (Asia/Shanghai)

The unchanged production configuration completed in **41.43 minutes**;
the final compiler changes completed in **18.65 minutes**
(**55.0% less elapsed time**). Both runs generated all **3072** rank files.

## Workload and measurement

The command comes from llm-train main's
`yamls/moe-150A9B-260923/stage-1.2.256k.pp3ep32cp32dp16.mbs1.192nodes.no-cross.vpp4-342.yaml`:
150B, 256K, PP3/VPP4 `333|333|333|342`, CP32/EP32, MBS1, 12 microbatches,
plan 96, runtime 1536, ref_kl coefficient 10, fixed reference checkpoint,
MXFP8, Block recomputation, MTP1 and frozen first-stage Vision.
The original training namespace and 32 rank-codegen workers were preserved.
Only compilation mode and task-owned output/shared-array locations were substituted.

Both runs used one local B200, the same source YAML checksum, separate empty
gencode and JIT cache directories, PYTHONHASHSEED=0, 8 OMP/MKL/OpenBLAS threads,
8 attribute writers and 4 Inductor compile threads. The container has a
96-CPU / 2400-GiB limit. Whole-program cProfile was disabled; nested phase
wall timers and 5-second resource samples were enabled. OS/NFS page caches
were not forcibly flushed. This is a complete compilation comparison,
including model construction, tracing, weights, every rank's code and process
shutdown; it excludes cluster provisioning, package installation and later
training/staging. There was no competing GPU workload or other heavy test
running during either timed compilation.

| Phase | Unchanged seconds | Final seconds |
| --- | ---: | ---: |
| Language model initialization | 93.848 | 91.030 |
| Language tracing | 759.903 | 334.625 |
| Primary attribute serialization | 174.948 | 155.374 |
| Non-persistent buffer serialization | 504.578 | 0.029 |
| Other tracing / IR conversion | 5.628 | 5.669 |
| IR serialization, graph processing and execution plan | 621.780 | 239.646 |
| Language rank codegen, including worker payload | 181.054 | 174.395 |
| Vision module, all phases | 78.816 | 73.759 |
| Process startup / remaining setup / shutdown | 65.442 | 44.604 |
| **Complete process** | **2485.998** | **1119.131** |

The graph-processing interval includes freeing the temporary traced model,
which now happens before partitioning. Some teardown cost consequently moves
into that interval. Phase totals and complete-process time are both reported.
Single paired measurements are not a statistical throughput study.

## Changes

- Index direct-child graph membership, maintaining indexes through mutations,
  nested segments, copies and restored graphs. Avoid repeated execution-plan
  dispatch and repeated scans of fixed device placement.
- Release unused bound-method metadata during tracing. Reuse an unchanged
  original CPU input after CUDA tracing rather than copying it back again;
  mutations and incompatible metadata retain the previous copy path.
- Add opt-in output caching for pure CPU metadata substitutes in reuse_cache.
  llm-train enables it for existing Linear/QKV/SwiGLU and three MoE substitutes.
  Every invocation retains its own current arguments, including layer weights.
  Already-mutated outputs, inference tensors and input aliases are not cached.
  The existing normalization tracing/compilation path is preserved.
- Replace duplicate non-persistent-buffer writes with a versioned index into
  completed attribute shards. This avoids approximately 278.7 GiB of duplicate
  reference-buffer serialization for this model.
- Release the temporary model's context-manager alias after attribute saving,
  allowing its CPU weights to be reclaimed before graph processing/codegen.

The sampled owned-process RSS peak fell from
**2016.9 GiB** to
**853.7 GiB**. These are compiler CPU
measurements, distinct from training CUDA allocated/reserved memory.

## Correctness

All 3072 emitted source files match after replacing checkout path prefixes
only. Import order, statements, identifiers and source-line comments were
preserved. Both modules' 1536 rank metadata records match. Logical content
hashes of all 870 serialized attributes and all 349 non-persistent buffers
match, including the legacy standalone archive versus the new index.
Independent traced IR pickle byte identity is not assumed.

The compiler branches separately, their pair, and the final compiler + memory
combination each passed 20 real VL training updates on six B200s. The 15B
surrogate uses PP3/VPP4-342, CP2/EP2/DP1, 16K global / 8K local tokens,
MBS1, 12 microbatches, BF16/MXFP8, Muon, MTP1, Block recomputation, first-stage
frozen Vision with a trainable projector, ref_kl=10, subset masks and
cross-file packing. A fixed real text/VL mixture supplied 413 images in
240 microbatches. Reference weights snapshot the initial 15B student.

Every rank's loss, KL, MTP loss, gradient norm and learning rate agree exactly
at every update. All batches and initial RNG agree. Full model/optimizer
content fingerprints agree at updates 0, 1 and 20; reference buffers remain
unchanged. This includes nonzero KL after the first update. Full per-element
gradient dumps and training on 1536 GPUs were not performed.

The final combined memory measurement used an otherwise idle machine, without
concurrent package hashing. Peaks are PyTorch CUDA allocated bytes, maximized
across all ranks and 20 steps; they are not reserved memory or sampled NVML
usage. Time is rank 0 median train_wall for updates 6–20.

| Training configuration | Peak allocated GiB | Median step seconds |
| --- | ---: | ---: |
| Original ref_kl baseline | 68.5020 | 8.3113 |
| Final compiler + both memory changes | 65.2383 | 8.3338 |

The allocation reduction is 3.264 GiB.
This local 15B result does not establish that the 150B job will fit.

The independent training-memory changes are in
[llm-train #181](https://github.com/msrallm/llm-train/pull/181) and
[NNScaler #147](https://github.com/msrasys/nnscaler/pull/147).
The final combined isolated training measurement is documented alongside them;
compiler-only validation is not a claim of training throughput improvement.

## Package compatibility and evidence

New indexed npbuffer packages require the updated runtime and their referenced
`fullmodel.pt.*` shards, including when restoring parameters from a checkpoint.
Legacy standalone archives remain readable. The actual YAML stages the whole
package. Do not remove referenced shards from a deployable package.

`summary.json` records exact source revisions, phase/resource measurements,
workload/source fingerprints and compact correctness results. Detailed raw
logs, input plans, per-rank audits and tensor hashes are retained under
`/data/tmp/ref-kl-opt-20261103`; older investigation artifacts are also available
in the pre-review branch history at NNScaler `6a7817f`. Previous IR-replay
807→424-second measurements were partial phases and are superseded for
complete timing by this report.

The original full compile succeeded after fixing the measurement guard:
raw cgroup usage included reclaimable file cache. The guard now records both
raw usage and non-reclaimable working headroom, retaining 128 GiB of the latter
and 256 GiB host available memory. Earlier monitor-triggered termination was
not proof of a compiler OOM or interference from another session. The earlier
atomic-rename monitoring race was also fixed in the task harness.

Heavy task-owned validation weight files are removed after comparisons;
retained code/IR/log directories are validation artifacts, not complete
deployable weight packages. Shared checkpoints and existing offload worktrees
were not modified.
