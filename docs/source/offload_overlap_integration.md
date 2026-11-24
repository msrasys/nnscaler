# Offload, recompute and module-stream integration

Both repositories use the experimental branch `yileiyang/integration_offload_overlap`.
It starts from the deployed revisions and preserves the original production and
feature worktrees. Integration and source-snapshot branches are committed and pushed.

## Sources and disabled behavior

- Tested nnscaler: `02b14b8dba8866896ddd8b23d2a7bf68541555b5`.
- Tested llm-train: `1f3c393ce468baf95a86a719ac08265e7b87d421`.

Pinned imports (offload snapshots include the original local edits):

- nnscaler production: `14c46e62da1c0f0233fab9dcf492197fff5cda0d`;
  offload snapshot: `69e1ade1656a2e0408130d007e0b74131311eab1`;
  overlap: `95ae10df0b09d8600c26eaeaa2115fad6054b82c`.
- llm-train production: `8839d63d58a96ea638cde0f01593f384f09ad641`;
  offload snapshot: `b15d2a9cb6b9359871fe3cdfede6f1ef198bee5a`;
  overlap: `1516cf59dfbd72b0cf6341df341e71789a64b337`.

Stock DeepEP binary SHA-256:
`91cfc0fdc03e7571c6e854377c8276602382465f65e80c2e13f3a9a3b49b3edc`.
The launcher rejects a different installed binary.

Offload module lists and module-stream rules default to empty; adjacent merging,
singleton recomputation and the separate routed-MoE adapter default to false.
The ordinary recipe retains its original checkpoint coverage and Torch checkpoint
entry point. Generated packages use trace metadata version 7.

The integration preserves deployed router/RMSNorm precision, optimizer and
cross-entropy behavior. Reducer ordering follows stable parameter construction
order. MoE temporary owners are released without resizing borrowed activation
storage; retained and higher-order graph tasks avoid destructive buffer reuse.

Activation scopes span the complete scheduled forward, including stream fragments,
and retain the final offload-group boundary. Recompute merging respects layout,
offload, side-effect and stream-invocation/readiness boundaries. CPU offload and
checkpoint inputs share one alias/lifetime contract. DeepEP uses the stock build;
buffer allocation and collective configuration agree on the same overrides.

## Original-recipe compatibility

Reference: NNScaler `14c46e62` and llm-train `8839d63d`, with only the two-line
Attention dict-literal tracing repair in llm-train `4a17b42b`. The unchanged pair
fails to compile this workload with `KeyError: max_seqlen_q`. Integration contains
the same repair; this is explicitly a repaired production reference.

Workload: 15B, six B200 GPUs, PP3/EP2, 8K, batch 1, accumulation 12, MXFP8,
frozen Vision and the original `Attention,MoEExpert` recompute selection.
On all six ranks, initial parameters/buffers/RNG, first complete gradients,
all 80 steps of loss/gradient-norm metrics and final parameter fingerprints
match exactly. The step-6 diagnostic parameters/buffers/RNG also match.

| Case | Native TFLOPS/GPU | Step interval (s) | Peak F/B allocated GiB/GPU | Final node PSS GiB |
|---|---:|---:|---:|---:|
| Repaired reference | 516.79 ± 5.49 | 4.7080 | 103.12 | 260.36 |
| Integration, new features off | 519.95 ± 5.50 | 4.6797 | 81.31 | 247.50 |

## Full feature matrix

Every arm uses the same expanded recompute selection and explicitly enables
`--recompute_single_ops`. The selection is Attention, MoEExpert, RMSNorm, RMSClip,
MixPrecisionLinear and MixPrecisionHeadLinear. This isolates boundary merging
from newly selecting single-operation checkpoints.

All eight arms completed 80 steps. All seven candidates have exact first-gradient
fingerprints, full loss/gradient-norm trajectories and final parameter fingerprints
against the expanded control on every rank. Audited input fingerprints at steps
1, 3, 40 and 80 also match.

| Merge | Offload | Module overlap | TFLOPS/GPU | Step interval (s) | Peak F/B GiB/GPU | Final node PSS GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 478.83 ± 5.13 | 5.0747 | 85.81 | 248.87 |
| 0 | 0 | 1 | 486.34 ± 4.89 | 4.9965 | 85.88 | 248.75 |
| 0 | 1 | 0 | 367.76 ± 3.28 | 6.5869 | 66.09 | 353.29 |
| 0 | 1 | 1 | 378.59 ± 3.39 | 6.3979 | 67.39 | 352.95 |
| 1 | 0 | 0 | 484.20 ± 4.87 | 5.0191 | 85.10 | 248.65 |
| 1 | 0 | 1 | 490.94 ± 5.10 | 4.9520 | 85.18 | 248.53 |
| 1 | 1 | 0 | 375.45 ± 3.14 | 6.4532 | 66.06 | 349.69 |
| 1 | 1 | 1 | 390.98 ± 3.31 | 6.2000 | 67.43 | 349.13 |

All features together save 18.37 GiB/GPU (21.4%)
in this comparison, with 22.2% longer measured step intervals.
Adjacent merging alone lowers this workload's peak by only about 0.7 GiB/GPU;
its existence does not imply a large global peak reduction. Small throughput
differences from a single run per arm are not universal speedup claims.

TFLOPS are the trainer's native metric. Performance uses steps 60–79; step
intervals exclude iterator fetching. F/B memory is sampled before peak counters
reset and maximized over all ranks/steps. PSS is an end-of-run snapshot, not a peak.
Parameter hashing/comparison runs outside the performance window.

## Additional checks and scope

- 104 CUDA component tests passed, including all eight model/Adam checkpoint
  round trips followed by another update. A small CPU cross-version test also
  passed old-to-new merged-checkpoint/Adam continuation.
- CPU compiler/runtime/configuration suites passed, including the joint feature
  matrix, serialization/default checks and 49 reducer/configuration checks.
- 45 real MoE kernel checks cover repeated backward and borrowed-storage safety.
- Plan audits verify real merges without expanding effective recomputation;
  generated code contains the expected offload and stream contexts. All offload
  arms perform actual D2H/H2D transfers; live/pending offload bytes return to zero.
- The full 15B matrix does not save/resume a complete 15B checkpoint. Resume
  validation above covers component models and the cross-version CPU case.
- The separate experimental routed-MoE adapter stays disabled in this matrix.
- A pre-existing attention dense-cross test-fixture failure reproduces on both
  reference and integration; it is recorded separately from passing test suites.

GPU work started only after the overlap session finished. Runs were serial and
bounded by disk/RAM guards; only owned regenerable weights and temporary parameter
references are removed after preserving source archives and numerical evidence.

Final resource audit: minimum free disk 316.1 GiB and minimum host available
memory 2513.3 GiB. All owned training processes and GPU contexts have exited.
The 33.4 GiB parameter reference and 2.7 GiB FFN probes were removed after
verification; compact manifests, fingerprints and source archives remain.

Local full manifests, commands, source archives, resource logs and comparisons:
`/home/yileiyang/agens/integration-offload-overlap-20261125/RESULTS.md` and `results.json`.

## Subsequent PR organization

The integration history preserves source snapshots and integration corrections.
For later review, separate the following areas and retain the joint regressions:

1. Common correctness/default-compatibility and lifetime fixes.
2. CPU activation store, transfer scheduling and offload policies.
3. Adjacent recompute merging and explicit singleton selection.
4. Module-stream planning/runtime and model adapters.

Land dependency-ordered PRs into `yileiyang/vl_pp` first, then prepare/revalidate
against `main`. This experiment creates no PR and merges no production branch.
