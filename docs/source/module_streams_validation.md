# Module stream validation (work in progress)

The framework feature passes CUDA and real MoE correctness checks. The requested
full hiding of DeepEP communication has **not** been demonstrated. Measurements
below distinguish communication tuning from stream scheduling.

## Workload and measurement

The unchanged llm-train `main` model runs on six NVIDIA B200 GPUs connected by
NV18: YOCO-MoE-15B-A6B-L4, PP3/EP2, 12 virtual stages, 8K sequences, 12
microbatches, MXFP8, frozen Vision, Attention/MoEExpert recomputation, and Muon.
This validation uses real VL batches and randomly initialized language weights;
it is not a resumed production 150B checkpoint. llm-train changes only its bash
launcher. The original 303000-update learning-rate horizon is preserved.

Each comparison completes 40 optimizer steps. Batches are hashed outside the
timed window and match on all six ranks for all 40 steps. The observer measures
through synchronized optimizer completion; the value for each step is the maximum
across ranks. The steady window is steps 21–40, excluding the profiled step 28.
Clean unprofiled repeats are still required before a final speedup claim.

| Configuration | Mean step (s) | Median (s) | Mean reduction |
| --- | ---: | ---: | ---: |
| Original DeepEP defaults, no module rules | 4.620034 | 4.607293 | — |
| Tuned DeepEP, no module rules | 4.440619 | 4.425621 | 3.88% |
| Tuned DeepEP + earlier shared split/after/async metadata | 4.538832 | 4.531060 | 1.76% |
| Tuned DeepEP + checkpoint recomputation stream | 4.410359 | 4.408788 | 4.54% |
| Tuned DeepEP + current shared split and recomputation | 4.463996 | 4.460448 | 3.38% |

The earlier shared-stream row predates the latest reduction in parameter tracking and
output-publication overhead. It is retained as a control, not a result for the
new implementation. Recompute-only is currently faster than the combined
configuration; additional resource and metadata-copy controls are in progress.
Tuned defaults use dispatch `(96,128,512,6,128)` and combine
`(96,64,512,6,128)`, in `deep_ep.Config` argument order. Installed DeepEP is
`1.2.1+567632d`. An isolated EP2 transport sweep validated all 81 tested
configurations; its best pair was 0.266 ms versus 1.397 ms for the original
configuration. That isolated improvement is not the training speedup.

## DeepEP timing

Per-device interval unions from the same profiled optimizer step 28:

| Rank | Baseline DeepEP (ms) | Baseline fraction | Tuned-only exposed fraction | Earlier shared-stream exposed fraction |
| --- | ---: | ---: | ---: | ---: |
| 0 | 507.89 | 10.28% | 4.79% | 3.48% |
| 1 | 478.42 | 9.70% | 2.43% | 2.02% |
| 2 | 482.47 | 9.79% | 2.63% | 2.59% |
| 3 | 575.15 | 11.67% | 4.70% | 2.53% |
| 4 | 439.40 | 9.04% | 2.94% | 2.11% |
| 5 | 343.95 | 7.06% | 1.48% | 1.18% |

Baseline and tuned-only communication have no concurrent useful compute. The
earlier shared-stream configuration overlaps 15–27% of its remaining DeepEP
kernel time. These fractions are per rank and cannot be added. Instrumented
steps include profiler overhead. Kernel concurrency alone does not establish a
critical-path improvement; shared SM and bandwidth contention can slow kernels.

## Correctness and resource use

- Initial CUDA stream runtime/lifetime suite: 74 tests passed. Subsequent combined
  regressions: 110 passed, including repeated invocations, before/after placement,
  host submission, reducer affinity, manual gradients, opaque owners and async
  metadata snapshots.
- Checkpoint/runtime/codegen subset: 41 CUDA-enabled tests passed. It verifies
  recomputation can finish while the incoming gradient stream is still pending,
  unchanged gradients, checkpoint early stopping, and repeated backward.
- Actual EP2 MoE dimensions: 4096 embedding, 1024 latent, 6656 expert FFN,
  32 experts, top-8, shared width 1664, MXFP8 and expert recomputation.
  Recompute-only scheduling gives bitwise-identical outputs and all gradients.
  Combining it with shared splitting gives identical outputs and parameter
  gradients; BF16 input-gradient relative L2 error is 8.708e-4.
- Host submission previously exceeded the input-gradient test's 0.003 relative
  bound (0.003794). A diagnostic fixed-order FP32 fan-in makes every output and
  gradient bitwise identical for all variants; production still uses its original
  BF16 fan-in. This control identifies accumulation order and does not relax the
  original test bound or establish long-run training equivalence.
- Over 40 steps, maximum logged loss differences versus baseline are 0.001402
  for tuning alone and 0.001514 for the earlier shared-stream run; maximum
  gradient-norm differences are 0.004532 and 0.005072 respectively.
- Completed controls peak at approximately 412 GiB process-tree RSS. Checkpoint
  stream runs peak around 119–122 thousand MiB GPU usage on the largest rank,
  compared with 106–107 thousand MiB for the earlier controls. Runs monitor free disk,
  host/cgroup memory and GPU allocation, with explicit stop thresholds.

The shared expert contributes only 22–24% as much kernel time as the original
DeepEP communication in the component trace. Its prefix can finish before
communication launches, and synchronous metadata copies can delay the caller.
Readiness snapshots, asynchronous copies, and independent checkpoint
recomputation address these distinct limits, but the end-to-end target remains
unmet in the completed controls above.

The original main-model compile also reproduced an existing `dict.update`
tracing failure in attention kwargs. NNScaler now retains those updates and
dynamic nested lookups; 65 relevant CPU tests passed. No llm-train model source
was changed.

Local evidence is stored under
`/home/yileiyang/agens/stream-overlap-20261116`: per-rank metrics and batch hashes,
compressed traces, communication summaries, launch arguments and resource logs.
