# Module stream validation

Full hiding of DeepEP communication has **not yet been demonstrated**. The
current comparison uses one fixed no-overlap baseline and changes stream
scheduling and its backend submission adapters. Communication tuning gains from
earlier experiments are excluded.

## Workload and measurement

The workload matches the offload session: YOCO-MoE-15B-A6B-L4, PP3/EP2 on six
B200 GPUs, 12 virtual stages, 8K sequences, batch size 1, update frequency 12,
MXFP8, frozen Vision, Attention/MoEExpert recomputation, Muon, universal loop 3,
and seed 2609271808. It uses real VL batches and randomly initialized language
weights. The original 303000-update learning-rate horizon is preserved.

Both modes use DeepEP configuration `[20,96,256,6,128]` for dispatch and combine,
4x CUDA launch queues, and identical global/FA4 deterministic settings. Here
96 is the token chunk size; the communication kernels use 20 SMs. Initial
parameters, buffers, RNG state and every input batch are hashed and compared.

Timing measures synchronized optimizer completion, taking the maximum across
all six ranks. Steps 21–40 form the timing window; only step 41 is profiled.
Nsight CUDA/NVTX traces attribute communication kernels to forward/backward
backend calls with zero unattributed DeepEP kernels. Torch-profiler and Nsight
runs are kept separate because their process instrumentation differs.

The current Nsight protocol results are:

| Configuration | Mean step | Reduction |
| --- | ---: | ---: |
| No module overlap | 5.216390 s | — |
| Shared prefix/suffix + backward input readiness + expert recomputation stream | 5.130144 s | 1.65% |

These are short-run results, not evidence of long-term training equivalence.
A final comparison without profiler injection is still required. The optional
native launch and backend kernel-graph paths are correctness-tested, but are
not faster than the table's best treatment in the completed full-model runs.

## Exposed communication

Per-device interval unions from the matched final-step traces are below.
“Exposed” means no useful computation overlaps that communication interval;
NCCL and other communication kernels do not count as useful computation.

| Rank | Baseline exposed (ms) | Treatment concurrent (ms) | Treatment exposed (ms) |
| --- | ---: | ---: | ---: |
| 0 | 330.13 | 137.52 | 199.03 |
| 1 | 266.60 | 139.60 | 178.05 |
| 2 | 257.60 | 175.74 | 173.15 |
| 3 | 342.82 | 145.52 | 347.28 |
| 4 | 221.08 | 118.26 | 196.03 |
| 5 | 225.08 | 109.70 | 115.67 |

Baseline rank 3 has 91.97 ms forward dispatch, 44.65 ms forward combine,
128.30 ms backward dispatch, 69.01 ms backward combine, and 8.89 ms layout.
Dispatch/combine alone occupy 333.93 ms, or 6.08% of its 5492.32 ms profiled
step. Including layout gives 342.82 ms / 6.24%. Fractions cannot be added across
ranks. Kernel durations include peer-readiness waits, and the profiled denominator
includes instrumentation overhead; the end-to-end A/B remains the acceptance
metric.

The treatment makes real communication/computation overlap, but rank 3's
communication duration increases to 492.80 ms. Its remaining forward dispatch
exposure is 165.89 ms. Consequently, the concurrent duration alone overstates
the end-to-end benefit. The results do not establish an optimization limit.

## Correctness and implementation checks

The generic NNScaler compiler/runtime uses module rules and backend launch
boundaries without MoE-specific names. Generated-source audits check both the
saved code-generation ABI and actual readiness/launch constructs. Loading an old
saved configuration preserves its ABI rather than silently assigning today's
version; one earlier stale-cache run was rejected and excluded.

Completed checks include generated forward/input-gradient/parameter-gradient
comparisons, parameter sharing, multiple live forwards, retained backward,
checkpoint tensor ownership, reducers, and explicit CUDA readiness tests.
Those readiness tests delay the incoming gradient and demonstrate that side
work starts after its inputs are ready and before communication completes.
Host ordering alone was insufficient: CPU submission could lead GPU execution
by more than 20 ms.

Actual EP2 MoE checks use embedding 4096, latent width 1024, routed width 6656,
32 experts, top-8 routing, shared width 1664, and 8192 local tokens. With
recomputation and a 3% weight update, native/shared overlap preserves outputs
and all parameter gradients bitwise; BF16 input-gradient relative L2 difference
is about 8.5e-4 from accumulation ordering. MXFP4 and MXFP8 backend kernel-graph
checks also passed, including weight refresh and multiple live graphs.

An experimental token-half split produced identical outputs but approximately
0.00339 relative L2 error in BF16 shared weight gradients, exceeding the
existing 0.003 check. Retaining those gradients in FP32 reduced the discrepancy
to approximately 5.9e-6. This prototype was archived rather than selected as
the default; its threshold was not relaxed.

## Backend submission under investigation

llm-train's opt-in `NNSCALER_DEEPEP_PADDED_DISPATCH=1` submits DeepEP payloads
without first waiting on the CPU for receive counts. It freezes a safe receive
capacity using the sum of all ranks' initial input lengths, permits shorter
inputs, and rejects growth. DeepEP marks padded rows with -1; exact expert
counts are reconstructed after the shared graph has been submitted. Dynamic
expert computation and routing are preserved, with no token truncation.

The padded path passed actual EP2 MoE correctness and a direct transport test
covering unequal rank lengths, shrinking inputs, partially invalid routes,
all-invalid routes, and both forward/backward communication. Its full PP3/EP2
mean step was 5.257910 s, 0.80% slower than the fixed baseline; it is not the
selected configuration. This path still reads exact counts back after
submission; it is not a fully static-shape MoE implementation.

The approach is motivated by Kimi K3 section 5.2.1: static shapes remove
per-layer host synchronization, and shared-expert GEMMs run on a separate
stream. This implementation continues to use DeepEP and does not implement
MoonEP's redundant-expert load balancing.

The next scheduling change gates expert recomputation on the downstream
backend's backward input readiness. In the previous trace, its first GEMM
typically started about 4 ms before backward dispatch. The generic
`recompute_ready_on` rule has passed the 77-case CUDA-enabled compiler/runtime
suite, including a delayed-input test that proves recomputation starts after
inputs are ready and before the pending gradient completes. Actual EP2 MoE
with recomputation and updated weights also passes: output and parameter
gradients are bitwise equal, with input-gradient relative L2 about 8.5e-4.
Full-model performance validation is in progress.

## Resources and evidence

The best completed treatment peaks at about 128 GiB device usage and 512 GiB
process-tree RSS. Kernel-graph caches are bounded to four graphs and 512 MiB of
logical input/output storage per device; graph scratch allocations are extra.
Nsight plus kernel graphs caused failures in full-model attempts, so those
traces were rejected. A specific cause has not been established.

Local reports, launch specifications, input/state hashes, traces, generated
source archives and guard logs are under
`/home/yileiyang/agens/stream-overlap-20261117`. Completed correctness gencode
weights are removed after source archival; the reusable full-model cache is
retained during iteration. The guard enforces RAM/disk/time limits and cleans
only owned orphan workers. GPU priority was explicitly assigned to this overlap
validation by the user; release it when GPU validation finishes.
