# Module stream validation

Full hiding of DeepEP communication has **not yet been demonstrated**. The
current comparison uses one fixed no-overlap baseline and changes stream
scheduling and its backend submission adapters. Communication tuning gains from
earlier experiments are excluded.

## 2026-11-18: local dInput/dWeight implementation

The optional `weight_backward_on` feature separates expert dInput and dWeight
inside the existing backward action. It follows the local communication window
in the `yileiyang/dw_split` references (NNScaler `4344a22a`, llm-train `7f49702b`),
while retaining ordinary autograd parameter edges instead of importing the FBW
pipeline scheduler or a device-wide pending-weight queue. It does not introduce
cross-microbatch work. The previous performance numbers below are **not**
measurements of this new feature.

CPU validation includes 66 compiler/runtime and 27 backend/MoE kernel-double
tests. Checks include
repeated module invocations, ordinary reducer completion counts with mixed
native gradients, BF16 parameters with FP32 gradients, retained backward,
parameter updates, checkpoint reuse, AMP and failure cleanup. After the offload
session released the GPUs, 98 CUDA-enabled framework tests passed. Real EP2
MXFP8 MoE tests use 16K global tokens, recomputation, per-token clipping 20000,
and a 3% weight update. Both dW variants produce bitwise-equal outputs and
parameter gradients; input-gradient relative L2 error is 0.000854, below the
unchanged 0.003 threshold.

The new performance protocol uses 80 steps, the actual rank-0 training metrics
over steps 60–79, no profiler injection, and no per-step observer synchronization.
Global deterministic algorithms and the fixed cuBLAS workspace remain enabled;
uninitialized-memory filling is disabled. These measurements cannot be compared
directly with the older fill-enabled, synchronized-observer numbers below.
NNScaler `fd6c303d` and llm-train `36bdb024` give:

| Configuration | Training s/step | TFLOPS/GPU | Step-time reduction |
| --- | ---: | ---: | ---: |
| No overlap | 4.489316 | 533.80 | — |
| Existing shared split + expert recomputation | 4.461282 | 537.16 | 0.62% |
| Additionally split expert dX/dW | 4.426563 | 541.37 | 1.40% |

All six ranks match initial tensors/RNG, selected full-input checkpoints and
per-step work. All 80 steps' loss, gradient norm and auxiliary losses are
identical between the last two configurations. Peak sampled GPU use rises from
134.10 GiB to 154.77 GiB when adding dW; these are driver-use samples, not active
tensor allocations. The gain is real in this run but small; full hiding is not
established.

A separate baseline trace has 265.17 ms/GPU of exposed dispatch plus combine,
5.47% of its annotated step, with zero unattributed DeepEP kernels. Forward
dispatch/combine contribute 76.27/36.90 ms and backward dispatch/combine
98.86/53.14 ms; layout is separately 8.49 ms. No computation overlaps baseline
communication. Treatment traces and remaining attention/shared-placement
experiments are pending. The first attention attempt stopped during compilation
because locally constructed kwargs dictionaries were conservatively rejected;
the compiler now proves local dictionary ownership instead of changing the
model or allowing external mutations.

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
| Additionally gate recomputation on backward input readiness | 5.189694 s | 0.51% |
| Limit side-stream DeepGEMM to 128 SMs | 5.161739 s | 1.05% |

These are short-run results, not evidence of long-term training equivalence.
The optional native launch and backend kernel-graph paths are correctness-tested,
but are not faster than the table's best treatment in the completed full-model runs.

## Timing without profiler injection

The matched 40-step runs use nnscaler `389c9925` and llm-train `49aaa81a`, with
no Nsight process injection and no active Torch profiler (`--profile-step 0`).
The same steps 21–40, six-rank maximum, transport settings, initial state and
input hash checks are retained.

| Configuration | Mean step | Standard deviation |
| --- | ---: | ---: |
| No module overlap | 5.191783 s | 0.059037 s |
| Default `run_stream_overlap.sh overlap` | 5.104408 s | 0.052740 s |

The measured reduction is **1.68%**, or 87.37 ms per step. This confirms the
earlier 1.65% result; it remains below the full-communication-hiding target.
Maximum loss and gradient-norm differences in the timing window are 0.000911
and 0.004484 respectively. The launcher selects eager shared prefix/suffix
execution with backward input readiness, ordinary early expert recomputation,
and independent asynchronous metadata copies. It does not select kernel graphs,
recomputation gating, padded dispatch or a modified DeepGEMM SM budget.

A final owner-event polling experiment passed 32 lifetime/gradient/runtime
checks but measured 5.120583 s with the same overlap rules and inputs, versus
5.104408 s above. That runtime change was archived and reverted as well; the
delivered executable code matches the selected measured configuration.

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

## Backend submission experiments

An experimental `NNSCALER_DEEPEP_PADDED_DISPATCH=1` path submitted DeepEP payloads
without first waiting on the CPU for receive counts. It freezes a safe receive
capacity using the sum of all ranks' initial input lengths, permits shorter
inputs, and rejects growth. DeepEP marks padded rows with -1; exact expert
counts are reconstructed after the shared graph has been submitted. Dynamic
expert computation and routing are preserved, with no token truncation.

The padded path passed actual EP2 MoE correctness and a direct transport test
covering unequal rank lengths, shrinking inputs, partially invalid routes,
all-invalid routes, and both forward/backward communication. Its full PP3/EP2
mean step was 5.257910 s, 0.80% slower than the fixed baseline. The backend change
was reverted from the delivered llm-train tree; the experiment remains in commit
`a0a59b42` and local reports. This path still reads exact counts back after
submission; it is not a fully static-shape MoE implementation.

The approach is motivated by Kimi K3 section 5.2.1: static shapes remove
per-layer host synchronization, and shared-expert GEMMs run on a separate
stream. This implementation continues to use DeepEP and does not implement
MoonEP's redundant-expert load balancing.

The optional `recompute_ready_on` rule gates expert recomputation on the downstream
backend's backward input readiness. In the previous trace, its first GEMM
typically started about 4 ms before backward dispatch. The generic
`recompute_ready_on` rule has passed the 77-case CUDA-enabled compiler/runtime
suite, including a delayed-input test that proves recomputation starts after
inputs are ready and before the pending gradient completes. Actual EP2 MoE
with recomputation and updated weights also passes: output and parameter
gradients are bitwise equal, with input-gradient relative L2 about 8.5e-4.
In the full model, backward dispatch exposure fell to 8–23 ms per rank, but
step time improved only 0.51%: delaying the recomputation also lost some useful
prefetch overlap. It is supported but not enabled by the default launcher.
The 128-SM DeepGEMM probe preserved outputs and parameter gradients bitwise,
but its 1.05% step improvement was also below the selected treatment; its
process-local backend patch was not integrated into NNScaler.

## Resources and evidence

In the unprofiled pair, maximum device usage rises from about 101 GiB to 126 GiB;
summed process-tree RSS peaks at about 431 GiB in both runs. Nsight runs reached
about 512 GiB RSS. Kernel-graph caches are bounded to four graphs and 512 MiB of
logical input/output storage per device; graph scratch allocations are extra.
Nsight plus kernel graphs caused failures in full-model attempts, so those
traces were rejected. A specific cause has not been established.

Local reports, launch specifications, input/state hashes, traces, generated
source archives and guard logs are under
`/home/yileiyang/agens/stream-overlap-20261117`. Completed correctness gencode
weights are removed after source archival; the reusable full-model cache is
retained during iteration. The guard enforces RAM/disk/time limits and cleans
only owned orphan workers. All GPU validation processes have exited, and the
GPU reservation was released at 18:44 on 2026-11-17 (Asia/Shanghai). About 16 GiB
of completed correctness gencode was removed during this continuation; one
30 GiB full-model cache remains for reproduction, with over 570 GiB disk free.

Both repositories use branch `yileiyang/stream_overlap`. The llm-train launcher
accepts only `baseline` and `overlap`; run it from `llm/` with the original
training command. Point `NNSCALER_ROOT` at this NNScaler checkout, or install
this branch, so the launch uses the new compiler/runtime. The machine-readable
summary is `/home/yileiyang/agens/stream-overlap-20261117/final-report.json`.
It explicitly records `full_communication_hiding_achieved: false`.
