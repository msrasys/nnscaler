# Module stream validation

## Production routed-expert acceptance (2026-11-19)

The committed `ModuleStreams`, backward dependencies, cooperative
`WeightGradientAccumulator` and checkpoint-consumer readiness feature now support
near-total DeepEP overlap in a real 15B PP3/EP2 run. The model backend divides
local/remote expert work into W13 and SwiGLU/W2 phases, schedules their reverse
phases and weight accumulation, and keeps MoE replay on the expert compute
stream. All work belongs to the current microbatch; the PP schedule is unchanged.

With six B200 GPUs, batch size 6, sequence 8192, accumulation 3 and
Attention/MoE plus CE recomputation, the final unprofiled complete-iteration mean
is **6.937011 → 6.680913 s**, saving **256.10 ms (3.69%)**, or **3.83% more
throughput**. Both modes use NNScaler `3716eca7`, llm-train `fc4a7bc7` and the
same isolated backend binary. The enabled mode includes optional same-node EP2
peer-copy transport; this result measures that combination, not scheduling alone.

The 60–79/80-step timing window, initial tensors/RNG, input checkpoints, work,
source, backend and observer audits match. All generated expert/replay calls and
forward/backward/weight callbacks have the expected counts. Complete first-step
nonexpert gradients are bitwise equal; expert W13/W2 gradients have maximum
relative L2 error 1.32e-5 from the fragmented FP32 reduction. Later training is
not bitwise identical. Peak sampled GPU use is 156.01 versus 166.51 GiB.

Separate matched step-22 captures use the same generated code and first-gradient
SHA as the full runs. Exposed backend intervals, including layout/preparation,
DMA and replay, fall **283.67 → 8.22 ms/GPU/step (97.10%)**. Useful compute covers
97.83% of the enabled backend intervals, with at least 95.17% coverage in every
forward/backward/recompute dispatch/combine category. All communication kernels
are attributed. Native transport exposure excluding layout is 3.63% of the
instrumented step, or 4.02% including backend preparation; the full iteration
gain is comparable. Timeline exposure is diagnostic, not an additive speedup
estimate.

The [model integration and complete measurement protocol](https://github.com/msrallm/llm-train/blob/yileiyang/stream_overlap/docs/stream-overlap.md)
and its `stream-overlap-results.json` include individual iteration timings,
configuration, limitations and artifact identities. This validation uses real VL
batches, randomly initialized language weights and frozen pretrained vision.
It does not establish long-training accuracy or performance beyond same-node EP2.

## Historical shared-expert and standalone dW experiment

The earlier small-batch experiment below did not demonstrate full hiding.
Correcting deferred-dWeight publication removed a regression, but adding it to
shared-expert overlap did not produce a reliable incremental gain. These
measurements explain the subsequent routed-expert/peer-copy implementation;
they are not the current acceptance result.

## Workload and protocol

YOCO-MoE-15B-A6B-L4, PP3/EP2/CP1 on six B200 GPUs, 12 virtual stages, 8K,
batch size 1, update frequency 12, universal loop 3, MTP1, MXFP8, frozen Vision,
Attention/MoEExpert recomputation, Muon and per-token clipping 20000. This matches
the local offload workload. No new cross-microbatch work or FBW pipeline schedule
is introduced. Both DeepEP directions use `[20,96,256,6,128]`; CUDA launch queues
are 4x. Global/FA4 deterministic modes and the fixed cuBLAS workspace are held
constant; uninitialized-memory filling is disabled.

Timing uses the actual unprofiled rank-0 `train_wall` over steps 60–79 of 80,
without per-step observer synchronization. Initial tensors/RNG, full-input
checkpoints at steps 1/3/36/80, and work per step are checked on all six ranks.
The original 303000-update learning-rate horizon is preserved. These runs use
real VL batches with randomly initialized language weights, not a resumed model.

The first cold-generation native run has small numerical differences from warm
controls despite identical initialization/inputs and equivalent generated code.
A warm native trace reproduces the previous warm native and standalone dW
numerical values. Both causal early/fixed pairs below have **identical 80-step
loss, gradient norm and auxiliary-loss metrics on all six ranks**. The overlap
reference and its dW variants also match each other. Do not describe every
native-versus-overlap run as bitwise identical.

## End-to-end results

The primary matrix uses NNScaler `eb6d2699`, llm-train `273c966f`, one cache and
one no-overlap configuration. The early-publication case is an explicitly
recorded runtime counterfactual on the same source, restoring main-stream
publication and accumulation. Standalone comparisons hold async metadata copies
off; the existing-overlap comparisons hold them on.

| Configuration | s/step | Peak sampled driver GiB |
| --- | ---: | ---: |
| No overlap | 4.484517 | 101.77 |
| Standalone dW, early publication | 4.511782 | 136.61 |
| Standalone dW, corrected publication | 4.480526 | 128.01 |
| Shared split + expert/attention recomputation | 4.415605 | 151.08 |
| Existing overlap + dW, early publication | 4.440813 | 167.29 |
| Existing overlap + dW, corrected publication | 4.411816 | 156.83 |

The correction recovers **31.255 ms** standalone and **28.997 ms** when combined
with existing overlap, respectively 0.69% and 0.65% of the early versions' step
times. Paired-step SEMs are 2.043 and 2.055 ms. Relative to the existing overlap
reference, corrected dW saves only **3.788 ms (0.086%)**, with paired-step SEM
1.967 ms. That is insufficient to select it as the preset default. The existing
preset improves this native control by 1.54%; an earlier matched batch measured
1.90%. These are measured batches, not a fixed performance guarantee.

## Communication exposure

Separate Nsight CUDA/NVTX runs capture matched step 41. The values below are
mean **ms per GPU per step**, not sums across GPUs. The no-overlap dispatch plus
combine exposure is 266.284 ms, 5.48% of its 4860.140 ms instrumented scope.
Layout preparation is an additional 8.487 ms and is excluded from this table.

| Exposed interval | No overlap | Existing overlap | Early dW | Corrected dW |
| --- | ---: | ---: | ---: | ---: |
| Forward dispatch | 80.651 | 80.977 | 85.411 | 82.095 |
| Backward dispatch | 98.297 | 68.045 | 71.107 | 68.467 |
| Dispatch total | 178.949 | 149.021 | 156.518 | 150.561 |
| Forward combine | 36.294 | 24.748 | 25.778 | 24.779 |
| Backward combine | 51.041 | 13.781 | 0.213 | 0.091 |
| All dispatch + combine | 266.284 | 187.550 | 182.509 | 175.432 |

“Backward dispatch” means the reverse transport of forward combine. The backward
of the **forward-dispatch operator** is combine instead; that operator's forward
plus backward is 80.651 + 51.041 = **131.693 ms** in this control. The prior turn's
capture measured 175.129 ms dispatch and 265.170 ms total dispatch/combine; use
one matched capture batch for comparisons rather than mixing these numbers.

All communication kernels are attributed. GEMM signature counts match the
control, every dW callback launches and publishes once, and no extra
recomputation or forward/recompute communication overlap appears in these traces.
Kernel intervals include resource competition and peer-arrival waits. Exposure
is a timeline diagnostic, not an additive prediction of wall-time gain.

## Confirmed dependency problem and remaining cost

The old parameter-gradient view node ran on main. Its gradient publication
joined W before independent upstream input backward could proceed. The correction
creates publication on W and gives compiler-proven exclusive parameter
accumulators/reducers W affinity, preserving existing hooks and ordinary
GraphTask completion. Mixed/native consumers retain their original affinity.

A delayed-W CUDA regression fails before the correction and passes afterward.
The full-model trace confirms 156 premature W-to-main waits on each of ranks
0–3 and 120 on ranks 4–5. After correction, every rank has 48 normal GraphTask-end
waits, **zero explicit early joins**, and W publication/accumulation. No work is
moved into another microbatch.

Resource contention remains. Expert dW GEMM duration is 284.840 ms with existing
overlap, 314.294 ms with early dW and 315.934 ms after the correction. Only
13.781 ms of backward combine was still exposed in the existing reference.
Each dW GEMM launches 148 blocks, with 230188 bytes of dynamic shared memory per
block. DeepEP uses 20 blocks of 768 threads; combine needs 98304 + 1024 shared
bytes and dispatch needs 196608 + 1024. These cannot coexist with those dW blocks
on one SM. Actual dW-GEMM/main-GEMM intersection is only 0.37–1.36 ms per GPU in
the corrected capture. Creating streams does not partition SM resources.

CUDA event-query host time is 115.935/157.137/163.852/167.152 ms for the four
captures. Explicit NNScaler owner polling contributes only about 3.4–3.8 ms in
the overlap cases. It is incorrect to attribute all event queries to that queue;
allocator tracking is also involved.

## Additional experiments

A later matched reference with the ordinary input policy measured 4.419665 s.
Its generated source is byte-identical to the earlier corrected dW case. The
following interventions retain initialization, input checkpoints, work and
numerical metrics. Cross-commit comparisons have an explicit audit showing the
default execution path unchanged.

| Intervention | s/step | Change from 4.419665 s |
| --- | ---: | ---: |
| dW on caller compute stream, after communication submission | 4.441055 | +21.390 ms |
| dW-only 128-SM limit, with corrected publication | 4.423510 | +3.845 ms |

The 128-SM micro test improved by about 0.22 ms, but full-model timing did not.
Its effective setting and restoration are audited; it is not selected.
Deferring only W2 while keeping W13 inline measured **4.432126 s**, 20.309 ms
slower than the corrected full-dW reference, with higher memory. It is rejected.

Two storage-retention policies were also tested. Keeping aliased gradients until
GraphTask completion exceeded the 170 GiB guard at step 9 (171.63 GiB). Keeping
only callback inputs until a completion event still exceeded it at step 7
(170.89 GiB). Neither is a complete performance measurement. Those optional
runtime/backend changes were removed from the final implementation; the
standard allocator-based lifetime protection is retained.

The legacy `dw_split` references (NNScaler `4344a22a`, llm-train `7f49702b`) also
wait `wgrad_done` before returning communication backward. Their default native
weight-group limit is 8.39M elements, below this model's 327.16M local expert
parameters. This can prevent large tasks from entering the intended window
unless overridden. It does **not** establish which settings the historical run
used; the full legacy FBW scheduler was not imported or benchmarked here.

## Correctness and artifacts

The final tree passes **188 CUDA/framework tests** and **9 CPU MoE-backend
tests**, followed by a guarded real EP2 native/dW comparison with bitwise-equal
outputs, input gradients and parameter gradients on both ranks.

Validation covers gradient-stream affinity, real reducer hooks, multiple live
forwards, retained backward, pruned parameter paths, mixed consumers, FP32
parameter gradients, autocast and exception cleanup. A delayed-accumulation CUDA
regression stresses an aliased gradient from an independent producer stream;
removing its lifetime protection corrupts the gradient under allocator reuse.
This guards against removing necessary stream protection to improve a benchmark.

Real EP2 MXFP8 checks use 16K global tokens, recomputation and per-token clipping
20000. The guarded native/named-stream/caller/SM-budget checks preserve outputs,
dX and parameter gradients bitwise. Early micro scripts were missing a main
guard and could be re-entered by compiler spawn workers; those micro timings are
excluded and the guarded checks were rerun. Full-model training uses a separate
entrypoint and did not have that failure.

Detailed logs, per-rank audits, traces, generated-source archives and experiment
scripts are under `/home/yileiyang/agens/stream-overlap-dw-refine-20261118`.
The production preset is `llm-train/scripts/run_stream_overlap_15b_pp3ep2.sh`.
The remaining forward dispatch exposure and lack of reliable incremental dW gain
are open optimization work; these measurements do not establish an upper limit.

### Staged replay integration (2026-11-18)

The runtime and compiler now support `recompute_prefetch_prefix` without
model-specific scheduling code. CPU validation covers prefix/tail gradients
and RNG, metadata changes, mutation detection, storage budgets, abandoned
forwards (including a retained launch point), concurrent-backward rejection,
non-differentiable outputs, explicit training-state snapshots, and composition
with submitted module callbacks. Existing checkpoint-prefetch tests remain
applicable. CUDA tests cover input-readiness alignment, exact gradients,
abandonment, failure cleanup, and host release of a ready gate.

A real 15B attention/MoE block with EP2, BF16 parameters and FP32 gradients
passed bitwise output/dX/all-parameter-gradient comparison after integration.
The staged-replay prototype also completed a matched 15B PP3/EP2 run at batch
size 2, sequence length 8192, accumulation 6. Over steps 60–79, complete iteration
time was 4.230742 s without overlap and 4.137791 s with overlap (2.20% reduction).
The training-body `train_wall` metric was 4.066909 s versus 3.971843 s (2.34%);
it excludes between-step overhead and is not the primary end-to-end metric. Both used
the same experimental compact DeepEP backend and native allocator; this is
not a comparison against an artificially slower allocator or transport.
The integrated runtime additionally keys cache bounds only on prefix inputs.
The result above does not establish complete communication hiding; optimization
and full-model profiling are ongoing.

The integrated runtime also completed a matched 80-step pair with the optional
8-warp compact transport: complete iteration 4.186883 s without overlap versus
4.110043 s with overlap (1.84% reduction), using steps 60–79. Training-body time
was 4.022076 s versus 3.947277 s (1.86%). Source commits, backend-library hashes,
initial state/RNG, full-input checkpoints and workload all match. Every first-step
parameter gradient was checked; maximum relative L2 error is 6.55e-8. Both runs
reference the same verified gradient binary: the new control's complete gradient
digest is byte-identical to that reference. This avoids another 67 GiB copy and
does not relax numerical checks. These gains remain below the task's target.

### Prequeued module execution (2026-11-18)

`launch_phase='prequeued'` separates ordinary side-graph construction from its
CUDA release at backend submission. Compiler tests cover unsplit and split
regions, dependent chunks sharing an anchor, separate anchors, and one/two
code-generation workers. Runtime CUDA tests check actual release timing,
bitwise output/input/weight gradients, and exception cleanup. The matching
DeepEP adapter tests cover dispatch/combine, prepared results, padded dispatch,
and callback failures (33 CPU tests).

The real 15B EP2 block preserves every output and gradient bitwise. Moving
shared forward/backward to dispatch-forward/combine-backward and attention
replay to the opposite pair did not improve block time: 44.54 ms versus
43.23 ms for the previous staged schedule (44.78 ms with no overlap). The new
phase is opt-in; this configuration is not a recommended performance preset.
Its trace shows attention tail submission misses the backward dispatch window.
Earlier tail submission and full-model performance remain under investigation.

### Independent branch weight backward (2026-11-18)

`weight_backward_independent=True` supports a whole eager streamed branch whose
activation does not depend on the weight anchor. A storage-free output token
orders callback registration before that anchor's backward; per-callback events
protect inputs produced on the branch stream. Existing data-dependent rules keep
their original validation and execution path.

Validation includes 82 relevant CPU compiler regressions, true two-worker/two-rank
code generation, 18 CUDA-enabled weight-runtime tests, and four real CUDA linear
backend cases. The new timing test verifies that dW starts during communication
and that upstream dX reaches its consumer before long dW finishes. BF16/MXFP8,
per-head/QKV outputs, clipping, FP32 parameter gradients, frozen weights and
weight updates preserve outputs and gradients bitwise against the ordinary path.

The 15B EP2 block also preserves output/dX/all parameter gradients bitwise.
No additional performance gain is established: the existing staged schedule was
43.215 ms; shared dW with side-stream attention tail was 43.337 ms, and a
caller-stream tail prototype was 43.252 ms. Adding caller-stream expert replay
also did not improve that block comparison. These variants remain opt-in and
are not recommended presets or evidence of complete communication hiding.

### Operator selection (2026-11-19)

`operators` selects a pure branch within a module invocation. Tests cover
repeated invocations, one/two code-generation workers, builtin metadata reads,
dependency/checkpoint rejection, cache invalidation, and multiple live CUDA
forwards. Focused suites pass 65 CPU tests and 13 CUDA tests.

The real 15B EP2 block preserves its output, auxiliary statistics, input gradient,
and all parameter gradients bitwise after moving MoE auxiliary statistics.
That placement did not improve performance: 43.87 ms on a separate stream and
43.96 ms on the attention replay stream, versus 43.25 ms for the existing
schedule and 44.22 ms without overlap. This is a validated selection feature,
not a recommended performance preset or an end-to-end gain claim.

### Operators with internal streams (2026-11-19)

`multi_stream_ops` declares operators whose implementations already manage
side streams. It enables scheduled lifetime tracking before per-rank emission,
so independent code-generation workers agree even when only one pipeline rank
contains a matching operator. CPU coverage includes matched/unmatched patterns,
checkpoints, one/two workers, cache invalidation, and legacy configuration loading.

The focused CPU suites pass 79 tests. Two CUDA validations run a two-rank,
four-stage interleaved pipeline with four microbatches and three optimizer steps.
Loss, all local parameter gradients, and updated weights match eager execution.
One validation has an internal-stream operator only on the first stage.

### Fragment weight-gradient accumulation (2026-11-19)

`WeightGradientAccumulator` provides one additive output buffer per parameter
and GraphTask. Focused CPU regressions cover repeated/retained backward, pruned
parameter edges, live forwards, legacy callbacks, reducer contribution counts,
frozen backend inputs, and failures without partial gradient publication.
Four CUDA tests validate FP32 gradients for FP32/BF16 parameters, checkpoint
replay, distinct producer streams, and dInput advancing before long dWeight
work completes. The corresponding BF16/MXFP8 MoE backend tests also pass with
and without checkpointing and with weight-gradient clipping enabled.

These establish the generic runtime and backend contracts. The production
routed-expert comparison at the top of this page now supplies the matched
end-to-end measurement for a model using fragment accumulation.

### Checkpoint saved tensors with multiple consumers (2026-11-19)

A checkpoint containing backward nodes on several CUDA streams previously
joined replay only on the stream that first requested a saved tensor. Another
consumer could read the reconstructed storage before its producer completed.
A warmed CUDA reproduction reads roughly 102 ms early and returns a gradient
of -32 instead of 12. Cold allocations can synchronize the device and conceal
this race.

The saved-tensor unpack hook now orders each consuming stream after replay and
protects its storage lifetime. Readiness is tracked per backward task and
released at task completion. The same reproduction produces exactly 12 after
the fix. All five CUDA checkpoint-stream tests pass, including either consumer
triggering replay, retained backward, gradient-readiness alignment, and opaque
input lifetime protection. This is a correctness fix, not a performance claim.
