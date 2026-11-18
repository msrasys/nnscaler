# Module stream validation

Full hiding of DeepEP communication has **not** been demonstrated. The 15B
preset keeps shared-expert splitting and expert/attention recomputation streams.
Deferred dWeight remains opt-in: correcting its publication removes a measured
regression, but adding it to that preset has not shown a reliable incremental gain.

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
