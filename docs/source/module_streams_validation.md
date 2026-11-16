# Module stream validation

The generic CUDA-stream feature is implemented and correctness-tested. **The
requested full hiding of DeepEP communication has not been achieved.** The best
completed clean comparison reduces step time by 4.60%; most of that improvement
comes from communication tuning. Additional pipeline experiments are not part
of the shipped implementation.

## Workload and protocol

Six NV18-connected NVIDIA B200 GPUs run the unchanged llm-train `main` model:
YOCO-MoE-15B-A6B-L4, PP3/EP2, 12 virtual stages, 8K sequences, 12 microbatches,
MXFP8, frozen Vision, Attention/MoEExpert checkpointing, and Muon. Batches are
real VL data. Language weights start from random initialization; this is not a
resumed production 150B checkpoint. llm-train changes only its bash launcher.
The original 303000-update learning-rate horizon is preserved.

The five clean controls use nnscaler `ce0ab3ab` and llm-train `4f3873a0`, with
clean source trees recorded in each launch manifest. All first 40 input batches
match by hash on every rank. Each step is measured through synchronized
optimizer completion, taking the maximum rank time. The measurement window is
steps 21–40. Runs C, D and E profile only step 41, **after** the timing window;
their step-41 input hashes also match on all ranks.

| Run | Configuration | Mean step (s) |
| --- | --- | ---: |
| A | Original DeepEP defaults; no stream rules | 4.551903 |
| B | Tuned DeepEP + expert recomputation stream | 4.364083 |
| C | Repeat original baseline | 4.596024 |
| D | Tuned DeepEP; no stream rules | 4.384227 |
| E | Repeat tuned DeepEP + recomputation stream | 4.363007 |

The two baselines average **4.573963 s**, and the two candidates average
**4.363545 s**: a **4.60%** step-time reduction. Tuning alone gives 4.15% relative
to the same baseline average. The candidate is only 0.47% faster than the single
tuning-only control. That incremental difference is smaller than the 0.97%
difference between the baseline repeats; it should not be presented as a
precisely isolated causal effect of stream scheduling.

The candidate configuration is:

```bash
export NNSCALER_MODULE_STREAMS='[{"module":"*.expert","stream":"expert_recompute","recompute_only":true}]'
export NNSCALER_DEEPEP_CONFIG='{"2":{"dispatch":[96,128,512,6,128],"combine":[96,64,512,6,128]}}'
```

DeepEP is `1.2.1+567632d`. An isolated EP2 transport sweep validated all 81 tested
configurations. Its best dispatch/combine pair took 0.266 ms versus 1.397 ms for
the original defaults. That isolated improvement is not the training speedup.

## Communication remaining

Matched step-41 traces from the frozen-code runs give per-device interval unions:

| Rank | Baseline DeepEP (ms) | Baseline fraction | Tuning-only exposed fraction | Recompute exposed fraction | Recompute communication overlap |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 656.65 | 13.35% | 4.34% | 3.50% | 12.60% |
| 1 | 481.59 | 9.81% | 2.84% | 2.94% | 10.13% |
| 2 | 515.86 | 10.52% | 3.51% | 3.20% | 6.14% |
| 3 | 528.26 | 10.77% | 3.52% | 3.30% | 16.04% |
| 4 | 434.38 | 8.98% | 3.68% | 2.65% | 7.17% |
| 5 | 350.91 | 7.25% | 1.52% | 1.57% | 5.71% |

Baseline and tuning-only communication have no concurrent useful compute.
Recompute still leaves approximately 1.6–3.5% of the profiled step exposed to
DeepEP. Fractions are per rank and cannot be summed. The trace denominator
includes profiler overhead; communication kernels also include peer-readiness
waits. Neither their duration nor overlap fraction alone predicts recoverable
critical-path latency. Some recomputation instead finishes during pipeline
receive waits, before the next dispatch starts.

## Correctness and memory

- The latest combined CUDA-enabled suite passes 116 tests. A strengthened
  checkpoint-owner regression and repeated-backward test additionally pass,
  verifying storage survives queued backward work after recomputation completes.
- Actual EP2 MoE dimensions are 4096 embedding, 1024 latent, 6656 expert FFN,
  32 experts, top-8 and shared width 1664, with MXFP8 and expert recomputation.
  Recompute-only, independent asynchronous metadata copies, and their combination
  produce bitwise-identical outputs, input gradients and parameter gradients.
- Shared splitting plus recomputation has identical outputs and parameter
  gradients; BF16 input-gradient relative L2 error is 8.708e-4. Host submission
  previously exceeded the input-gradient test's 0.003 bound (0.003794). A
  diagnostic fixed-order FP32 fan-in makes all variants bitwise identical;
  production retains its original BF16 fan-in. The original bound was not relaxed.
- Across the clean measurement window, candidate B/E maximum loss differences
  from A are 0.000671/0.001276, and gradient-norm differences are
  0.002629/0.002848. Baseline C itself differs from A by up to 0.001484 in loss
  and 0.004798 in gradient norm. These short runs do not establish long-term
  training equivalence.
- Process-tree RSS peaks around 412 GiB. Maximum device usage rises from about
  102 GiB in baseline runs to about 118 GiB with recomputation streams. The
  PyTorch live-allocation peak is nearly unchanged; reserved memory and device
  usage increase. This extra memory matters for configurations near capacity.

The original main-model compile also reproduced an existing `dict.update`
tracing failure in attention kwargs. NNScaler now retains those updates and
dynamic nested lookups; 65 relevant CPU tests passed. No llm-train model source
was changed.

## Other experiments

Earlier 40-step runs with profiling inside the measurement window, excluding the
profiled step, gave 4.620 s for baseline, 4.441 s for tuning only, 4.539 s for the
earlier shared-stream implementation, and 4.464 s for current shared streams
plus recomputation. Independent async metadata copies plus recomputation gave
4.439 s. These exploratory values are not mixed with the clean protocol above.

The shared expert provides only 22–24% as much kernel time as the original
DeepEP communication in the component trace. Shared splitting overlaps some
communication but its scheduling and resource costs offset the gain for this
shape. An external DeepGEMM resource probe (shared GEMMs 84 SMs, recompute 116,
DeepEP 64) gave 4.481 s and was not integrated into the framework.

An external two-stream pipeline prototype is investigating independent
microbatches. Four-stage/two-rank tests match losses, all optimizer gradients and
weights across four updates, including a deliberately delayed stateful CUDA
loss. The real model required explicit serialization of its stateful MoE loss;
simply changing microbatch streams raced on cumulative expert counts. The
corrected 12-step probe completed, but its short-window time was 4.463 s versus
4.370 s for the matched tuning-only control. It is not a validated performance
feature and has not been committed to either repository.

Further probes used a separate input stream, independent intranode DeepEP
buffers with handle-to-buffer ownership, schedule-aware forward/backward stream
assignment, and transfer placement following producers/consumers. The two-buffer
transport test passed exact results across cached dispatch/combine and deliberately
changed caller streams. Actual traces confirmed two active communication queues.
Nevertheless, matched short-window times (steps 6–11) remained worse:

| Probe | Mean step (s) |
| --- | ---: |
| Tuning-only control | 4.369649 |
| Independent input stream, shared communication buffer | 4.445104 |
| Two communication buffers, 32-SM communication | 4.556345 |
| Schedule-aware assignment, two buffers, 64 SMs, async metadata | 4.450410 |
| Additionally aligned transfer streams | 4.584429 |

All compared batches match by hash. These real-model probes passed scalar
training diagnostics; they are not full parameter-gradient equivalence audits.
The last probe still exposed 2.2–4.9% of the profiled step to communication.
These experiments were therefore left outside the shipped framework. They do
not provide evidence of full communication hiding.

Local evidence lives in `/home/yileiyang/agens/stream-overlap-20261116`: per-rank
metrics and batch hashes, compressed traces, launch manifests, resource logs,
and generated-source archives. Runs enforce disk/memory/time bounds and clean
owned orphan workers after failure. Generated sources and experiment helpers are archived. Large generated weight
and graph caches are removed after the completed comparisons; no GPU jobs remain.
