# Module stream validation (work in progress)

This branch has not yet demonstrated the requested end-to-end speedup. The
current implementation passes CUDA regressions; optimized training comparisons
and communication tuning are in progress.

## Real training baseline

On six NVIDIA B200 GPUs, the unchanged llm-train main model ran 12 optimizer
steps successfully: YOCO-MoE-15B-A6B-L4, PP3/EP2, 12 virtual stages, 8K sequences,
12 microbatches, MXFP8, frozen Vision, and Attention/MoEExpert recomputation.
Input batches were hashed outside the timing window. The observer synchronized
after optimizer work; the six ordinary sampled steps after warmup averaged
4.5913 seconds using the slowest rank. This is an exploratory baseline; final
timing will use repeated unprofiled runs after step 20.

One profiled optimizer step gave these per-rank interval unions:

| Rank | Profiled step (ms) | DeepEP dispatch/combine (ms) | Fraction | Concurrent compute |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4960.1 | 477.0 | 9.62% | 0 |
| 1 | 4973.4 | 488.9 | 9.83% | 0 |
| 2 | 4957.0 | 451.2 | 9.10% | 0 |
| 3 | 4953.6 | 543.3 | 10.97% | 0 |
| 4 | 4890.2 | 387.3 | 7.92% | 0 |
| 5 | 4888.6 | 356.0 | 7.28% | 0 |

These are per-device fractions, not additive across ranks. Instrumented timing
includes profiler overhead. A candidate must show a corresponding reduction in
unprofiled step time, not merely concurrent kernel intervals.

## Correctness completed

- Initial CUDA stream runtime/lifetime suite: 74 tests passed.
- Generated repeated-module CUDA forward/input-gradient/parameter-gradient
  equivalence passed.
- The actual llm-train MoE ran on EP2 with dimensions 4096/1024 latent/6656 expert,
  32 experts, top-8 and shared width 1664. Both split and whole-branch scheduling
  matched baseline outputs exactly, with and without expert recomputation.
  Split-mode parameter gradients were bitwise identical. The largest whole-mode
  parameter-gradient relative L2 error was 4.5e-8. BF16 input-gradient relative
  L2 error was approximately 8.7e-4, associated with changed accumulation order.
- The latest combined suite passes 110 tests, including before/after placement,
  host submission, accumulator affinity, manual-gradient fences, opaque backward
  owners, asynchronous metadata copies and earlier stream-lifetime regressions.
- The CPU-only subset additionally covers reducer serialization and guards;
  55 tests passed with CUDA hidden.

## Observed limitation and next comparisons

The first standalone MoE traces showed that simply moving the shared expert to
another stream covered only about 7% of DeepEP kernel time. Its prefix often
finished before dispatch was submitted. A synchronous CPU-to-CUDA expert-count
copy also delayed the dispatch operator's return until communication completed.
Those observations motivated saved readiness events, optional host submission
and bounded asynchronous metadata copies.

The component's shared-expert linears and SwiGLU contained only about 22–24% as
much GPU kernel time as dispatch/combine. Thus shared-expert scheduling alone
cannot be assumed to hide all communication for this shape. Communication
granularity/SM allocation and additional independent computation must be measured
before drawing a final performance conclusion. No result from those proposed
comparisons is claimed here.

The initial main-model compile also reproduced an existing `dict.update` tracing
failure in attention kwargs. The framework now lowers updates to tracked writes
and retains dynamic nested lookup values; 65 relevant CPU tests passed. No
llm-train model implementation was changed.
