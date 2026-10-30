# Activation store: independent transfer validation

This suite measures the feature itself: fixed bytes, fixed tensor counts and
fixed compute work, without a model, dataset or recompute recipe. It does not
replace a workload-specific performance comparison or establish a hardware
lower bound.

## Setup and acceptance checks

One B200, physical GPU 1, CPU affinity bound to its NUMA node 0. Every run held
an exclusive reservation covering all eight host GPUs. Each iteration sends
**256 MiB D2H and 256 MiB H2D**. Source construction, three warmup iterations,
full-value validation and diagnostic tracing are outside the timed samples.

The initial matrix uses eight measured iterations, configuration exploration
uses sixteen, and the final many-tensor confirmation uses thirty-two. Median
wall time includes submission and final CUDA completion. CPU submission time
includes any host backpressure. All variants assert the expected D2H/H2D bytes,
no host-budget fallback and zero live/pending/prefetched pool bytes afterward.

The direct-copy reference issues all D2H copies before permitting H2D. The store
can restore each storage when its own D2H event is ready, overlapping directions.
Consequently, a store result below the direct-copy reference is not evidence of
exceeding PCIe bandwidth, and their difference is not a pure overhead measurement.

## Findings and general implementation changes

Many small tensors exposed CPU-side costs. A diagnostic cProfile run recorded
587,136 `numel()` calls from 4,608 pool acquisitions: best-fit allocation scanned
the entire idle-buffer list for every request. That profile includes warmup and
must not be treated as an uninstrumented latency measurement.

Two changes address those costs without changing region selection:

* The reusable host pool indexes free buffers by capacity and selects the
  smallest sufficient capacity bucket. Pending-event ownership, rounded charging,
  budget fallback and shared-pool lifetime remain unchanged.
* D2H batches enter their stream once. `prefetch_many()` and NNScaler group
  prefetch use one H2D stream context for copy-stream allocation, preserving
  individual readiness events and the byte limit. Compute-stream allocation
  retains its existing allocation/dependency path. The stream context is restored
  even on an exception; handles from another store are rejected before submission.

## Confirmed measurements

For **128 × 2 MiB**, thirty-two measured iterations per configuration:

| Configuration | Round-trip wall ms | CPU submission ms |
| --- | ---: | ---: |
| Previous implementation, one batch of 128, individual prefetch calls | 15.890 | 15.501 |
| Capacity buckets + batched transport, one batch of 128 | 13.163 | 11.034 |
| Same implementation, configuration changed to 32 tensors per batch | 12.637 | 11.337 |

The implementation change reduces measured wall time by **17.2%** and CPU
submission time by **28.8%**. Selecting a smaller batch brings the total measured
wall-time reduction to **20.5%**, without another runtime code change. Smaller
is not always better: batches of four measured about 14.34 ms in the sixteen-
iteration exploration. The best setting is therefore exposed as a scheduling
choice, not encoded for a particular model or tensor shape.

For **16 × 16 MiB**, the initial matrix measured round-trip wall time changing
from 7.745 to 7.376 ms, with submission time changing from 1.907 to 1.521 ms.
The direct-copy references stayed essentially unchanged: about 9.70 ms for
sixteen tensors and 10.58 ms for 128 tensors.

The equal-compute tests use two identical blocks of BF16 matrix multiplications:

| Configuration | Serial compute/copy ms | Overlapped compute/copy ms |
| --- | ---: | ---: |
| 16 × 16 MiB, new implementation | 8.796 | 7.889 |
| 128 × 2 MiB, batches of 32, confirmation | 14.034 | 12.979 |

This demonstrates usable asynchronous overlap with equal work. It does not
prove all DMA is hidden. Many-tensor round trips still exceed the simpler direct-
copy reference; version tracking, ownership and per-tensor API work remain real
CPU costs.

## Budgets, safety and resources

At 16 × 16 MiB with batches of four, reducing the D2H inflight threshold from
512 to 32 MiB increased round-trip latency from about 7.57 to 8.94 ms. Reducing
the prefetch window to 32 MiB produced 112 demand loads over eight iterations
without changing total transfer bytes or correctness. Demand-load counts alone
therefore do not determine the exposed delay; inspect the actual timeline.

The store's pinned pool was **256 MiB** under a 1 GiB budget. A separate direct-
copy reference also holds 256 MiB of pinned buffers. Whole-probe process-tree RSS
peaked around **1.6 GiB**. PyTorch GPU allocated/reserved peaks were approximately
**288/822 MiB**; the sampled device-level used-memory maximum was about **1682 MiB**,
including CUDA/runtime allocations. Reserved peaks also include cached arenas
from earlier cases and should not be attributed to one mode's activations.

After the changes, **193 runtime/integration tests**, all eight pipeline flag
combinations, and real Attention/MoE PP2/VPP4 gradient/parameter comparisons
passed. A separate mutation-error test verified stream restoration and subsequent
successful prefetch. The policy test's largest reported gradient absolute
difference was approximately 9.1e-13. These correctness groups overlap and are
not summed into a count of unique tests.

All guards completed without a resource stop or remaining owned processes.
Including the two-GPU numerical checks, the largest process-tree RSS was about
8.25 GiB and sampled single-GPU use stayed below 2.8 GiB. Free disk space stayed
above 848 GiB. No model checkpoints were created.

## Reproduction and evidence

Initialize the CUDA environment, reserve the GPU, and run:

```bash
python benchmark/benchmark_activation_store.py \
  --tensors 128 --tensor-mib 2 --batch-tensors 32 \
  --inflight-mib 512 --prefetch-mib 512 --prefetch-mode batch \
  --iterations 32 --output /tmp/activation-probe.json
```

Use the same GPU-local CPU binding when comparing runs. The benchmark refuses
to overwrite results and caps payload, CPU-pool size and iteration counts.

Configurations, raw samples, bounded traces, source hashes and resource records
are in the [evidence manifest](../activation-store-results/20261030/performance/manifest.json).
The earlier implementation is frozen as `v4`, the tested implementation as `v5`.
The additional stream-error test was run against the same runtime in the working
checkout. Full-model measurements from October 29 are not reused as evidence for
these implementation changes.
