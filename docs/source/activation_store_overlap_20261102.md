# Activation store: equal-work overlap validation

The store can hide most transfer time when independent computation covers the
copies. This is a component experiment with fixed payloads, without a model or
recompute policy; it does not establish zero overhead for distributed training.

## Measurement

One B200 (GPU 0), CPU and memory bound to NUMA node 0, an exclusive reservation
covering the host's eight GPUs. Each case uses four warmup and sixteen measured
iterations. Two identical blocks of 400 BF16 4096 × 4096 matrix multiplications
provide independent work. The compute-only reference retains the same payload
on GPU through both blocks. The overlap case submits D2H before the first block,
prefetches before the second, and then loads the saved values.

Source construction, warmup, full-value validation and diagnostic tracing are
excluded from the samples. Wall time includes submission and final CUDA
completion. Each transfer mode verifies exact D2H/H2D bytes, no host-budget
fallback, and zero live/pending/prefetched pool bytes afterward.

| Payload per direction | Tensors per batch | Compute only, ms | With overlapped copies, ms | Store round trip without compute, ms |
| --- | ---: | ---: | ---: | ---: |
| 16 × 16 MiB | 16 | 83.135 | 84.386 | 10.317 |
| 16 × 16 MiB | 4 | 83.731 | 84.255 | 7.728 |
| 128 × 2 MiB | 128 | 83.509 | 87.369 | 14.924 |
| 128 × 2 MiB | 32 | 83.593 | 88.188 | 12.126 |
| 16 × 64 MiB | 16 | 83.901 | 84.628 | 38.898 |
| 16 × 64 MiB | 4 | 83.869 | 84.755 | 29.583 |

These are per-case medians. The large-tensor cases add about 0.6–1.5% to the
compute-only reference. The many-small-tensor cases add about 4.6–5.5%; per-value
ownership, version tracking and submission remain real CPU work.

Dividing a payload into more batches shortens round-trip latency here by about
19–25%: a restore can start after an earlier D2H batch completes, while later
batches are still copying. It does not consistently improve a schedule whose
copies already overlap sufficient computation. In particular, splitting the
128-tensor batch did not improve its overlapped result. Batch size is therefore
a configuration choice, not a universally faster default.

Batching controls submission and completion boundaries. The store still copies
discontiguous storages separately; it does not allocate an entire-payload GPU
packing buffer or change user storage identities. A simpler direct-copy reference
waits for the entire D2H payload before H2D, so a lower store round-trip time is
not evidence of exceeding link bandwidth or a pure API-overhead measurement.

## Memory and scope

The shared store pool budget is 2 GiB, and D2H/prefetch windows are each 1 GiB.
A separate direct-copy reference owns additional pinned buffers. The largest
whole-process RSS was below 4 GiB; resource guards completed without a stop or
remaining owned processes. No model checkpoints were written.

Peak GPU allocation includes the initial and final payload. This experiment
does not demonstrate a lower GPU peak: training memory savings require multiple
activation lifetimes and are measured separately. Reserved memory also includes
allocator caches from earlier modes and cannot be attributed to one mode alone.
The initial unbound-CPU probes and October 30 measurements are not mixed into
this NUMA-bound comparison.

## Reproduction

Initialize the host CUDA environment and reserve its GPU before running:

```bash
CUDA_VISIBLE_DEVICES=0 numactl --cpunodebind=0 --membind=0 \
  python benchmark/benchmark_activation_store.py \
  --tensors 16 --tensor-mib 64 --batch-tensors 4 \
  --iterations 16 --warmup 4 --inflight-mib 1024 --prefetch-mib 1024 \
  --cpu-budget-mib 2048 --compute-repeats 400 --compute-dim 4096 \
  --output /tmp/activation-overlap.json
```

Use the NUMA node local to the selected GPU. Exact configurations, samples,
bounded traces, source hashes and resource records are in the
[evidence manifest](../activation-store-results/20261102/manifest.json).
Runtime `acdca489` passed 304 runtime/CUDA checks, pipeline combinations, and
real PP2/EP4 gradient/parameter comparisons. The benchmark adds an actual
compute-only reference in `27af2685`; it does not alter the runtime.
