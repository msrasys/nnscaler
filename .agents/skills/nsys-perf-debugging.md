---
name: nsys-perf-debugging
description: Use NVIDIA Nsight Systems (nsys) to profile and debug nnscaler training performance. USE WHEN the user reports low TFLOPS / tokens-per-second, suspected pipeline bubble, or suspected communication-compute serialization in a parallelized nnscaler run, and you need to capture a trace, localize the bottleneck to an nnscaler subsystem, change the right layer, and re-measure without being misled by the compile cache.
---

# Skill: nsys-perf-debugging

Use this skill for the nsys → diagnose → change → re-measure loop on nnscaler training. Before you start, read [AGENTS.md](../../AGENTS.md) §2 (pipeline), §3 (routing table), and §4 (footguns — especially cache invalidation).

---

## 0. Prerequisites

- `nsys` (Nsight Systems) installed and on `PATH`. Verify: `nsys --version`.
- An nnscaler training entry that reproduces the perf issue. Prefer the **smallest config that still shows the symptom** — short iteration loops dominate the quality of every step below.
- Know which command + `ComputeConfig` produced the current numbers. Record these alongside every trace.

---

## 1. Capture a baseline trace

Wrap your training script with `nsys profile`. A reasonable default for distributed PyTorch with NCCL:

```bash
nsys profile \
    -o baseline \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    -f true \
    python <train_entry> <args>
```

Notes:
- Use `--capture-range=cudaProfilerApi` and wrap a few warm iterations + a few measured iterations with `torch.cuda.profiler.start()` / `stop()` (or `cudaProfilerStart/Stop`). Profiling the whole run produces multi-GB traces that are slow to open.
- For multi-rank runs, profile only **one or two ranks** of interest — typically rank 0 plus a suspected slow stage. Set `NSYS_OUTPUT` per rank or use `--output=trace_rank%q{RANK}`.
- Add NVTX ranges in the training loop (`torch.cuda.nvtx.range_push/pop`) around `forward`, `backward`, `optimizer.step`, and around each pipeline micro-batch if your entry doesn't already. This makes the timeline readable.

Record alongside the trace: git SHA, branch, command, `ComputeConfig`, world size, TP/DP/PP shape, and the four headline numbers: **TFLOPS, tokens/s, peak memory, iteration time**.

---

## 2. Read the generated code first

Before opening the trace, open `.nnscaler/_parallel_modules/__main__/{ModelName}/_/gencode{rank}.py` for the rank you profiled. This is the fastest way to learn what the compiler actually emitted: which `torch.cuda.stream(...)` each collective lands on, where `synchronize()` calls sit, where activations are released. The trace will make a lot more sense afterward, and many "mystery" serializations are obvious from the source.

---

## 3. Localize from the CLI

Coding agents run headless — use `nsys stats` and `nsys analyze` to extract structured summaries from the `.nsys-rep` file, not the GUI. Useful reports:

```bash
# Top CUDA kernels by total GPU time (compute hotspots)
nsys stats --report cuda_gpu_kern_sum --format csv,column baseline.nsys-rep | head -40

# NCCL / communication kernel breakdown
nsys stats --report cuda_gpu_kern_sum --format csv baseline.nsys-rep \
    | awk -F, 'NR==1 || /nccl|ncclKernel/'

# Per-stream activity — reveals whether comm and compute share a stream
nsys stats --report cuda_gpu_trace --format csv baseline.nsys-rep \
    | awk -F, 'NR==1 || $0 ~ /nccl/' | head -40

# CUDA API calls — large cudaStreamSynchronize / cudaEventSynchronize totals
# indicate forced serialization
nsys stats --report cuda_api_sum --format csv,column baseline.nsys-rep | head -30

# NVTX range summary — per-phase (fwd/bwd/opt/microbatch) wall time
nsys stats --report nvtx_pushpop_sum --format csv,column baseline.nsys-rep | head -40

# Built-in expert-system rules (idle GPU, low occupancy, sync misuse, etc.)
nsys analyze baseline.nsys-rep
```

Work top-down on the output:

1. **Which rank / pipeline stage** is on the critical path? Compare per-rank traces. The slow rank is the one with the longest wall time between iteration-start and iteration-end NVTX ranges; idle ranks are waiting on it.
2. **Which phase** dominates? Use the `nvtx_pushpop_sum` report — `forward` / `backward` / `optimizer` / per-microbatch totals.
3. **Which kernel category** dominates? Aggregate `cuda_gpu_kern_sum` by name prefix:
   - `ncclKernel_*` / `nccl*` → communication.
   - `ampere_*gemm*`, `cutlass*`, `flash_*`, `attn*` → compute.
   - `Memcpy*` / `Memset*` → data movement.
   - Large gaps in `cuda_gpu_trace` with high `cudaStreamSynchronize` totals in `cuda_api_sum` → CPU/Python overhead or forced sync.
4. **Comm/compute overlap check**: in `cuda_gpu_trace`, identify the stream IDs of NCCL kernels vs. compute kernels. If they share a stream ID, overlap is structurally impossible. If they are on different streams but their `[Start, Start+Duration]` intervals don't overlap, look for a `cudaStreamWaitEvent` / `cudaEventSynchronize` between them in `cuda_api_sum`.

Then map symptom → suspect subsystem using AGENTS.md §3 and the table below.

---

## 4. Symptom → suspect subsystem (nnscaler-specific)

| What you see in nsys | First place to look | Notes |
|---|---|---|
| Long idle gaps in the middle of an iteration on most ranks (pipeline bubble) | `pipeline_nmicros`, `pipeline_scheduler` in `ComputeConfig.pas_config`; [nnscaler/graph/schedule/predefined.py](../../nnscaler/graph/schedule/predefined.py) | Try `sched_1f1b_interleaved`; tune `num_stages` |
| NCCL kernel (`ncclAllReduce` / `AllGather` / `ReduceScatter`) on the same stream as compute, or on a separate stream but the compute stream is event-waiting for its full duration | `use_multi_streams` flag, `_get_node_stream()` in [nnscaler/codegen/schedule/schedule.py](../../nnscaler/codegen/schedule/schedule.py); `StreamConfig` in [nnscaler/graph/schedule/schedplan.py](../../nnscaler/graph/schedule/schedplan.py); stream contexts in [nnscaler/runtime/device.py](../../nnscaler/runtime/device.py) | Confirm in `gencode{rank}.py` that the collective is wrapped in `torch.cuda.stream(...)` |
| Many tiny NCCL kernels back-to-back | Adapter fusion in [nnscaler/execplan/planpass/fusion.py](../../nnscaler/execplan/planpass/fusion.py) (`DiffFusion.nnfuse`); gradient bucketing in [nnscaler/runtime/adapter/reducer.py](../../nnscaler/runtime/adapter/reducer.py) | |
| `ncclSend` / `ncclRecv` at stage boundary blocks the next compute on the same stage | Schedule order in [nnscaler/graph/schedule/interleaved_1f1b.py](../../nnscaler/graph/schedule/interleaved_1f1b.py) (SEND_F / RECV_F / SEND_B / RECV_B placement); `move()` in [nnscaler/runtime/adapter/collectives.py](../../nnscaler/runtime/adapter/collectives.py) | If schedule issues comm too late, no amount of stream juggling will overlap it |
| Spurious `cudaStreamSynchronize` / event-wait near a collective | Codegen sync emission in [nnscaler/codegen/schedule/schedule.py](../../nnscaler/codegen/schedule/schedule.py); `synchronize_streams` in [nnscaler/runtime/module.py](../../nnscaler/runtime/module.py) | |
| OOM with otherwise-good throughput | Recomputation (`recompute_modules`), micro-batch count, schedule choice; tensor lifecycle in [nnscaler/codegen/lifecycle.py](../../nnscaler/codegen/lifecycle.py) | |
| Autodist picked an obviously bad plan | Profile DB likely stale, or cost-model gap; [nnscaler/autodist/cost_database.py](../../nnscaler/autodist/cost_database.py), [nnscaler/autodist/op_partition.py](../../nnscaler/autodist/op_partition.py) | Delete `~/.cache/nnscaler/autodist/...` before retrying |
| Compute kernel itself is slow (one big GEMM / attn) | Custom op / kernel choice in [nnscaler/customized_ops/](../../nnscaler/customized_ops/); op dim annotations | Not an nnscaler-orchestration problem — fix the op |

If the symptom is comm/compute serialization, also verify in this order: (a) is `use_multi_streams` even on? (b) is the comm wrapped in a non-default stream in `gencode{rank}.py`? (c) does the schedule issue the comm early enough to overlap?

---

## 5. Hypothesize and change

Write one sentence per change: *"I expect changing X in [file] to reduce Y by Z%, because …"*. If you can't fill in `because`, go back to §3 — you are guessing.

You may touch multiple subsystems from AGENTS.md §3 in a single iteration if the hypothesis genuinely requires it (e.g. a new prim in `ir/adapter/` plus its emission in `codegen/`). Just be aware that the more layers you change at once, the harder attribution becomes — if the re-measure is ambiguous, split the change.

Edit the **generator / schedule / policy**, never the emitted `gencode{rank}.py` (see AGENTS.md §4).

---

## 6. Invalidate the cache (the #1 cause of "my change did nothing")

When you change anything in `nnscaler/graph/`, `nnscaler/execplan/`, `nnscaler/codegen/`, `nnscaler/autodist/`, `nnscaler/policies.py`, or `nnscaler/ir/adapter/`, you must invalidate:

| Cache | Path | How to invalidate |
|---|---|---|
| Generated per-rank module | `.nnscaler/_parallel_modules/__main__/{ModelName}/_/gencode{rank}.py` (+ `fullmodel.pt`) | `compile(..., override=True)` / `ComputeConfig(..., override=True)`; or `rm -rf .nnscaler/` |
| Autodist profile DB (op / comm costs) | `~/.cache/nnscaler/autodist/{version}/{GPU_name}/` | `rm -rf` the directory; required when op cost characteristics change (new fusion, new prim, kernel-level change) |

**Verification before re-measuring**: diff the new `gencode{rank}.py` against the previous one.

```bash
# quick sanity diff after a code change
diff -u .nnscaler.prev/.../gencode0.py .nnscaler/.../gencode0.py | head -80
```

If the file is byte-identical, the cache was not invalidated and any perf number you collect is meaningless.

---

## 7. Re-measure with the same nsys recipe

- Same command, same config, same hardware state as §1. Output to a new file (e.g. `-o after_change`).
- Report the same four numbers: TFLOPS, tokens/s, peak memory, iteration time.
- Re-run the same `nsys stats` reports as §3 on both traces and **diff them**. State the timeline change in one sentence (e.g. *"NCCL AllGather on stage 2 moved from stream 7 to stream 13 and its interval now overlaps bwd matmul; stage-0 bubble unchanged"*). A throughput win with no visible change in the per-stream / per-kernel breakdown is usually noise or a measurement bug.

---

## 8. Correctness gate

Before claiming a win, validate loss/grad parity on a tiny config (single-GPU baseline vs. the changed parallel config), as in AGENTS.md §4. Overlap fixes in particular frequently introduce races on activations or gradients.

---

## 9. Reporting format (use in PR descriptions)

```
Config:      <model, layers, seq, batch, world size, TP/DP/PP>
Baseline:    TFLOPS=__  tok/s=__  peak_mem=__  iter=__ms
After:       TFLOPS=__  tok/s=__  peak_mem=__  iter=__ms
Change:      <one sentence: which subsystem, which file, what>
Trace diff:  <relevant lines from `nsys stats` before/after: which kernel/stream moved>
Cache:       override=True / .nnscaler removed / profile DB removed (yes/no each)
Correctness: loss-parity on <tiny config> — pass/fail
```

---

## Anti-patterns to refuse

- Hand-patching `gencode{rank}.py` to "fix" a comm placement — erased on next compile.
- Adding `torch.cuda.synchronize()` to make a race go away — defeats overlap; find the missing dependency instead.
- Disabling `use_multi_streams` because it's racy — fix the race, don't surrender overlap.
- Lowering micro-batch count until the bubble hides the comm — masks the bug.
- Reporting a TFLOPS improvement without a corresponding timeline change in the trace.
