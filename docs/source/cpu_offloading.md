# CPU activation offloading

Activation memory of a pipeline stage is dominated by the tensors saved for
backward by the micro-batches in flight between their forward and backward.
Recomputation trades that memory for compute; offloading moves the tensors to
pinned host memory right after the forward and brings them back before the
backward, using copy engines that otherwise idle.  The two compose: a module
can be recomputed *and* have the inputs its checkpoint keeps offloaded, so that
nothing of it stays resident.

## 1. Design

```
 planning (per model)      code generation (per plan)        runtime (per process)
 ------------------------  --------------------------------  ---------------------------------
 OpPlan.offload_id    -->  with self.cpu_offloading_hooks(g) -->  hooks.CPUOffloadContext
 OpPlan.recompute_id  -->    ckpt.checkpoint(...) inside it        | pack / unpack
                           schedule: hint_forward(seg, dist)        v
                                     prefetch_next_backward(seg) tracker.ExecutionTracker  (what / when)
                                                                   | batches, FIFO contexts, prefetch policy
 Executor.fexecute/backward ---------------------------------->    v
                                                                 transport.Transfer / CopyEngine  (how)
                                                                   | one slab + one event pair per group,
                                                                   | D2H/H2D streams, windows, chunking,
                                                                   | comm-exclusive barriers, fake modes
                                                                   v
                                                                 host_memory.PinnedPool      (where)
                                                                   | best-fit slabs, NUMA membind
                                                                 stats / config                (observe / tune)
```

### 1.1 Planning: `offload_id` and `recompute_id`

A policy marks consecutive forward operators with the same non-negative
`offload_id`; everything autograd saves while they run is offloaded.  The
granularity is whatever module boundary the policy chooses (an `Attention`
call, a whole `Block`, one kernel).  Constraints match `recompute_id`:
consecutive operators, one pipeline stage.
`nnscaler.policies.apply_module_offload(graph, offload_types, recompute_types=())`
groups by module type and FQN like `apply_module_recompute` and assigns both
ids where the two module sets overlap; llm-train's fixed policy exposes the
same through `--offload_modules` / `--recompute_modules` (with the
`Attention.core` spelling for "only the attention kernel").

An operator may carry both ids.  The group is then recomputed and only the
inputs its checkpoint keeps are offloaded.  "Recompute what is cheap to
recompute, offload the rest" is therefore expressed entirely in the plan, e.g.
`--recompute_modules Attention,MoEExpert --offload_modules Block`.

### 1.2 Code generation

`ModuleCodeGen` splits a segment into runs of equal `(recompute_id, offload_id)`
and emits each run as plain code, as `with self.cpu_offloading_hooks(g):`, as
`ckpt.checkpoint(...)`, or as the checkpoint call inside the offload context.
Adapters and multirefs inserted between marked operators inherit the ids of
their neighbour so groups stay contiguous after partitioning.

For scheduled pipelines `ScheduleCodeGen` also emits two hints derived from the
static schedule it unrolls: `hint_forward(segment, distance)` before every
forward (how many segment executions separate it from its own backward; below
`min_backward_distance` the execution is not offloaded) and
`prefetch_next_backward(segment)` before every backward (which backward comes
next).  Both are optional; eager execution falls back to demand-driven prefetch.

### 1.3 Runtime

**Tracker** (`tracker.py`).  `Executor.fexecute(name)` opens an execution
context; the offload groups executed inside are recorded in forward order and
the finished context is queued per segment name.  `Executor.backward(name)`
(or `backward_input`, released after `backward_weight`) pops the oldest one.
The queue mirrors the executor's own queue of detached inputs exactly: every
forward with autograd enabled is queued, also when it offloaded nothing, and a
sequence check turns any drift into an error instead of a wrong release.
Within a context groups are needed in reverse forward order, across
micro-batches a segment's backward consumes contexts in forward order.  Both
hold for 1F1B, interleaved 1F1B and gradient accumulation, so no warm-up
iteration is needed to learn the order (Megatron's fine-grained offloader
records it during the first step instead).  The prefetch policy reloads the
demanded group and exactly `prefetch_groups` (or `prefetch_bytes`) earlier
groups that saved something; `begin_backward` reloads the last groups of the
context it pops and, when the scheduler announced the following backward, the
last groups of that one too.

**Hooks** (`hooks.py`).  Pack skips non-CUDA, non-strided and small tensors
(`min_bytes`; every saved tensor is counted in a size histogram so the
threshold can be chosen from data), keeps a lightweight reference for
parameters/buffers (ZeRO-3 safe) and otherwise registers the view with the
group's transfer: a dictionary lookup, no CUDA call.  Unpack rebuilds the view
with `as_strided` on the reloaded bytes.

**Transport** (`transport.py`).  The unit of transport is a *transfer*: all
tensors one execution of one offload group saved.  When the group ends the
transfer is committed: views of the same storage that overlap are merged into
regions (a block output saved as a whole and as slices is copied once), the
regions are laid out in one pinned slab, one event recorded on the compute
stream orders every copy after the kernels that produced the views, one D2H
copy per region follows on the D2H stream and one event marks their
completion.  Sources are `record_stream`ed so the allocator cannot reuse them
early and dropped so the memory is released once the copy finished.  The reload
is one device buffer (allocated on the consumer stream, whose allocator pool it
belongs to) and one H2D copy, or `copy_chunk_bytes` chunks each with its own
event so the first tensors of a group can be consumed before the rest arrived.
This is what keeps the host-side cost at tens of microseconds per saved tensor
and lets the copy engines run long transfers; Megatron's offloader bulk-copies
per layer for the same reason.  Optional policies: bounded in-flight windows
(`max_inflight_copies`, counted in transfers), chunking and
communication-exclusive windows (`comm_exclusive`, see §3: transfers requested
inside a window are issued when it closes).  Fake modes keep the device
tensors while exercising everything else (A/B tests).  A backward that runs
inside a group (for example `torch.autograd.grad` in the forward) commits the
transfer early; tensors saved afterwards stay on the device
(`skipped_late_bytes`).

**Host memory** (`host_memory.py`).  Slabs are pooled best-fit: sizes round up
to a quantum of 1/32-1/16 of their magnitude (at most 6.25% slack) so MoE
groups whose bytes change every micro-batch still reuse slabs, `acquire` takes
the smallest free slab whose last reader has finished, and `pool_max_bytes`
turns a further allocation into a wait for the best fitting busy slab
(`cudaHostAlloc` costs ~0.3 ms/MiB).  Pages are placed on the GPU's NUMA node
with `set_mempolicy` around the allocation, which works even when the launcher
confines all ranks to one socket.

**Stats** (`stats.py`).  Per-step counters, CUDA-event timings (copy durations,
compute-stream stall time, D2H backlog), the saved-tensor size histogram,
per-group profiles and a summary table; `all_rank_max` reduces the timings
across ranks because the pipeline waits for its slowest rank.  Host time is
split into hook bookkeeping (`pack_seconds`, `commit_seconds`,
`unpack_seconds`), copy issue (`d2h_issue_seconds`, `h2d_issue_seconds`) and
pinned allocation (`acquire_seconds`, only while the pool grows), so a slow
first step is not mistaken for hook overhead.  Finished timing events are
folded at every segment backward, so nothing accumulates when a trainer never
polls `offload_stats()`; polling once per step is what makes the "/step"
figures of the report exact.

## 2. Configuration

Programmatic: `nnscaler.runtime.cpu_offload.configure(OffloadConfig(...))`
before the first step.  Environment: `CPU_OFFLOADING_<NAME>` read on first use.

| field | env | default | meaning |
| --- | --- | --- | --- |
| `enabled` | `DISABLE=1` to turn off | on | master switch |
| `min_bytes` | `MIN_BYTES` | 1 MiB | smaller tensors stay on the device |
| `min_backward_distance` | `MIN_BACKWARD_DISTANCE` | 1 | do not offload a forward whose backward is this close (segment executions) |
| `prefetch_groups` | `PREFETCH_GROUPS` | 2 | groups reloaded ahead of the demanded one |
| `prefetch_bytes` | `PREFETCH_MB` | 0 | alternative byte budget for the prefetch depth |
| `max_inflight_copies` | `MAX_INFLIGHT_COPIES` | 0 | bound of transfers (groups) in flight per direction |
| `copy_chunk_bytes` | `COPY_CHUNK_MB` | 0 | split copies into chunks; reloads get one event per chunk |
| `comm_exclusive` | `COMM_EXCLUSIVE` | `off` | `off` / `strict` / `soft` windows around all-to-all (copies requested inside wait for the end; `strict` also makes the window wait for copies in flight) |
| `numa_membind` | `NUMA_MEMBIND` | on | pinned pages on the GPU's NUMA node |
| `numa_affinity` | `NUMA_AFFINITY` | off | also pin the main thread to the GPU's CPUs |
| `pool_max_bytes` | `POOL_MAX_GB` | 0 | bound of pinned memory owned by the pool (waits instead of growing) |
| `timing` | `TIMING` | on | CUDA-event timings |
| `debug` | `DEBUG` | off | log the saved tensors of each group once |
| `profile_every` | `PROFILE_EVERY` | 0 | log the per-group table and size histogram every N steps (a step is one `offload_stats()` snapshot; nnscaler's trainer and llm-train take one per training step) |
| `fake_mode` | `FAKE_COPY` | `off` | `no_copy` / `d2h_only` / `both_keep` A/B modes |

## 3. Tuning workflow

1. **Measure the ceiling.** Run with `fake_mode=no_copy`: same plan, nothing
   moves.  The difference to the recompute-only configuration is the compute
   saved by not recomputing; the difference to the real run is the cost of the
   transfers.
2. **Look at the per-group table** (`profile_every`, or `offload_report()`):
   bytes per step, copy durations (effective bandwidth), stall time per group,
   and the saved-tensor size histogram.  Effective bandwidth far below the
   single-GPU figure points at host memory placement (check the `NUMA node` log
   line) or at contention; stall time points at reloads arriving late (raise
   `prefetch_groups`/`prefetch_bytes`); many tensors carrying few bytes point
   at `min_bytes` (the histogram marks the buckets it currently excludes); a
   host-side `hooks` time that is a visible fraction of the step is the same
   signal.
3. **Reduce bytes before tuning transport.** Offload the tensors that are
   expensive to recompute (attention kernel outputs, expert inputs), recompute
   the cheap ones (norms, activations, projections); use `Attention.core`
   style specs and the combined recompute+offload form.  Raise
   `min_backward_distance` to skip micro-batches whose backward is near: the
   peak is set by the longest-lived activations, so most of the memory win
   comes from a fraction of the bytes.
4. **Transport knobs last.**  `comm_exclusive` matters only when the node runs
   an NVLink all-to-all in every layer: PCIe copies and NVLink traffic contend
   inside the GPU's I/O hub (on 8x B200 DeepEP `intranode::dispatch` ran 4x
   slower with copies in flight).  The strict window restores the kernel but
   makes the compute stream wait at every window entry, which cost more than it
   saved with 576 windows per step; it is therefore off by default.

## 4. Measurements

8x B200, YOCO-MoE-15B-A6B-L4, PP2 x EP4, interleaved 1F1B, 8 micro-batches,
steps 6-20 mean, rank 0:

| configuration | seq | step (s) | peak (GB) |
| --- | ---: | ---: | ---: |
| no recompute, no offload | 8k | 4.06 | 162.8 |
| recompute MoE only (ceiling for attention offload) | 8k | 4.60 | 95.1 |
| recompute attention + MoE (production) | 8k | 4.96 | 67.3 |
| attention kernel offload + recompute the rest, copies disabled (A/B) | 8k | 5.08 | 83.2 |
| attention kernel offload + recompute the rest | 8k | 6.58 | 77.2 |
| same with strict communication windows | 8k | 7.11 | 77.2 |
| production | 16k | 8.86 | 99.9 |
| attention kernel offload + recompute the rest | 16k | 12.58 | 119.5 |

Loss trajectories are identical across configurations; framework overhead is
~0.1 s/step, copies run at 36-54 GB/s and no reload stalls the compute stream.
The remaining gap on this node is the PCIe/NVLink hub contention above, so
attention offload cannot beat attention recompute there, whose whole cost is
0.36 s/step at 8k.

4x A6000 (PCIe only, no NVLink), YOCO-MoE-A0.4B, PP2 x EP2, interleaved 1F1B,
8 micro-batches, seq 4096, last of 4-5 steps, rank 0:

| configuration | micro-batch | step (s) | peak (GB) | tokens/s | moved per step |
| --- | ---: | ---: | ---: | ---: | ---: |
| no recompute, no offload | 1 | 7.9 | 29.3 | 8.1k | - |
| recompute attention + MoE (production) | 1 | 8.8 | 19.6 | 7.2k | - |
| production | 2 | 13.9 | 27.4 | 9.4k | - |
| production | 3 | 19.4 | 35.2 | 10.1k | - |
| offload whole Blocks, no recompute | 1 | 12.0 | 14.4 | 5.3k | 61 GB |
| recompute attention + MoE, offload the rest (combined form) | 1 | 11.0 | 12.4 | 5.8k | 27 GB |
| combined form, copies disabled (`fake_mode=no_copy`) | 1 | 8.5 | 19.7 | 7.5k | - |
| combined form | 2 | 16.6 | 13.2 | 7.9k | 53 GB |
| combined form | 4 | 28.9 | 14.6 | 9.0k | 107 GB |
| combined form, `min_bytes` 2 MiB (31% fewer tensors, 4% fewer bytes) | 1 | 11.1 | 12.7 | 5.8k | 26 GB |
| combined form, `comm_exclusive=soft` around the torch all-to-all | 1 | 9.4-11.9 | 12.5 | 5.4-6.8k | 27 GB |
| combined form, `comm_exclusive=strict` | 1 | 11.4-13.0 | 12.5 | 4.9-5.6k | 27 GB |
| combined form, `min_backward_distance=2` | 1 | 10.4-12.5 | 12.5 | 5.1-6.1k | 25 GB |

Loss trajectories are identical.  Step times of these short runs vary by
about one second from step to step, so the window and distance variants are
within noise here; a longer run is needed to rank them.  The combined form keeps less resident than
offloading everything, moves 44% of the bytes, and holds the peak nearly flat
while the micro-batch grows (12.4 -> 14.6 GB from 1 to 4, against 19.6 -> 35.2
GB for the production form from 1 to 3).  The `no_copy` A/B run matches the
production step time: the hooks, bookkeeping and prefetch logic cost nothing on
the critical path; the whole penalty of the real run is the bytes on the PCIe
bus, which here also carries the all-to-all and the pipeline traffic.  Hence on
this box the memory freed does not convert into throughput (9.0k tokens/s at
micro-batch 4 against 10.1k for production at micro-batch 3), and reducing the
host-side work further (`min_bytes`) changes nothing.  Offloading pays off when
the recomputed work is expensive relative to the bytes it saves, when the copy
path does not share a link with collective communication, and when the memory
freed buys a larger micro-batch or sequence; the fake mode and the per-step
timings quantify both sides on a new configuration before committing to it.
