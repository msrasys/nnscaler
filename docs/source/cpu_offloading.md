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
`prefetch_next_backward(seg1, seg2, ...)` before every backward (the next eight
backwards, in order).  Both are optional; eager execution falls back to
demand-driven prefetch.

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
records it during the first step instead).

The unit of reloading follows the pipeline's own unit of work.  With
`prefetch_unit=backward` (default) a segment execution (the F/B of a virtual
stage for one micro-batch) is reloaded as a whole, groups needed first going
first, and the `prefetch_backwards` announced executions after it are reloaded
ahead while it computes: `1` hides one execution's copy time behind one
execution's compute, a GPU whose compute is fast relative to the host link
uses `2` or `3`.  `prefetch_bytes` bounds the device memory these ahead-of-demand
reloads may occupy (reloaded but not yet consumed bytes; the demanded execution
always loads).  With `prefetch_unit=group` (and without an executor) groups
inside the running execution are reloaded `prefetch_groups` ahead of demand
instead.  Both reload the demanded tensors on demand when no prefetch covered
them, which the profile reports as stalls.

**Hooks** (`hooks.py`).  Pack skips non-CUDA, non-strided and small tensors
(`min_bytes`; every saved tensor is counted in a size histogram so the
threshold can be chosen from data), keeps a lightweight reference for
parameters/buffers (ZeRO-3 safe) and otherwise registers the view with the
group's transfer: a dictionary lookup, no CUDA call.  Unpack rebuilds the view
with `as_strided` on the reloaded bytes.

**Transport** (`transport.py`).  The unit of transport is a *transfer*: all
tensors one execution of one offload group saved (or a slice of them when
`commit_bytes` flushes a large group early so its copy overlaps the rest of
the group).  When the group ends the transfer is committed: views of the same
storage that overlap are merged into regions (a block output saved as a whole
and as slices is copied once; with `dedup_scope=context` a view the previous
group of the same execution already shipped is served from that copy), the
regions are laid out in one pinned slab and grouped into chunks of about
`copy_chunk_bytes`, one event recorded on the compute stream orders every copy
after the kernels that produced the views, one D2H copy per region follows on
the D2H stream and one event marks their completion.  Sources are
`record_stream`ed so the allocator cannot reuse them early and dropped so the
memory is released once the copy finished.  The reload is one device buffer
and one H2D copy per chunk, each with its own event: the first tensors of a
group can be consumed before the rest arrived and a chunk's device memory goes
back as soon as its views were consumed.  Reload buffers are allocated on the
H2D stream (`reload_on_copy_stream`), so a reload never queues behind the
compute already issued on the consumer stream; the allocator's pool for that
stream acts as the staging area.  This is what keeps the host-side cost at tens
of microseconds per saved tensor and lets the copy engines run long transfers;
Megatron's offloader bulk-copies per layer for the same reason.  Optional
policies: bounded in-flight windows
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
| `prefetch_unit` | `PREFETCH_UNIT` | `backward` | `backward`: reload whole segment executions (pipeline F/B units); `group`: groups inside the running execution |
| `prefetch_backwards` | `PREFETCH_BACKWARDS` | 1 | announced backward executions reloaded ahead of the running one (raise on GPUs that compute faster than the host link copies) |
| `prefetch_groups` | `PREFETCH_GROUPS` | 2 | with `prefetch_unit=group`: groups reloaded ahead of the demanded one |
| `prefetch_bytes` | `PREFETCH_MB` | 0 | device-memory budget for ahead-of-demand reloads (reloaded, not yet consumed) |
| `commit_bytes` | `COMMIT_MB` | 0 | flush a group's saved tensors once this many bytes accumulated instead of at its end |
| `copy_chunk_bytes` | `COPY_CHUNK_MB` | 0 | reload in chunks of about this size, one device buffer and event each |
| `reload_on_copy_stream` | `RELOAD_ON_COPY_STREAM` | on | allocate reload buffers on the H2D stream (off: on the consumer stream, waiting for its backlog) |
| `dedup_scope` | `DEDUP_SCOPE` | `group` | `context`: also share views the previous group of the same execution shipped (holds that group's device memory one group longer) |
| `max_inflight_copies` | `MAX_INFLIGHT_COPIES` | 0 | bound of transfers in flight per direction |
| `comm_exclusive` | `COMM_EXCLUSIVE` | `off` | `off` / `strict` / `soft` windows around all-to-all (copies requested inside wait for the end; `strict` also makes the window wait for copies in flight) |
| `numa_membind` | `NUMA_MEMBIND` | on | pinned pages on the GPU's NUMA node |
| `numa_affinity` | `NUMA_AFFINITY` | off | also pin the main thread to the GPU's CPUs |
| `pool_max_bytes` | `POOL_MAX_GB` | 0 | bound of pinned memory owned by the pool (waits instead of growing) |
| `pool_trim_steps` | `POOL_TRIM_STEPS` | 0 | release pinned slabs unused for this many steps |
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
   `prefetch_backwards`, or `prefetch_groups` with the group unit;
   `prefetch_bytes` caps the device memory this costs); many tensors carrying
   few bytes point
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

Same node, combined form (`--recompute_modules Attention,MoEExpert
--offload_modules Block`) with the per-execution reload, steps 6-20 mean:

| configuration | seq | mbs | step (s) | peak (GiB) | tokens/s | moved per step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| production | 8k | 1 | 4.86 | 67.3 | 53.2k | - |
| combined form, defaults | 8k | 1 | 8.25 | 50.6 | 31.3k | 102 GiB |
| combined form, copies disabled (`no_copy`) | 8k | 1 | 5.45 | 67.5 | 47.4k | - |
| combined form, `prefetch_backwards=2` | 8k | 1 | 8.19 | 53.7 | 31.6k | 102 GiB |
| combined form, `comm_exclusive=soft` | 8k | 1 | 8.39 | 50.6 | 30.9k | 102 GiB |
| combined form, `prefetch_backwards=2`, `min_bytes` 8 MiB, `min_backward_distance=2` | 8k | 1 | 7.70 | 53.5 | 33.5k | 99 GiB |
| production | 8k | 2 | 8.62 | 99.6 | 60.2k | - |
| combined form, best knobs | 8k | 2 | 14.45 | 72.0 | 36.0k | 198 GiB |
| production | 16k | 1 | 8.73 | 99.9 | 58.7k | - |
| combined form, best knobs | 16k | 1 | 14.04 | 72.3 | 36.5k | 198 GiB |

Mechanically everything behaves as designed: no reload stalls (0.001 s/step),
copies at the link's measured limits (37 GB/s D2H, 50 GB/s H2D), the peak
stays flat from micro-batch 1 to 2 (50.6 -> 72.0 GiB against 67.3 -> 99.6 for
production), loss differences against production are the same size as
production's own run-to-run differences (DeepEP is not deterministic).
Economically it does not pay on this node: the `no_copy` run puts the
framework overhead at 0.6 s/step (12%, mostly the saved-tensor hook calls), the
other 2.8 s/step are the bytes contending with DeepEP in the I/O hub, and the
memory freed buys nothing because production already fits at micro-batch 2 and
the peak is dominated by weights and optimizer state (102 GiB moved per step
lower the peak by 17 GiB).  Offloading activations would start to matter where
production runs out of memory (sequences of 32k and above on this node), and
the remaining lever within the feature is to move fewer bytes
(`min_backward_distance`) or to place copies where the hub is idle.

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
| combined form, `prefetch_unit=backward` (default), `prefetch_backwards=1`, no stalls | 1 | 10.5 | 14.0 | 6.1k | 27 GB |
| same, `prefetch_backwards=2` | 1 | 11.2 | 14.8 | 5.7k | 27 GB |
| same, `prefetch_unit=group` (74 stalls, 0.1 ms) | 1 | 10.6 | 12.5 | 6.0k | 27 GB |
| same, `commit_bytes` 8 MiB (splits 27 MiB groups: intra-group sharing lost, 30 GB moved) | 1 | 10.6 | 14.2 | 6.0k | 30 GB |
| same, `dedup_scope=context` (nothing to share in this model) | 1 | 10.4 | 14.0 | 6.1k | 27 GB |
| same, `reload_on_copy_stream=off` | 1 | 10.8 | 14.0 | 5.9k | 27 GB |
| same, `copy_chunk_bytes` 16 MiB | 1 | 11.4 | 14.0 | 5.6k | 27 GB |

Loss trajectories are identical.  Step times of these short runs vary by
about one second from step to step, so the window, distance and transport
variants are within noise here; a longer run is needed to rank them.  What the
counters do show: reloading per segment execution (`prefetch_unit=backward`)
removes every stall that the group-level lookahead still had, at the price of
about 1.5 GB more resident on this model (the running and the next execution's
saved tensors); `prefetch_bytes` is the brake if that matters, and
`prefetch_unit=group` the low-memory alternative.  `commit_bytes` is meant for
groups of gigabytes, not these 27 MiB groups, where splitting only loses the
intra-group sharing (3.1 GB per step here).  The combined form keeps less resident than
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
