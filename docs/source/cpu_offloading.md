# CPU activation offloading

Activation memory of a pipeline stage is dominated by the saved tensors of the
micro-batches that are in flight between their forward and their backward.
Recomputation trades this memory for compute (attention recompute roughly costs
one extra attention forward per layer).  CPU offloading instead copies the saved
tensors to pinned host memory right after the forward and brings them back just
before the backward, using copy engines that otherwise sit idle.

The feature has three layers.  Only the first one is specific to how a model is
planned; the other two are shared by every parallelization strategy.

## 1. Planning: `OpPlan.offload_id`

A policy marks consecutive forward operators with the same non-negative
`offload_id`.  All tensors that autograd saves while those operators run are
offloaded.  The granularity is therefore "whatever module boundary the policy
chooses": an `Attention` call, a whole transformer `Block`, an MoE expert group.
Constraints are the same as `recompute_id` (consecutive operators, one pipeline
stage, never both recompute and offload on the same operator).

`IRGraph.offload(nodes)` is the graph-level entry; `nnscaler.policies.fn`
applies the ids from `OpPlan`.  Adapters and multirefs inserted between marked
operators inherit the group of their neighbour so a group stays contiguous
after partitioning.

### Combining recompute and offload

An operator may carry both ids.  Code generation then wraps the checkpoint
call in the offload context: the recompute group saves only its inputs for
backward and exactly those are offloaded, so the group leaves nothing resident
on the device.  This is how "recompute what is cheap, offload the rest, keep
no activation resident" is expressed, e.g. in llm-train
`--recompute_modules Attention,MoEExpert --offload_modules Block`: attention
and expert work is recomputed from offloaded inputs, every other saved tensor
of the Block (residual inputs, norm inputs, router, shared experts, latent
projections) is offloaded directly.

## 2. Code generation

`ModuleCodeGen` wraps every offload group of a segment in

```python
with self.cpu_offloading_hooks(<group_id>):
    ...  # operators of the group
```

which installs `torch.autograd.graph.saved_tensors_hooks`.  When the graph is
scheduled (pipeline parallelism), `ScheduleCodeGen` also emits two hints into
`_train_step`, derived from the static schedule it is already unrolling:

* `hint_forward(segment, distance)` before every `fexecute`: the number of
  segment executions between this forward and its own backward.  A distance
  below `CPU_OFFLOADING_MIN_BACKWARD_DISTANCE` (default 1) disables offloading
  for that execution, e.g. the last pipeline stage in 1F1B, where F and B of a
  micro-batch are adjacent and offloading could only add a stall.
* `prefetch_next_backward(segment)` before every backward: the segment whose
  backward comes next, so its first group can be reloaded while the current
  backward computes.

Both hints are optional.  Without them (eager execution, non-scheduled graphs)
the runtime falls back to demand-driven prefetching.

## 3. Runtime: `nnscaler.runtime.cpu_offloading`

### Execution contexts give the backward order

`Executor.fexecute(name, ...)` opens an execution context for the segment; the
offload groups that run inside it are appended in forward order; finished
contexts are queued per segment name.  `Executor.backward(name, ...)` pops the
oldest queued context of that name.  This encodes two facts that hold for
1F1B, interleaved 1F1B and plain gradient accumulation:

* within one segment execution, groups are consumed in reverse forward order
  (last group first);
* across micro-batches, a given segment's backward consumes contexts in the
  order their forwards ran (FIFO).

So no profiling or warm-up iteration is needed to learn the order, unlike
Megatron's fine-grained offloading which records it during the first step.

### Pack

For each saved tensor: skip non-CUDA, non-strided, parameters/buffers (a
lightweight reference is kept instead, which also preserves ZeRO-3 semantics)
and tensors smaller than `CPU_OFFLOADING_MIN_BYTES` (1 MiB).  Otherwise the byte
range of its storage that the view touches is copied on the dedicated D2H
stream after `wait_stream(compute)`; the device storage is `record_stream`ed
on the D2H stream so the caching allocator cannot reuse it before the copy
has finished, and the Python reference is dropped so the memory is released as
soon as the forward no longer needs it.  Views of the same storage range inside
one group share one host region (q/k/v views of a fused projection, transposes
saved by several operators).  Strong references to the packed storages are
kept until the group ends so addresses cannot be recycled and mistaken for a
duplicate.

Host buffers come from a pinned pool with geometric size classes (ratio 1.25),
so tensors whose shapes vary every step (MoE expert inputs) still reuse
buffers.  `cudaHostAlloc` costs about 0.3 ms per MiB, so pooling is what makes
steady-state overhead small.

### Unpack and prefetch

Unpacking a tensor issues (or waits for) the H2D copy of its region on the
H2D stream and makes the compute stream wait on the copy event.  It also
prefetches the `CPU_OFFLOADING_PREFETCH_GROUPS` preceding groups of the same
context.  `begin_backward` and `prefetch_next_backward` prefetch the last
group(s) of the context that is about to run.  `end_backward` returns the host
buffers of the context to the pool.  Views are rebuilt with `as_strided`, so
non-contiguous saved tensors come back with their original strides.

### Diagnostics

`offload_stats()` returns per-step counters: bytes offloaded, deduplicated,
prefetched, number of unpacks that had to wait for a copy (`stalls`), time
spent in the hooks, pinned pool size, contexts still pending.  llm-train logs
them as `offload_*` training metrics.  `CPU_OFFLOADING_DEBUG=1` logs the
shape, dtype and size of every tensor saved by the first execution of each
offload group, which is the quickest way to see what a planning choice moves.

## Host memory locality

Pinned buffers are allocated on the NUMA node of the thread that first touches
them.  When GPUs of one node stream through pinned memory that lives on the
other socket, the copies share the inter-socket link.  On an 8x B200 node the
concurrent per-GPU D2H rate varied between 18 and 52 GB/s with default
placement and reached 35-57 GB/s when every process was bound to its GPU's
socket.  Launchers frequently confine all ranks to the CPUs of one socket, so
the manager does not rely on CPU affinity: by default it wraps every pinned
allocation in `set_mempolicy(MPOL_BIND, <gpu numa node>)`
(`CPU_OFFLOADING_NUMA_MEMBIND=0` disables), reading the node from
`/sys/bus/pci/devices/<gpu>/numa_node`.  `CPU_OFFLOADING_NUMA_AFFINITY=1`
additionally pins the main thread to the GPU's local CPUs, the measure
Megatron's `set_ideal_affinity_for_current_gpu` takes.  The decision is logged
once per rank as `[cpu_offload] GPU i: NUMA node n, pinned memory membind=on`.
`offload_d2h_s` / `offload_h2d_s` (summed copy durations) and `offload_stall_s`
(time the compute stream waited for a reload) tell whether copies are slow or
merely late.

## Bounded copy windows

CUDA multiplexes streams onto a small number of hardware work queues
(`CUDA_DEVICE_MAX_CONNECTIONS`, default 8).  If thousands of memcpys sit in
the copy streams' queues, kernel launches on the compute stream block behind
them on the host even though the streams are independent, and copies and
compute end up serialised.  An Nsight trace of whole-Block offload on 4x A6000
showed exactly that: 16 s of kernels in a 38 s window, 21 s with no kernel
running, and only 30% of the memcpy time overlapping with kernels, while the
compute stream waited on reload events for 0.4 s in total.

`CPU_OFFLOADING_MAX_INFLIGHT_COPIES` bounds the memcpys queued per direction;
further copies wait in a software queue and are issued from the next
pack/unpack/hint call (`pump`) once earlier copies completed, and a tensor
demanded before its reload was issued bypasses the window (`forced_h2d`).
On the A6000 reproduction a window of 8 did not improve step time (the host
cost turned out to be the per-copy CUDA API work and pinned allocations, not
queue blocking), so the window is off by default (0) and kept as a knob.  With
window 2 the reloads arrived late (`stall_s` doubled); if a window is used it
should stay well above the prefetch depth times the regions per group.

Per copy, the host spends roughly 80 µs issuing a D2H (two events, wait,
memcpy, record_stream) and 120 µs issuing an H2D (allocation on the consumer
stream plus the same).  With 720 regions per step (attention kernel offload on
the 15B model) that is ~0.1 s per step; with ~6000 regions per step (every
tensor of every Block) it reaches ~3 s and starves the GPU.  Coalescing the
regions of a group into one copy is the natural next step for such
configurations.

## Communication-exclusive windows

An Nsight comparison on 8x B200 (PP2 x EP4, attention-kernel offload) between
real copies and the fake-copy mode showed every compute kernel unchanged while
DeepEP `intranode::dispatch` went from 1.3 ms to 5.5 ms per call and
`combine` from 1.3 to 1.6 ms; summed over 288 dispatches per step that is the
whole 1.4 s/step gap.  The all-to-all runs over NVLink, the copies over PCIe,
but both pass through the GPU's I/O hub.  Kimi K2 reports the same effect
("offloading may slightly affect EP traffic") and GLM-5 schedules offload
around dispatch/combine.

`comm_exclusive()` marks such a region on the current stream.  On entry the
compute stream waits for every copy issued so far and a barrier event is
recorded; copies issued until the region ends are ordered after it, so copies
and the all-to-all never run at the same time and the copies drain in the
compute windows between them.  llm-train wraps DeepEP dispatch/combine and the
torch all-to-all fallback (`arch/offload_hooks.py`).  Copies can be split with
`CPU_OFFLOADING_COPY_CHUNK_MB` so the wait at region entry is short;
The windows are off by default; `CPU_OFFLOADING_COMM_EXCLUSIVE=1` enables the strict mechanism, `=2` is a soft
variant in which the window never waits for copies (copies merely avoid
starting inside it).  `comm_windows` in the stats counts the regions per step.

Measured on the 8x B200 node with attention-kernel offload (58 GB/step): the
strict window brings `intranode::dispatch` back to 1.3 ms, but the compute
stream then spends 5-6 s per 20 s waiting at window entries (576 windows per
step, one every ~16 ms, versus 5 ms per 200 MB copy), and the step gets slower
(7.11 s) than with copies overlapping the all-to-all (6.58 s) or than
attention recompute (4.96 s).  With an all-to-all in every layer there is not
enough copy-free compute time between windows for 2.75 s of copy-engine work
per step.  PCIe activation offload therefore does not pay off for this
MoE/EP configuration; it needs either far fewer bytes, or models without an
intranode all-to-all in every layer.

## Sizing against the link

Measured on an 8x B200 node (PCIe Gen5): D2H 38.6 GB/s, H2D 55.6 GB/s,
pinned allocation 0.38 s/GB.  Offloading every tensor saved by a 15B YOCO-MoE
`Attention` module at seq 8192 moved 121 GB per step on rank 0 (PP2 x EP4,
8 micro-batches), more than the link can carry within the 4.6 s step, and the
step slowed to 10.1 s.  Offloading only the attention kernel's saved tensors
(q, k, v, out, lse) and recomputing the projections/norms around it cuts the
volume by ~3.6x; the rule of thumb is that offloaded bytes per step divided by
the slower link direction must stay well below the step time, with the rest of
the slack covering the burstiness of 1F1B phases.

## Choosing what to offload

The copy time of a group must hide behind the compute that runs between its
forward and its backward.  Per transformer block and micro-batch, attention
saves roughly `4 * S * H * D` elements (q, k, v, o) plus the log-sum-exp; a
whole block additionally saves the MoE inputs, expert activations and
normalisation inputs.  With PCIe Gen5 (~50 GB/s per direction) and a 1F1B
window of several micro-batches there is ample slack for attention tensors on
large models, while offloading whole blocks of an MoE model can saturate the
link in the cooldown phase (every step is backward-only there).  Mixing is the
intended usage: recompute the cheap, large-footprint pieces (RMSNorm, SwiGLU,
MoE activation), offload the expensive ones (attention, expert FC1 inputs).

## Measurements and when to use the feature

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

Loss trajectories are identical across configurations.  Framework overhead is
about 0.1 s/step, copies run at 36-54 GB/s and no reload ever stalls the
compute stream; the remaining gap is the interaction between PCIe copies and
the expert-parallel all-to-all described above.  On this hardware and model,
attention offload cannot beat attention recompute, whose whole cost is 0.36
s/step at 8k.

Offloading pays off when the recomputed work is expensive relative to the
bytes it saves *and* the GPUs do not run an NVLink all-to-all in every layer
at the same time: long sequences on dense or context-parallel models, models
whose attention is a large fraction of the step, or expert parallelism that
does not share the GPUs with the copies.  Before committing to it on a new
configuration, run `CPU_OFFLOADING_FAKE_COPY=1` to measure the ceiling and the
real run to measure the loss; `offload_stall_s`, `offload_d2h_s` and
`offload_h2d_s` tell whether the gap is scheduling (fixable with prefetch
depth) or transport (not).
