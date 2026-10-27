# CPU activation offloading

Select saved activations independently of the parallel layout:

```python
from nnscaler import ComputeConfig
from nnscaler.policies import apply_module_offload, OpPlan
from nnscaler.runtime.cpu_offloading import CPUOffloadConfig

config = ComputeConfig(2, 2, cpu_offload=CPUOffloadConfig(
    max_cpu_bytes=75 * 1024**3,   # per process, shared across participating modules
    min_tensor_bytes=1024**2,
    prefetch_groups=1,
    max_prefetch_bytes=512 * 1024**2,
))
# Before partition/replication:
apply_module_offload(graph, Attention)
# Or select an explicit contiguous region:
graph.offload(nodes)
# Or in a function policy:
yield OpPlan(node, stage_id=0, offload_id=0, recompute_id=0)
```

Offload and recompute may select the same region. The generated saved-tensor
context wraps the non-reentrant checkpoint call: checkpoint inputs go to CPU,
while recomputed intermediates are produced and consumed on GPU. Offload-only
regions save backward-required intermediates as well. Outputs are saved only
when a consumer needs them. Partial selection inside one recompute region is
rejected instead of silently splitting its recomputation semantics.

## Transfer and lifetime

A selected region collects saved tensors and submits copies at its boundary on
an independent D2H stream. Copies retain the original bytes, dtype, strides and
storage offsets. Shared storage/views with the same version share one record,
including across regions and live forward scopes of a compiled module. Tensor
identity is not inferred from shape or allocator address alone.

A generated schedule keys each forward scope by `(segment, microbatch)`. Before
the preceding local compute action, it issues a hint for the actual next
backward scope. This works with ordinary PP and interleaved VPP; “one action
before” means a local F/B action, not an optimizer step. Only the entry group is
prefetched, not the entire stage. Group output autograd prehooks prefetch prior
groups during backward. Ordinary non-pipeline autograd uses the same group
hooks. Demand loading remains a correctness fallback for branched graphs,
insufficient lookahead or an exhausted speculative budget.

If a forward is immediately followed by its own backward, its final selected
group stays resident. If the original storage is still held by a pipeline input,
output or YOCO cache at restore time, it is reused without H2D or a second GPU
allocation. This does **not** free external owners. The runtime never resizes
live user storage or overwrites pipeline inputs to force a memory reduction.

Executor-retained segment inputs are excluded per forward scope before any CPU
allocation (`skip_retained_inputs=True`). Saves discarded before group commit,
such as a checkpoint with no saved intermediates, also avoid allocation and DMA.
Only live records reserve pinned buffers, before the group's copies are submitted.

`min_offload_gap` optionally keeps an entire scope on GPU when fewer than this
many local F/B actions separate its forward and backward. The default 0 preserves
ordinary selection; 2 additionally keeps immediately-followed-backward scopes
resident. Scopes without schedule distance metadata retain ordinary offloading.
`scope_prefetch_groups` (default 1) controls how many backward-entry groups the
schedule preloads. `prefetch_groups` controls lookahead inside backward; setting
it to 0 still disables all speculative loading. Both use the same byte budget.

Reload buffers are allocated on the current compute stream by default, so they
can reuse its allocator arena. The H2D stream joins the already-enqueued compute
prefix before writing into the allocation, then signals readiness to backward.
`reload_on_compute_stream=False` retains the copy-stream allocation path for
controlled comparisons. This does not remove transfer dependencies or pin all
future groups on the GPU.

Restored GPU storage is retained by the shared saved record until the final
backward consumer releases it. Repeated saves/reads therefore do not cause
repeated H2D. With `retain_graph=True`, saved records remain alive until the
graph is released. CPU buffers cannot return to the reusable pool until their
last D2H/H2D event completes. Every GPU consumer records its CUDA stream, so
allocator reuse cannot race pending kernels.

## Budgets and exclusions

`max_cpu_bytes` bounds rounded pinned-block capacity owned by a **shared process
pool**, including idle reusable and in-flight buffers. Modules in one process must use the same
CPU cap. The process pool persists between module instances, so destroying a
module does not reset its allocation accounting. Allocation is lazy and uses
actual tensor byte sizes. ATen rounds the underlying pinned allocation to a
power of two; the budget charges that rounded capacity, while the Tensor keeps
its requested size. `allocated_cpu_bytes` reports Tensor capacity and
`allocated_pinned_bytes` reports the charged pinned blocks.
Reuse chooses the smallest available buffer large enough for a request. A
buffer stays charged while idle; the pool does not evict old size classes to
allocate new ones. Workloads with very variable shapes can therefore fall back
to resident storage before all idle bytes are useful.

An optional `pool_slab_bytes` enables event-safe suballocation from persistent
power-of-two pinned slabs. Leases have exact logical lengths and 256-byte aligned
slots; adjacent free slots coalesce after transfer events complete. The budget
charges whole slabs, including free space. All modules in a process must agree
on this setting. `live_pinned_bytes` then counts slabs with active leases once,
while `live_cpu_bytes` counts their logical lease lengths. Slabs are opt-in:
individual buffers can be more efficient when activation sizes already match
allocator size classes. Inspect real pool capacity and process memory before
choosing a slab size.

When the budget or host headroom is insufficient, that tensor stays on GPU;
there is no implicit recomputation. `/proc/meminfo` and visible cgroup v2 ancestor
limits constrain the allocation headroom check. The budget is not a bound on
the entire process, OS allocation overhead, or PyTorch's separate pinned allocator cache. Leave room for
model construction, optimizer state, dataloaders and other processes.

`max_inflight_bytes` applies backpressure before the next offload region starts,
waiting for older events until queued source bytes are at or below the threshold.
A region then submits its whole batch without host synchronization between copies.
The queue can exceed the threshold by one group, and the current region also has
its normal compute/staging allocations. Producer streams are captured during save
and joined once per stream at group commit. `max_prefetch_bytes` bounds speculative
GPU copies, not activations already demanded by backward. A storage larger than
that cap is loaded on demand. `prefetch_groups=0` disables speculative requests.

Parameters, registered buffers and views are excluded, preserving ZeRO-3
parameter-view reconstruction. `mark_activation_resident(tensor)` excludes its
storage and views, for example a cached quantized weight. Non-CUDA/unsupported
layouts, conjugate/negative views, inference tensors, small tensors and slices
whose backing storage exceeds eight times their logical bytes are kept resident.
Custom autograd code must use `save_for_backward`; tensors kept directly on a
custom context are outside saved-tensor hooks.

Use `module.activation_offload_stats()` to inspect D2H/H2D bytes, original-storage
reuse, budget fallback, prefetch activity and live/allocated CPU bytes. Counters
are cumulative and shared with other modules using the same process pool.
Moving or sleeping a module with live offloaded activations is rejected.
Generate fresh code after selecting regions or changing offload configuration.
