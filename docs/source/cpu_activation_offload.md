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

Region selection, saved-activation storage and pipeline input lifetime are
independent features. A policy declares eligible regions; `CPUOffloadConfig`
controls transfer, prefetch and host budgets. `ComputeConfig.release_pipeline_inputs`
controls executor ownership even when CPU offload is disabled. Eligibility does
not force a copy: storage still retained by the executor is skipped by default,
and normal layout, size and budget checks still apply.

## Transfer and lifetime

A selected region collects saved tensors and submits copies at its boundary on
an independent D2H stream. Copies retain the original bytes, dtype, strides and
storage offsets. Shared storage/views with the same version share one record,
including across regions and live forward scopes of a compiled module. Tensor
identity is not inferred from shape or allocator address alone.

Releasing a graph also releases offload contexts and scope/group metadata when
Python cyclic garbage collection is disabled. The forward scope owns its group
list; groups only weakly reference that scope and retain the manager and previous
groups needed for backward lookahead. Exiting a region drops the saved-hook
context manager to break its bound-method cycle. Autograd continues to own the
unpack callback while a saved value is live, including with `retain_graph=True`.
Output traversal is iterative so a self-referencing visitor cannot retain
autograd nodes, boundary inputs or their gradients.

### Experimental pipeline input release

`ComputeConfig(..., release_pipeline_inputs=True)` lets a generated pipeline
schedule replace selected input variables with weak identity references after
their final data read. The executor captures dInput in a small autograd boundary
with an empty anchor instead of retaining a detached input leaf. Saved-tensor
hooks can then release the original GPU allocation when no other owner needs
it. No tensor's data pointer, shape or storage is modified to force reclamation.
The gradient target owns its data and follows the leaf-gradient layout contract;
an expanded incoming cotangent is not handed back as a borrowed input gradient.

An outer eager graph uses the generic executor connector before forward:

```python
from nnscaler.runtime.executor import register_input_backward

register_input_backward(
    feature,
    context_factory=backward_context,  # optional factory, e.g. gradient sync
    on_data_release=drop_feature_references,
    on_complete=finish_microbatch,
)
```

With early release requested, the connector retains a non-leaf's gradient edge
and invokes `on_data_release` at the generated final-read marker. Leaf tensors
and older PyTorch without gradient edges stay alive until backward; their early
release callback is not invoked. `on_complete` runs after successful backward,
outside the supplied context. Autograd still owns any saved values it needs.
The lower-level `register_input_grad_callback(..., release_input=True)` remains
available for custom gradient handling; that callback must retain its own edge.
Weak registrations use identity objects so tensor-id reuse cannot redirect an
older callback to a newer tensor.

The old `CPUOffloadConfig.release_pipeline_inputs=True` setting is a deprecated
alias. Constructing or loading `ComputeConfig` migrates it to the top-level
field without mutating the caller's CPU config. An explicitly supplied top-level
boolean takes precedence. New serialized configs store the canonical value.

This option is disabled by default and requires regenerated code. It applies
to ordinary pipeline backward, and rejects split input/weight backward. The
analysis keeps local backward roots, returned tensors and attributes; adapter
and segment-hook reads still count as data uses. Unknown global backward hooks
keep tensor targets. The internal output-only loss-scaling hook supports the
new targets; changing to an incompatible hook after forward raises an error.

Bridged inputs must remain immutable, and custom operators must support a
non-leaf input view. Normal alias ownership and CUDA stream lifetimes remain
intact. Outside references, including a caller's tensor or a Vision callback,
can still keep the allocation alive. This does not promise that every pipeline
input will leave the GPU. The preceding implementation passed CUDA ownership,
PP/VPP and full-model checks. The October 29 API separation has CPU coverage for
configuration migration, code generation with offload/lifetime independently
enabled, outer backward and ownership. GPU and performance validation of this
refactor is pending; earlier measurements do not validate the new revision.

## Immutability contract

An activation saved by an offload context and every alias of its storage must
remain immutable until backward releases the saved values (including retained
graphs). This applies outside the selected region too. In-place work before
the first save, after all saves are released, or on unrelated tensors is allowed.
The runtime supports one immutable version per live saved storage; it does not
keep several snapshots to accommodate writes to that storage.

The runtime retains version counters without retaining the original activation
allocation. It checks them before repeated saves, D2H submission, prefetch and
unpack. Restored views share the reload buffer's version counter, so modifying
one cannot silently corrupt a later read through another view. Resident and
budget-fallback saves also check versions. Violations raise an error identifying
the observation point and expected/current versions. Checks introduce no CUDA
synchronization. They detect violations at these boundaries, rather than
intercepting writes or guaranteeing race-free execution of invalid programs.

Custom operators must use `save_for_backward` for saved activations and maintain
PyTorch's version counters for **all** writes. Ordinary PyTorch in-place
operations do this. A raw-pointer CUDA/Triton writer must use the appropriate
mutation registration, `ctx.mark_dirty` for a custom autograd Function, or
`torch.autograd.graph.increment_version(tensor)`. Reporting a mutation makes it
detectable; it does not authorize modifying a live saved activation.

Writes through `.data` that bypass version tracking, raw kernels that do not
report mutations, and aliases created with independent version counters (for
example manual `set_(storage)`) are unsupported. Equal numeric `_version`
values do not prove those aliases share a counter. Version checks cannot detect
arbitrary hidden writes; the caller must uphold this operator contract. Normal
views and `detach()` aliases share the original version counter and are covered.
No graph alias analysis, automatic mutation repair, or extra tensor cloning is
performed to support otherwise invalid mutations.

## Scheduling

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
Groups whose shared storages are already restored do not consume lookahead
depth, so repeated checkpoint boundaries cannot hide the next pending transfer.

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
power of two; the pool exposes that already-charged capacity in its reusable
tensor so slightly larger requests in the same size class can reuse it. Only
requested activation bytes are transferred. `allocated_cpu_bytes` reports Tensor capacity and
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
limits constrain the allocation headroom check. Clean inactive file-cache pages
are treated as reclaimable (conservatively subtracting all dirty/writeback file
pages); active file cache, anonymous and pinned memory remain charged. Missing
cache statistics fall back to raw cgroup usage. This estimates allocation
headroom without requesting cache eviction. The budget is not a bound on
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
