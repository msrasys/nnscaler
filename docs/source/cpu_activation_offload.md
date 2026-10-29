# CPU activation offloading

The October 30 store/adapter refactor has passed CPU and CUDA correctness checks;
see the [architecture validation](activation_store_validation_20261030.md).
Independent transfer probes are queued behind the current GPU job. Full-model
numbers linked below measure the October 29 implementation.

The [storage architecture](activation_store_design.md) separates host memory,
CUDA transport, saved-value ownership and graph/autograd adapters. An explicit
`ActivationStore` also supports values retained for a later forward consumer:

```python
from nnscaler.runtime.activation_store import ActivationStore
from nnscaler.runtime.activation_config import CPUOffloadConfig

store = ActivationStore(CPUOffloadConfig(max_cpu_bytes=8 * 1024**3), device='cuda')
with store.batch() as batch:
    saved = batch.save(activation)
del activation                         # drop this owner's original payload
# Other independent computation can run here.
saved.prefetch()                        # bounded, nonblocking request
activation = saved.load()               # joins readiness on the consumer stream
output = consume(activation)            # ordinary forward, with gradients preserved
del activation
saved.evict()                           # drop the store's restored GPU reference
# A later consumer may prefetch/load again from the immutable CPU copy.
saved.release()                         # last use of this handle; idempotent
output.sum().backward()
store.close()                           # only after all this store's saves are released
```

`load()` preserves ordinary autograd connectivity when `save()` ran with gradient
tracking enabled. `no_grad` saves are data-only. `release()` leaves already loaded
tensors and their gradient edges valid, and rejects future handle access. `evict()`
does not copy back to CPU: loaded values are immutable, so the existing host copy
remains authoritative. Aliased handles share one storage record. Re-saving a
restored view in the same store reuses that record; repeated eviction checks
versions of still-live older restored views as well.

An external tensor alias or an autograd save by a producer/consumer can still
retain the allocation. Leaf gradient accumulation also retains its leaf. For an
automatic NNScaler graph, use the region and lifecycle adapters below; arbitrary
Python caches must explicitly keep handles in place of tensors. This API is not
a transparent replacement for all tensors inside traced Python code. Custom
operators still need the immutable/version-counter contract below.

Plain PyTorch code can put autograd saves into the same store as explicit handles:

```python
with store.saved_tensors(exclude=(*model.parameters(), *model.buffers())) as saved:
    activation = model(inputs)
    handle = store.save(activation)
del activation
# The producer's own saved values can now use CPU handles as well.
saved.prefetch()  # optional, at a caller-selected point before backward
```

The saved-tensor adapter preserves parameter/buffer views through `exclude` and
shares an already active `store.batch()` when present. Its data tokens do not
create gradient anchors; explicit forward handles do. Both adapters share the
same hook cleanup path, including exceptions and disabled cyclic GC.

`saved.status()` reports logical residency and original-storage ownership;
`store.residency_reason(tensor)` explains static eligibility. `store.snapshot()`
reports managed records and the shared pool, without counting one allocation
twice through aliases. Its logical residency excludes pending CUDA frees and
allocator cache; use PyTorch allocated/reserved counters for physical peaks.
Closing one store does not close the process pool used by other stores.

Observation is opt-in and bounded, without patching runtime methods:

```python
with store.trace(max_events=10000) as trace:
    # save / prefetch / load / evict / release
    ...
metadata = trace.to_dict()  # scalar metadata; includes dropped_events
```

Run `benchmark/benchmark_activation_store.py --output /tmp/store-probe.json`
on an exclusively reserved GPU after initializing the CUDA environment. The
probe compares direct copies, store round trips and equal-work serial/overlapped
schedules. Tensor size/count, transfer windows and compute duration are explicit
arguments. Source creation, validation and the diagnostic trace are excluded
from measured samples. The probe caps payload and CPU memory and rejects an
existing output file. Compare the same transferred bytes; a residency fallback
must not be presented as faster transfer bandwidth.

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

A separately compiled module can override the activation store through
`ModuleParallelizeConfig(compute_config=OptionalComputeConfig(cpu_offload=...))`.
`None` inherits the parent configuration. This also supports a parent with no
offload regions: the child supplies its own configuration and policy annotations.
Modules with the same pool configuration share the process budget; enabling
another module does not allocate another allowance. Region selection stays in
each module's policy, and transfer scheduling stays in the common runtime.

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
enabled, outer backward and ownership. The refactor has since passed runtime,
PP/VPP and Vision GPU correctness checks.
The October 29 implementation passed full-model 64-step local VL comparisons; see the
[application benchmark report](https://github.com/msrallm/llm-train/blob/yileiyang/cpu_offloading/docs/activation-offload-results/20261029/FINAL_REPORT.md)
for the tested configurations and CPU/GPU measurements.

### Experimental pipeline output release

`ComputeConfig(..., release_pipeline_outputs=True)` replaces eligible non-scalar
backward roots with gradient-edge references after their final scheduled data
read. It does not change tensor data pointers, shapes or storage. Combining it
with `release_pipeline_inputs=True` removes both executor input ownership and
output-root ownership from eligible boundaries, allowing saved-activation
offload to reclaim storage. Autograd saves and external owners still retain
values they require.

The lifecycle pass treats sends, ordinary operations and hook metadata as data
uses. Returned values, scalar losses, leaf roots and roots needed by unknown
backward hooks stay as tensors. Switching to an incompatible hook after release
raises an error. Split backward is unsupported. References share weak identity
tokens to distinguish repeated references from recycled tensor ids, without
annotating caller tensors or changing serialization.

When an asynchronous P2P send has already registered a legacy backward edge,
the reference takes ownership of it and cancels the pending placeholder update.
The send handler retains its own tensor until communication completes. There
is then one owner of the backward edge and no need to mutate the sent tensor.
Normal stream bookkeeping also remains active for runtime tensor fallbacks.

This option defaults to false and requires regenerated code. Runtime tests,
eight offload/input/output combinations, Vision and real Attention/MoE PP/VPP
checks pass on GPU, including per-step executor cleanup. The local full-model
comparison found little peak-memory benefit from output references alone;
region selection and other live owners still matter. This does not promise
that all activation storage leaves the GPU or that all DMA is hidden.

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

`max_inflight_bytes` applies backpressure before a new transfer batch is submitted,
after forward computation has been enqueued and live saves have obtained host
buffers. Empty or budget-rejected batches do not wait. Older transfer events bring
queued source bytes at or below the threshold, then the whole batch is submitted
without host synchronization between copies. Computation can overlap that wait.
The queue can exceed the threshold by one group, and the current region also has
its normal compute/staging allocations. These workspaces may overlap the prior
batch; this threshold is not a bound on total GPU memory. Producer streams are captured during save
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
For a model tree, use `nnscaler.runtime.cpu_offloading.activation_offload_pool_stats(model)`.
It returns one snapshot per initialized shared pool, so multiple compiled modules
cannot cause duplicate accounting. It does not initialize offloading or retain
runtime resources in the result. A shared pool's counters include all its users,
including modules outside the queried subtree; these are not per-module totals.
`nnscaler.runtime.executor.observe_forward_outputs(callback)` provides a scoped,
read-only view of segment outputs for diagnostics, including outputs that never
enter `save_for_backward`. The callback receives `(segment_name, outputs)`;
its return value is ignored. Keep only metadata or weak references when measuring
lifetime. Nested observers compose, and registration is removed on context exit.
Moving or sleeping a module with live offloaded activations is rejected.
Generate fresh code after selecting regions or changing offload configuration.
