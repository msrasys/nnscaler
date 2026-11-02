# Activation storage architecture

## Contract

An activation store owns a saved, immutable value; a handle names a view of that
value. The caller transfers its *retention responsibility* by keeping the handle
instead of the tensor. Existing tensor aliases and autograd saves remain valid
owners: the store never edits their data pointers or promises to free them.
CUDA work already reading an allocation must complete before allocator reuse.

`save → prefetch → load → evict → load → release` is the explicit data lifecycle.
Prefetch is speculative and bounded; load waits only on its own readiness event.
Eviction drops the store's restored GPU copy while retaining the immutable host
copy. Release closes that handle, not other views or tensors already returned to
consumers. A released handle rejects further access. Budgets include reusable
and in-flight pinned buffers; exhausted host capacity keeps the value resident.
Supported CUDA handles are strided tensors. Resident strided values still join
producer readiness and record consumer streams, including a load before batch
commit or after host-budget refusal. Unsupported layouts are only passed through
by the autograd adapter with normal PyTorch ownership/stream semantics.

For a later forward consumer, the handle preserves the autograd edge separately
from the payload. A scalar-backed gradient anchor carries the original logical
shape, and a restoration Function forwards gradients through that edge. Neither
Function saves the original payload. Leaf gradients and upstream backward saves
can still retain storage; this is normal autograd ownership, not an eviction bug.
Saved-tensor hooks use the same storage path without creating gradient anchors.

## Responsibilities

* **Configuration and admission:** immutable settings, explicit eligibility and
  residency reasons; no model names, tensor-shape exceptions or inferred writes.
* **Host memory:** process-wide budget, rounded allocator accounting, reusable
  buffers, pending completion events and host/cgroup headroom.
* **Transport:** device streams, producer/consumer dependencies, transfer batches
  and backpressure. It has no module, checkpoint or pipeline knowledge.
* **Submission and restore queues:** `activation_batch.py` separates source retention
  until forward submission from weak-reference backward restore order. Physical
  batch targets and DMA chunk sizes are independent of logical F/B scheduling.
  Chunking does not split a storage's ownership or reduce its full allocation.
* **Store and handles:** shared-storage identity, view metadata, version checks,
  saved-value ownership, gradient edges and explicit restore/evict/release.
* **Adapters:** saved-tensor contexts and pipeline scheduling select regions and
  issue prefetch hints. Existing graph annotations remain the user-facing NNScaler
  integration; application recipes do not implement transfer mechanisms.
* **Observation:** opt-in metadata events and counters; optional bounded CUDA-event
  pairs measure consumer dependency intervals without retaining tensors. Normal
  execution records no timing events. Export never synchronizes and marks pending
  samples explicitly. Independent probes
  fix the transferred bytes, tensor count and reuse schedule before comparing
  runtime configurations. Full-model throughput does not establish transport
  efficiency by itself.

## Validation boundaries

Transport batches share immutable completion fences within a common producer
readiness group; restore fences retain the original backward consumer groups.
An early ready value must not wait for a later producer or consumer group. A fence
is published only after its CUDA event has been recorded: a different autograd
thread must never wait on an unrecorded event. Pool reaping queries a shared
fence once per pass, and consecutive loads on the same consumer stream omit
redundant waits. Storage identities and version checks remain per saved value.

`backpressure_mode="host"` retains the legacy `max_inflight_bytes` admission
threshold. `"stream"` uses that threshold for CUDA execution pacing and requires
an independent, positive `max_pending_bytes` at least as large. CUDA waits retire
execution credits only; physically unfinished submitted source bytes remain
charged until their completion is observed. The host still waits at the pending
threshold to bound memory. Admission may overshoot by one indivisible batch.
Uncommitted forward saves, other GPU owners and allocator cache are separate
from these submitted-source counters; use the physical batch target and actual
GPU memory measurements as well. This is not an unlimited background queue.

The reference ideas were group completion fences and stream-side pacing in
Megatron Core's `fine_grained_activation_offload.py`. This implementation keeps
NNScaler's own storage/version and host-budget contracts; it does not resize
user storages or assume all views can be discarded. No Megatron code was copied.

Host submission and GPU execution are distinct. The calling thread prepares host
buffers and submits CUDA copy operations; there is no dedicated CPU copy worker.
The D2H and H2D streams run transfers asynchronously, and the queues track their
ownership, readiness and byte budgets. A logical batch shares completion fences
where dependencies permit; it can still contain several storage copies or DMA
chunks. It does not concatenate a stage into another GPU staging tensor. GPU-side
pacing avoids a host wait at the soft limit; the separate hard pending limit can
still block submission to bound unfinished source allocations.

The optional checkpoint-input submission adapter observes weak references to
the saved-value tokens created by that invocation. Shared storage can still be
owned by another region or explicit handle; those owners must not make a
checkpoint that needs no replay keep submitting unused inputs. The observation
changes only the next call's submission timing, never which values autograd
retains. A dynamic branch that starts needing replay remains correct through
ordinary region-exit submission.

CPU checks cover configuration, ownership, exceptional exits and gradient-edge
semantics. CUDA checks must additionally cover views, version violations, delayed
streams, multiple producer/consumer streams, repeated eviction, shared budgets
and non-leaking teardown. Numerical checks compare input and parameter gradients.
Independent transfer probes report CPU submission time, event/copy counts, GPU
copy intervals, requested/allocated memory and fallbacks; they do not label a
transfer hidden solely because it overlaps a communication kernel.

Opaque writes without version updates remain unsupported. Arbitrary external
Python caches need to adopt handles explicitly; this is not transparent GPU
virtual memory. A graph adapter may automate adoption only when it can establish
the corresponding data-use and ownership boundaries.
