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

The [API contract](cpu_activation_offload.md) defines mutation, ownership and
budget rules. [Action scheduling](activation_action_schedule.md) defines producer
readiness, transfer windows and completion-fence ordering.
