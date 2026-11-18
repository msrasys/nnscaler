# Concurrent module streams

`ComputeConfig.module_streams` schedules independent module branches on named
CUDA streams inside a compiled segment. Model `forward` code does not need to
change. PyTorch autograd executes each backward operator on its forward stream
and synchronizes tensor dependencies between streams.

```python
from nnscaler import ComputeConfig, ModuleStreamRule

config = ComputeConfig(
    plan_ngpus=8,
    runtime_ngpus=8,
    module_streams=(ModuleStreamRule(
        module='*.shared',
        stream='shared_expert',
        launch_before='*.dispatch',
        split_before='down_proj',
        resume_before='*.combine',
    ),),
)
```

The same configuration is available to existing launchers through a JSON
environment variable. Its value is resolved when `ComputeConfig` is constructed
and stored in the serialized configuration. Changing it invalidates generated
code reuse, while allowing reuse of the unmodified traced graph.

```bash
export NNSCALER_MODULE_STREAMS='[{"module":"*.shared","stream":"shared_expert","launch_before":"*.nnscaler_moe_dispatch","split_before":"down_proj","resume_before":"*.nnscaler_moe_combine"}]'
```

`module` matches qualified module paths (including all descendant operators).
`launch_before` matches an operator signature in the enclosing module invocation.
Repeated invocations of the same module, such as a universal transformer loop,
are scheduled separately. `split_before` names a child module relative to the
selected module; the suffix starts before `resume_before`. Omit both split fields
to launch the whole branch. Without a launch anchor, a branch stays in its
original position and can overlap subsequent independent modules.

## Local input/weight backward overlap

Forward splitting and backward readiness can be selected independently:

```python
ModuleStreamRule(
    module='*.shared', stream='shared_expert',
    launch_before='*.nnscaler_moe_dispatch', split_before='down_proj',
    resume_before='*.nnscaler_moe_combine', overlap_backward=True,
    backward_on='*.nnscaler_moe_combine',
)
```

`backward_on` keeps the eager forward placement, but gates the selected backward
subgraph at the named backend's input-ready event. It also raises that subgraph's
autograd ready-queue priority: an early forward prefix would otherwise remain
behind intervening expert work even when its gradient is ready. Input graph
boundaries and leaf accumulators keep their original priorities, and dependencies
still control readiness. No additional backward GraphTask or microbatch is queued.

The explicit backward anchor must follow all selected forward chunks, remain on
the caller stream outside checkpoints, and expose the existing readiness protocol.
It cannot also be a deferred/native forward launch point. The option requires
PyTorch's autograd node priority API, checked at runtime, and tensor/builtin
container graph boundaries. It changes sequence numbers used by autograd
profiling; use the backend NVTX ranges to attribute these schedules. Generated
code ABI 14 preserves the required input boundaries and output control references.

`weight_backward_on` delays a cooperative module's parameter-gradient kernels
until an upstream backend has submitted its backward communication:

```python
ModuleStreamRule(
    module='*.expert', stream='expert_wgrad',
    weight_backward_on='*.nnscaler_moe_dispatch',
)
```

This borrows the dInput/dWeight separation used by zero-bubble scheduling, but
does not change the PP schedule or mix microbatches. A region belongs to one
module invocation and one autograd GraphTask. Its dWeight finishes within that
backward; it is not queued for a later microbatch. The anchor must precede the
module, outside checkpoints, with a differentiable data path into the module.
The rule can be combined with a separate `recompute_only` rule for that module.

The backend wrapper calls `nnscaler.prepare_weight_backward(parameters)` before
its autograd Function. When a binding is returned, it passes
`binding.parameters` to the Function and retains the binding in its context.
Backward computes dInput, registers
`binding.defer(callback, inputs)`, and returns `None` for the bound weight
gradients. The callback returns those gradients in the original parameter
order. Tensor inputs retained by the callback should be detached references;
do not resize storage shared with other consumers. The caller lists all inputs
whose storage the callback reads so the runtime can record their CUDA stream.
Callbacks perform pure gradient computation; they do not issue collectives or
publish reducer contributions themselves. Register them from the module's
ordinary backward, not from a nested autograd GraphTask.

The asynchronous anchor uses `prepare_async_input` before Function.apply and
captures `capture_async_backward()` in forward. Its backward calls
`record_ready()` before communication, then `submitted()` after communication
submission and before inserting the result wait. Both waits must still be
inserted on exception paths. Without `submitted()`, parameter backward provides
a later launch fallback. The ordinary autograd parameter edges, gradient dtype,
accumulation hooks and reducer contribution counts are retained; no manual
reducer-buffer publication is required. Parameter edges are created immediately
before their anchor to avoid retaining an entire stage's weight gradients.

Generated-code ABI 15 keeps both the parameter-gradient view node and
compiler-proven exclusive parameter accumulators on the weight stream. Joining
that work on the caller stream at each parameter publication would unnecessarily
stall upstream input backward. Ordinary autograd completes the caller-stream
dependency at the end of the GraphTask instead. Mixed/native parameter consumers
retain their original accumulator affinity, and pruned parameter paths still
join outstanding work in GraphTask cleanup. No work is moved into another
microbatch.

Support is first-order, single-device, ZeRO 0/1, with non-reentrant checkpoints.
Concurrent backward calls on the same forward invocation are unsupported;
separate live forward invocations keep separate task state.
Missing backend binding/readiness fails explicitly. As of the initial
implementation, CPU compiler/autograd and MoE kernel-double tests pass; CUDA
correctness, actual memory use and PP3/EP2 performance still require validation.

## Explicit asynchronous launch points

For cooperative backends, `launch_phase='input_ready'` with
`launch_on`/`resume_on` builds the ordinary side graph at the actual backend
input boundary. The adapter calls `nnscaler.prepare_async_input(value)` after
preparing inputs and immediately before its `autograd.Function.apply`, and uses
the returned tensor as that input. This records input readiness on the caller,
launches side work waiting for that event, and binds backward control edges.
It avoids deferred tensor copies and nested VJPs while preserving ordinary
autograd behavior. Dependencies between native regions at the same launch point
are resolved in order on the same side stream. Missing backend preparation
raises an error.

A backend can use `nnscaler.native_async_launch_pending()` to submit its raw
asynchronous work first, under `torch.no_grad()`, then call `prepare_async_input`
and attach its autograd function to the already submitted result. Insert the
result wait afterwards, including when a callback fails. This keeps Python side
graph construction from delaying communication submission. DeepEP uses this
two-phase form; its communication algorithm and backward remain the same.

`launch_phase='prequeued'` builds the ordinary side graph before entering the
anchor and queues its CUDA work behind a host-releasable gate. The backend must
call `prepare_async_input` to bind backward dependencies, then
`notify_async_launch()` after submitting communication and before waiting for
its result. Preparing the input does not release the gate. This separates host
graph construction from the communication window while retaining real tensor
outputs and ordinary autograd edges. The input-ready event still governs data
dependencies, and scope exit joins all side work within the same microbatch.

Use `launch_on` (and `resume_on` for a split region). Dependent regions at the
same anchor must use the same stream and are built in dependency order. A
single anchor cannot mix `submitted`, `input_ready`, and `prequeued` regions.
Prequeued execution requires eager CUDA and the gate helper described below;
`kernel_graphs` is supported only with `input_ready`. On exceptions, scope
cleanup opens pending gates before joining their streams. Gate watchdog fallback
preserves progress but can lose the intended overlap, so measure end-to-end
performance as well as correctness before choosing this phase.

`kernel_graphs=True` additionally enables cooperative backend kernel-graph
helpers inside an `input_ready` region. A backend calls
`nnscaler.run_module_kernel_graph(function, tensor_inputs, **immutable_kwargs)`
from its own no-grad autograd forward. All changing tensor state, including
quantized weights, must be explicit inputs. The existing autograd function
continues to compute gradients. Replays copy current inputs and clone outputs,
so updates and multiple outstanding forwards cannot reuse stale values.

Cache misses execute eagerly and retain detached input snapshots. NNScaler's
Trainer calls `prepare_module_kernel_graphs()` after a complete optimizer step;
standalone loops must call it at that same safe boundary. Capture may synchronize
CUDA and must never run inside an in-flight pipeline/communication region.
Steady execution only replays graphs. The cache is bounded to four graphs and
512 MiB of logical input/output storage per device; excess shapes execute eagerly.
Backend scratch memory is additional and should be measured in GPU memory stats.
`module_kernel_graph_stats()` reports captures, replays and capacity fallbacks.
`clear_module_kernel_graphs()` releases the cache at a completed-step boundary.
Only pure dense tensor kernels are supported: no RNG, host reads, mutations or
observable aliases. Backend configuration outside the captured arguments must
remain fixed or the cache must be cleared. The llm-train MX linear adapter opts
in to this helper; enabling the rule does not automatically capture arbitrary
operators which have no such adapter.

```python
ModuleStreamRule(
    module='*.shared', stream='shared_expert',
    launch_on='*.nnscaler_moe_dispatch', split_before='down_proj',
    resume_on='*.nnscaler_moe_combine', launch_phase='input_ready',
    overlap_backward=True,
)
```

`overlap_backward=True` also works with `launch_before`/`resume_before`.
That path submits the ordinary autograd graph on the side stream, then adds
the backward control edge before calling the anchor. It avoids deferred output
copies and a nested gradient engine, and can overlap an anchor's host metadata
wait. Inputs keep their original gradient-accumulator stream placement. It
requires synchronous host submission; `host_concurrency` and after-call anchors
cannot supply the side outputs before the control edge is created.

`launch_on` and `resume_on` prepare a tensor region before an anchor, then
submit its GPU computation when the anchor calls `nnscaler.notify_async_launch()`.
The backend should signal immediately after submitting asynchronous device work,
before waiting for its result. A missing signal is an error. This avoids placing
the side computation after an anchor's blocking metadata copies.

```python
ModuleStreamRule(
    module='*.shared', stream='shared_expert',
    launch_on='*.nnscaler_moe_dispatch',
    split_before='down_proj', resume_on='*.nnscaler_moe_combine',
    overlap_backward=True,
)
```

With `overlap_backward=True`, the compiler also adds autograd control edges to
the anchor inputs. The side region's backward becomes ready after the anchor's
backward has submitted its work. Control edges return `None` gradients, so they
do not enqueue arithmetic or inherit the communication result's CUDA wait.
For a split shared expert, this pairs its suffix backward with the combine
operator's backward dispatch, and its prefix backward with the dispatch
operator's backward combine.

Backends can align this work with the actual GPU input readiness by saving
`nnscaler.capture_async_backward()` on their autograd context in forward.
In backward, call the returned point's `record_ready()` after preparing the
communication inputs and before enqueuing communication or its result wait.
The control node makes each side stream wait for that input event. This matters
when CPU submission runs many milliseconds ahead of GPU execution: merely
ordering host calls can let the side computation finish before communication
starts. Captured points must record readiness; a missing record raises an error.

The implementation separates three responsibilities:

- The compiler selects modules, validates tensor dependencies and inserts the
  forward launch points and backward control edges.
- The runtime prepares the autograd edges, restores grad/autocast state inside
  custom-function forward calls, and runs an isolated functional inner graph.
  Input and parameter gradients return through the ordinary outer autograd graph.
- Backend adapters expose launch timing only. They keep their communication
  algorithm, arguments and result waits. There are no MoE-specific names in the
  generic scheduling/runtime implementation.

Deferred regions currently require dense tensor inputs/attributes and fresh,
non-aliasing tensor outputs, ordinary returned autograd gradients, and no nested
checkpoint region. Manual parameter-gradient writes, inference mode, CUDA graph
capture and higher-order gradients are not supported by this opt-in path.
Output copies preserve the inner graph's saved values; a saved lifetime token
releases them when backward finishes or an unused graph dies. `retain_graph`
and original tensor hooks retain their normal first-order semantics.

The launch-point path is undergoing CUDA validation. CPU tests verify generated
code, outputs, input/parameter gradients, retained graphs and backward ordering;
these checks alone do not establish device concurrency or a performance gain.

For asynchronous operators which enqueue a wait on the caller stream, use
`launch_after`/`resume_after` instead of the corresponding `before` fields.
NNScaler records input readiness *before* the anchor, then submits the side
branch after the anchor's Python call returns. The side stream waits for the
saved event, avoiding the communication wait that the anchor just inserted on
the caller. Inputs must already be available before that anchor, or produced
earlier on the same side stream. This can prevent a short branch from finishing
before communication is even submitted, and changes the backward enqueue order
as PyTorch reverses the graph. Profile both launch placements for the workload.

Two additional options address host submission stalls:

- `host_concurrency=True` submits the selected branch on a host worker dedicated
  to its device/stream. The caller continues immediately and resolves the task
  at the first consumer. Grad mode, inference mode and CUDA autocast are copied
  to the worker. Selected operators must be thread-safe as well as pure; custom
  caches must respect stream dependencies. Python state mutation, arbitrary
  thread-local modes and RNG-bearing custom operators are outside this contract.
- `async_anchor_copies=True`, together with `launch_after`, converts synchronous
  CPU-to-CUDA metadata copies of at most 4096 bytes inside the anchor into pinned
  asynchronous transfers. Sources are snapshotted before returning, preserving
  synchronous-copy semantics even if the caller mutates the original CPU tensor.
  Differentiable CPU tensors and larger transfers retain their original behavior.
  This prevents small metadata copies from waiting for previously enqueued
  communication on the caller stream. No model-specific operator is patched.

Host submission and CPU metadata snapshots require eager execution of the
generated functions; they reject CUDA graph capture.

Metadata snapshots can also be enabled independently of module placement, using
`ComputeConfig.async_copy_ops` or the equivalent environment variable:

```bash
export NNSCALER_ASYNC_COPY_OPS='["*.nnscaler_moe_dispatch"]'
```

These operator-signature globs wrap only the selected calls in `AsyncTensorCopies`,
with the same size, dtype and lifetime rules as `async_anchor_copies`. This is
useful when the host can submit subsequent work while an asynchronous operator
finishes, even without an independent forward module. The setting is serialized
for generated-code reuse and does not change the traced graph.

A `recompute_only=True` rule leaves forward execution in place and runs an
existing non-reentrant checkpoint's recomputation on the named stream:

```bash
export NNSCALER_MODULE_STREAMS='[{"module":"*.expert","stream":"expert_recompute","recompute_only":true}]'
```

Optionally set `recompute_ready_on='*.async_backend'` to align recomputation
with the GPU input readiness of that downstream backend's backward. The backend
captures `nnscaler.capture_async_backward()` in forward and calls its
`record_ready()` immediately before backward communication. Recomputation then
waits for that input event, alongside the original forward-completion event;
it does not wait for communication completion. This prevents a host that runs
ahead of GPU execution from launching recomputation too early to overlap it.

The selected backend must be the checkpoint's sole direct differentiable
consumer, in the same enclosing module invocation. Splits, other consumers and
checkpointed anchors are rejected. The backend must expose the readiness
protocol; a missing capture or event raises an error. This rule composes with
shared-branch backward controls at the same backend, using one readiness event.
It changes only when the existing checkpoint recomputes, and does not create
additional recomputation or change the forward order.

Recomputation uses saved forward inputs, so it can start while an incoming
gradient is still being communicated. It waits for the original forward's
completion event; the original backward stream joins it before consuming saved
tensors. The rule must cover a complete checkpoint and cannot specify forward
launch/split options. It does not enable checkpointing itself. Current support
requires PyTorch's saved-tensor hook inspection API, eager execution, one CUDA
device per checkpoint, and pure selected operators. It composes the active
checkpoint hooks and preserves checkpoint early stopping and repeated backward.

Checkpoint-local kwargs dictionaries are supported: the compiler proves that
the dictionary is created from a literal inside the region, and follows direct
updates of that same object. External dictionary mutations, nested aliases and
tensor writes remain rejected. This does not relax forward branch reordering.

To move existing recomputation to an earlier communication window, use
`recompute_prefetch_on` with `recompute_prefetch_phase='forward'` or `'backward'`:

```python
ModuleStreamRule(
    '*.self_attn', stream='attention_recompute', recompute_only=True,
    recompute_prefetch_on='*.nnscaler_moe_dispatch',
    recompute_prefetch_phase='forward', recompute_prefetch_bytes=4 * 1024**3,
)
```

The compiler selects one later, dependent asynchronous operation in the same
enclosing module invocation. Forward prefetch uses the backend's input-ready
launch point; backward prefetch uses its captured backward submission point.
The checkpoint replays on the named stream and retains its saved tensors for
the original backward to consume. A successful prefetch replaces that
backward's lazy replay, so it does not add another recomputation. In particular,
this can supply independent computation during forward communication even when
the shared branch is too short. It does not change the checkpoint's outputs.

The byte limit is a shared per-device budget for saved activation storage,
excluding existing input and parameter storage. All prefetch rules must use the
same limit. A region that cannot reserve its estimated storage falls back to
ordinary lazy recomputation. Temporary replay workspace is additional. Forward
prefetch extends activation residency until backward; increasing the budget can
therefore increase peak memory. Replay work is joined before leaving its pipeline
segment, and never borrowed from another microbatch. The first backward consumer
waits for replay completion, with allocator tracking on consuming streams.

Prefetch requires a complete, contiguous, pure checkpoint and PyTorch's installed
non-reentrant checkpoint hook contract. It preserves saved RNG/autocast state and
supports repeated first-order backward calls. Nested checkpoints, concurrent
backward of the same forward, higher-order differentiation, GraphExecGroup and
CUDA graph capture are unsupported. It cannot be combined with
`recompute_ready_on` or forward movement of the checkpoint/anchor. Its benefit
depends on available communication time, memory budget and resource contention;
measure full training steps with and without the rule.

The compiler checks tensor and object dependencies before moving computation.
Local layout adapters inserted inside the selected branch move with it, even
though those synthesized nodes do not carry the original module path.
It preserves complete recompute regions and rejects splits through their
boundaries, ambiguous anchors and known mutations/random operations. Rules are
local to each pipeline segment and never move operations between stages. ZeRO
0/1 is supported. Selecting a module is an assertion that it and its custom
operators are pure: hidden mutable state, external side effects, RNG use inside
custom operators, and unrepresented alias mutations are not safe to reorder.

Each fork waits on the caller's submitted work. The caller waits on a region's
completion event only at the first consumer, and drains remaining events when
leaving the segment. Storage is recorded on consuming streams; opaque owners
are retained through completion. Stream dependency waits do not synchronize the
host; host tasks are resolved when their outputs are needed.
Streams are cached per device and name; completion events are per invocation,
so pipeline microbatches cannot overwrite another invocation's dependency.
Parameters used exclusively on one named stream have their reducer gradient
accumulators created on that stream. Parameters also used by other streams keep
their ordinary placement. Reducers order cross-stream writes, including custom
operators which accumulate gradients manually and return `None`, and wait for
all contributing streams before launching a bucket collective. Opaque owners
are retained through backward GPU completion via output gradient hooks.

For MoE, launching shared-expert up/gate projections before dispatch and its
down projection before combine can cover communication with independent GEMMs.
Launching before dispatch also makes computation available while a dynamic
DeepEP dispatch obtains its receive counts on the host. This follows the shared
expert schedule in [Kimi K3, Figure 11 and Section 5.2.1](https://github.com/MoonshotAI/Kimi-K3/blob/main/k3_tech_report.pdf).
Kimi's MoonEP additionally uses balanced static shapes; module scheduling alone
does not remove DeepEP's receive-count host synchronization.

For a separately gated shared expert, move the gate with the shared branch.
In llm-train, put a `*.shared_gate` rule before the `*.shared` rule, use the same
stream and first launch anchor, and enable backward overlap on both. If setting
`backward_on`, give both rules that same boundary. This preserves the reference
gradient merge order at their common input for this MoE topology. Moving only
the FFN changes that order; rounding differences at a BF16 input can be amplified
by upstream FP8 quantization even when forward outputs match. Verify the input
and all parameter gradients when configuring a new branch topology.

Performance must be measured with both unprofiled training wall time and GPU
timelines. Sum interval unions per GPU instead of summing kernel durations across
streams. Report the baseline dispatch/combine fraction, communication with no
concurrent useful computation, and the achieved step-time reduction. Concurrent
kernels can contend for SMs and bandwidth; timeline overlap alone does not
establish that all communication latency has been hidden.

`examples/module_streams/profile_train.py` wraps an existing Trainer entrypoint,
records per-rank step metrics and optionally exports one compressed CUDA trace.
It sets the local stopping bound after constructing the original LR scheduler.
Run the same inputs/configuration with and without scheduling, and use separate
observer output directories. For example, from the training entrypoint directory:

```bash
torchrun --nproc_per_node=8 /path/to/nnscaler/examples/module_streams/profile_train.py \
  --steps 24 --profile-step 8 --output /path/to/results/profile \
  nnscaler_train.py [original training arguments ...]
python -m nnscaler.profiler.stream_overlap /path/to/results/profile/*.trace.json.gz \
  --output /path/to/results/communication.json
```

Repeat with `--profile-step 0` for unprofiled timing. The analyzer uses annotated
step spans as the denominator, excludes NCCL/NVSHMEM kernels from useful compute,
and reports both total and uncovered communication as interval unions. Inspect
`communication_kernel_names` to verify that the regex covers the installed
DeepEP implementation; `--communication-regex` can override it. Profiled timing
includes instrumentation overhead and should not be used as the final speedup.

The report also separates `dispatch`, `combine`, and `layout`, and attributes
communication to forward/backward CPU launch scopes via trace correlation IDs.
GPU-only traces keep an `unknown` phase. `--compute-regex` additionally reports
overlap with a chosen subset of compute kernels, for example shared-expert GEMMs.
Per-operation or phase unions must not be added when their intervals overlap.

Optional DeepEP default tuning is available through `NNSCALER_DEEPEP_CONFIG`,
applied by `nnscaler.init()` before buffers are initialized. For example:

```bash
export NNSCALER_DEEPEP_CONFIG='{"2":{"dispatch":[96,128,512,6,128],"combine":[96,64,512,6,128]}}'
```

Each five-integer configuration is `(SMs, NVL send chunk, NVL receive capacity,
RDMA send chunk, RDMA receive capacity)`, matching `deep_ep.Config`. Overrides
currently support intranode EP2/4/8 and only change library defaults; explicit
application configs retain precedence. Dispatch/combine SM counts and NVL
capacities must match. These numbers must be tuned for the actual shape and
hardware: optimizing isolated communication may leave too few SMs for overlap.
Report tuning-only and tuning-plus-overlap controls separately.

## Staging a checkpoint replay

`recompute_prefetch_prefix` splits **replay**, leaving the original forward
checkpoint whole. This is useful when one independent communication boundary
has forward and backward work to hide:

```python
ModuleStreamRule(
    "*.attention",
    stream="attention_replay",
    recompute_only=True,
    recompute_prefetch_on="*.dispatch",
    recompute_prefetch_prefix=("*.project_qkv", "*.project_query"),
    recompute_prefetch_bytes=8 * 1024**3,
)
```

The first matching top-level operator assignment ends the prefix. NNScaler
makes parameters explicit inputs to generated replay functions. It pre-enqueues
that prefix on the named stream and releases it after the asynchronous backend
submits its forward work. During backward, it rebuilds the remaining graph
after the same backend records **input readiness**, before consuming the
incoming gradient. A backend must implement both `notify_async_launch()` and
`capture_async_backward()` / `record_ready()`. The compiler preserves normal
forward dependency joins and waits for the prefix before leaving the segment;
no work is borrowed from another microbatch.

A cold prefix runs during backward to learn its retained storage footprint.
Subsequent invocations with the same prefix input metadata reserve from the
shared per-device budget before pre-enqueueing. The key includes only arguments
read by the prefix: changing tail-only metadata does not force another cold
replay. Storage identities account for aliases, and cache use still checks
original input versions. Budget exhaustion falls back to ordinary backward
replay. The budget limits retained prefix storage, not temporary kernel
workspace or the rest of the model. Prefix allocation must be shape-stable;
growth beyond its learned bound raises an error.

The original function runs under `no_grad`, as in reentrant checkpointing.
Both replay stages restore RNG and autocast state. Parameter gradient dtypes
are preserved, including BF16 parameters with FP32 gradients. This mode is
first-order and eager only. Nested checkpoints, concurrent backwards of one
forward invocation, CUDA graph capture, and data-dependent prefix allocation
are unsupported. The selected group must have explicit read-only attributes;
stateful module access and a cut inside an operator context manager are rejected.
Whole-replay and staged-replay rules cannot share one anchor, nor can a staged
rule use the same anchor as an `input_ready` or `prequeued` module callback. Submitted
module callbacks can share the anchor.

CUDA pre-enqueue uses driver stream memory operations and a native host
watchdog. The helper is built lazily into the PyTorch extension cache and
requires CUDA toolkit headers and a C++ compiler. A completed caller can release
the gate from the host; otherwise a stream-ordered write preserves readiness.
The watchdog can release a stalled gate without the GIL or a runnable CUDA
queue. A watchdog fallback preserves computation but may lose overlap and
should be counted during performance validation. Async allocator dependency
reuse is temporarily constrained while a gate is pending and restored at the
segment boundary. Abandoned forwards and failures release their gates and
retain storage through the GPU completion event.
