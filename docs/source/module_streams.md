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

## Explicit asynchronous launch points

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

Recomputation uses saved forward inputs, so it can start while an incoming
gradient is still being communicated. It waits for the original forward's
completion event; the original backward stream joins it before consuming saved
tensors. The rule must cover a complete checkpoint and cannot specify forward
launch/split options. It does not enable checkpointing itself. Current support
requires PyTorch's saved-tensor hook inspection API, eager execution, one CUDA
device per checkpoint, and pure selected operators. It composes the active
checkpoint hooks and preserves checkpoint early stopping and repeated backward.

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
