# Concurrent module streams

`ComputeConfig.module_streams` schedules independent, pure module branches on
named CUDA streams inside a compiled segment. It preserves pipeline order and
checks data dependencies, mutations and checkpoint boundaries before emission.
PyTorch autograd runs each backward operator on its forward stream.

```python
from nnscaler import ComputeConfig, ModuleStreamRule

config = ComputeConfig(
    plan_ngpus=2,
    runtime_ngpus=2,
    module_streams=(ModuleStreamRule(
        module='*.shared', stream='shared_expert',
        launch_after='*.dispatch', split_before='down_proj',
        resume_after='*.combine', overlap_backward=True,
    ),),
)
```

`module` matches qualified module paths. Launch anchors are signature globs
inside the enclosing invocation; repeated calls have independent boundaries.
`split_before` names a child relative to the selected module. Omit the split and
resume fields to move a whole branch. `launch_after`/`resume_after` capture input
readiness before their anchors and submit the branch afterwards. Without a
launch anchor, the branch stays in its original position.

For communication operators, register a submission/binding implementation:

```python
from nnscaler import AsyncResult, register_async_op

register_async_op(dispatch, submit=enqueue_dispatch, complete=bind_dispatch)
```

`enqueue_dispatch(*args, **kwargs)` returns `AsyncResult(prepared, wait)`. It
submits communication and protects storage used on its private streams. `wait()`
joins that work to the consuming stream. `bind_dispatch(prepared, *args, **kwargs)`
then attaches the operator's autograd behavior without submitting it again.
It must use the supplied inputs, which can include backward-ordering identities.
Differentiable preparation, such as selecting routing probabilities, may run
during submission; the transport's controlled input edge is bound at completion.
Such preparation can change autograd's ready-queue order. Its binding can use
`runtime.backward.prioritize_backward` with the original input boundaries to
keep the caller's backward submission ahead of the side branch.

By default NNScaler also guards submitted inputs and payloads until the join.
A backend that already records **all** input/output storage on its private
streams and the caller, and retains opaque owners, may return
`AsyncResult(..., manages_lifetime=True)` to avoid redundant traversal. This is
an explicit backend ownership contract; other operators keep the default guard.

For registered async operators, after-launch rules emit submission first, then
the independent module, then completion. The compiler keeps inputs alive across
these phases and validates that the module does not consume unfinished outputs.
An exception in the module still joins submitted work before releasing storage.

`operators` optionally selects signature globs inside a module invocation.
Exclusive local layout adapters and builtin metadata reads follow that branch;
other producer computations do not move implicitly. Operator selection cannot
split a module or select a recomputation region.

Rules can also be supplied as JSON through `NNSCALER_MODULE_STREAMS`. Resolved
rules and their compiler ABI are serialized in the compute configuration and
invalidate generated-code reuse when changed. Regenerate plans made with older
experimental launch modes. ZeRO 0/1 are supported.

## Backward and checkpoint readiness

`overlap_backward=True` supports before-launch rules and after-launch rules with
registered async operators. It adds
control edges so an asynchronous anchor submits its backward work before the
independent branch runs. The backend calls `capture_async_backward()` during
forward, then calls the captured object's `record_ready()` immediately before
submitting backward communication and installing its result wait.

`backward_on` can select a later anchor independently of forward placement.
The compiler retains the required graph edges and prioritizes the selected
backward branch without moving its input or parameter accumulators. The anchor
must remain on the caller stream and follow all selected forward chunks.

```python
ModuleStreamRule('*.mlp', stream='experts', recompute_only=True)
```

A recompute-only rule keeps the original forward placement and selects the
stream for a complete non-reentrant checkpoint replay. Each backward consumer
waits for the reconstructed tensors and protects their storage lifetime.
`recompute_ready_on` additionally gates replay on a unique downstream backend's
input readiness. The compiler rejects ambiguous consumers and partial regions.
It searches the nearest enclosing invocation, so a selected compute child may
use a backend outside its checkpoint wrapper; the sole-gradient-consumer check
still applies.

## Runtime branch API

Custom backends can manage their own phases through `ModuleStreams`:

```python
from nnscaler import ModuleStreams

with ModuleStreams() as streams:
    with streams.fork('branch', 'compute', inputs):
        result = compute(*inputs)
    streams.publish((result,))
    streams.join('branch')
```

Forks wait for their input producers; joins use the individual region's event.
`publish` protects output storage and opaque owners through their consumers.
`stream(name)` returns a shared named stream on the current device.
`handoff(name, tensors)` orders that stream after the current producer and
places tensor backward edges there without adding GPU arithmetic.
`after_backward(value, dependencies, point=None)` adds control edges carrying
no tensor gradients, avoiding a premature wait on an asynchronous consumer.

Declare an operator that manages internal streams with
`ComputeConfig(..., multi_stream_ops=('my_backend.operator',))` or
`NNSCALER_MULTI_STREAM_OPS`. This enables scheduled lifetime protection on all
pipeline ranks and codegen workers. The operator still owns its CUDA ordering
and must join its forward work before the next microbatch uses the caller.

Explicit module/checkpoint scopes manage their own cross-stream lifetimes and
do not enable whole-plan `track_stream` processing. Unmanaged internal streams
and explicit scheduler streams keep the conservative executor protection.

## CUDA replay within a module stream

Set `cuda_graph=True` on a pure module-stream rule to reduce Python and kernel
submission overhead. It can also be used on both pieces of a split module. The
first optimizer step runs eagerly and measures the number of live forwards for
each region. `build_optimizer` captures the regions at the next `zero_grad`,
after previous backwards finish, and synchronizes pipeline ranks before replay.
For a custom optimizer loop, call
`nnscaler.runtime.module_graph.prepare_module_graphs(model)` at that boundary.

Adjacent graph regions on the same stream are coalesced when their input
readiness and backward anchors match. This avoids separate forks and graph
launches between independent sibling modules without moving their execution.

Each live forward owns a native PyTorch graph pair and its captured state, so
FIFO and other backward orders do not overwrite another forward's activations.
Returned outputs and gradients have independent storage. `module_graph_stats(model)` reports
calibration capacity and actual replay counts without GPU synchronization.

Regions must have fixed, contiguous CUDA tensor boundaries and stable parameter
storage. Checkpointed regions, hooks, internal streams and cooperative dW cannot
be captured together. All training paths must be exercised during calibration.
Replay supports one first-order backward per forward; inference stays eager.
Backend caches must register a `register_capture_context` factory when a cache
hit would omit work required after parameter updates, such as weight quantization.

`cuda_graph='forward'` captures only forward execution. Backward uses the retained
slot-local autograd tape, allocating temporary gradients normally instead of
reserving backward buffers for every live forward. This saves memory without
adding recomputation. Selected backends must support retained, stateless backward
execution; parameter versions are still checked at the public boundary.
For after-launch regions whose inputs are available before the anchor, the
compiler stages input copies, output allocation and stream bookkeeping as soon
as those inputs are ready, without crossing mutations or checkpoint boundaries.
Graph replay and its autograd edges are created after submission. Computation
still waits for the anchor's input readiness, even when the graph's input copy
finishes earlier, so it does not occupy SMs before communication can start.

## Cooperative weight gradients

Cooperative operators can also be selected by the compiler:

```python
ModuleStreamRule(
    '*.expert.compute', operators='*.expert_ffn',
    stream='expert_dw', weight_backward_only=True,
)
```

This keeps forward placement and checkpointing intact, wraps matching operators
in a weight-gradient scope, and establishes the parameter gradient stream before
reducer hooks are built. Thus parameter publication does not insert a dW wait
onto the unrelated dX path. The selected backend must bind every trainable
parameter through `prepare_weight_backward`; unsupported operators fail explicitly.
Recomputed calls use the original parameter values and do not publish gradients
from the discarded replay graph. Weight-only rules cannot also move forward work.

Add `backward_on='*.dispatch'` to delay dW submission until an earlier async
operator's backward has submitted communication. The compiler requires that
anchor to be an upstream dependency outside the selected checkpoint. Parameter
aliases carry autograd control edges; the backend's input-readiness event gates
the dW stream without waiting for communication results. Deferred callbacks and
their inputs belong to one backward task. This also preserves partial-gradient
requests and the original checkpoint boundary. The anchor must be a registered
async operator using `capture_async_backward()` and `record_ready()`.

`accumulate_weight_grads=True` allows a cooperative backend to add directly to
native parameter gradients, avoiding a temporary dW buffer and its final add.
CUDA parameters must have exclusive compiler-proven affinity to the dW stream.
Tensor hooks, reducer-owned parameters and partial `autograd.grad` requests
retain ordinary gradient publication. This option requires serialized backward
tasks and does not support custom AccumulateGrad node hooks. Callbacks always
add to the supplied buffer, which may contain earlier microbatch contributions.

`WeightGradientAccumulator(stream, parameters)` is a context manager scoped to
one forward invocation. A cooperative backend binds parameters using
`prepare_weight_backward(parameters)` before its autograd `Function.apply` and
uses the returned parameter aliases. In backward,
`binding.accumulate(callback, inputs)` supplies zero-initialized gradient buffers
on first use; the callback adds contributions and returns those same buffers.
Buffer dtypes follow each parameter's `grad_dtype`, or its ordinary dtype.

Ready dW contributions run on the named stream. Ordinary autograd/reducer edges
publish the accumulated result once, after all contributors have registered.
`binding.defer(callback, inputs)` supports a callback returning fresh gradients,
with an additional accumulation copy. Frozen parameters may appear in backend
calls but must not belong to the accumulator's trainable parameter set.
Retained backwards get fresh buffers; pruned parameter paths release theirs.
No task is carried into another microbatch or optimizer step. Higher-order
backward, CUDA graph capture and concurrent backwards of one region are unsupported.

`BackwardValue(device)` transfers intermediate tensors between paired backward
operators without making them visible activation gradients. Construct it on the
consumer's forward stream, call `put(tuple_of_tensors)` in the producer backward
and `pop()` on the original consumer stream. It records cross-stream readiness
and discards unconsumed values at GraphTask completion.

## DeepEP configuration

`nnscaler.runtime.deepep.get_config_override(ranks, operation)` validates and
returns optional `NNSCALER_DEEPEP_CONFIG` overrides. Backends apply the returned
config explicitly; NNScaler does not replace methods in the installed library.
