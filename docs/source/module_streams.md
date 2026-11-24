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
        launch_before='*.dispatch', split_before='down_proj',
        resume_before='*.combine', overlap_backward=True,
    ),),
)
```

`module` matches qualified module paths. Launch anchors are signature globs
inside the enclosing invocation; repeated calls have independent boundaries.
`split_before` names a child relative to the selected module. Omit the split and
resume fields to move a whole branch. `launch_after`/`resume_after` capture input
readiness before their anchors and submit the branch afterwards. Without a
launch anchor, the branch stays in its original position.

`operators` optionally selects signature globs inside a module invocation.
Exclusive local layout adapters and builtin metadata reads follow that branch;
other producer computations do not move implicitly. Operator selection cannot
split a module or select a recomputation region.

Rules can also be supplied as JSON through `NNSCALER_MODULE_STREAMS`. Resolved
rules and their compiler ABI are serialized in the compute configuration and
invalidate generated-code reuse when changed. Regenerate plans made with older
experimental launch modes. ZeRO 0/1 are supported.

## Backward and checkpoint readiness

`overlap_backward=True` applies to `launch_before`/`resume_before` rules. It adds
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

## Cooperative weight gradients

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
