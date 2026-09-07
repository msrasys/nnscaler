# F/B/W Split Backward and Custom Implementations

This document describes nnScaler's F/B/W (Forward / Backward-input / Backward-weight) design, the limitations of the default implementation, and how to replace it without changing scheduler code generation. It also explains two runnable examples in detail: DeepSeek DualPipe-style explicit dW callbacks and d9d-style graph-preserving phased backward.

Related code:

- Default runtime: [nnscaler/runtime/executor.py](../../nnscaler/runtime/executor.py)
- Default graph-splitting algorithm: [nnscaler/runtime/_patch_torch_pipelining_backend.py](../../nnscaler/runtime/_patch_torch_pipelining_backend.py)
- DeepSeek example: [tests/cli/test_customized_fbw_deekseek.py](../../tests/cli/test_customized_fbw_deekseek.py)
- d9d example: [tests/cli/test_customized_fbw_d9d.py](../../tests/cli/test_customized_fbw_d9d.py)

## 1. Why Split Backward into B and W

A conventional autograd traversal computes two kinds of gradients together:

- **B (backward-input)** computes the stage input gradients, dX. A pipeline must send dX to the preceding stage as soon as possible, so dX is on the inter-stage critical path.
- **W (backward-weight)** computes the local parameter gradients, dW. dW does not cross a stage boundary and can usually be delayed to fill a pipeline bubble.

One microbatch therefore changes from `F -> backward` to `F -> B -> W`. The scheduler prioritizes B and places W into a later gap. This changes execution order, but it must not change final parameter gradients, reducer behavior, or optimizer state.

Enable the default FBW implementation through `ComputeConfig`:

```python
from nnscaler.parallel import ComputeConfig

compute_config = ComputeConfig(
    plan_ngpus=2,
    runtime_ngpus=2,
    use_end2end=True,
    use_fbw=True,
)
```

FBW is supported only for end-to-end training, not inference. The pipeline scheduler determines the actual order, and the matching W is not necessarily called immediately after a B.

## 2. The Scheduler/Executor Boundary

Generated schedules do not inline the backward algorithm. They call module-level runtime symbols:

```python
input_grads = nnscaler.runtime.executor.backward_input(
    name, input_tensors, output_tensors, output_tensor_grads, weights
)

# Other microbatch operations may run here.
nnscaler.runtime.executor.backward_weight(name, weights)
```

This defines the scheduling contract:

| Entry point | Responsibility |
| --- | --- |
| `backward_input` | Consume one microbatch's output gradients, return its stage input gradients as early as possible, and retain enough state to compute dW later |
| `backward_weight` | Consume the oldest pending B state for the same segment, then compute and accumulate dW |

`name` identifies a segment. Several microbatches for one segment can be waiting for W simultaneously, so state must be stored in a **per-segment FIFO**. Every B must pair with exactly one W. Even a B with no work must publish an empty state for its matching W to consume.

## 3. The Default FBW Flow

### 3.1 F: Isolate Each Stage's Autograd Graph

Before executing a segment, `Executor.fexecute`:

1. Waits for asynchronous communication associated with its inputs.
2. Applies `detach()` and then `requires_grad_()` to stage inputs that require gradients, giving each stage an independent autograd graph.
3. Stores `(original input object id, detached input)` pairs in the `Executor._detach[name]` FIFO and runs forward with the detached inputs.

B later retrieves the leaf inputs that were actually used by this forward. It cannot use the original cross-stage tensors supplied by the scheduler because they do not belong to the stage-local graph.

### 3.2 B: Compute dX and Capture the Starting Points for dW

`Executor.backward_input` performs these steps:

1. Calls `Executor.sync_tensors` for `output_tensor_grads` with pending asynchronous communication. The final stage may use `None` for a scalar loss, while an intermediate stage normally receives real gradient tensors.
2. Pops the detached inputs for this microbatch from `Executor._detach[name]`.
3. Deduplicates repeated `(output, output_grad)` pairs and applies the optional backward pre-hook.
4. Calls `stage_backward_input(outputs, output_grads, inputs, weights)`.
5. Returns dX immediately and pushes the resulting `param_groups` into the `Executor._weight_backward_states[name]` FIFO for W.

`stage_backward_input` constructs a reverse graph from the outputs, stage inputs, and parameter `AccumulateGrad` nodes. It locates the boundary between the input-gradient closure and parameter-gradient closures. Each parameter group records:

- `params`: the parameters' `AccumulateGrad` nodes;
- `intermediates`: graph nodes on the B/W boundary;
- `grads`: the per-output-edge gradients captured by pre-hooks while B crosses that boundary.

It then runs a partial backward whose targets are only the stage inputs. For native C++ autograd nodes, PyTorch can inspect the current GraphTask's requested edges and prune unrelated dW branches. B retains only the boundary state needed by W and disconnects the no-longer-needed stage-output side of the graph.

### 3.3 W: Re-enter at the Boundary and Accumulate dW

`Executor.backward_weight` pops the oldest state from `_weight_backward_states[name]`. The default `stage_backward_weight` then processes each parameter group:

1. It creates `GradientEdge(intermediate, output_index)` for every valid output edge. Different outputs of one node must not be merged into output 0.
2. It re-enters the retained weight closure at those edges with the gradients captured by B and computes dW.
3. It sends each computed dW through backward on the leaf parameter instead of assigning `weight.grad` directly.

The last step is essential for nnScaler. Reducers use post-hooks on parameter `AccumulateGrad` nodes to move gradients into contiguous buffers. Returning a value from `torch.autograd.grad` or assigning `weight.grad = dweight` does not fire those hooks.

### 3.4 Special Cases

- **No output tensors**: B publishes an empty state and W consumes it without work, preserving schedule pairing.
- **No grad-requiring stage input**: This normally describes the first stage. There is no cross-stage dX to return, so B saves outputs and output gradients and W runs the complete backward later.
- **A view output**: The default entry point makes a differentiable clone because the underlying helper detaches outputs in place and a view cannot use `detach_()`.
- **Asynchronous communication**: Output gradients need to be synchronized once at B entry. W state or callbacks originate from that synchronized traversal and should not synchronize the same gradient again.
- **Multiple microbatches**: Both `_detach` and `_weight_backward_states` are per-segment FIFOs. A single dictionary value would overwrite an older pending microbatch.

## 4. Problems Already Fixed by the Default Implementation

nnScaler's backend is based on PyTorch's pipeline FBW algorithm but includes runtime-specific corrections:

1. **Reducer hooks**: It first computes dW, then invokes backward on each leaf weight so that `AccumulateGrad` runs.
2. **Multi-output intermediates**: It creates a separate `GradientEdge(node, output_index)` for each valid output edge instead of merging all gradients into output 0.
3. **Multiple boundary nodes**: W clamps captured boundary gradients so that a descendant intermediate cannot feed an ancestor again and double-count a contribution.
4. **The first stage**: When there is no grad-requiring stage input, it delays the complete backward to W.
5. **View outputs**: It provides a differentiable clone before entering the helper that performs an in-place detach.

These are current behaviors. They are historical bugs that nnScaler has fixed, not limitations that remain in the default path.

## 5. Limitations of the Default FBW Implementation

### 5.1 Incomplete Partial-Backward Semantics for Python `autograd.Function`

In PyTorch 2.10, `ctx.needs_input_grad` in a Python `autograd.Function.backward` is initialized from each forward input's `requires_grad`. It is not refreshed to describe the targets of the current partial-backward GraphTask.

For example, an opaque linear may receive `(activation, weight)`, both requiring gradients. B requests only the activation gradient, but `ctx.needs_input_grad` can still be `(True, True)`. Native C++ nodes can use per-edge engine demand to prune dW. A Python custom Function has no equivalent public query and may still execute its expensive dW formula during B.

### 5.2 PyTorch 2.10 Compatibility Between Python PyNode and `GradientEdge`

Default W uses public `torch.autograd.grad` to re-enter the graph from saved `GradientEdge` objects. When a boundary is a Python custom Function's PyNode, the PyTorch 2.10 public wrapper accesses `_FunctionBase._input_metadata` and can fail with:

```text
RuntimeError: Attribute '_input_metadata' is invalid for this instance of _C._FunctionBase
```

The default path is therefore best suited to graphs made of native PyTorch autograd nodes. Parameterized opaque Python `autograd.Function` nodes require additional handling. This is an interface gap between Python custom Functions and per-edge partial backward, not a scheduler-ordering problem.

### 5.3 Graph Ownership for Python Custom Functions

Splitting B from W requires more than retaining a Python `grad_fn` object. The underlying C++ graph and saved tensors must remain alive until W finishes. After an output is detached in place, retaining only its Python PyNode may not retain the complete underlying graph. A custom graph-preserving implementation must hold an undetached output or an equivalent graph owner explicitly.

### 5.4 Dependence on PyTorch Internals

Boundary discovery, `GradientEdge`, Node pre-hooks, and engine behavior are tied closely to PyTorch autograd internals. After changing PyTorch versions, revalidate:

- whether B computes only dX;
- whether W computes and accumulates dW exactly once;
- whether per-edge behavior for Python custom Functions has changed;
- whether reducer hooks, asynchronous communication, and multi-output nodes remain correct.

## 6. Replacing the Default Implementation with `custom_fbw`

Generated code resolves `nnscaler.runtime.executor.backward_input` and `backward_weight` dynamically when it runs. The `custom_fbw` context manager can therefore replace both symbols temporarily without changing scheduler code generation:

```python
from typing import Iterable, Optional

import torch
import nnscaler.runtime.executor as executor


def my_backward_input(
    name: str,
    input_tensors: list[torch.Tensor],
    output_tensors: list[torch.Tensor],
    output_tensor_grads: list[Optional[torch.Tensor]],
    weights: Iterable[torch.nn.Parameter],
):
    # Compute and return dX as early as possible.
    # Push the matching dW state into a FIFO keyed by name.
    ...


def my_backward_weight(
    name: str,
    weights: Iterable[torch.nn.Parameter],
) -> None:
    # Pop the oldest state for name, then compute and accumulate dW once.
    ...


with executor.custom_fbw(my_backward_input, my_backward_weight):
    trainer.run()
```

`custom_fbw` has the following semantics:

- Both arguments must be callable and must retain the signatures of the default entry points.
- Its `finally` block restores the original module symbols, including after an exception.
- Contexts can be nested; exiting an inner context restores the outer replacement.
- Replacement is **module-global state within each worker process**, not thread-local or task-local. Every distributed worker must enter the context, and trainers using different FBW implementations must not run concurrently in one process.
- The context should cover the complete Trainer construction and execution lifetime so that initialization and execution cannot observe different entry points.

A custom implementation must satisfy at least these invariants:

1. Resolve asynchronous output-gradient communication before B consumes the gradients, for example with `executor.sync_tensors`.
2. Use the detached stage inputs from `Executor._detach` for this forward, or reuse `Executor.backward` as the DeepSeek example does.
3. Match the default dX return convention: `None` for zero gradients, a Tensor for one, and a tuple for multiple gradients.
4. Publish exactly one state per B and consume exactly one state per W, using a per-segment FIFO.
5. Handle `None` output gradients, duplicate outputs, no outputs, no grad-requiring stage inputs, and multi-output nodes correctly.
6. Route dW through parameter `AccumulateGrad` nodes so that reducer hooks fire.
7. Discard unpublished partial state after an exception, and provide clear/check operations that detect unmatched B or W calls.
8. Compare both model parameters and optimizer state against a baseline. Parameter equality after a short run can hide missing or duplicate gradients.

`custom_fbw` only replaces and restores the runtime entry points. It does not maintain this state for the implementation and does not change the schedule produced by `use_fbw=True`.

## 7. DeepSeek DualPipe Style: Explicit dW Callbacks

The runnable example is [tests/cli/test_customized_fbw_deekseek.py](../../tests/cli/test_customized_fbw_deekseek.py). The filename retains the spelling `deekseek`; the source project and design are **DeepSeek DualPipe**.

### 7.1 Core Idea

W does not traverse the autograd graph again. During B, each custom Function that supports deferred dW:

1. Computes and returns dX normally.
2. Retains only the detached tensors needed by its dW formula.
3. Creates a dW callback and places it in the current B cache.
4. Returns `None` for the weight input, preventing this traversal from accumulating dW.

The example's opaque linear is equivalent to:

```python
input_grad = output_grad @ weight.T

def accumulate_weight_grad():
    weight_grad = saved_input.T @ saved_output_grad
    torch.autograd.backward((weight,), grad_tensors=(weight_grad,))

WeightGradStore.put(accumulate_weight_grad)
return input_grad, None
```

The callback invokes backward on the leaf weight instead of writing `.grad`, so reducer hooks still run.

### 7.2 Pairing B and W

`_deepseek_backward_input` performs this sequence:

1. Synchronize output gradients.
2. Start one B transaction with `WeightGradStore.begin(name)`.
3. Call ordinary `Executor.backward`. Participating custom Functions compute dX and register callbacks without returning dW.
4. On success, call `flush()` to publish the complete callback batch atomically to `queues[name]`.
5. On failure, call `abort()` so W cannot execute a partial microbatch batch.

`_deepseek_backward_weight` pops one batch from the `queues[name]` FIFO and invokes each callback. An empty batch is a valid paired state; a missing batch indicates a B/W pairing error.

### 7.3 Benefits and Limitations

Benefits:

- W does not depend on `GradientEdge`, avoiding the Python PyNode public-wrapper problem in PyTorch 2.10.
- A callback retains only the tensors required by the explicit dW formula instead of a complete autograd graph, making memory lifetime easier to control.
- The work assigned to B and W is explicit, which fits linear layers, MoE experts, and other hot operators with simple dW formulas.

Limitations:

- Every parameterized operator whose dW should truly be delayed must participate in the protocol and implement an explicit callback.
- B uses an ordinary full backward. An unmodified parameterized operator still computes and accumulates dW during B. A mixed model can be numerically correct, but it does not move all W work off the critical path.
- Callback captures of tensors, weights, and communication state require careful lifetime management. W must not wait again on an output gradient already synchronized at B entry.
- The example store is process-global and assumes autograd traversals in one worker are neither concurrent nor reentrant.

## 8. d9d Style: Preserve the Graph and Re-enter It by Phase

The runnable example is [tests/cli/test_customized_fbw_d9d.py](../../tests/cli/test_customized_fbw_d9d.py). Instead of creating a separate callback for every dW, it retains the autograd graph and enters the same custom Function twice under different gradient phases.

### 8.1 Gradient-Direction Context

Because PyTorch 2.10 `ctx.needs_input_grad` cannot describe the current partial-backward targets, the example adds a process-global side channel:

- `INPUTS` phase: a custom Function computes only dX;
- `WEIGHTS` phase: a custom Function computes only dW;
- both phases: it performs an ordinary complete backward.

The custom Function still checks `ctx.needs_input_grad`, but it also checks `_GlobalGradContext.allows(direction)`. The context restores the preceding phase in `finally`, preventing one exception from corrupting later microbatches.

### 8.2 B: Capture the Boundary and Retain the Graph

`_d9d_backward_input` reimplements the default B path:

1. Record and synchronize output gradients, then retrieve detached stage inputs from `Executor._detach[name]`.
2. Deduplicate outputs and apply the backward pre-hook.
3. Build the reverse graph and locate boundary `param_groups` between input and parameter closures.
4. Register pre-hooks on each intermediate and retain the gradient tuple per output index as B crosses it.
5. In the `INPUTS` phase, run partial backward with stage inputs as targets and `retain_graph=True`.
6. Push `param_groups` and undetached outputs into the FIFO. The outputs act as `graph_owners`, keeping the Python PyNode's underlying C++ graph and saved tensors alive through W.

Retaining only a Python PyNode reference is insufficient. The example deliberately retains outputs to prevent premature destruction of the underlying graph.

### 8.3 W: Re-enter Every Real Output Edge

`_d9d_backward_weight` pops the matching state and then:

1. Maps parameter `AccumulateGrad` nodes back to leaf Parameters.
2. Creates `GradientEdge(intermediate, output_index)` for every non-`None` captured gradient. It never sums multi-output gradients and assigns them all to edge 0.
3. Clamps boundaries when there are multiple intermediates, preventing duplicate propagation from a descendant to an ancestor.
4. Calls `_engine_run_backward(..., accumulate_grad=True)` in the `WEIGHTS` phase.

The example calls private `_engine_run_backward` because the PyTorch 2.10 public `autograd.grad/backward` wrappers hit the `_input_metadata` error for Python `_FunctionBase` edges. `accumulate_grad=True` routes dW through parameter `AccumulateGrad` and triggers reducers.

When a segment has no grad-requiring stage input, B has no input closure to execute and delays the complete backward to W. W must enable both `INPUTS` and `WEIGHTS`: even without a cross-stage dX, dX must still propagate between operators inside the stage so that earlier layers receive dW. A weights-only phase in this branch leaves earlier-stage parameters unchanged.

### 8.4 Benefits and Limitations

Benefits:

- dW remains in the custom Function's normal backward formula; no per-B explicit dW callback is required.
- Boundary capture is graph-based and can handle several parameters sharing complex activation paths.
- The phase explicitly prevents participating custom Functions from executing expensive dW during B.

Limitations:

- B uses `retain_graph=True` and retains graph owners and saved tensors through W, generally using more memory than explicit callbacks.
- A custom Function must obey the phase context. An unmodified Python Function still has the `ctx.needs_input_grad` limitation.
- The example depends on `_engine_run_backward`, Node pre-hooks, and nnScaler graph-analysis helpers, all of which are version-sensitive internal APIs.
- Its phase and state store are process-global and are unsuitable for concurrent or reentrant autograd without a task-local redesign.

## 9. Choosing Between the Two Custom Designs

| Dimension | DeepSeek callbacks | d9d graph-preserving |
| --- | --- | --- |
| Source of W work | Explicit dW formula callbacks | Re-entry into backward from saved graph boundaries |
| Complete graph retained until W | No | Yes |
| Custom operator changes | Implement dX and a separate dW callback | Gate dX and dW in backward by phase |
| Dependency on PyTorch `GradientEdge` | None | Yes; the example bypasses the public wrapper with the private engine |
| Typical memory behavior | Retains only dW inputs, usually lower | Uses `retain_graph=True`, usually higher |
| Best fit | Hot operators with an explicit dW formula | Operators that should reuse their normal backward over complex graph boundaries |
| Primary risks | Missed parameterized operators and callback lifetime | Private APIs, graph ownership, and leaked phase state |

Neither design is a transparent switch for an arbitrary model. Parameterized custom Functions must cooperate with the chosen protocol, and each implementation needs end-to-end validation against its target PyTorch version, pipeline schedule, asynchronous communication mode, and reducer configuration.

## 10. Validation Strategy

Both CLI examples run three training variants: a conventional-backward baseline, custom FBW, and custom FBW with asynchronous communication. They compare final model and optimizer checkpoints. They also verify that:

- every registered DeepSeek callback executes;
- both dX and dW phases execute in the d9d path;
- runtime coverage includes both a scalar loss's `None` output gradient and a cross-stage Tensor output gradient;
- every per-segment FIFO is empty after training.

When extending the design to a new operator or schedule, retain those checks and add coverage for:

1. B/W pairing under multiple microbatches and interleaved schedules;
2. distinct gradients on every output edge of a multi-output custom Function;
3. missing or duplicate accumulation when one parameter is used along several paths;
4. reducer and ZeRO hooks across different dtype buckets;
5. cleanup of phase, callback cache, and FIFOs after an exception;
6. peak memory and the actual distribution of B/W kernels on the target PyTorch version.

The DeepSeek example is adapted from [DeepSeek DualPipe](https://github.com/deepseek-ai/DualPipe), and the d9d example is adapted from [d9d](https://github.com/d9d-project/d9d). See each example's file header for copyright and license details.
