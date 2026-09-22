# Partition Constraints Guide

Nnscaler allows users to guide the parallelization strategy by specifying constraints. This is useful when you have specific knowledge about how certain operators should be partitioned or if you want to enforce specific behaviors like recomputation or pipeline stages.

There are two scenarios for providing constraints, they cannot be used at the same time:
1.  **Partition constraints in Autodist**: When using autodist, you can use **Autodist YAML Configuration** to define valid partition dimensions for specific operators.
2.  **Partition constraints in Function Policy**: You can provide a **Function Policy** that yields `OpPlan` objects to explicitly define the plan for operators. There are no auto search for partitioning except for your definition.

## Method 1: Partition constraints in Autodist

You can use a **Autodist YAML Configuration** file to specify which dimensions are allowed for partitioning for specific operators. This is often used to prevent Autodist from partitioning certain operators in ways that are known to be inefficient or problematic (e.g., forcing replication).

### Configuration Format

The configuration is a list of dictionaries, each describing a constraint rule.

```yaml
- allowed_partition_dims:
  - 0,0  # List of allowed (input_index, dim_index) pairs
  name: torch.sum
  parent_module: 'MoE' # Optional: Filter by parent module class name
  replica_allowed: false
```

### Fields

*   **`name`** (required): The fully qualified name or signature of the operator (e.g., `torch.sum`, `arch.ffn.ffn_func`).
*   **`allowed_partition_dims`** (required): A list of strings representing allowed partition strategies.
    *   Format: `"input_idx,dim_idx"`.
    *   Example: `"0,0"` means the operator can be partitioned along dimension 0 of input 0.
    *   If the list is empty, the operator might be forced to replicate (depending on `replica_allowed`).
*   **`parent_module`** (optional): If specified, the constraint only applies to operators that are children of a module with this class name. This is useful for targeting specific parts of the model (e.g., only `torch.sum` inside `MoE` layer).
*   **`replica_allowed`** (optional, default: `true`): Whether replication is a valid strategy. If `false`, Autodist *must* find a partition strategy from `allowed_partition_dims`.

### Example

Below is an example of a custom operator:

```yaml

# Constraint for a custom op
- allowed_partition_dims:
  - 0,0
  name: arch.all2all_moe.nnscaler_all2all_moe_gmm
  parent_module: 'MoE'
  replica_allowed: false

```

To use this file, pass its path to `AutoDistConfig`:

```python
cfg = AutoDistConfig(
    ...,
    partition_constraints_path='/path/to/constraints.yaml'
)
```


## Method 2: Partition constraints in Function Policy

For fine-grained control, you can provide a **Function Policy** (the `pas_policy` argument in `parallelize` or `pas_policy` argument in `TrainerArgs`). This function yields `OpPlan` objects which explicitly specify the partitioning strategy for specific nodes.

### Usage

Define a function `policy(graph, cfg)` that iterates over the graph nodes and yields `OpPlan` objects.

**Important Considerations:**
If you choose to manually partition operators (especially for complex communication patterns), you often need to define `OpPlan` for **all connected operators** that share the partition logic, this means:
1. you must define OpPlans for all ops you want to partition.
2. the default OpPlans is replicated  if you don't define `OpPlan` for ops.
3. The only exception is when you define `OpPlan.partition` to `auto`, which will try to partition the op based on its input partitions.

### `OpPlan` Parameters

The `OpPlan` class defines the strategy for a single operator.

```python
class OpPlan:
    def __init__(self, op, partition='auto', recompute_id=-1, offload_id=-1, stage_id=-1, ...):
        ...
```

*   **`op`**: The graph node (`IRFwOperation`) this plan applies to.
*   **`partition`**: define the partitioning strategy.
    *   `OpPartition(input=i, dim=d)`: Partition the operator based on the `d`-th dimension of its `i`-th input tensor.
    *   `'auto'` (default): Tries to follow the partition of its inputs. If no input is partitioned, it just replicate the operator.
    *   `None`: Force the operator to be replicated (no partitioning).
*   **`recompute_id`** (default: -1):
    *   Used to group operators for Recompute (Gradient Checkpointing).
    *   Operators with the same non-negative `recompute_id` will be grouped into a single recomputation block.
    *   These operators with the same `recompute_id` should be consecutive in the graph.
*   **`offload_id`** (default: -1):
    *   Used to group operators whose tensors saved for backward are offloaded to CPU.
    *   Operators with the same non-negative `offload_id` must be consecutive and belong to the same pipeline stage.
    *   `offload_id` and `recompute_id` cannot both be set on the same operator.
    *   Dense strided tensors are copied asynchronously through pinned CPU memory. Unsupported layouts remain on their original device.
    *   Each CPU-offload context forms one batch linked to the preceding live context in process-local forward order. On unpack, the runtime loads the demanded tensor first and then follows the linked batches in reverse pack order. `ParallelModule` reads `CPU_OFFLOADING_PREFETCH_LEVEL` once when the module is constructed; it defaults to 2 when unset, prefetching the next two tensors. Positive levels define a tensor lookahead: 1 prefetches the next tensor after the demanded tensor, 2 prefetches the next two, and so on, continuing into preceding batches when necessary. Level 0 loads only on demand. Negative levels count whole batches instead: -1 prefetches the current batch, -2 covers the current and immediately preceding batches, and larger absolute values follow more preceding batches. Set `module.cpu_offloading_prefetch_level` after construction to override the value for that module instance.
    *   CPU-offload contexts must execute sequentially on one thread. Nested contexts and concurrent use from multiple threads are not supported.
*   **`stage_id`** (default: -1):
    *   Used for Pipeline Parallelism assignment.
    *   These operators with the same `stage_id` should be consecutive in the graph.
*   **`pre_hook` / `post_hook`**:
    *   You can attach custom Python functions to be executed before or after the operator. See source code for signature details.

### Pipeline auxiliary outputs

With an end-to-end pipeline function policy, auxiliary views are inserted on the
original graph after stage IDs are resolved, before split bookkeeping,
recompute/offload grouping, shared-parameter multiref, and `graph.staging`.
A read-only pass resolves auxiliary output and unused cross-stage input detach
sites. The insertion pass updates consumers, graph outputs, and plans using
the native `Detach` operator, then re-infers gradient metadata for both the
source and detached tensors and updates their consumers' existing backward outputs.
This includes consumers that keep the original input, since their gradient
value maps can change when auxiliary gradient contributions are removed.
The no-grad detach operators need no backward nodes; the other backward nodes
are preserved rather than rebuilt globally.
Only inserted detach outputs are marked `requires_grad=False`; existing
forward flags are preserved. Detached outputs and boundaries isolate auxiliary
branches from the loss, so conservative `True` flags inside those branches do
not reconnect runtime autograd. Inferring output flags from input flags would
be unsafe for operators that create grad-enabled tensors internally. An output
that no longer requires grad at runtime can retain a redundant detach.
Staging then infers the stage interfaces without auxiliary-output postprocessing.
New operators inherit the recompute/offload region at their insertion position
through their `OpPlan`, just like handwritten operators.
The original activation stays differentiable through
its last loss-dependent stage; later auxiliary-only consumers receive detached views.
Shared parameters' auxiliary-only remote uses are detached at the owning stage,
before multiref can create separate differentiable aliases for those consumers.
Returned parameters are detached in the same original-graph pass.
Inserted detaches use `OpPlan(partition='auto')` and the ordinary TP path.
Dimension partitions can be propagated; value-partitioned inputs are combined
by the usual adapters before a replicated detach. Producerless parameters
use the normal replicated fallback, so their storage layout may change.
No TP-specific detach handling, later backward pruning, adapter/codegen
compensation, or runtime zero filling is added.
User-written `.data` and `.detach()` outputs follow the same activation analysis.

The analysis is conservative for opaque operators: a live output retains all
declared input gradients. External inputs retain the existing behavior.
Stages with no differentiable
outputs remain unsupported.

Unused parameters in TP reducers retain the normal reducer semantics:
`reducer_none_grad=False` can materialize zeros. To preserve eager `grad=None`
and avoid weight decay on unused weights, enable `reducer_none_grad` and keep
used and unused parameters in separate buckets.

The runtime requires every declared input gradient to be present. Missing
gradients raise an error rather than being replaced with zeros.

### Example: Custom Partitioning and Recomputation

This example demonstrates how to:
1.  Use helper functions: `get_pas_ops` (filter for relevant ops), `get_layer_index` (extract layer ID from name), and `get_called_self_module_name` (identify sub-module names like `gate_proj`).
2.  Filter operations by the module class chain (e.g., targeting `FFNDropout`).
3.  Assign `recompute_id` and `stage_id` dynamically based on the model's layer index.
4.  Apply different partition strategies based on the specific module being called.

```python
from nnscaler.policies import OpPlan, OpPartition, get_layer_index, get_called_self_module_name, get_pas_ops
import torch

def custom_policy(graph):
    for node in get_pas_ops(graph):
        if FFNDropout not in node.module_class_chain: # work only on FFN module
            continue

        ffn_idx = get_layer_index(node.fqn)
        module_called = get_called_self_module_name(node.call_expr)

        if node.fn == torch.nn.functional.linear:
            if module_called in ['gate_proj', 'up_proj']:
                yield OpPlan(node, recompute_id=ffn_idx, stage_id=ffn_idx, partition=OpPartition(input=1, dim=0))
            else:
                # down_proj
                yield OpPlan(node, recompute_id=ffn_idx, stage_id=ffn_idx, partition=OpPartition(input=1, dim=1))
        else:
            # other ops
            yield OpPlan(node, recompute_id=ffn_idx, stage_id=ffn_idx, partition='auto')
```
