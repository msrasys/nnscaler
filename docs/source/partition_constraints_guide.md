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
    *   Used to group operators whose tensors saved for backward are offloaded to pinned CPU memory and prefetched back before their backward.
    *   Operators with the same non-negative `offload_id` must be consecutive and belong to the same pipeline stage.
    *   `offload_id` may be combined with `recompute_id`: the group is recomputed and the inputs the checkpoint keeps for backward are offloaded, so the group leaves nothing resident on the device.
    *   Dense strided CUDA tensors of at least `CPU_OFFLOADING_MIN_BYTES` (default 1 MiB) are copied on a dedicated D2H stream; parameters/buffers and small or non-strided tensors stay on the device. Views sharing a storage region are copied once.
    *   Prefetch order follows the execution contexts opened by `Executor.fexecute`/`Executor.backward`: within one segment execution groups are reloaded in reverse forward order (`CPU_OFFLOADING_PREFETCH_GROUPS` groups ahead, default 2); across micro-batches the backward of a segment consumes contexts in forward order (FIFO), which is exact for 1F1B, interleaved 1F1B and gradient accumulation. Scheduled pipeline code additionally calls `prefetch_next_backward(name)` before every backward and `hint_forward(name, distance)` before every forward; a forward whose own backward is fewer than `CPU_OFFLOADING_MIN_BACKWARD_DISTANCE` (default 1) segment executions away is not offloaded.
    *   Pinned host buffers are pooled in geometric size classes and reused across steps (`CPU_OFFLOADING_POOL_MAX_GB` bounds the free list). Pinned pages are allocated on the GPU's NUMA node via `set_mempolicy` (`CPU_OFFLOADING_NUMA_MEMBIND=0` disables); `CPU_OFFLOADING_NUMA_AFFINITY=1` additionally pins the main thread to the GPU's local CPUs. `CPU_OFFLOADING_DEBUG=1` logs what each group saves, `CPU_OFFLOADING_FAKE_COPY=1` keeps tensors on the GPU for A/B tests, `CPU_OFFLOADING_TIMING=0` disables the CUDA-event timings (`stall_ms`, `d2h_ms`, `h2d_ms`). `CPU_OFFLOADING_MAX_INFLIGHT_COPIES` (default 0 = unbounded) can bound the memcpys queued per direction. `CPU_OFFLOADING_DISABLE=1` turns the contexts into no-ops; `CPU_OFFLOADING_PREFETCH_LEVEL` from the first prototype is accepted but ignored.
    *   See `nnscaler/runtime/cpu_offloading.py` for the runtime and `docs/source/cpu_offloading.md` for the design.
*   **`stage_id`** (default: -1):
    *   Used for Pipeline Parallelism assignment.
    *   These operators with the same `stage_id` should be consecutive in the graph.
*   **`pre_hook` / `post_hook`**:
    *   You can attach custom Python functions to be executed before or after the operator. See source code for signature details.

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
