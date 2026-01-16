# Autodist Constraints Guide

Autodist allows users to guide the parallelization strategy by specifying constraints. This is useful when you have specific knowledge about how certain operators should be partitioned or if you want to enforce specific behaviors like recomputation or pipeline stages.

There are two primary ways to provide constraints, only one of them can be used at a time:
1.  **YAML Configuration**: Using `allowed_partition_dims` to define valid partition dimensions for specific operators.
2.  **Python Policy Generator**: providing a generator function that yields `OpPlan` objects to explicitly define the plan for operators.

## Method 1: YAML Configuration

You can use a YAML file to specify which dimensions are allowed for partitioning for specific operators. This is often used to prevent Autodist from partitioning certain operators in ways that are known to be inefficient or problematic (e.g., forcing replication).

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


## Method 2: Python Policy Generator (`OpPlan`)

For fine-grained control, you can provide a Python generator function (the `policy` argument in `parallelize` or `autodist_wrapper`). This function yields `OpPlan` objects which tell Autodist exactly how to handle specific nodes.

### Usage

Define a function `policy(graph, cfg)` that iterates over the graph nodes and yields `OpPlan` objects.

**Important Considerations:**
If you choose to manually partition operators (especially for complex communication patterns), you often need to define `OpPlan` for **all connected operators** that share the partition logic (e.g., cast ops like `float`, `to`, or element-wise ops). If you miss them, Autodist might fail to infer valid partitionings for those intermediate nodes.


### `OpPlan` Parameters

The `OpPlan` class defines the strategy for a single operator.

```python
class OpPlan:
    def __init__(self, op, partition='auto', recompute_id=-1, stage_id=-1, ...):
        ...
```

*   **`op`**: The graph node (`IRFwOperation`) this plan applies to.
*   **`partition`**: define the partitioning strategy.
    *   `OpPartition(input=i, dim=d)`: Partition the operator based on the `d`-th dimension of its `i`-th input tensor.
    *   `'auto'` (default): Let Autodist automatically infer the best partitioning strategy.
    *   `None`: Force the operator to be replicated (no partitioning).
*   **`recompute_id`** (default: -1):
    *   Used to group operators for Rematerialization (Gradient Checkpointing).
    *   Operators with the same non-negative `recompute_id` will be grouped into a single recomputation block.
*   **`stage_id`** (default: -1):
    *   Used for Pipeline Parallelism assignment.
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
