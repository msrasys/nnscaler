# Open-model parallelism validation

The reduced Kimi K3, GLM-5, and DeepSeek-V3.2 examples were validated on four
NVIDIA RTX A6000 GPUs with PyTorch 2.13.0+cu126. Every row below includes
nnScaler code generation and execution of the generated module.

| Model / scenario | Configuration | Parallelism observed | Runtime check |
| --- | --- | --- | --- |
| GLM-5 AutoDist | `plan=4`, `runtime=4` | Four-way expert parameter sharding and expert-output all-reduce | 2 training steps |
| DeepSeek-V3.2 AutoDist | `plan=4`, `runtime=4` | Four-way EP plus sequence/head activation sharding | 2 training steps |
| Kimi K3 AutoDist | `plan=4`, `runtime=4`, EP constraint | Four-way SiTU expert parameter sharding, expert-output all-reduce, and attention TP | 2 inference steps |
| GLM-5 / DeepSeek pipeline | 4 layers, 4 microbatches, `pipeline_stages=2`, `max_partition_degree=2` | Two pipeline stages with up to two-way intra-stage SPMD activation/tensor sharding | 2 training steps |
| GLM-5 pipeline replicas | `plan=2`, `runtime=4`, `pipeline_stages=2`, `max_partition_degree=1` | Two pipeline stages replicated twice, reducer synchronization, and ZeRO-1 | 2 training steps |
| GLM-5 hybrid EP/DP | `plan=2`, `runtime=4`, EP constraint | Two-way EP inside each plan, two replica groups, and ZeRO-1 | 2 training steps |
| GLM-5 async reducer | `plan=2`, `runtime=4`, no ZeRO, async reducer, replicated-parameter reduction | Async gradient synchronization across replicas | 2 training steps |
| GLM-5 ZeRO reduce-scatter | `plan=2`, `runtime=4`, ZeRO-1 reduce-scatter | Reduce-scattered gradients and gathered parameters | 2 training steps |

The training entrypoints check finite loss and gradients. For replicated plans,
they also compare corresponding gradient and parameter shards after
synchronization. Kimi K3 remains inference-only because the official reference
MoE implementation does not provide a training path.

## Four-GPU commands

The following GLM-5 commands illustrate the main combinations. DeepSeek-V3.2
accepts the same flags. Install each example's requirements first.

```bash
# Unconstrained 4-GPU AutoDist
python examples/glm5/compile.py --mode compile \
  --plan-ngpus 4 --runtime-ngpus 4
torchrun --standalone --nproc_per_node=4 examples/glm5/compile.py \
  --mode run --plan-ngpus 4 --runtime-ngpus 4

# Two pipeline stages, each with up to two-way intra-stage SPMD
python examples/glm5/compile.py --mode compile \
  --plan-ngpus 4 --runtime-ngpus 4 --num-layers 4 --microbatches 4 \
  --pipeline-stages 2 --max-partition-degree 2
torchrun --standalone --nproc_per_node=4 examples/glm5/compile.py \
  --mode run --plan-ngpus 4 --runtime-ngpus 4 --num-layers 4 \
  --microbatches 4 --pipeline-stages 2 --max-partition-degree 2

# Two-way EP in each of two data-parallel replicas, with ZeRO-1
python examples/glm5/compile.py --mode compile \
  --plan-ngpus 2 --runtime-ngpus 4 \
  --partition-constraints-path examples/glm5/ep_constraints.yaml
torchrun --standalone --nproc_per_node=4 examples/glm5/compile.py \
  --mode run --plan-ngpus 2 --runtime-ngpus 4 \
  --partition-constraints-path examples/glm5/ep_constraints.yaml
```

Kimi K3 uses its model-specific constraint:

```bash
python examples/kimi_k3/compile.py --mode compile \
  --plan-ngpus 4 --runtime-ngpus 4 \
  --partition-constraints-path examples/kimi_k3/ep_constraints.yaml
torchrun --standalone --nproc_per_node=4 examples/kimi_k3/compile.py \
  --mode run --plan-ngpus 4 --runtime-ngpus 4 \
  --partition-constraints-path examples/kimi_k3/ep_constraints.yaml
```

## Inspect generated plans

Code generation alone does not prove that a useful partition was selected.
Inspect parameter slices, pipeline placement, collectives, and local expert
ranges with:

```bash
python examples/inspect_parallel_plan.py /path/to/generated/output
```

The four-GPU validation also passed:

- the TP, hybrid DP, synchronous/async reducer, ZeRO-1, and reduce-scatter
  end-to-end matrix in `tests/parallel_module/test_end2end.py`;
- reducer and expert-parallel code-generation tests;
- AutoDist pipeline-stage and partition-constraint tests;
- runtime synchronous and asynchronous point-to-point/collective tests.

The corresponding generated modules and JSON inspection reports are preserved
under:

```text
/Data/yomia/nnscaler-gencode/feat-latest-open-model-operators-4gpu/
```
