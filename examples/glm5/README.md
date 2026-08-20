# GLM-5 nnScaler integration probe

This example compiles a reduced [GLM-5](https://huggingface.co/zai-org/GLM-5)
whose dimensions are small but whose DeepSeek Sparse Attention (DSA), MLA,
and sparse-MoE paths are enabled.

See [the four-GPU validation matrix](../open_models_parallelism.md) for EP,
TP, pipeline, replica DP, reducer, and ZeRO commands.

## Operator coverage

| Operator/path | nnScaler status | Notes |
| --- | --- | --- |
| Linear, RMSNorm primitives, SiLU, matmul, reshape/transpose, RoPE | Available | Covered by nnScaler's PyTorch operator mapping. |
| `topk`, `gather`, `where`, `nonzero`, `masked_fill` | Available | Present in `nnscaler/graph/parser/mapping.py`. |
| Compact `einsum` equations | Available | This integration adds annotations for standard equations such as `bshd,btd->bsht`. |
| `scatter` / `scatter_` | Available, replicated | This integration adds a conservative non-partitioned annotation used by DSA masks and MoE routing. |
| FlashAttention/GQA | Available as a custom annotation | Reusable from `examples/transformers_utils/flash_attn_anno.py`. |
| MLA projections | Partial | Composed from supported primitives; no first-class fused MLA operator. |
| DSA indexer and sparse attention | Partial | The eager Transformers path is used by this probe. FlashMLA/DeepGEMM kernels are not registered with nnScaler. |
| MoE routing | Partial | The existing DeepSeek-Coder-V2-Lite example registers a fixed-shape route operator; GLM-5 uses different expert counts and routing. |
| Expert dispatch/grouped GEMM | Partial | An example-level `grouped_gemm` implementation exists, but there is no generic GLM-5/DeepSeek-V3.2 expert-parallel operator. |
| MTP/speculative head | Not covered | This probe targets the training CausalLM backbone only. |

The reduced probe selects Transformers' `batched_mm` expert implementation
because its primitive `bmm` graph is partitionable today. It validates the
reduced operator graph, not the optimized grouped-GEMM production kernel or
the full-checkpoint memory footprint.

## Run

Use a separate environment for this example. Keep the environment outside the
repository when the checkout is synchronized by OneDrive. The following is the
validated CUDA 12.4 setup:

```bash
python -m venv ~/.venvs/nnscaler-glm5
source ~/.venvs/nnscaler-glm5/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt -r examples/glm5/requirements.txt
python -m pip install -e . --no-deps
python examples/glm5/compile.py --mode eager
python examples/glm5/compile.py --mode compile --plan-ngpus 2 --runtime-ngpus 2
torchrun --standalone --nproc_per_node=2 examples/glm5/compile.py \
  --mode run --plan-ngpus 2 --runtime-ngpus 2
```

Choose the matching official PyTorch wheel index for a different CUDA runtime,
but keep PyTorch at 2.13.0. Installing the repository editable is required to
build nnScaler's native `dp_solver` extension.

The compile command runs nnScaler `autodist` and writes generated code to
the system temporary directory by default. Run mode loads that generated
module, performs forward and backward, checks loss and gradients for finite
values, and applies AdamW updates.

The reduced model was validated with PyTorch 2.13.0+cu126 and Transformers
5.3.0 on two and four NVIDIA RTX A6000 GPUs.

Production work still requires a generic expert-parallel grouped-GEMM
kernel, all-to-all token dispatch, and registered FlashMLA/DeepGEMM kernels.
The reduced example uses a tensorized expert kernel with an AutoDist EP
transform rule; `ep_constraints.yaml` can force expert partitioning.
