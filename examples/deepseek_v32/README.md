# DeepSeek-V3.2 nnScaler integration probe

This example preserves the defining operator paths of
[DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2):
DeepSeek Sparse Attention (DSA), MLA projections, sparse MoE, and the DSA
indexer. The dimensions and layer count are reduced so code generation does
not require downloading the 671B FP8 checkpoint.

See [the four-GPU validation matrix](../open_models_parallelism.md) for EP,
sequence/head sharding, pipeline, replica DP, reducer, and ZeRO commands.

The pinned Transformers revision is required because DeepSeek-V3.2 support is
newer than the 5.3.0 release. The example uses the partitionable `batched_mm`
expert implementation. Production grouped-GEMM/expert-parallel kernels,
FlashMLA/DeepGEMM, and FP8 UE8M0 checkpoint loading remain future work.

```bash
python -m venv ~/.venvs/nnscaler-deepseek-v32
source ~/.venvs/nnscaler-deepseek-v32/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt -r examples/deepseek_v32/requirements.txt
python -m pip install -e . --no-deps
python examples/deepseek_v32/compile.py --mode eager
python examples/deepseek_v32/compile.py --mode compile --plan-ngpus 2 --runtime-ngpus 2
torchrun --standalone --nproc_per_node=2 examples/deepseek_v32/compile.py \
  --mode run --plan-ngpus 2 --runtime-ngpus 2
```

Keep this environment separate from GLM-5 and Kimi K3: DeepSeek-V3.2 requires
the pinned unreleased Transformers revision above. On OneDrive-backed
checkouts, place the environment outside the repository. Use the matching
official PyTorch wheel index for a different CUDA runtime, but keep PyTorch at
2.13.0.

The reduced training graph (including backward, DSA, MLA, and tensorized MoE)
was validated with PyTorch 2.13.0+cu126 on two and four NVIDIA RTX A6000 GPUs,
with finite loss and gradients and AdamW parameter updates.

Four-GPU validation additionally shards the routed-expert parameters. Use
`ep_constraints.yaml` when deterministic EP selection is required.
