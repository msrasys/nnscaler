# Kimi K3 nnScaler integration probe

[Kimi K3](https://huggingface.co/moonshotai/Kimi-K3) is Moonshot AI's
latest open-weight model as of 2026-08-08. The official checkpoint uses 69
KDA linear-attention layers, 24 gated-MLA layers, sparse MoE, MoonViT-V2, and
MXFP4 weights.

This example reduces the text model to two layers while preserving one KDA
layer, one MLA layer, sparse MoE routing, latent experts, and SiTU-GLU. It:

- registers the `fla-core` KDA kernel as an nnScaler custom operator;
- replaces the reference implementation's CPU-synchronized expert loop with
  equivalent batched tensor operations;
- uses eager MLA so the operator graph can be partitioned;
- constructs unquantized BF16/FP32 weights from config instead of downloading
  the 1.5 TB MXFP4 checkpoint.

See [the four-GPU validation matrix](../open_models_parallelism.md) for the
validated attention TP and four-way expert-parallel command.

The official sparse-MoE reference path is inference-only, so this probe uses
`ComputeConfig(inference_only=True)`. Training support requires an upstream
Kimi K3 training MoE implementation and MXFP4-aware partition rules.

```bash
python -m venv ~/.venvs/nnscaler-kimi-k3
source ~/.venvs/nnscaler-kimi-k3/bin/activate
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt -r examples/kimi_k3/requirements.txt
python -m pip install -e . --no-deps
python examples/kimi_k3/compile.py --mode eager
python examples/kimi_k3/compile.py --mode compile --plan-ngpus 2 --runtime-ngpus 2
torchrun --standalone --nproc_per_node=2 examples/kimi_k3/compile.py \
  --mode run --plan-ngpus 2 --runtime-ngpus 2
```

Keep this environment separate from DeepSeek-V3.2 because the examples require
different Transformers revisions. On OneDrive-backed checkouts, place the
environment outside the repository. Use the matching official PyTorch wheel
index for a different CUDA runtime, but keep PyTorch at 2.13.0.

Production gaps remain: MXFP4/MXFP8 kernels and checkpoint loading, optimized
expert parallelism, the multimodal MoonViT-V2 tower, and distributed KDA state
handling.

For the reduced probe, individual expert Linear weights are merged into three
3D parameters so AutoDist can shard the expert dimension. Loading an official
checkpoint therefore requires the inverse name/shape conversion.

The reduced KDA + MLA + sparse-MoE graph was validated with PyTorch
2.13.0+cu126, Transformers 5.3.0, `fla-core` 0.5.2, and two and four NVIDIA
RTX A6000 GPUs. The remote Kimi code is pinned to the revision in
`MODEL_REVISION`.
