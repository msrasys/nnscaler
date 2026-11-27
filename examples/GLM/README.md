# GLM Family (`GLM-4`, `GLM-4-MoE`, `GLM-4-MoE-Lite`, `GLM-5 DSA`, `GLM-5-Next`) — Zero-GPU `nnScaler` Adaptation

Hardware-decoupled `nnScaler` model adaptation and declarative parallelization package for the
**GLM** model family, built following
[`.github/skills/nnscaler-model-adaptation/SKILL.md`](../../.github/skills/nnscaler-model-adaptation/SKILL.md).

## Supported Architectures

| Preset | HF Architecture | Key Mechanisms & Rewrites |
|---|---|---|
| `glm4-9b` / `micro-glm4` | `Glm4ForCausalLM` | Dense GQA (`fallback_replicate` when `num_kv_heads < tp_size`) + interleaved `Glm4MLP.gate_up_proj` for `tp_col`. |
| `glm4-moe-355b` / `micro-glm4-moe` | `Glm4MoeForCausalLM` | GQA + biased `o_proj` decoupling + `glm_moe_route` + `glm_moe_grouped_gemm` (`EP`) / `glm_grouped_linear` (`EP` & `ETP`). |
| `micro-glm4-moe-lite` | `Glm4MoeLiteForCausalLM` | Head-parallel MLA (`glm_mla_concat_kv`) + Sparse MoE. |
| `glm5-744b-dsa` / `micro-glm-moe-dsa` | `GlmMoeDsaForCausalLM` | Head-parallel MLA + `glm_dsa_index_mask` (`@torch.no_grad()` interleaved-RoPE DSA Sparse Indexer) + 256-expert MoE. |
| `micro-glm5-next` | `Glm5NextForConditionalGeneration` | Hybrid KDA (`glm_causal_conv1d` + `glm_chunk_kda`) + NoPE-MLA/DSA + HyperConnections + MoE. |

## Quickstart (0 GPUs Required)

```bash
# 1. Compile GLM-4.5/4.6 355B-A32B (46L, 128E, PP=4 x EP/TP=8, 32 GPUs)
PYTHONPATH=. python examples/GLM/compile.py \
    --model glm4-moe-355b \
    --strategy examples/GLM/strategies/glm4_moe_355b_ep8_pp4.json \
    --out-dir /tmp/glm4_moe_355b_bundle

# 2. Compile GLM-5 744B-A40B with DSA Sparse Attention (PP=4 x EP/TP=8, 32 GPUs)
PYTHONPATH=. python examples/GLM/compile.py \
    --model glm5-744b-dsa --num-layers-override 8 \
    --strategy examples/GLM/strategies/glm5_dsa_744b_ep8_pp4.json \
    --out-dir /tmp/glm5_dsa_bundle

# 3. Run Full Verification Suite (CPU Golden Parity + Zero-GPU Compilation)
PYTHONPATH=. python examples/GLM/tests/run_all_tests.py
```

KDA explicitly uses the unwrapped PyTorch reference on CPU and for FP32; optional
FLA is selected only for supported FP16/BF16 CUDA inputs. Convolution remains
`F.conv1d`. See [optional-kernel dispatch and CUDA parity tests](../qwen3p5/README.md#optional-cuda-kernels).
