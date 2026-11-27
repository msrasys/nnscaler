# Kimi Family (`Kimi-Linear-48B-A3B` & `Kimi-K2.5`) — Zero-GPU `nnScaler` Adaptation

Hardware-decoupled `nnScaler` model adaptation and declarative parallelization package for the
**Kimi** model family (**Kimi-Linear-48B-A3B** hybrid KDA + NoPE-MLA + MoE, and **Kimi-K2.5**
1T-class Multimodal VLM with 27L ViT + 61L DeepSeek-V3 MLA/384-Expert MoE), built following
[`.github/skills/nnscaler-model-adaptation/SKILL.md`](../../.github/skills/nnscaler-model-adaptation/SKILL.md).

---

## Architectural Blockers Solved

| Subsystem | Upstream HuggingFace Blocker | `examples/Kimi` Solution |
|---|---|---|
| **Kimi Delta Attention (`KDA`)** | `@use_kernelized_func` dispatches to FLA `chunk_kda` + `causal_conv1d_fn` with concatenated `[3*H*D]` conv1d. | Split depthwise conv into `kimi_causal_conv1d` (`b (h d^) s^`) + `kimi_chunk_kda` (`b s^ h d^`), keeping `q/k/v_proj`, `conv1d`, `forget_gate.f_b_proj`, `b_proj`, `g_b_proj`, `o_norm`, and `o_proj` 100% head-parallel across `TP`. |
| **MLA `expand_kv`** (`KimiLinearAttention` & `Kimi-K2.5`) | Inplace `.copy_` + `.expand(-1, H, -1, -1)` breaks FX SSA and freezes head axis. | 3D `kv_b_proj` + `kimi_mla_concat_kv` (`b h s dn^, b 1^ s dr^ -> b h s dk^`). |
| **Sparse MoE** (`256` / `384` experts) | `@use_experts_implementation` (`scatter_`, `sort`, `histc`, `cumsum`). | `kimi_moe_route` + default fused `kimi_moe_grouped_gemm` (`EP`) and opt-in expanded `kimi_grouped_linear` (`EP` / `ETP`). |
| **Kimi-K2.5 Vision & VLM** | `lengths.tolist()` in `Kimi_K25VisionAttention`, `.tolist()` split+cat in `get_image_features`, inplace mask indexing. | `kimi_k25_vision_varlen_sdpa` (`1^ h s^ d^`) + `kimi_k25_embed_scatter` + `torch.where`. |
| **Vocabulary Output Head** (`163,840` vocab) | `F.cross_entropy(..., reduction='mean')` locks `N^ C^`. | `KimiCausalLMTaskWrapper` (`sum / valid_token_count`) unlocks `vocab_parallel`. |

---

## Quickstart (0 GPUs Required)

```bash
# 1. Compile Kimi-Linear-48B-A3B (27L: 20 KDA + 7 MLA, 256 experts, EP=8/TP=8)
PYTHONPATH=. python examples/Kimi/compile.py \
    --model kimi-linear-48b \
    --strategy examples/Kimi/strategies/kimi_linear_48b_ep8_tp8.json \
    --out-dir /tmp/kimi_linear_48b_bundle

# 2. Compile Kimi-K2.5 Multimodal VLM (micro or full 32-GPU PP=4 x EP/TP=8)
PYTHONPATH=. python examples/Kimi/compile.py \
    --model micro-kimi-k25 --multimodal \
    --strategy examples/Kimi/strategies/kimi_linear_expanded_etp4.json \
    --expand-moe-ops false \
    --out-dir /tmp/kimi_k25_mm_bundle

# 3. Run Full Verification Suite (CPU Golden Parity + Zero-GPU Compilation)
PYTHONPATH=. python examples/Kimi/tests/run_all_tests.py
```
