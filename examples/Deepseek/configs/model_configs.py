# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model configuration presets, zero-GPU `meta` model builder, and dummy sample factory for DeepSeek.

Covers:
  - Fast CPU golden-parity & unit-test presets:
    `micro-v2-lite`, `micro-v2`, `micro-v3`, `micro-v32`
  - Official HuggingFace frontier presets:
    `deepseek-v2-lite` (15.7B-A2.4B, 27L, 64E),
    `deepseek-v2`      (236B-A21B,   60L, 160E),
    `deepseek-v3`      (671B-A37B,   61L, 256E),
    `deepseek-r1`      (671B-A37B,   61L, 256E),
    `deepseek-v3.2`    (671B-A37B + DSA Sparse Indexer, 61L, 256E)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    DeepseekV2Config,
    DeepseekV2ForCausalLM,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    DeepseekV32Config,
    DeepseekV32ForCausalLM,
)

from examples.Deepseek.rewrites.model_rewrites import (
    DeepseekCausalLMTaskWrapper,
    RewriteConfig,
    apply_deepseek_rewrites,
)


def _v2_preset(
    *,
    vocab_size: int,
    hidden_size: int,
    intermediate_size: int,
    moe_intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    first_k_dense_replace: int,
    kv_lora_rank: int,
    q_lora_rank: int | None,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    n_routed_experts: int,
    n_shared_experts: int,
    num_experts_per_tok: int,
    n_group: int,
    topk_group: int,
    topk_method: str,
    routed_scaling_factor: float,
) -> dict[str, Any]:
    return {
        "model_type": "deepseek_v2",
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "moe_intermediate_size": moe_intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_attention_heads,
        "first_k_dense_replace": first_k_dense_replace,
        "kv_lora_rank": kv_lora_rank,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "n_group": n_group,
        "topk_group": topk_group,
        "topk_method": topk_method,
        "routed_scaling_factor": routed_scaling_factor,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "mlp_bias": False,
        "_attn_implementation": "sdpa",
    }


def _v3_preset(
    *,
    model_type: str = "deepseek_v3",
    vocab_size: int = 129280,
    hidden_size: int = 7168,
    intermediate_size: int = 18432,
    moe_intermediate_size: int = 2048,
    num_hidden_layers: int = 61,
    num_attention_heads: int = 128,
    first_k_dense_replace: int = 3,
    kv_lora_rank: int = 512,
    q_lora_rank: int = 1536,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    n_routed_experts: int = 256,
    n_shared_experts: int = 1,
    num_experts_per_tok: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    routed_scaling_factor: float = 2.5,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "model_type": model_type,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "moe_intermediate_size": moe_intermediate_size,
        "num_hidden_layers": num_hidden_layers,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_attention_heads,
        "first_k_dense_replace": first_k_dense_replace,
        "kv_lora_rank": kv_lora_rank,
        "q_lora_rank": q_lora_rank,
        "qk_nope_head_dim": qk_nope_head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "v_head_dim": v_head_dim,
        "n_routed_experts": n_routed_experts,
        "n_shared_experts": n_shared_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "n_group": n_group,
        "topk_group": topk_group,
        "norm_topk_prob": True,
        "routed_scaling_factor": routed_scaling_factor,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "_attn_implementation": "sdpa",
    }
    if extra:
        cfg.update(extra)
    return cfg


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    # --- Fast CPU Micro Presets ---
    "micro-v2-lite": _v2_preset(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        first_k_dense_replace=1,
        kv_lora_rank=24,
        q_lora_rank=None,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        topk_method="greedy",
        routed_scaling_factor=1.0,
    ),
    "micro-v2": _v2_preset(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        first_k_dense_replace=1,
        kv_lora_rank=24,
        q_lora_rank=48,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        topk_method="group_limited_greedy",
        routed_scaling_factor=2.0,
    ),
    "micro-v3": _v3_preset(
        model_type="deepseek_v3",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=8,
        first_k_dense_replace=1,
        kv_lora_rank=24,
        q_lora_rank=48,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=2.5,
    ),
    "micro-v32": _v3_preset(
        model_type="deepseek_v32",
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=8,
        first_k_dense_replace=1,
        kv_lora_rank=24,
        q_lora_rank=48,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=2.5,
        extra={
            "index_n_heads": 4,
            "index_head_dim": 16,
            "index_topk": 8,
            "mlp_bias": False,
        },
    ),
    # --- Official HuggingFace Frontier Presets ---
    "deepseek-v2-lite": _v2_preset(
        vocab_size=102400,
        hidden_size=2048,
        intermediate_size=10944,
        moe_intermediate_size=1408,
        num_hidden_layers=27,
        num_attention_heads=16,
        first_k_dense_replace=1,
        kv_lora_rank=512,
        q_lora_rank=None,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=64,
        n_shared_experts=2,
        num_experts_per_tok=6,
        n_group=1,
        topk_group=1,
        topk_method="greedy",
        routed_scaling_factor=1.0,
    ),
    "deepseek-v2": _v2_preset(
        vocab_size=102400,
        hidden_size=5120,
        intermediate_size=12288,
        moe_intermediate_size=1536,
        num_hidden_layers=60,
        num_attention_heads=128,
        first_k_dense_replace=1,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        n_routed_experts=160,
        n_shared_experts=2,
        num_experts_per_tok=6,
        n_group=8,
        topk_group=3,
        topk_method="group_limited_greedy",
        routed_scaling_factor=16.0,
    ),
    "deepseek-v3": _v3_preset(model_type="deepseek_v3"),
    "deepseek-r1": _v3_preset(model_type="deepseek_v3"),
    "deepseek-v3.2": _v3_preset(
        model_type="deepseek_v32",
        extra={
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 2048,
            "mlp_bias": False,
        },
    ),
}


def list_model_presets() -> list[str]:
    return sorted(MODEL_PRESETS.keys())


def load_model_config_dict(preset_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(preset_or_path, dict):
        return copy.deepcopy(preset_or_path)
    key = str(preset_or_path)
    if key in MODEL_PRESETS:
        return copy.deepcopy(MODEL_PRESETS[key])
    path = Path(key)
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    raise ValueError(f"Unknown DeepSeek preset or missing config file: {preset_or_path!r}")


def instantiate_hf_config(cfg_dict: dict[str, Any]):
    raw = copy.deepcopy(cfg_dict)
    raw.setdefault("_attn_implementation", "sdpa")
    model_type = str(raw.get("model_type", "deepseek_v3"))
    if model_type == "deepseek_v2":
        return DeepseekV2Config(**raw)
    if model_type == "deepseek_v3":
        return DeepseekV3Config(**raw)
    if model_type == "deepseek_v32":
        return DeepseekV32Config(**raw)
    raise ValueError(f"Unsupported DeepSeek model_type: {model_type!r}")


def _resolve_model_cls(model_type: str):
    if model_type == "deepseek_v2":
        return DeepseekV2ForCausalLM
    if model_type == "deepseek_v3":
        return DeepseekV3ForCausalLM
    if model_type == "deepseek_v32":
        return DeepseekV32ForCausalLM
    raise ValueError(f"Unsupported DeepSeek model_type: {model_type!r}")


def _materialize_non_persistent_rope_buffers(model: nn.Module) -> None:
    """Ensure small non-persistent RoPE `inv_freq` / `e_score_correction_bias` buffers exist cleanly.

    Wait: keep all parameters and buffers on `meta` when `weightless=True` so `nnScaler` meta
    tracing treats the whole module uniformly (`param_init_strategy='shard'`).
    """
    return


def build_deepseek_model(
    preset_or_path: str | Path | dict[str, Any],
    *,
    rewrite_cfg: RewriteConfig | None = None,
    weightless: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[DeepseekCausalLMTaskWrapper, dict[str, Any]]:
    """Build a patched `DeepseekCausalLMTaskWrapper` on `meta` (0 RAM, 0 GPUs) or CPU."""
    cfg_dict = load_model_config_dict(preset_or_path)
    hf_cfg = instantiate_hf_config(cfg_dict)
    model_cls = _resolve_model_cls(hf_cfg.model_type)
    rw = rewrite_cfg or RewriteConfig()

    if weightless:
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device("meta"):
                hf_model = model_cls(hf_cfg)
        finally:
            torch.set_default_dtype(orig_dtype)
    else:
        hf_model = model_cls(hf_cfg).to(dtype=dtype, device="cpu")

    hf_model.train()
    patch_counts = apply_deepseek_rewrites(hf_model, rw)
    wrapper = DeepseekCausalLMTaskWrapper(hf_model, sum_ce_loss=rw.sum_ce_loss)
    wrapper.train()

    meta_info = {
        "model_class": model_cls.__name__,
        "model_type": hf_cfg.model_type,
        "weightless_meta": weightless,
        "dtype": str(dtype),
        "num_hidden_layers": int(hf_cfg.num_hidden_layers),
        "first_k_dense_replace": int(hf_cfg.first_k_dense_replace),
        "num_moe_layers": max(0, int(hf_cfg.num_hidden_layers) - int(hf_cfg.first_k_dense_replace)),
        "n_routed_experts": int(hf_cfg.n_routed_experts),
        "num_attention_heads": int(hf_cfg.num_attention_heads),
        "patch_counts": patch_counts,
        "rewrite_config": rw.to_dict(),
    }
    return wrapper, meta_info


def build_dummy_samples(
    preset_or_cfg: str | Path | dict[str, Any],
    *,
    batch_size: int = 2,
    seq_len: int = 64,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    cfg_dict = load_model_config_dict(preset_or_cfg)
    vocab_size = int(cfg_dict.get("vocab_size", 129280))
    gen = torch.Generator(device="cpu")
    gen.manual_seed(2025)

    input_ids = torch.randint(0, min(vocab_size, 1024), (batch_size, seq_len), generator=gen, dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.randint(0, min(vocab_size, 1024), (batch_size, seq_len), generator=gen, dtype=torch.long)
    valid_count = torch.full((batch_size, 1), float(seq_len - 1), dtype=torch.float32)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "valid_token_count": valid_count.to(device),
    }
