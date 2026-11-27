# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model configuration presets and weightless (`meta`) builders for the GLM family."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    Glm4Config,
    Glm4ForCausalLM,
    Glm4MoeConfig,
    Glm4MoeForCausalLM,
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
    Glm5NextConfig,
    Glm5NextForConditionalGeneration,
    GlmMoeDsaConfig,
    GlmMoeDsaForCausalLM,
)


def list_presets() -> list[str]:
    return [
        "micro-glm4",
        "glm4-9b",
        "micro-glm4-moe",
        "glm4-moe-355b",
        "micro-glm4-moe-lite",
        "micro-glm-moe-dsa",
        "glm5-744b-dsa",
        "micro-glm5-next",
    ]


def build_glm_config(
    preset: str,
    *,
    num_layers_override: int | None = None,
) -> Any:
    key = preset.lower().strip()
    if key == "micro-glm4":
        return Glm4Config(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=num_layers_override or 2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            pad_token_id=0,
            eos_token_id=2,
            use_cache=False,
        )

    if key == "glm4-9b":
        return Glm4Config(
            num_hidden_layers=num_layers_override or 40,
            use_cache=False,
        )

    if key == "micro-glm4-moe":
        return Glm4MoeConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=num_layers_override or 3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            n_shared_experts=1,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            pad_token_id=0,
            eos_token_id=2,
            use_cache=False,
        )

    if key == "glm4-moe-355b":
        return Glm4MoeConfig(
            num_hidden_layers=num_layers_override or 46,
            use_cache=False,
        )

    if key == "micro-glm4-moe-lite":
        return Glm4MoeLiteConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=num_layers_override or 3,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            kv_lora_rank=32,
            q_lora_rank=48,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=16,
            pad_token_id=0,
            eos_token_id=2,
            use_cache=False,
        )

    if key == "micro-glm-moe-dsa":
        return GlmMoeDsaConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=num_layers_override or 3,
            num_attention_heads=4,
            num_key_value_heads=4,
            n_shared_experts=1,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_group=1,
            topk_group=1,
            first_k_dense_replace=1,
            kv_lora_rank=32,
            q_lora_rank=48,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=16,
            index_n_heads=4,
            index_head_dim=32,
            index_topk=16,
            pad_token_id=0,
            eos_token_id=2,
            use_cache=False,
        )

    if key == "glm5-744b-dsa":
        return GlmMoeDsaConfig(
            num_hidden_layers=num_layers_override or 78,
            use_cache=False,
        )

    if key == "micro-glm5-next":
        n_layers = num_layers_override or 4
        layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n_layers)]
        mlp_types = ["dense" if i == 0 else "sparse" for i in range(n_layers)]
        return Glm5NextConfig(
            text_config={
                "vocab_size": 512,
                "hidden_size": 128,
                "intermediate_size": 256,
                "moe_intermediate_size": 128,
                "num_hidden_layers": n_layers,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "n_shared_experts": 1,
                "n_routed_experts": 8,
                "num_experts_per_tok": 2,
                "n_group": 1,
                "topk_group": 1,
                "first_k_dense_replace": 1,
                "mlp_layer_types": mlp_types,
                "kv_lora_rank": 32,
                "q_lora_rank": 48,
                "qk_nope_head_dim": 16,
                "qk_rope_head_dim": 0,
                "v_head_dim": 16,
                "index_n_heads": 4,
                "index_head_dim": 32,
                "index_topk": 16,
                "linear_num_key_heads": 4,
                "linear_num_value_heads": 4,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "layer_types": layer_types,
                "pad_token_id": 0,
                "eos_token_id": 2,
                "use_cache": False,
            },
            vision_config={
                "hidden_size": 64,
                "intermediate_size": 128,
                "depth": 2,
                "num_heads": 4,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "out_hidden_size": 128,
            },
            use_cache=False,
        )

    raise ValueError(f"Unknown GLM preset: {preset!r}. Expected one of {list_presets()}")


def build_glm_model(
    preset: str,
    *,
    weightless: bool = True,
    dtype: torch.dtype = torch.float32,
    num_layers_override: int | None = None,
) -> nn.Module:
    cfg = build_glm_config(preset, num_layers_override=num_layers_override)
    cls_map: dict[type, type[nn.Module]] = {
        Glm4Config: Glm4ForCausalLM,
        Glm4MoeConfig: Glm4MoeForCausalLM,
        Glm4MoeLiteConfig: Glm4MoeLiteForCausalLM,
        GlmMoeDsaConfig: GlmMoeDsaForCausalLM,
        Glm5NextConfig: Glm5NextForConditionalGeneration,
    }
    cls = cls_map[type(cfg)]
    ctx = torch.device("meta") if weightless else nullcontext()
    with ctx:
        model = cls(cfg)
    if not weightless:
        model = model.to(dtype=dtype)
    return model


def build_glm_sample(
    preset: str,
    *,
    batch_size: int = 2,
    seq_len: int = 32,
) -> dict[str, torch.Tensor]:
    cfg = build_glm_config(preset)
    vocab_size = cfg.text_config.vocab_size if isinstance(cfg, Glm5NextConfig) else cfg.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = input_ids.clone()
    valid_token_count = torch.full((batch_size, 1), seq_len - 1, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "valid_token_count": valid_token_count,
    }
