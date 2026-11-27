# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Model configuration presets and weightless (`meta`) builders for Kimi models."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
from transformers import KimiLinearConfig, KimiLinearForCausalLM
from transformers.models.deepseek_v3 import DeepseekV3Config
from transformers.models.kimi_k25 import Kimi_K25Config, Kimi_K25ForConditionalGeneration


def list_presets() -> list[str]:
    return [
        "micro-kimi-linear",
        "kimi-linear-48b",
        "micro-kimi-k25",
        "kimi-k2.5",
    ]


def build_kimi_config(
    preset: str,
    *,
    num_layers_override: int | None = None,
) -> Any:
    key = preset.lower().strip()
    if key == "micro-kimi-linear":
        n_layers = num_layers_override or 4
        layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n_layers)]
        mlp_types = ["dense" if i == 0 else "sparse" for i in range(n_layers)]
        return KimiLinearConfig(
            vocab_size=512,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=n_layers,
            num_attention_heads=4,
            num_key_value_heads=4,
            num_shared_experts=1,
            kv_lora_rank=32,
            q_lora_rank=None,
            qk_rope_head_dim=16,
            v_head_dim=32,
            qk_nope_head_dim=32,
            num_expert_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            num_experts=8,
            linear_head_dim=16,
            linear_num_heads=4,
            linear_conv_kernel_dim=4,
            layer_types=layer_types,
            mlp_layer_types=mlp_types,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=False,
        )

    if key == "kimi-linear-48b":
        n_layers = num_layers_override or 27
        layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n_layers)]
        mlp_types = ["dense" if i == 0 else "sparse" for i in range(n_layers)]
        return KimiLinearConfig(
            num_hidden_layers=n_layers,
            layer_types=layer_types,
            mlp_layer_types=mlp_types,
            use_cache=False,
        )

    if key == "micro-kimi-k25":
        n_layers = num_layers_override or 2
        return Kimi_K25Config(
            text_config=DeepseekV3Config(
                vocab_size=512,
                hidden_size=128,
                intermediate_size=256,
                moe_intermediate_size=128,
                num_hidden_layers=n_layers,
                num_attention_heads=4,
                num_key_value_heads=4,
                n_shared_experts=1,
                n_routed_experts=8,
                num_experts_per_tok=2,
                n_group=2,
                topk_group=1,
                first_k_dense_replace=1,
                kv_lora_rank=32,
                q_lora_rank=48,
                qk_nope_head_dim=16,
                qk_rope_head_dim=16,
                v_head_dim=16,
                use_cache=False,
            ),
            vision_config={
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "patch_size": 14,
                "merge_kernel_size": [2, 2],
                "pos_emb_height": 8,
                "pos_emb_width": 8,
                "pos_emb_time": 2,
            },
            projection_hidden_size=64,
            use_cache=False,
        )

    if key == "kimi-k2.5":
        n_layers = num_layers_override or 61
        return Kimi_K25Config(
            text_config=DeepseekV3Config(
                vocab_size=163840,
                hidden_size=7168,
                intermediate_size=18432,
                moe_intermediate_size=2048,
                num_hidden_layers=n_layers,
                num_attention_heads=64,
                num_key_value_heads=64,
                n_shared_experts=1,
                n_routed_experts=384,
                num_experts_per_tok=8,
                n_group=8,
                topk_group=4,
                first_k_dense_replace=1,
                kv_lora_rank=512,
                q_lora_rank=1536,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                use_cache=False,
            ),
            vision_config={
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "patch_size": 14,
                "merge_kernel_size": [2, 2],
                "pos_emb_height": 64,
                "pos_emb_width": 64,
                "pos_emb_time": 4,
            },
            use_cache=False,
        )

    raise ValueError(f"Unknown Kimi preset: {preset!r}. Expected one of {list_presets()}")


def build_kimi_model(
    preset: str,
    *,
    weightless: bool = True,
    dtype: torch.dtype = torch.float32,
    num_layers_override: int | None = None,
) -> nn.Module:
    cfg = build_kimi_config(preset, num_layers_override=num_layers_override)
    if isinstance(cfg, KimiLinearConfig):
        cls: type[nn.Module] = KimiLinearForCausalLM
    elif isinstance(cfg, Kimi_K25Config):
        cls = Kimi_K25ForConditionalGeneration
    else:
        raise TypeError(f"Unsupported config type: {type(cfg)}")

    ctx = torch.device("meta") if weightless else nullcontext()
    with ctx:
        model = cls(cfg)
    if not weightless:
        model = model.to(dtype=dtype)
    return model


def build_kimi_sample(
    preset: str,
    *,
    batch_size: int = 2,
    seq_len: int = 32,
    multimodal: bool = False,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    cfg = build_kimi_config(preset)
    vocab_size = cfg.text_config.vocab_size if isinstance(cfg, Kimi_K25Config) else cfg.vocab_size
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = input_ids.clone()
    valid_token_count = torch.full((batch_size, 1), seq_len - 1, dtype=torch.long)
    sample: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "valid_token_count": valid_token_count,
    }
    if multimodal and isinstance(cfg, Kimi_K25Config):
        vcfg = cfg.vision_config
        kh, kw = vcfg.merge_kernel_size
        # 1 image of grid (t=1, h=4, w=4) -> 16 patches -> 4 merged tokens per batch item
        t, h, w = 1, 4, 4
        n_merged = (h // kh) * (w // kw)
        total_patches = batch_size * t * h * w
        pixel_values = torch.randn((total_patches, 3, vcfg.patch_size, vcfg.patch_size), dtype=dtype)
        image_grid_thw = torch.tensor([[t, h, w]] * batch_size, dtype=torch.long)
        # Place `n_merged` image tokens at start of each sequence
        for b in range(batch_size):
            input_ids[b, :n_merged] = cfg.image_token_id
            labels[b, :n_merged] = -100
        valid_token_count = (labels[:, 1:] != -100).sum(dim=1, keepdim=True)
        sample["input_ids"] = input_ids
        sample["labels"] = labels
        sample["valid_token_count"] = valid_token_count
        sample["pixel_values"] = pixel_values
        sample["image_grid_thw"] = image_grid_thw
    return sample
