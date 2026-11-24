"""Official Qwen3.5 model family architecture presets (`0.8B` .. `397B-A17B` + VL + micro test presets).

Allows zero-network, zero-GPU, zero-checkpoint compilation of any Qwen3.5 model directly on CPU `meta` tensors.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def _layer_types(num_layers: int, interval: int = 4) -> list[str]:
    return [
        "full_attention" if (idx + 1) % interval == 0 else "linear_attention"
        for idx in range(num_layers)
    ]


_DEFAULT_VISION_CFG: dict[str, Any] = {
    "depth": 27,
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_heads": 16,
    "in_channels": 3,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 1024,
    "num_position_embeddings": 2304,
}


QWEN3_5_PRESETS: dict[str, dict[str, Any]] = {
    # ---------------- Micro unit-test presets (fast CPU eager & compile parity) ----------------
    "micro-dense": {
        "model_type": "qwen3_5_text",
        "vocab_size": 512,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "layer_types": _layer_types(4, 4),
        "tie_word_embeddings": True,
    },
    "micro-moe": {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 512,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 32,
        "linear_value_head_dim": 32,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64,
        "num_experts_per_tok": 2,
        "num_experts": 8,
        "layer_types": _layer_types(4, 4),
        "tie_word_embeddings": False,
    },
    "micro-vl": {
        "model_type": "qwen3_5",
        "text_config": {
            "vocab_size": 512,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "layer_types": _layer_types(4, 4),
            "tie_word_embeddings": True,
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 128,
            "num_position_embeddings": 64,
        },
        "image_token_id": 500,
        "video_token_id": 501,
        "vision_start_token_id": 502,
        "vision_end_token_id": 503,
        "tie_word_embeddings": True,
    },
    # ---------------- Stage 1: Official Qwen3.5 Dense Models ----------------
    "qwen3.5-0.8b": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "layer_types": _layer_types(24, 4),
        "tie_word_embeddings": True,
    },
    "qwen3.5-2b": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": 24,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "layer_types": _layer_types(24, 4),
        "tie_word_embeddings": True,
    },
    "qwen3.5-4b": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 2560,
        "intermediate_size": 9216,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "layer_types": _layer_types(32, 4),
        "tie_word_embeddings": True,
    },
    "qwen3.5-9b": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "num_hidden_layers": 32,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "layer_types": _layer_types(32, 4),
        "tie_word_embeddings": False,
    },
    "qwen3.5-27b": {
        "model_type": "qwen3_5_text",
        "vocab_size": 248320,
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "layer_types": _layer_types(64, 4),
        "tie_word_embeddings": False,
    },
    # ---------------- Stage 2: Official Qwen3.5 Sparse MoE Models ----------------
    "qwen3.5-35b-a3b": {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 248320,
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "num_experts_per_tok": 8,
        "num_experts": 256,
        "layer_types": _layer_types(40, 4),
        "tie_word_embeddings": False,
    },
    "qwen3.5-122b-a10b": {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 248320,
        "hidden_size": 3072,
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "num_experts_per_tok": 8,
        "num_experts": 256,
        "layer_types": _layer_types(48, 4),
        "tie_word_embeddings": False,
    },
    "qwen3.5-397b-a17b": {
        "model_type": "qwen3_5_moe_text",
        "vocab_size": 248320,
        "hidden_size": 4096,
        "num_hidden_layers": 60,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 64,
        "moe_intermediate_size": 1024,
        "shared_expert_intermediate_size": 1024,
        "num_experts_per_tok": 8,
        "num_experts": 512,
        "layer_types": _layer_types(60, 4),
        "tie_word_embeddings": False,
    },
    # ---------------- Stage 3: Multimodal Qwen3.5-VL Preset ----------------
    "qwen3.5-0.8b-vl": {
        "model_type": "qwen3_5",
        "text_config": {
            "vocab_size": 248320,
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 16,
            "layer_types": _layer_types(24, 4),
            "tie_word_embeddings": True,
        },
        "vision_config": copy.deepcopy(_DEFAULT_VISION_CFG),
        "tie_word_embeddings": True,
    },
}


def list_model_presets() -> list[str]:
    return sorted(QWEN3_5_PRESETS.keys())


def load_model_config_dict(preset_or_path: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load a Qwen3.5 configuration dictionary from a preset name, JSON file path, or dict."""
    if isinstance(preset_or_path, dict):
        return copy.deepcopy(preset_or_path)

    key = str(preset_or_path).strip().lower()
    if key in QWEN3_5_PRESETS:
        return copy.deepcopy(QWEN3_5_PRESETS[key])

    path = Path(preset_or_path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    if path.is_dir() and (path / "config.json").is_file():
        with (path / "config.json").open("r", encoding="utf-8") as fh:
            return json.load(fh)

    raise ValueError(
        f"Unknown model preset or config path {preset_or_path!r}. Available presets: {list_model_presets()}"
    )
