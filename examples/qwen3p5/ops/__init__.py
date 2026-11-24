# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Custom nnScaler operator registrations (with CPU/meta fake_fn) for Qwen3.5."""

from .gdn_ops import (
    qwen3_5_causal_conv1d,
    qwen3_5_chunk_gated_delta_rule,
    register_gdn_ops,
)
from .moe_ops import (
    qwen3_5_grouped_linear,
    qwen3_5_moe_grouped_gemm,
    qwen3_5_moe_permute,
    qwen3_5_moe_route,
    qwen3_5_moe_unpermute,
    register_moe_ops,
)
from .vision_ops import (
    qwen3_5_masked_scatter,
    qwen3_5_vision_varlen_attn,
    register_vision_ops,
)


def register_all_qwen3_5_ops() -> dict[str, str]:
    status: dict[str, str] = {}
    status.update(register_gdn_ops())
    status.update(register_moe_ops())
    status.update(register_vision_ops())
    return status


__all__ = [
    "qwen3_5_causal_conv1d",
    "qwen3_5_chunk_gated_delta_rule",
    "qwen3_5_moe_route",
    "qwen3_5_moe_grouped_gemm",
    "qwen3_5_moe_permute",
    "qwen3_5_grouped_linear",
    "qwen3_5_moe_unpermute",
    "qwen3_5_vision_varlen_attn",
    "qwen3_5_masked_scatter",
    "register_gdn_ops",
    "register_moe_ops",
    "register_vision_ops",
    "register_all_qwen3_5_ops",
]
