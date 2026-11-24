# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Custom nnScaler operators for DeepSeek MLA, V3.2 Sparse Attention (DSA), and Sparse MoE."""

from examples.Deepseek.ops.mla_ops import (
    deepseek_mla_concat_kv,
    deepseek_v32_dsa_index_mask,
    register_mla_ops,
)
from examples.Deepseek.ops.moe_ops import (
    deepseek_grouped_linear,
    deepseek_moe_grouped_gemm,
    deepseek_moe_permute,
    deepseek_moe_unpermute,
    deepseek_v2_moe_route,
    deepseek_v3_moe_route,
    register_moe_ops,
)


def register_all_deepseek_ops() -> dict[str, str]:
    status: dict[str, str] = {}
    status.update(register_mla_ops())
    status.update(register_moe_ops())
    return status


__all__ = [
    "deepseek_mla_concat_kv",
    "deepseek_v32_dsa_index_mask",
    "deepseek_v2_moe_route",
    "deepseek_v3_moe_route",
    "deepseek_moe_grouped_gemm",
    "deepseek_moe_permute",
    "deepseek_grouped_linear",
    "deepseek_moe_unpermute",
    "register_mla_ops",
    "register_moe_ops",
    "register_all_deepseek_ops",
]
