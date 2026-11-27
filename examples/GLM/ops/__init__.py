# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .kda_ops import (
    glm_causal_conv1d,
    glm_chunk_kda,
    register_kda_ops,
)
from .mla_ops import (
    glm_dsa_index_mask,
    glm_mla_concat_kv,
    register_mla_ops,
)
from .moe_ops import (
    glm_grouped_linear,
    glm_moe_grouped_gemm,
    glm_moe_permute,
    glm_moe_route,
    glm_moe_unpermute,
    register_moe_ops,
)


def register_glm_ops() -> dict[str, str]:
    """Idempotently register all custom GLM operators (MLA, DSA, KDA, MoE)."""
    status: dict[str, str] = {}
    status.update(register_mla_ops())
    status.update(register_kda_ops())
    status.update(register_moe_ops())
    return status


__all__ = [
    "glm_mla_concat_kv",
    "glm_dsa_index_mask",
    "glm_causal_conv1d",
    "glm_chunk_kda",
    "glm_moe_route",
    "glm_moe_grouped_gemm",
    "glm_moe_permute",
    "glm_grouped_linear",
    "glm_moe_unpermute",
    "register_mla_ops",
    "register_kda_ops",
    "register_moe_ops",
    "register_glm_ops",
]
