# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .kda_ops import (
    kimi_causal_conv1d,
    kimi_chunk_kda,
    register_kda_ops,
)
from .mla_ops import (
    kimi_mla_concat_kv,
    register_mla_ops,
)
from .moe_ops import (
    kimi_grouped_linear,
    kimi_moe_grouped_gemm,
    kimi_moe_permute,
    kimi_moe_route,
    kimi_moe_unpermute,
    register_moe_ops,
)
from .vision_ops import (
    kimi_k25_embed_scatter,
    kimi_k25_vision_meta_indices,
    kimi_k25_vision_pos_emb,
    kimi_k25_vision_varlen_sdpa,
    register_vision_ops,
)


def register_kimi_ops() -> dict[str, str]:
    """Idempotently register all custom Kimi operators (KDA, MLA, MoE, Vision)."""
    status: dict[str, str] = {}
    status.update(register_kda_ops())
    status.update(register_mla_ops())
    status.update(register_moe_ops())
    status.update(register_vision_ops())
    return status


__all__ = [
    "kimi_causal_conv1d",
    "kimi_chunk_kda",
    "kimi_mla_concat_kv",
    "kimi_moe_route",
    "kimi_moe_grouped_gemm",
    "kimi_moe_permute",
    "kimi_grouped_linear",
    "kimi_moe_unpermute",
    "kimi_k25_vision_varlen_sdpa",
    "kimi_k25_embed_scatter",
    "kimi_k25_vision_pos_emb",
    "kimi_k25_vision_meta_indices",
    "register_kda_ops",
    "register_mla_ops",
    "register_moe_ops",
    "register_vision_ops",
    "register_kimi_ops",
]
