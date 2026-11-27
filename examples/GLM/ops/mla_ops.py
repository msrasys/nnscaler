# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Custom nnScaler operators for GLM MLA (`Glm4MoeLite`, `GlmMoeDsa`, `Glm5Next`) and DSA Indexer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import apply_rotary_pos_emb_interleave

from nnscaler.graph.parser.register import CustomizedOps, register_op


def glm_mla_concat_kv(k_nope: torch.Tensor, k_rot: torch.Tensor) -> torch.Tensor:
    """Concatenate per-head `k_nope` `[B, H, S, Dn]` with shared MQA `k_rot` `[B, 1, S, Dr]`."""
    k_rot_exp = k_rot.expand(-1, k_nope.shape[1], -1, -1)
    return torch.cat((k_nope, k_rot_exp), dim=-1)


def glm_mla_concat_kv_fake(k_nope: torch.Tensor, k_rot: torch.Tensor) -> torch.Tensor:
    b, h, s, dn = k_nope.shape
    dr = k_rot.shape[-1]
    return k_nope.new_empty(
        (b, h, s, dn + dr),
        requires_grad=torch.is_grad_enabled() and (k_nope.requires_grad or k_rot.requires_grad),
    )


def glm_mla_concat_kv_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 2:
        raise ValueError(f"glm_mla_concat_kv expects 2 tensor inputs, got {len(args)}")
    dn = args[0].shape[-1]
    dr = args[1].shape[-1]
    return f"b h s dn^, b 1^ s dr^ -> b h s {dn + dr}^"


@torch.no_grad()
def glm_dsa_index_mask(
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask_3d: torch.Tensor,
    wq_b_weight: torch.Tensor,
    wk_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    n_heads: int = 64,
    head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    index_topk: int = 2048,
    k_norm_eps: float = 1e-6,
) -> torch.Tensor:
    """GLM-MoE-DSA / GLM5-Next `@torch.no_grad()` Sparse Attention Indexer mask."""
    batch_size, seq_len, _ = hidden_states.shape
    q = F.linear(q_resid, wq_b_weight).view(batch_size, seq_len, n_heads, head_dim)
    q_pe, q_nope = torch.split(q, [qk_rope_head_dim, head_dim - qk_rope_head_dim], dim=-1)

    k = F.layer_norm(F.linear(hidden_states, wk_weight), (head_dim,), k_norm_weight, k_norm_bias, eps=k_norm_eps).unsqueeze(2)
    k_pe, k_nope = torch.split(k, [qk_rope_head_dim, head_dim - qk_rope_head_dim], dim=-1)

    q_pe, k_pe = apply_rotary_pos_emb_interleave(q_pe, k_pe, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pe, q_nope], dim=-1)
    k = torch.cat([k_pe, k_nope], dim=-1).squeeze(2)

    scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * (head_dim ** -0.5)
    scores = F.relu(scores)
    weights = F.linear(hidden_states.to(weights_proj_weight.dtype), weights_proj_weight).float() * (n_heads ** -0.5)
    index_score = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    if attention_mask_3d is not None:
        m = attention_mask_3d[:, None, :] if attention_mask_3d.ndim == 2 else attention_mask_3d
        if m.dtype == torch.bool:
            index_score = index_score.masked_fill(~m, float("-inf"))
        else:
            index_score = index_score + m

    topk = min(index_topk, index_score.shape[-1])
    topk_indices = index_score.topk(topk, dim=-1)[1]
    mask = torch.full_like(index_score, float("-inf"), dtype=hidden_states.dtype).scatter_(-1, topk_indices, 0.0)
    if attention_mask_3d is not None:
        m = attention_mask_3d[:, None, :] if attention_mask_3d.ndim == 2 else attention_mask_3d
        if m.dtype == torch.bool:
            mask = mask.masked_fill(~m, float("-inf"))
        else:
            mask = mask + m.to(mask.dtype)
    return mask.unsqueeze(1)


def glm_dsa_index_mask_fake(
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attention_mask_3d: torch.Tensor,
    wq_b_weight: torch.Tensor,
    wk_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    k_norm_bias: torch.Tensor,
    weights_proj_weight: torch.Tensor,
    n_heads: int = 64,
    head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    index_topk: int = 2048,
    k_norm_eps: float = 1e-6,
) -> torch.Tensor:
    b, s = hidden_states.shape[:2]
    t = attention_mask_3d.shape[-1]
    return hidden_states.new_empty((b, 1, s, t))


def glm_dsa_index_mask_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 10:
        raise ValueError(f"glm_dsa_index_mask expects 10 tensor inputs, got {len(args)}")
    b_dim = args[0].shape[0]
    cos_b = "1^" if args[2].shape[0] == 1 and b_dim != 1 else "b"
    sin_b = "1^" if args[3].shape[0] == 1 and b_dim != 1 else "b"
    mask_b = "1^" if args[4].shape[0] == 1 and b_dim != 1 else "b"
    mask_ndim = len(args[4].shape)
    mask_anno = f"{mask_b} t^" if mask_ndim == 2 else f"{mask_b} s^ t^"
    return (
        f"b s^ h^, b s^ q^, {cos_b} s^ dr^, {sin_b} s^ dr^, {mask_anno}, "
        f"nhd^ q^, dh^ h^, dh^, dh^, nh^ h^ -> b 1^ s^ t^"
    )


def register_mla_ops() -> dict[str, str]:
    """Idempotently register GLM MLA and DSA operators."""
    status: dict[str, str] = {}
    specs = (
        (glm_mla_concat_kv, glm_mla_concat_kv_anno, glm_mla_concat_kv_fake),
        (glm_dsa_index_mask, glm_dsa_index_mask_anno, glm_dsa_index_mask_fake),
    )
    for fn, anno, fake_fn in specs:
        sig = f"{fn.__module__}.{fn.__qualname__}"
        if sig in CustomizedOps.kOpMap:
            status[sig] = "already_registered"
            continue
        register_op(anno, fake_fn=fake_fn)(fn)
        status[sig] = "registered"
    return status
