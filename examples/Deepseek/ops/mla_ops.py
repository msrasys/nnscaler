# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Custom nnScaler operators for Multi-head Latent Attention (MLA) and DeepSeek-V3.2 Sparse Attention (DSA).

1. `deepseek_mla_concat_kv`:
   Replaces HF's inplace slice-`.copy_` (`key_states[..., :dn].copy_(k_nope)` / `key_states[..., dn:].copy_(k_rot)`)
   and avoids `Expand` (`1 -> num_heads`) locking the head axis with a numeric freeze (`128^`).
   Annotated `b h s dn^, b 1^ s dr^ -> b h s dk^` so `k_nope` shards cleanly along `h` (`TP`) while shared MQA
   `k_rot` (`[B, 1, S, Dr]`) stays replicated in forward and automatically All-Reduces gradients across TP in backward.

2. `deepseek_v32_dsa_index_mask`:
   Wraps `@torch.no_grad()` `DeepseekV32Indexer` + scatter boolean mask construction (`[B, 1, S, T]`), avoiding
   `nnScaler` parser errors on broadcasted 4D matmul (`[B, S, H, D] @ [B, 1, D, T]`) and dynamic `topk` / `scatter`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from nnscaler.graph.parser.register import CustomizedOps, register_op
from nnscaler.ir.cten import IRObject, IRTensor


def _unwrap_int(val: Any) -> int:
    while isinstance(val, IRObject):
        val = val.value
    return int(val)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_half_split_rope_unsqueeze2(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos_u = cos.unsqueeze(2)
    sin_u = sin.unsqueeze(2)
    q_embed = (q * cos_u) + (_rotate_half(q) * sin_u)
    k_embed = (k * cos_u) + (_rotate_half(k) * sin_u)
    return q_embed, k_embed


# ============================================================================
# 1. MLA Head-Sharding Key Concatenation (`k_nope [B, H, S, Dn]` + `k_rot [B, 1, S, Dr]`)
# ============================================================================

def deepseek_mla_concat_kv(k_nope: torch.Tensor, k_rot: torch.Tensor) -> torch.Tensor:
    """Out-of-place head-parallel concatenation of per-head `k_nope` and shared MQA `k_rot`."""
    k_rot_expanded = k_rot.expand(-1, k_nope.shape[1], -1, -1)
    return torch.cat((k_nope, k_rot_expanded), dim=-1)


def deepseek_mla_concat_kv_fake(k_nope: torch.Tensor, k_rot: torch.Tensor) -> torch.Tensor:
    out_dim = k_nope.shape[-1] + k_rot.shape[-1]
    return k_nope.new_empty((*k_nope.shape[:-1], out_dim))


def mla_concat_kv_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) != 2:
        raise ValueError(f"deepseek_mla_concat_kv expects 2 tensor inputs, got {len(args)}")
    k_nope, k_rot = args[0], args[1]
    if not isinstance(k_nope, IRTensor) or len(k_nope.shape) != 4:
        raise ValueError(f"Expected 4D k_nope [B, H, S, Dn], got {k_nope}")
    if not isinstance(k_rot, IRTensor) or len(k_rot.shape) != 4:
        raise ValueError(f"Expected 4D k_rot [B, 1, S, Dr], got {k_rot}")
    out_dim = int(k_nope.shape[-1]) + int(k_rot.shape[-1])
    return f"b h s dn^, b 1^ s dr^ -> b h s {out_dim}^"


# ============================================================================
# 2. DeepSeek-V3.2 Sparse Attention (DSA) Indexer Mask Builder
# ============================================================================

@torch.no_grad()
def deepseek_v32_dsa_index_mask(
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
    """Compute DeepSeek-V3.2 DSA boolean sparse mask `[B, 1, S, T]` (`True` = masked out)."""
    batch_size, seq_len, _ = hidden_states.shape
    q = F.linear(q_resid, wq_b_weight).view(batch_size, seq_len, n_heads, head_dim)
    q_rot, q_pass = torch.split(q, [qk_rope_head_dim, head_dim - qk_rope_head_dim], dim=-1)

    k = F.linear(hidden_states, wk_weight)
    k = F.layer_norm(k, (head_dim,), k_norm_weight, k_norm_bias, k_norm_eps).unsqueeze(2)
    k_rot, k_pass = torch.split(k, [qk_rope_head_dim, head_dim - qk_rope_head_dim], dim=-1)

    q_rot, k_rot = _apply_half_split_rope_unsqueeze2(q_rot, k_rot, cos, sin)
    q = torch.cat((q_rot, q_pass), dim=-1)
    k = torch.cat((k_rot, k_pass), dim=-1).squeeze(2)

    softmax_scale = head_dim**-0.5
    scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * softmax_scale
    scores = F.relu(scores)

    weights = F.linear(hidden_states.to(weights_proj_weight.dtype), weights_proj_weight).float() * (n_heads**-0.5)
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    if attention_mask_3d.dtype == torch.bool:
        index_scores = index_scores.masked_fill(~attention_mask_3d, float("-inf"))
    else:
        index_scores = index_scores + attention_mask_3d

    key_len = index_scores.shape[-1]
    topk = min(index_topk, key_len)
    topk_indices = index_scores.topk(topk, dim=-1).indices

    index_mask = (
        torch.ones((batch_size, seq_len, key_len), dtype=torch.bool, device=hidden_states.device)
        .scatter(-1, topk_indices.long(), False)
        .unsqueeze(1)
    )
    return index_mask


def deepseek_v32_dsa_index_mask_fake(
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
    batch_size, seq_len = hidden_states.shape[:2]
    key_len = attention_mask_3d.shape[-1]
    return torch.zeros((batch_size, 1, seq_len, key_len), dtype=torch.bool, device=hidden_states.device)


def dsa_index_mask_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 10:
        raise ValueError(f"deepseek_v32_dsa_index_mask expects 10 tensor inputs, got {len(args)}")
    b_dim = args[0].shape[0]
    cos_b = "1^" if args[2].shape[0] == 1 and b_dim != 1 else "b"
    sin_b = "1^" if args[3].shape[0] == 1 and b_dim != 1 else "b"
    mask_b = "1^" if args[4].shape[0] == 1 and b_dim != 1 else "b"
    return (
        f"b s^ h^, b s^ q^, {cos_b} s^ dr^, {sin_b} s^ dr^, {mask_b} s^ t^, "
        f"nhd^ q^, dh^ h^, dh^, dh^, nh^ h^ -> b 1^ s^ t^"
    )


def register_mla_ops() -> dict[str, str]:
    """Idempotently register DeepSeek MLA and V3.2 DSA operators."""
    status: dict[str, str] = {}
    specs = (
        (deepseek_mla_concat_kv, mla_concat_kv_anno, deepseek_mla_concat_kv_fake),
        (deepseek_v32_dsa_index_mask, dsa_index_mask_anno, deepseek_v32_dsa_index_mask_fake),
    )
    for fn, anno, fake_fn in specs:
        sig = f"{fn.__module__}.{fn.__qualname__}"
        if sig in CustomizedOps.kOpMap:
            status[sig] = "already_registered"
            continue
        register_op(anno, fake_fn=fake_fn)(fn)
        status[sig] = "registered"
    return status
