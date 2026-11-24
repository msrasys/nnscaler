# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stage-3 Multimodal Qwen3.5 vision-language custom operators + fake_fns."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from nnscaler.graph.parser.register import CustomizedOps, register_op
from nnscaler.ir.cten import IRTensor


def qwen3_5_vision_varlen_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scaling: float = 0.125,
) -> torch.Tensor:
    """Packed variable-length bidirectional ViT attention (`q, k, v: [S, H, D]`)."""
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).detach().cpu().tolist()
    qt = q.transpose(0, 1)  # [H, S, D]
    kt = k.transpose(0, 1)
    vt = v.transpose(0, 1)
    q_splits = torch.split(qt, lengths, dim=1)
    k_splits = torch.split(kt, lengths, dim=1)
    v_splits = torch.split(vt, lengths, dim=1)

    outs: list[torch.Tensor] = []
    for qc, kc, vc in zip(q_splits, k_splits, v_splits):
        scores = torch.matmul(qc, kc.transpose(-1, -2)) * float(scaling)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(qc.dtype)
        outs.append(torch.matmul(probs, vc))
    merged = torch.cat(outs, dim=1)  # [H, S, D]
    return merged.transpose(0, 1).contiguous()


def qwen3_5_vision_varlen_attn_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scaling: float = 0.125,
) -> torch.Tensor:
    return torch.empty_like(q)


def vision_varlen_attn_anno(*args: Any, **kwargs: Any) -> str:
    q, k, v, cu_seqlens = args[:4]
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, IRTensor) or len(t.shape) != 3:
            raise ValueError(f"Expected 3D {name} [S, H, D], got {t}")
    if not isinstance(cu_seqlens, IRTensor) or len(cu_seqlens.shape) != 1:
        raise ValueError(f"Expected 1D cu_seqlens, got {cu_seqlens}")
    # Head axis `h` is freely partitionable for ViT Tensor Parallelism!
    return "s^ h d^, s^ h d^, s^ h d^, cu^ -> s^ h d^"


def qwen3_5_masked_scatter(
    inputs_embeds: torch.Tensor,
    mask: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    """Scatter multimodal `[N_vis, H]` features into `[B, S, H]` token embeddings."""
    expanded_mask = mask.expand_as(inputs_embeds)
    return inputs_embeds.masked_scatter(expanded_mask, features.to(inputs_embeds.dtype))


def qwen3_5_masked_scatter_fake(
    inputs_embeds: torch.Tensor,
    mask: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(inputs_embeds)


def masked_scatter_anno(*args: Any, **kwargs: Any) -> str:
    # Partitionable along hidden_size `h`!
    return "b^ s^ h, b^ s^ 1, v^ h -> b^ s^ h"


def register_vision_ops() -> dict[str, str]:
    status: dict[str, str] = {}
    specs = (
        (qwen3_5_vision_varlen_attn, vision_varlen_attn_anno, qwen3_5_vision_varlen_attn_fake),
        (qwen3_5_masked_scatter, masked_scatter_anno, qwen3_5_masked_scatter_fake),
    )
    for fn, anno, fake_fn in specs:
        sig = f"{fn.__module__}.{fn.__name__}"
        if CustomizedOps.exist(sig):
            status[sig] = "already_registered"
        else:
            register_op(anno, fake_fn=fake_fn)(fn)
            status[sig] = "registered"
    return status
