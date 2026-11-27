# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Custom nnScaler operator for Multi-Head Latent Attention (MLA) KV concatenation."""

from __future__ import annotations

from typing import Any

import torch

from nnscaler.graph.parser.register import CustomizedOps, register_op


def kimi_mla_concat_kv(k_nope: torch.Tensor, k_rot: torch.Tensor) -> torch.Tensor:
    """Concatenate per-head `k_nope` `[B, H, S, Dn]` with shared MQA `k_rot` `[B, 1, S, Dr]`."""
    k_rot_exp = k_rot.expand(-1, k_nope.shape[1], -1, -1)
    return torch.cat((k_nope, k_rot_exp), dim=-1)


def kimi_mla_concat_kv_fake(k_nope: torch.Tensor, k_rot: torch.Tensor) -> torch.Tensor:
    b, h, s, dn = k_nope.shape
    dr = k_rot.shape[-1]
    return k_nope.new_empty(
        (b, h, s, dn + dr),
        requires_grad=torch.is_grad_enabled() and (k_nope.requires_grad or k_rot.requires_grad),
    )


def mla_concat_kv_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 2:
        raise ValueError(f"kimi_mla_concat_kv expects 2 tensor inputs, got {len(args)}")
    dn = args[0].shape[-1]
    dr = args[1].shape[-1]
    return f"b h s dn^, b 1^ s dr^ -> b h s {dn + dr}^"


def register_mla_ops() -> dict[str, str]:
    """Idempotently register Kimi MLA operator."""
    sig = f"{kimi_mla_concat_kv.__module__}.{kimi_mla_concat_kv.__qualname__}"
    if sig in CustomizedOps.kOpMap:
        return {sig: "already_registered"}
    register_op(mla_concat_kv_anno, fake_fn=kimi_mla_concat_kv_fake)(kimi_mla_concat_kv)
    return {sig: "registered"}
