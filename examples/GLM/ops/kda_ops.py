# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Custom nnScaler operators for GLM-5-Next KDA (`Glm5NextTextLinearAttention`)."""

from __future__ import annotations

import inspect
from typing import Any

import torch
import torch.nn.functional as F
from transformers.models.glm5_next.modeling_glm5_next import chunk_kimi_delta_attention

from nnscaler.graph.parser.register import CustomizedOps, register_op

from examples._kernel_utils import fla_kernel


_reference_chunk_kda = inspect.unwrap(chunk_kimi_delta_attention)


def glm_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation: str | None = "silu",
) -> torch.Tensor:
    """Depthwise causal 1D convolution on `[B, H, D, S]` with weight `[H, D, K]`."""
    b, h, d, seq_len = x.shape
    channels = h * d
    x_3d = x.reshape(b, channels, seq_len)
    w_2d = weight.reshape(channels, weight.shape[-1])
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        x_3d.to(weight.dtype),
        weight=w_2d.unsqueeze(1),
        bias=None,
        padding=padding,
        groups=channels,
    )[:, :, :seq_len]
    if activation == "silu":
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported activation in glm_causal_conv1d: {activation}")
    return out.to(x.dtype).view(b, h, d, seq_len)


def glm_causal_conv1d_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    activation: str | None = "silu",
) -> torch.Tensor:
    return torch.empty_like(
        x, requires_grad=torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad),
    )


def glm_causal_conv1d_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 2:
        raise ValueError(f"glm_causal_conv1d expects 2 tensor inputs, got {len(args)}")
    return "b h d^ s^, h d^ k^ -> b h d^ s^"


def glm_chunk_kda(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Chunked Kimi Delta Attention `[B, S, H, D]` -> `[B, S, H, Dv]`."""
    if (
        query.ndim != 4 or query.shape != key.shape or query.shape != g.shape
        or value.shape[:3] != query.shape[:3] or beta.shape != query.shape[:3]
    ):
        raise ValueError("KDA expects matching Q/K/g [B,S,H,D], V [B,S,H,Dv], and beta [B,S,H]")
    kernel = fla_kernel("kda", query, key, value)
    if kernel is not None and g.device == beta.device == query.device:
        query, key, value = (tensor.contiguous() for tensor in (query, key, value))
        g, beta = g.float().contiguous(), beta.contiguous()
    else:
        kernel = _reference_chunk_kda
    out, _ = kernel(
        query, key, value, g=g, beta=beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return out


def glm_chunk_kda_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    b, s, h, _ = query.shape
    dv = value.shape[-1]
    return query.new_empty(
        (b, s, h, dv),
        requires_grad=torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (query, key, value, g, beta)
        ),
    )


def glm_chunk_kda_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 5:
        raise ValueError(f"glm_chunk_kda expects 5 tensor inputs, got {len(args)}")
    return "b s^ h d^, b s^ h d^, b s^ h dv^, b s^ h d^, b s^ h -> b s^ h dv^"


def register_kda_ops() -> dict[str, str]:
    """Idempotently register GLM-5-Next KDA operators."""
    status: dict[str, str] = {}
    specs = (
        (glm_causal_conv1d, glm_causal_conv1d_anno, glm_causal_conv1d_fake),
        (glm_chunk_kda, glm_chunk_kda_anno, glm_chunk_kda_fake),
    )
    for fn, anno, fake_fn in specs:
        sig = f"{fn.__module__}.{fn.__qualname__}"
        if sig in CustomizedOps.kOpMap:
            status[sig] = "already_registered"
            continue
        register_op(anno, fake_fn=fake_fn)(fn)
        status[sig] = "registered"
    return status
