# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Gated DeltaNet wrapper operators + IRDimops annotations + CPU/meta fake_fns."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from nnscaler.graph.parser.register import CustomizedOps, register_op
from nnscaler.ir.cten import IRObject, IRTensor

try:
    from causal_conv1d import causal_conv1d_fn as _cuda_causal_conv1d_fn
except Exception:  # pragma: no cover - optional CUDA package
    _cuda_causal_conv1d_fn = None

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule
except Exception:  # pragma: no cover - optional Triton/CUDA package
    _fla_chunk_gated_delta_rule = None


def _unwrap_ir_value(value: Any) -> Any:
    if isinstance(value, IRTensor):
        return value
    if isinstance(value, IRObject):
        return _unwrap_ir_value(value.value)
    if isinstance(value, tuple):
        return tuple(_unwrap_ir_value(v) for v in value)
    if isinstance(value, list):
        return [_unwrap_ir_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _unwrap_ir_value(v) for k, v in value.items()}
    return value


def qwen3_5_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    """Depthwise causal Conv1d (`[B, C, S]`) supporting both 2D `[C, W]` and 3D `[C, 1, W]` weights."""
    if x.is_cuda and _cuda_causal_conv1d_fn is not None:
        w2d = weight.squeeze(1) if weight.dim() == 3 else weight
        return _cuda_causal_conv1d_fn(x, w2d, bias, activation=activation)

    _, channels, seq_len = x.shape
    w3d = weight.unsqueeze(1) if weight.dim() == 2 else weight
    padding = w3d.shape[-1] - 1
    out = F.conv1d(
        x.to(w3d.dtype),
        weight=w3d,
        bias=bias.to(w3d.dtype) if bias is not None else None,
        padding=padding,
        groups=channels,
    )[..., :seq_len]
    if activation in ("silu", "swish"):
        out = F.silu(out)
    elif activation is not None:
        raise ValueError(f"Unsupported causal_conv1d activation: {activation}")
    return out.to(x.dtype)


def qwen3_5_causal_conv1d_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    return torch.empty_like(
        x,
        requires_grad=torch.is_grad_enabled() and any(
            tensor is not None and tensor.requires_grad for tensor in (x, weight, bias)
        ),
    )


def causal_conv1d_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 2:
        raise ValueError(f"qwen3_5_causal_conv1d expects at least (x, weight), got {len(args)} args")
    x, weight = args[0], args[1]
    bias = args[2] if len(args) >= 3 else kwargs.get("bias", None)

    if not isinstance(x, IRTensor) or len(x.shape) != 3:
        raise ValueError(f"Expected rank-3 IRTensor x [B, C, S], got {x}")
    if not isinstance(weight, IRTensor) or len(weight.shape) not in (2, 3):
        raise ValueError(f"Expected rank-2 [C, W] or rank-3 [C, 1, W] IRTensor weight, got {weight}")
    if x.shape[1] != weight.shape[0]:
        raise ValueError(f"Channel mismatch between x {x.shape} and weight {weight.shape}")

    w_anno = "c 1 w^" if len(weight.shape) == 3 else "c w^"
    if isinstance(bias, IRTensor):
        if len(bias.shape) != 1 or bias.shape[0] != x.shape[1]:
            raise ValueError(f"Expected 1D bias matching C={x.shape[1]}, got {bias.shape}")
        return f"b c s^, {w_anno}, c -> b c s^"
    return f"b c s^, {w_anno} -> b c s^"


def _l2norm_fp32(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def _reference_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Out-of-place differentiable PyTorch reference for Gated DeltaNet (`[B, T, H, D]`)."""
    orig_dtype = v.dtype
    if v.shape[2] != q.shape[2]:
        group = v.shape[2] // q.shape[2]
        q = q.repeat_interleave(group, dim=2)
        k = k.repeat_interleave(group, dim=2)

    qf = q.transpose(1, 2).to(torch.float32)  # [B, H, T, Dk]
    kf = k.transpose(1, 2).to(torch.float32)
    vf = v.transpose(1, 2).to(torch.float32)  # [B, H, T, Dv]
    gf = g.transpose(1, 2).to(torch.float32)  # [B, H, T]
    bf = beta.transpose(1, 2).to(torch.float32)

    if use_qk_l2norm_in_kernel:
        qf = _l2norm_fp32(qf)
        kf = _l2norm_fp32(kf)
    if scale is None:
        scale = qf.shape[-1] ** -0.5
    qf = qf * float(scale)

    bsz, num_heads, seq_len, dk = qf.shape
    dv = vf.shape[-1]
    state = qf.new_zeros((bsz, num_heads, dk, dv))
    step_outputs: list[torch.Tensor] = []

    for t in range(seq_len):
        q_t = qf[:, :, t]
        k_t = kf[:, :, t]
        v_t = vf[:, :, t]
        decay_t = gf[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = bf[:, :, t].unsqueeze(-1)

        state = state * decay_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out_t = (state * q_t.unsqueeze(-1)).sum(dim=-2)
        step_outputs.append(out_t)

    out = torch.stack(step_outputs, dim=2).transpose(1, 2).contiguous()
    return out.to(orig_dtype)


def qwen3_5_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    """Chunk Gated Delta Rule (`[B, T, Hv, Dv]`) with FLA Triton fast-path & CPU fallback."""
    if q.is_cuda and _fla_chunk_gated_delta_rule is not None:
        out, _ = _fla_chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        return out
    return _reference_gated_delta_rule(
        q,
        k,
        v,
        g=g,
        beta=beta,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def qwen3_5_chunk_gated_delta_rule_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> torch.Tensor:
    return torch.empty_like(
        v, requires_grad=torch.is_grad_enabled() and any(tensor.requires_grad for tensor in (q, k, v, g, beta)),
    )


def chunk_gated_delta_rule_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 5:
        raise ValueError(
            f"qwen3_5_chunk_gated_delta_rule expects 5 tensor inputs (q, k, v, g, beta), got {len(args)}"
        )
    q, k, v, g, beta = args[:5]
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, IRTensor) or len(tensor.shape) != 4:
            raise ValueError(f"Expected rank-4 IRTensor {name} [B, T, H, D], got {tensor}")
    for name, tensor in (("g", g), ("beta", beta)):
        if not isinstance(tensor, IRTensor) or len(tensor.shape) != 3:
            raise ValueError(f"Expected rank-3 IRTensor {name} [B, T, H], got {tensor}")

    if tuple(q.shape) != tuple(k.shape):
        raise ValueError(f"Expected matching q and k shapes, got {q.shape} vs {k.shape}")
    if q.shape[:2] != v.shape[:2] or g.shape[:2] != v.shape[:2] or tuple(g.shape) != tuple(beta.shape):
        raise ValueError(
            f"Mismatched [B, T] dims across q={q.shape}, v={v.shape}, g={g.shape}, beta={beta.shape}"
        )
    if g.shape[2] != v.shape[2]:
        raise ValueError(f"Expected g/beta heads ({g.shape[2]}) == v heads ({v.shape[2]})")

    if q.shape[2] == v.shape[2]:
        hk_anno = hv_anno = "h"
    elif v.shape[2] % q.shape[2] == 0:
        group_size = v.shape[2] // q.shape[2]
        hk_anno = "hk"
        hv_anno = f"(hk {group_size})"
    else:
        raise ValueError(f"v heads ({v.shape[2]}) must be divisible by q/k heads ({q.shape[2]})")

    return (
        f"b t^ {hk_anno} dk^, b t^ {hk_anno} dk^, b t^ {hv_anno} dv^, "
        f"b t^ {hv_anno}, b t^ {hv_anno} -> b t^ {hv_anno} dv^"
    )


def register_gdn_ops() -> dict[str, str]:
    """Idempotently register GDN wrapper ops with CPU/meta fake_fns."""
    status: dict[str, str] = {}
    ops_spec = (
        (
            qwen3_5_causal_conv1d,
            causal_conv1d_anno,
            qwen3_5_causal_conv1d_fake,
        ),
        (
            qwen3_5_chunk_gated_delta_rule,
            chunk_gated_delta_rule_anno,
            qwen3_5_chunk_gated_delta_rule_fake,
        ),
    )
    for fn, anno, fake_fn in ops_spec:
        sig = f"{fn.__module__}.{fn.__name__}"
        if CustomizedOps.exist(sig):
            status[sig] = "already_registered"
        else:
            register_op(anno, fake_fn=fake_fn)(fn)
            status[sig] = "registered"
    return status
