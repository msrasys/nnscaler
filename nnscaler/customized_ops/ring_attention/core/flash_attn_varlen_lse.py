from typing import Optional, Tuple

import torch
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

from .utils import get_default_args

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_cute_varlen_func
except ImportError:
    flash_attn_cute_varlen_func = None

try:
    from flash_attn.cute.interface import _flash_attn_bwd
except ImportError:
    _flash_attn_bwd = None


def _raise_missing_dlse(backend: str):
    raise RuntimeError(
        f"attention z-loss requires a FlashAttention {backend} backward with dlse support")


class FlashAttnVarlenLSEFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        use_cute: bool,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        if use_cute:
            assert flash_attn_cute_varlen_func is not None, "flash_attn.cute is not available"
            cute_window_size = tuple(None if w == -1 else w for w in window_size)
            out, softmax_lse = flash_attn_cute_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=cute_window_size,
                deterministic=deterministic,
                return_lse=True,
            )
        else:
            out, softmax_lse, _ = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )

        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.use_cute = use_cute
        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlse: Optional[torch.Tensor]):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        if ctx.use_cute:
            assert _flash_attn_bwd is not None, "flash_attn.cute is not available"
            cute_window_size = tuple(None if w == -1 else w for w in ctx.window_size)
            params = get_default_args(_flash_attn_bwd).copy()
            params.update({
                "dout": dout,
                "q": q,
                "k": k,
                "v": v,
                "out": out,
                "lse": softmax_lse,
                "dq": dq,
                "dk": dk,
                "dv": dv,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": ctx.max_seqlen_q,
                "max_seqlen_k": ctx.max_seqlen_k,
                "softmax_scale": ctx.softmax_scale,
                "causal": ctx.causal,
                "window_size_left": cute_window_size[0],
                "window_size_right": cute_window_size[1],
                "deterministic": ctx.deterministic,
            })
            if "dlse" in params:
                params["dlse"] = dlse
            elif dlse is not None:
                _raise_missing_dlse("cute")
            _flash_attn_bwd(**params)
            return (dq, dk, dv) + (None,) * 11

        params = get_default_args(_flash_attn_varlen_backward).copy()
        params.update({
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": softmax_lse,
            "dq": dq,
            "dk": dk,
            "dv": dv,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": ctx.max_seqlen_q,
            "max_seqlen_k": ctx.max_seqlen_k,
            "dropout_p": ctx.dropout_p,
            "softmax_scale": ctx.softmax_scale,
            "causal": ctx.causal,
            "alibi_slopes": ctx.alibi_slopes,
            "deterministic": ctx.deterministic,
        })
        if "window_size" in params:
            params["window_size"] = ctx.window_size
        else:
            params["window_size_left"] = ctx.window_size[0]
            params["window_size_right"] = ctx.window_size[1]
        if "dlse" in params:
            params["dlse"] = dlse
        elif dlse is not None:
            _raise_missing_dlse("varlen")
        _flash_attn_varlen_backward(**params)
        return (dq, dk, dv) + (None,) * 11


def flash_attn_varlen_lse_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    use_cute: bool = False,
):
    return FlashAttnVarlenLSEFunc.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale, causal, window_size, alibi_slopes,
        deterministic, use_cute)
