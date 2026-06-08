#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import dataclass
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.distributed as dist
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

@dataclass
class ZigZagAllGatherVarlenMetadata:
    cu_seqlens_q_front: torch.Tensor
    cu_seqlens_k_front: torch.Tensor
    cu_seqlens_q_end: torch.Tensor
    cu_seqlens_k_end: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k_front: int
    max_seqlen_k_end: int
    q_front_idx: torch.Tensor
    q_end_idx: torch.Tensor


_METADATA_CACHE_MAXSIZE = 128
_METADATA_CACHE = OrderedDict()


def _build_branch_cu_seqlens(
    q_lens: torch.Tensor,
    active_k_lens: torch.Tensor,
    full_seq_lens: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    zeros = torch.zeros_like(q_lens, dtype=dtype, device=device)
    q_entries = torch.stack([q_lens.to(dtype=dtype, device=device), zeros], dim=1).reshape(-1)
    k_entries = torch.stack(
        [
            active_k_lens.to(dtype=dtype, device=device),
            (full_seq_lens - active_k_lens).to(dtype=dtype, device=device),
        ],
        dim=1,
    ).reshape(-1)

    cu_seqlens_q = torch.zeros(q_entries.numel() + 1, dtype=dtype, device=device)
    cu_seqlens_k = torch.zeros(k_entries.numel() + 1, dtype=dtype, device=device)
    cu_seqlens_q[1:] = torch.cumsum(q_entries, dim=0)
    cu_seqlens_k[1:] = torch.cumsum(k_entries, dim=0)

    max_seqlen_q = q_lens.max().item()
    max_seqlen_k = k_entries.max().item()
    return cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k


def _build_q_split_indices(
    slice_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_front_idx = []
    q_end_idx = []
    offset = 0
    for slice_size in slice_sizes.tolist():
        q_front_idx.append(torch.arange(offset, offset + slice_size, device=slice_sizes.device))
        offset += slice_size
        q_end_idx.append(torch.arange(offset, offset + slice_size, device=slice_sizes.device))
        offset += slice_size
    return torch.cat(q_front_idx), torch.cat(q_end_idx)


def prepare_zigzag_allgather_attn_varlen_metadata(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    world_size: int,
    rank: int,
) -> ZigZagAllGatherVarlenMetadata:
    cache_key = (tuple(cu_seqlens_q.tolist()), tuple(cu_seqlens_k.tolist()), world_size, rank)
    if cache_key in _METADATA_CACHE:
        return _METADATA_CACHE[cache_key]

    total_slices = 2 * world_size
    q_seq_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    k_seq_lens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    assert q_seq_lens.numel() == k_seq_lens.numel(), (
        "cu_seqlens_q and cu_seqlens_k must describe the same number of sequences"
    )
    assert torch.all(q_seq_lens % total_slices == 0), (
        "Each sequence length must be divisible by 2 * world_size. "
        f"Got q_seq_lens={q_seq_lens.tolist()}, world_size={world_size}."
    )

    slice_sizes = q_seq_lens // total_slices
    kv_extra_lens = k_seq_lens - q_seq_lens
    front_k_lens = (rank + 1) * slice_sizes + kv_extra_lens
    end_k_lens = (total_slices - rank) * slice_sizes + kv_extra_lens

    (
        cu_seqlens_q_front,
        cu_seqlens_k_front,
        max_seqlen_q,
        max_seqlen_k_front,
    ) = _build_branch_cu_seqlens(
        slice_sizes,
        front_k_lens,
        k_seq_lens,
        cu_seqlens_q.dtype,
        cu_seqlens_q.device,
    )
    (
        cu_seqlens_q_end,
        cu_seqlens_k_end,
        _,
        max_seqlen_k_end,
    ) = _build_branch_cu_seqlens(
        slice_sizes,
        end_k_lens,
        k_seq_lens,
        cu_seqlens_q.dtype,
        cu_seqlens_q.device,
    )
    q_front_idx, q_end_idx = _build_q_split_indices(slice_sizes)

    metadata = ZigZagAllGatherVarlenMetadata(
        cu_seqlens_q_front=cu_seqlens_q_front,
        cu_seqlens_k_front=cu_seqlens_k_front,
        cu_seqlens_q_end=cu_seqlens_q_end,
        cu_seqlens_k_end=cu_seqlens_k_end,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k_front=max_seqlen_k_front,
        max_seqlen_k_end=max_seqlen_k_end,
        q_front_idx=q_front_idx,
        q_end_idx=q_end_idx,
    )
    if len(_METADATA_CACHE) >= _METADATA_CACHE_MAXSIZE:
        _METADATA_CACHE.popitem(last=False)
    _METADATA_CACHE[cache_key] = metadata
    return metadata


def _run_flash_attn_varlen(
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
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    use_cute: bool,
    window_size: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_cute:
        assert flash_attn_cute_varlen_func is not None, "flash_attn.cute is not available"
        cute_window_size = tuple(None if w == -1 else w for w in window_size)
        out, lse = flash_attn_cute_varlen_func(
            q,
            k,
            v,
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
        return out, lse

    out, lse, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=True,
    )
    return out, lse


def _run_flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    use_cute: bool,
    window_size: Tuple[int, int],
    dlse: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    if use_cute:
        assert _flash_attn_bwd is not None, "flash_attn.cute is not available"
        cute_window_size = tuple(None if w == -1 else w for w in window_size)
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
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": softmax_scale,
            "causal": causal,
            "window_size_left": cute_window_size[0],
            "window_size_right": cute_window_size[1],
            "deterministic": deterministic,
        })
        if "dlse" in params:
            params["dlse"] = dlse
        elif dlse is not None:
            raise RuntimeError(
                "attention z-loss requires a FlashAttention cute backward with dlse support")
        _flash_attn_bwd(**params)
        return dq, dk, dv

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
        "max_seqlen_q": max_seqlen_q,
        "max_seqlen_k": max_seqlen_k,
        "dropout_p": dropout_p,
        "softmax_scale": softmax_scale,
        "causal": causal,
        "alibi_slopes": alibi_slopes,
        "deterministic": deterministic,
    })
    if "window_size" in params:
        params["window_size"] = window_size
    else:
        params["window_size_left"] = window_size[0]
        params["window_size_right"] = window_size[1]
    if "dlse" in params:
        params["dlse"] = dlse
    elif dlse is not None:
        raise RuntimeError(
            "attention z-loss requires a FlashAttention varlen backward with dlse support")
    _flash_attn_varlen_backward(**params)
    return dq, dk, dv


class ZigZagAllGatherAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        process_group,
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        alibi_slopes: Optional[torch.Tensor],
        deterministic: bool,
        use_cute: bool,
        window_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        output, softmax_lse, metadata, front_output, front_lse, end_output, end_lse = (
            _zigzag_allgather_attn_varlen_forward_impl(
                q, k, v, cu_seqlens_q, cu_seqlens_k, process_group,
                dropout_p, softmax_scale, causal, alibi_slopes,
                deterministic, use_cute, window_size)
        )
        ctx.save_for_backward(q, k, v, front_output, front_lse, end_output, end_lse)
        ctx.metadata = metadata
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.use_cute = use_cute
        ctx.window_size = window_size
        return output, softmax_lse

    @staticmethod
    def backward(ctx, dout: torch.Tensor, dlse: Optional[torch.Tensor]):
        q, k, v, front_output, front_lse, end_output, end_lse = ctx.saved_tensors
        metadata = ctx.metadata

        q_front = q[metadata.q_front_idx]
        q_end = q[metadata.q_end_idx]
        dout_front = dout[metadata.q_front_idx].contiguous()
        dout_end = dout[metadata.q_end_idx].contiguous()
        dlse_front = dlse[..., metadata.q_front_idx].contiguous() if dlse is not None else None
        dlse_end = dlse[..., metadata.q_end_idx].contiguous() if dlse is not None else None

        dq_front, dk_front, dv_front = _run_flash_attn_varlen_backward(
            dout_front,
            q_front,
            k,
            v,
            front_output,
            front_lse,
            metadata.cu_seqlens_q_front,
            metadata.cu_seqlens_k_front,
            metadata.max_seqlen_q,
            metadata.max_seqlen_k_front,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.alibi_slopes,
            ctx.deterministic,
            ctx.use_cute,
            ctx.window_size,
            dlse_front,
        )
        dq_end, dk_end, dv_end = _run_flash_attn_varlen_backward(
            dout_end,
            q_end,
            k,
            v,
            end_output,
            end_lse,
            metadata.cu_seqlens_q_end,
            metadata.cu_seqlens_k_end,
            metadata.max_seqlen_q,
            metadata.max_seqlen_k_end,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.alibi_slopes,
            ctx.deterministic,
            ctx.use_cute,
            ctx.window_size,
            dlse_end,
        )

        dq = torch.empty_like(q)
        dq[metadata.q_front_idx] = dq_front
        dq[metadata.q_end_idx] = dq_end
        dk = dk_front + dk_end
        dv = dv_front + dv_end
        return (dq, dk, dv) + (None,) * 10


def _zigzag_allgather_attn_varlen_forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    process_group,
    dropout_p: float,
    softmax_scale: Optional[float],
    causal: bool,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    use_cute: bool,
    window_size: Tuple[int, int],
):
    assert process_group is not None and dist.get_world_size(process_group) > 1, (
        "zigzag_allgather_attn_varlen_func only handles the multi-GPU CP branch"
    )

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    metadata = prepare_zigzag_allgather_attn_varlen_metadata(
        cu_seqlens_q,
        cu_seqlens_k,
        world_size=world_size,
        rank=rank,
    )

    q_front = q[metadata.q_front_idx]
    q_end = q[metadata.q_end_idx]

    front_output, front_lse = _run_flash_attn_varlen(
        q_front,
        k,
        v,
        metadata.cu_seqlens_q_front,
        metadata.cu_seqlens_k_front,
        metadata.max_seqlen_q,
        metadata.max_seqlen_k_front,
        dropout_p,
        softmax_scale,
        causal,
        alibi_slopes,
        deterministic,
        use_cute,
        window_size,
    )
    end_output, end_lse = _run_flash_attn_varlen(
        q_end,
        k,
        v,
        metadata.cu_seqlens_q_end,
        metadata.cu_seqlens_k_end,
        metadata.max_seqlen_q,
        metadata.max_seqlen_k_end,
        dropout_p,
        softmax_scale,
        causal,
        alibi_slopes,
        deterministic,
        use_cute,
        window_size,
    )

    output = front_output.new_empty((q.shape[0],) + front_output.shape[1:])
    output[metadata.q_front_idx] = front_output
    output[metadata.q_end_idx] = end_output

    softmax_lse = front_lse.new_empty(front_lse.shape[:-1] + (q.shape[0],))
    softmax_lse[..., metadata.q_front_idx] = front_lse
    softmax_lse[..., metadata.q_end_idx] = end_lse

    return output, softmax_lse, metadata, front_output, front_lse, end_output, end_lse


def zigzag_allgather_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    process_group,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    use_cute: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
) -> Tuple[torch.Tensor, torch.Tensor]:
    return ZigZagAllGatherAttnVarlenFunc.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, process_group,
        dropout_p, softmax_scale, causal, alibi_slopes, deterministic,
        use_cute, window_size)
