#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Context Parallel Sliding Window Attention Implementation.

Key insight: For causal sliding window attention, each rank only needs
min(offset_in_seq, window_size_left) KV tokens from the previous rank,
not a full all_gather. This enables efficient single-hop A2A communication
and a single flash_attn computation (no ring loop).
"""

from dataclasses import dataclass
from collections import OrderedDict
from typing import Tuple, Optional, List

import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from .utils import get_default_args

try:
    from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd
except ImportError:
    _flash_attn_fwd = None
    _flash_attn_bwd = None


@dataclass
class SlidingWindowMetadata:
    """Precomputed metadata for sliding window CP attention."""
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k: int
    input_split_sizes: List[int]   # forward A2A send splits
    output_split_sizes: List[int]  # forward A2A recv splits


# Global cache: avoids recomputing metadata every layer
_METADATA_CACHE_MAXSIZE = 128
_METADATA_CACHE = OrderedDict()


def _make_cache_key(cu_seqlens: torch.Tensor, window_size_left: int, rank: int, world_size: int) -> Tuple:
    return (tuple(cu_seqlens.tolist()), window_size_left, rank, world_size)

def prepare_sliding_window_metadata(
    cu_seqlens: torch.Tensor,
    window_size_left: int,
    rank: int,
    world_size: int,
) -> SlidingWindowMetadata:
    """
    Compute per-rank cu_seqlens for sliding window CP attention.

    Args:
        cu_seqlens: Global cumulative sequence lengths [0, s1, s2, ..., total].
        window_size_left: Sliding window size (left context).
        rank: Current rank in the CP group.
        world_size: Total number of ranks in the CP group.

    Returns:
        SlidingWindowMetadata with local cu_seqlens_q, cu_seqlens_k,
        and cached A2A split sizes.
    """
    cache_key = _make_cache_key(cu_seqlens, window_size_left, rank, world_size)
    if cache_key in _METADATA_CACHE:
        return _METADATA_CACHE[cache_key]

    total_length = cu_seqlens[-1].item()
    assert total_length % world_size == 0, (
        f"total_length {total_length} must be divisible by world_size {world_size}"
    )
    length_per_rank = total_length // world_size
    assert window_size_left <= length_per_rank, (
        f"window_size_left {window_size_left} must be <= length_per_rank {length_per_rank}. "
        f"Multi-hop communication is not supported."
    )

    chunk_start = rank * length_per_rank
    chunk_end = (rank + 1) * length_per_rank

    # Find sequence boundaries that overlap with this rank's chunk
    left = torch.searchsorted(cu_seqlens, chunk_start)
    right = torch.searchsorted(cu_seqlens, chunk_end)
    if cu_seqlens[left].item() != chunk_start:
        left -= 1
    left = left.item()
    right = right.item()

    # --- cu_seqlens_q: local Q boundaries (same as ring_attn_varlen) ---
    cu_seqlens_q = cu_seqlens[left: right + 1].clone()
    cu_seqlens_q -= chunk_start
    cu_seqlens_q[0] = 0
    cu_seqlens_q[-1] = length_per_rank

    # --- recv_size: how many tokens to receive from previous rank ---
    recv_size = 0
    if rank > 0 and cu_seqlens[left].item() < chunk_start:
        offset_in_seq = chunk_start - cu_seqlens[left].item()
        recv_size = min(offset_in_seq, window_size_left)

    # --- cu_seqlens_k: extended K boundaries ---
    # The first (straddling) sequence gets recv_size extra K tokens prepended.
    # All other sequences have K length == Q length.
    # Result: cu_seqlens_k[i] = cu_seqlens_q[i] + recv_size for i >= 1, cu_seqlens_k[0] = 0
    cu_seqlens_k = cu_seqlens_q.clone()
    cu_seqlens_k[1:] += recv_size

    # --- send_size: how many tokens to send to next rank ---
    send_size = 0
    if rank < world_size - 1:
        next_chunk_start = chunk_end
        right_idx = torch.searchsorted(cu_seqlens, next_chunk_start).item()
        if right_idx > 0 and cu_seqlens[right_idx].item() > next_chunk_start:
            offset_in_seq = next_chunk_start - cu_seqlens[right_idx - 1].item()
            send_size = min(offset_in_seq, window_size_left)

    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < world_size - 1 else None
    input_split_sizes = [0] * world_size
    output_split_sizes = [0] * world_size
    if send_size > 0 and next_rank is not None:
        input_split_sizes[next_rank] = send_size
    if recv_size > 0 and prev_rank is not None:
        output_split_sizes[prev_rank] = recv_size

    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()

    metadata = SlidingWindowMetadata(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        input_split_sizes=input_split_sizes,
        output_split_sizes=output_split_sizes,
    )
    if len(_METADATA_CACHE) >= _METADATA_CACHE_MAXSIZE:
        _METADATA_CACHE.popitem(last=False)
    _METADATA_CACHE[cache_key] = metadata
    return metadata


def _all_to_all_varlen(
    tensor: torch.Tensor,
    input_split_sizes: List[int],
    output_split_sizes: List[int],
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """All-to-all with variable split sizes along dim 0."""
    output_size = sum(output_split_sizes)
    output = tensor.new_empty(output_size, *tensor.shape[1:]).contiguous()
    dist.all_to_all_single(
        output,
        tensor.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )
    return output


def _a2a_communicate_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: SlidingWindowMetadata,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    A2A send/recv of KV tokens between neighboring ranks.

    - Sends last send_size tokens of k/v to next rank.
    - Receives recv_size tokens of k/v from prev rank.

    Returns:
        (extended_k, extended_v) with received tokens prepended.
    """
    input_split_sizes = metadata.input_split_sizes
    output_split_sizes = metadata.output_split_sizes
    send_size = sum(input_split_sizes)
    recv_size = sum(output_split_sizes)

    # Prepare send buffers
    if send_size > 0:
        send_k = k[-send_size:].contiguous()
        send_v = v[-send_size:].contiguous()
    else:
        send_k = k.new_empty((0, k.shape[1], k.shape[2]))
        send_v = v.new_empty((0, v.shape[1], v.shape[2]))

    recv_k = _all_to_all_varlen(send_k, input_split_sizes, output_split_sizes, group)
    recv_v = _all_to_all_varlen(send_v, input_split_sizes, output_split_sizes, group)

    # Construct extended K/V
    if recv_size > 0:
        extended_k = torch.cat([recv_k, k], dim=0)
        extended_v = torch.cat([recv_v, v], dim=0)
    else:
        extended_k = k
        extended_v = v

    return extended_k, extended_v


def _a2a_communicate_grad(
    dk_recv: Optional[torch.Tensor],
    dv_recv: Optional[torch.Tensor],
    dk_local: torch.Tensor,
    dv_local: torch.Tensor,
    metadata: SlidingWindowMetadata,
    group: dist.ProcessGroup,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    A2A send/recv of KV gradients (reverse direction from forward).

    - Sends dk_recv/dv_recv back to prev rank (gradient for received tokens).
    - Receives dk_sent/dv_sent from next rank (gradient for sent tokens).

    Returns:
        (dk_from_next, dv_from_next) or (None, None) if send_size == 0.
    """
    # Backward direction is reverse of forward:
    # send grad-for-recv to prev, recv grad-for-send from next.
    input_split_sizes = metadata.output_split_sizes
    output_split_sizes = metadata.input_split_sizes
    recv_size = sum(input_split_sizes)
    send_size = sum(output_split_sizes)

    if recv_size > 0 and dk_recv is not None:
        send_dk = dk_recv.contiguous()
        send_dv = dv_recv.contiguous()
    else:
        send_dk = dk_local.new_empty((0, dk_local.shape[1], dk_local.shape[2]))
        send_dv = dv_local.new_empty((0, dv_local.shape[1], dv_local.shape[2]))

    dk_from_next = _all_to_all_varlen(send_dk, input_split_sizes, output_split_sizes, group)
    dv_from_next = _all_to_all_varlen(send_dv, input_split_sizes, output_split_sizes, group)
    if send_size == 0:
        dk_from_next = None
        dv_from_next = None

    return dk_from_next, dv_from_next


def sliding_window_forward(
    process_group: dist.ProcessGroup,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    metadata: SlidingWindowMetadata,
    softmax_scale: Optional[float],
    dropout_p: float = 0.0,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    use_cute: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for sliding window CP attention.

    1. A2A communicate KV with neighboring ranks.
    2. Single flash_attn_varlen call with extended K/V.
    """
    # Step 1: A2A communication
    extended_k, extended_v = _a2a_communicate_kv(k, v, metadata, process_group)

    cu_seqlens_q = metadata.cu_seqlens_q
    cu_seqlens_k = metadata.cu_seqlens_k
    max_seqlen_q = metadata.max_seqlen_q
    max_seqlen_k = metadata.max_seqlen_k

    # Step 2: Single flash attention call
    if use_cute:
        assert _flash_attn_fwd is not None, "flash_attn.cute is not available"
        window_size_cute = tuple(None if w == -1 else w for w in window_size)
        params = get_default_args(_flash_attn_fwd).copy()
        params.update({
            "q": q,
            "k": extended_k,
            "v": extended_v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": softmax_scale,
            "causal": True,
            "window_size_left": window_size_cute[0],
            "window_size_right": window_size_cute[1],
        })
        out, lse = _flash_attn_fwd(**params)
    else:
        params = get_default_args(_flash_attn_varlen_forward).copy()
        params.update({
            "q": q,
            "k": extended_k,
            "v": extended_v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": True,
            "alibi_slopes": alibi_slopes,
            "return_softmax": True and dropout_p > 0,
        })
        if "window_size" in params:
            params["window_size"] = window_size
        else:
            params["window_size_left"] = window_size[0]
            params["window_size_right"] = window_size[1]
        outputs = _flash_attn_varlen_forward(**params)
        if len(outputs) == 8:
            out, _, _, _, _, lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            out, lse, _, _ = outputs

    return out, lse, extended_k, extended_v


def sliding_window_backward(
    process_group: dist.ProcessGroup,
    dout: torch.Tensor,
    q: torch.Tensor,
    extended_k: torch.Tensor,
    extended_v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    metadata: SlidingWindowMetadata,
    softmax_scale: Optional[float],
    dropout_p: float = 0.0,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[torch.Tensor] = None,
    deterministic: bool = False,
    use_cute: bool = False,
):  # pragma: no cover
    """
    Backward pass for sliding window CP attention.

    1. Reuse extended K/V cached from forward.
    2. flash_attn backward -> dq, dk_extended, dv_extended.
    3. A2A send dk_recv back to prev rank, recv dk_from_next from next rank.
    4. Accumulate gradient for sent tokens.
    """
    recv_size = sum(metadata.output_split_sizes)
    send_size = sum(metadata.input_split_sizes)

    cu_seqlens_q = metadata.cu_seqlens_q
    cu_seqlens_k = metadata.cu_seqlens_k
    max_seqlen_q = metadata.max_seqlen_q
    max_seqlen_k = metadata.max_seqlen_k

    # Allocate gradient buffers
    dq = torch.empty_like(q)
    dk_extended = torch.empty_like(extended_k)
    dv_extended = torch.empty_like(extended_v)

    # Step 2: flash_attn backward
    if use_cute:
        assert _flash_attn_bwd is not None, "flash_attn.cute is not available"
        window_size_cute = tuple(None if w == -1 else w for w in window_size)
        params = get_default_args(_flash_attn_bwd).copy()
        params.update({
            "dout": dout,
            "q": q,
            "k": extended_k,
            "v": extended_v,
            "out": out,
            "lse": softmax_lse,
            "dq": dq,
            "dk": dk_extended,
            "dv": dv_extended,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": softmax_scale,
            "causal": True,
            "window_size_left": window_size_cute[0],
            "window_size_right": window_size_cute[1],
            "deterministic": deterministic,
        })
        _flash_attn_bwd(**params)
    else:
        params = get_default_args(_flash_attn_varlen_backward).copy()
        params.update({
            "dout": dout,
            "q": q,
            "k": extended_k,
            "v": extended_v,
            "out": out,
            "softmax_lse": softmax_lse,
            "dq": dq,
            "dk": dk_extended,
            "dv": dv_extended,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": True,
            "alibi_slopes": alibi_slopes,
            "deterministic": deterministic,
        })
        if "window_size" in params:
            params["window_size"] = window_size
        else:
            params["window_size_left"] = window_size[0]
            params["window_size_right"] = window_size[1]
        _flash_attn_varlen_backward(**params)

    # Step 3: Split dk_extended into recv portion and local portion
    dk_recv = dk_extended[:recv_size] if recv_size > 0 else None
    dk_local = dk_extended[recv_size:]
    dv_recv = dv_extended[:recv_size] if recv_size > 0 else None
    dv_local = dv_extended[recv_size:]

    # Step 4: A2A gradient communication (reverse direction)
    dk_from_next, dv_from_next = _a2a_communicate_grad(
        dk_recv, dv_recv, dk_local, dv_local, metadata, process_group,
    )

    # Step 5: Accumulate gradients for sent tokens
    if send_size > 0 and dk_from_next is not None:
        dk_local[-send_size:] += dk_from_next
        dv_local[-send_size:] += dv_from_next

    return dq, dk_local, dv_local


class SlidingWindowAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        metadata,
        dropout_p,
        softmax_scale,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        use_cute,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        k = k.contiguous()
        v = v.contiguous()

        out, softmax_lse, extended_k, extended_v = sliding_window_forward(
            group,
            q,
            k,
            v,
            metadata,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            use_cute=use_cute,
        )

        ctx.save_for_backward(q, extended_k, extended_v, out, softmax_lse)
        ctx.attn_metadata = metadata
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.use_cute = use_cute
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):  # pragma: no cover
        q, extended_k, extended_v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = sliding_window_backward(
            ctx.group,
            dout,
            q,
            extended_k,
            extended_v,
            out,
            softmax_lse,
            ctx.attn_metadata,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            use_cute=ctx.use_cute,
        )
        return (dq, dk, dv) + (None,) * 11


def sliding_window_attn_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    metadata,
    dropout_p=0.0,
    softmax_scale=None,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    use_cute=False,
):
    """
    Public entry point for sliding window CP attention.

    Args:
        q: [total_q, nheads_q, dim]
        k: [total_k, nheads_k, dim] (local portion)
        v: [total_v, nheads_v, dim] (local portion)
        cu_seqlens_q: Global cumulative sequence lengths.
        cu_seqlens_k: Global cumulative sequence lengths.
        metadata: SlidingWindowMetadata from prepare_sliding_window_metadata.
        window_size: (window_size_left, window_size_right), typically (W, 0).
        group: Process group for CP.
    """
    return SlidingWindowAttnFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        metadata,
        dropout_p,
        softmax_scale,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        use_cute,
    )
