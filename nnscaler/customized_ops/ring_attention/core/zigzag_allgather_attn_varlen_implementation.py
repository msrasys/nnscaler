#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from flash_attn import flash_attn_varlen_func

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_cute_varlen_func
except ImportError:
    flash_attn_cute_varlen_func = None

from ..varlen_utils import shuffle_varlen, unshuffle_varlen


@dataclass
class ZigZagAllGatherVarlenMetadata:
    cu_seqlens_q_front: torch.Tensor
    cu_seqlens_k_front: torch.Tensor
    cu_seqlens_q_end: torch.Tensor
    cu_seqlens_k_end: torch.Tensor
    max_seqlen_q: int
    max_seqlen_k_front: int
    max_seqlen_k_end: int
    slice_sizes: torch.Tensor


_METADATA_CACHE_MAXSIZE = 128
_METADATA_CACHE = OrderedDict()


def _get_cp_ranks(process_group) -> List[int]:
    world_size = dist.get_world_size(process_group)
    return [dist.get_global_rank(process_group, rank) for rank in range(world_size)]


def _split_shuffled_q(
    shuffled_q: torch.Tensor,
    slice_sizes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_front_slices = []
    q_end_slices = []
    offset = 0
    for slice_size in slice_sizes.tolist():
        q_front_slices.append(shuffled_q[offset : offset + slice_size])
        offset += slice_size
        q_end_slices.append(shuffled_q[offset : offset + slice_size])
        offset += slice_size
    return torch.cat(q_front_slices, dim=0), torch.cat(q_end_slices, dim=0)


def _merge_shuffled_output(
    front_output: torch.Tensor,
    end_output: torch.Tensor,
    slice_sizes: torch.Tensor,
) -> torch.Tensor:
    output = []
    front_offset = 0
    end_offset = 0
    for slice_size in slice_sizes.tolist():
        output.append(front_output[front_offset : front_offset + slice_size])
        front_offset += slice_size
        output.append(end_output[end_offset : end_offset + slice_size])
        end_offset += slice_size
    return torch.cat(output, dim=0)


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

    metadata = ZigZagAllGatherVarlenMetadata(
        cu_seqlens_q_front=cu_seqlens_q_front,
        cu_seqlens_k_front=cu_seqlens_k_front,
        cu_seqlens_q_end=cu_seqlens_q_end,
        cu_seqlens_k_end=cu_seqlens_k_end,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k_front=max_seqlen_k_front,
        max_seqlen_k_end=max_seqlen_k_end,
        slice_sizes=slice_sizes,
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
) -> torch.Tensor:
    if use_cute:
        assert flash_attn_cute_varlen_func is not None, "flash_attn.cute is not available"
        cute_window_size = tuple(None if w == -1 else w for w in window_size)
        out, _ = flash_attn_cute_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=cute_window_size,
            deterministic=deterministic,
        )
        return out

    return flash_attn_varlen_func(
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
        return_attn_probs=False,
    )


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
) -> torch.Tensor:
    assert process_group is not None and dist.get_world_size(process_group) > 1, (
        "zigzag_allgather_attn_varlen_func only handles the multi-GPU CP branch"
    )
    
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    cp_ranks = _get_cp_ranks(process_group)

    metadata = prepare_zigzag_allgather_attn_varlen_metadata(
        cu_seqlens_q,
        cu_seqlens_k,
        world_size=world_size,
        rank=rank,
    )

    shuffled_q = shuffle_varlen(q, cu_seqlens_q, cp_ranks, process_group)
    q_front, q_end = _split_shuffled_q(shuffled_q, metadata.slice_sizes)

    front_output = _run_flash_attn_varlen(
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
    end_output = _run_flash_attn_varlen(
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

    shuffled_output = _merge_shuffled_output(
        front_output,
        end_output,
        metadata.slice_sizes,
    )
    return unshuffle_varlen(shuffled_output, cu_seqlens_q, cp_ranks, process_group)
