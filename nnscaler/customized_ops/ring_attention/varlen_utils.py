#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Utilities for variable-length sequence processing in ring attention.
Contains shuffle and unshuffle functions for context parallel processing.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor
import torch.distributed as dist


@dataclass
class _A2AMeta:
    send_perm: Tensor
    inv_send_perm: Tensor
    input_split_sizes: List[int]
    output_split_sizes: List[int]
    recv_perm: Optional[Tensor] = None       # None means identity
    inv_recv_perm: Optional[Tensor] = None   # None means identity


_A2A_CACHE_MAXSIZE = 128
_A2A_CACHE = OrderedDict()


def _all_to_all_varlen(tensor: Tensor, input_split_sizes: List[int],
                        output_split_sizes: List[int], group: dist.ProcessGroup) -> Tensor:
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


class _ShuffleVarlenA2A(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, send_perm, inv_send_perm, recv_perm, inv_recv_perm,
                input_split_sizes, output_split_sizes, group):
        ctx.inv_send_perm = inv_send_perm
        ctx.inv_recv_perm = inv_recv_perm
        ctx.input_split_sizes = input_split_sizes
        ctx.output_split_sizes = output_split_sizes
        ctx.group = group

        send_data = t.index_select(0, send_perm)
        recv_data = _all_to_all_varlen(send_data, input_split_sizes, output_split_sizes, group)

        if recv_perm is not None:
            return recv_data.index_select(0, recv_perm)
        return recv_data

    @staticmethod
    def backward(ctx, grad_output):
        # Undo recv_perm if present
        if ctx.inv_recv_perm is not None:
            grad_output = grad_output.index_select(0, ctx.inv_recv_perm)

        # Reverse A2A: swap input/output split sizes
        grad_permuted = _all_to_all_varlen(
            grad_output, ctx.output_split_sizes, ctx.input_split_sizes, ctx.group
        )
        # Undo send permutation
        grad_t = grad_permuted.index_select(0, ctx.inv_send_perm)
        return grad_t, None, None, None, None, None, None, None


def _compute_a2a_metadata(cu_seqlens_padded: Tensor, cp_size: int, cp_rank: int,
                           device: torch.device):
    """Compute and cache all_to_all metadata for both shuffle and unshuffle.

    Returns (shuffle_meta, unshuffle_meta) as _A2AMeta dataclasses.
    """
    cache_key = (tuple(cu_seqlens_padded.tolist()), cp_size, cp_rank)
    if cache_key in _A2A_CACHE:
        _A2A_CACHE.move_to_end(cache_key)
        return _A2A_CACHE[cache_key]

    total_slices = 2 * cp_size
    seq_lens = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    assert torch.all(seq_lens % total_slices == 0), (
        f"Each sequence length must be divisible by 2*cp_size={total_slices}. "
        f"Got seq_lens={seq_lens.tolist()}"
    )
    slice_sizes = seq_lens // total_slices
    total_tokens = cu_seqlens_padded[-1].item()
    chunk_size = total_tokens // cp_size
    num_seqs = seq_lens.numel()

    # --- Shuffle metadata ---
    # For each local token, determine destination (= shuffled) rank
    my_start = cp_rank * chunk_size
    global_pos = torch.arange(my_start, my_start + chunk_size, device=device)

    seq_idx = torch.searchsorted(cu_seqlens_padded[1:], global_pos, right=True)
    pos_in_seq = global_pos - cu_seqlens_padded[seq_idx]
    token_slice_sizes = slice_sizes[seq_idx]
    slice_idx = pos_in_seq // token_slice_sizes

    # slice i → dest rank i (front) if i < cp_size, else dest rank (total_slices-1-i) (end)
    dest_rank = torch.where(slice_idx < cp_size, slice_idx, total_slices - 1 - slice_idx)

    assert dest_rank.min().item() >= 0, (
        f"Negative dest_rank detected (min={dest_rank.min().item()}). "
        f"This usually means sequence lengths are not divisible by 2*cp_size={total_slices}. "
        f"seq_lens={seq_lens.tolist()}, cp_rank={cp_rank}"
    )
    assert dest_rank.max().item() < cp_size, (
        f"dest_rank out of range (max={dest_rank.max().item()}, cp_size={cp_size}). "
        f"seq_lens={seq_lens.tolist()}, cp_rank={cp_rank}"
    )

    shuffle_send_perm = torch.argsort(dest_rank, stable=True)
    shuffle_inv_send_perm = torch.empty_like(shuffle_send_perm)
    shuffle_inv_send_perm[shuffle_send_perm] = torch.arange(chunk_size, device=device)
    shuffle_input_splits = torch.bincount(dest_rank.long(), minlength=cp_size).tolist()

    # Compute zigzag positions for this rank (= shuffle output order)
    zigzag_positions = []
    for s in range(num_seqs):
        sl = slice_sizes[s].item()
        seq_start = cu_seqlens_padded[s].item()
        zigzag_positions.append(
            torch.arange(seq_start + cp_rank * sl, seq_start + (cp_rank + 1) * sl, device=device)
        )
        zigzag_positions.append(
            torch.arange(seq_start + (total_slices - cp_rank - 1) * sl,
                         seq_start + (total_slices - cp_rank) * sl, device=device)
        )
    zigzag_pos = torch.cat(zigzag_positions)

    # Output split sizes for shuffle = tokens received from each source rank
    source_ranks_shuffle = zigzag_pos // chunk_size
    shuffle_output_splits = torch.bincount(source_ranks_shuffle.long(), minlength=cp_size).tolist()

    # Shuffle recv_perm is identity (zigzag order = ascending position order,
    # and all_to_all output is ascending across source ranks)
    shuffle_meta = _A2AMeta(shuffle_send_perm, shuffle_inv_send_perm,
                            shuffle_input_splits, shuffle_output_splits,
                            recv_perm=None, inv_recv_perm=None)

    # --- Unshuffle metadata ---
    # Input is in zigzag order; destination is original rank
    unshuffle_dest = zigzag_pos // chunk_size
    unshuffle_send_perm = torch.argsort(unshuffle_dest, stable=True)
    unshuffle_inv_send_perm = torch.empty_like(unshuffle_send_perm)
    unshuffle_inv_send_perm[unshuffle_send_perm] = torch.arange(len(zigzag_pos), device=device)
    unshuffle_input_splits = torch.bincount(unshuffle_dest.long(), minlength=cp_size).tolist()

    # Unshuffle output split sizes: for each position in my original range,
    # which shuffled rank held it? That's the source rank in unshuffle's all_to_all.
    orig_shuffled_rank = dest_rank  # reuse: dest_rank[i] = shuffled rank for position my_start+i
    unshuffle_output_splits = torch.bincount(orig_shuffled_rank.long(), minlength=cp_size).tolist()

    # Unshuffle DOES need recv_perm: after all_to_all, tokens arrive grouped by source
    # (shuffled) rank, but positions from different shuffled ranks may not be in ascending order.
    # recv_perm reorders from arrival order to ascending position order.
    # Arrival order = stable sort of original positions by their shuffled rank = shuffle_send_perm
    # So recv_perm = shuffle_inv_send_perm (maps from arrival order to ascending order)
    unshuffle_recv_perm = shuffle_inv_send_perm
    unshuffle_inv_recv_perm = shuffle_send_perm

    unshuffle_meta = _A2AMeta(unshuffle_send_perm, unshuffle_inv_send_perm,
                              unshuffle_input_splits, unshuffle_output_splits,
                              recv_perm=unshuffle_recv_perm,
                              inv_recv_perm=unshuffle_inv_recv_perm)

    result = (shuffle_meta, unshuffle_meta)
    if len(_A2A_CACHE) >= _A2A_CACHE_MAXSIZE:
        _A2A_CACHE.popitem(last=False)
    _A2A_CACHE[cache_key] = result
    return result


def shuffle_varlen(t: Tensor, cu_seqlens_padded: Tensor, cp_ranks: List[int],
                   cp_group: dist.ProcessGroup) -> Tensor:
    """
    Shuffle tensor data for variable-length sequences in context parallel processing.

    Uses all_to_all instead of all_gather to avoid materializing the full global tensor,
    reducing peak memory from O(cp_size * local) to O(2 * local).
    """
    cp_size = dist.get_world_size(group=cp_group)
    assert cp_size > 1, "cp_size should be greater than 1"
    cp_rank = dist.get_rank(group=cp_group)

    shuffle_meta, _ = _compute_a2a_metadata(cu_seqlens_padded, cp_size, cp_rank, t.device)

    return _ShuffleVarlenA2A.apply(
        t, shuffle_meta.send_perm, shuffle_meta.inv_send_perm,
        shuffle_meta.recv_perm, shuffle_meta.inv_recv_perm,
        shuffle_meta.input_split_sizes, shuffle_meta.output_split_sizes, cp_group,
    )


def unshuffle_varlen(t: Tensor, cu_seqlens_padded: Tensor, cp_ranks: List[int],
                     cp_group: dist.ProcessGroup) -> Tensor:
    """
    Unshuffle tensor data to restore original variable-length sequence order.
    This is the reverse operation of shuffle_varlen.

    Uses all_to_all instead of all_gather to avoid materializing the full global tensor.
    """
    cp_size = dist.get_world_size(group=cp_group)
    assert cp_size > 1, "cp_size should be greater than 1"
    cp_rank = dist.get_rank(group=cp_group)

    _, unshuffle_meta = _compute_a2a_metadata(cu_seqlens_padded, cp_size, cp_rank, t.device)

    return _ShuffleVarlenA2A.apply(
        t, unshuffle_meta.send_perm, unshuffle_meta.inv_send_perm,
        unshuffle_meta.recv_perm, unshuffle_meta.inv_recv_perm,
        unshuffle_meta.input_split_sizes, unshuffle_meta.output_split_sizes, cp_group,
    )
