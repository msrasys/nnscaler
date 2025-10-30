#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Utilities for variable-length sequence processing in ring attention.
Contains shuffle and unshuffle functions for context parallel processing.
"""

from typing import List
import torch
from torch import Tensor
import torch.distributed as dist
from nnscaler.runtime.adapter.nn import allgather_reducescatter


def shuffle_varlen(t: Tensor, cu_seqlens_padded: Tensor, cp_ranks: List[int], cp_group: dist.ProcessGroup) -> Tensor:
    """
    Shuffle tensor data for variable-length sequences in context parallel processing.
    
    Args:
        t: Input tensor to shuffle (local portion from each rank)
        cu_seqlens_padded: Cumulative sequence lengths (global)
        cp_ranks: List of ranks in the context parallel group
        cp_group: Process group for context parallel communication
        
    Returns:
        Shuffled tensor
    """
    # Get context parallel size and rank
    cp_size = torch.distributed.get_world_size(group=cp_group)
    assert cp_size > 1, "cp_size should be greater than 1"
    cp_rank = torch.distributed.get_rank(group=cp_group)

    # Calculate the chunk sizes for each sequence
    total_slices_of_any_sequence = 2 * cp_size
    slice_sizes = (
        cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    ) // total_slices_of_any_sequence

    # Process each tensor directly instead of using keys_to_change loop
    def process_tensor(val):
        if val is None:
            return val
        # Determine which dimension is the sequence dimension
        # Ensure cu_seqlens_padded[-1] is a Python int, not a 0-dim tensor
        if isinstance(cu_seqlens_padded[-1], torch.Tensor):
            seq_len_val = cu_seqlens_padded[-1].item()
        else:
            seq_len_val = cu_seqlens_padded[-1]

        # Handle 1D tensors (like position_ids that don't have batch dimension)
        if val.ndim == 1:
            if val.shape[0] == seq_len_val:
                current_seq_dim = 0
            else:
                raise ValueError(
                    "1D tensor shape doesn't match expected sequence length. Make sure the"
                    " inputs are in THD format and padded correctly."
                )
        elif val.ndim >= 2:
            if val.shape[1] == seq_len_val:
                current_seq_dim = 1
            elif val.shape[0] == seq_len_val:
                current_seq_dim = 0
            else:
                raise ValueError(
                    "Make sure the inputs are in THD format and padded correctly."
                )
        else:
            raise ValueError("Tensor must be at least 1D")

        # On this particular rank, for each sequence, get two slices, one from the beginning
        # and one from the end.
        cp_rank_slices = []
        for slice_size, seq_start in zip(slice_sizes, cu_seqlens_padded[:-1]):
            # 1st segment
            cp_rank_slices.append(
                torch.arange(
                    seq_start + (cp_rank * slice_size),
                    seq_start + ((cp_rank + 1) * slice_size),
                    device=val.device,
                )
            )

            # 2nd segment
            cp_rank_slices.append(
                torch.arange(
                    seq_start + ((total_slices_of_any_sequence - cp_rank - 1) * slice_size),
                    seq_start + ((total_slices_of_any_sequence - cp_rank) * slice_size),
                    device=val.device,
                )
            )

        return val.index_select(current_seq_dim, torch.cat(cp_rank_slices))

    full_tensor = allgather_reducescatter(t, 0, cp_ranks)
    return process_tensor(full_tensor)


def unshuffle_varlen(t: Tensor, cu_seqlens_padded: Tensor, cp_ranks: List[int], cp_group: dist.ProcessGroup) -> Tensor:
    """
    Unshuffle tensor data to restore original variable-length sequence order.
    This is the reverse operation of shuffle_varlen.
    
    Args:
        t: Shuffled tensor to unshuffle (local portion from each rank)
        cu_seqlens_padded: Cumulative sequence lengths (global)
        cp_ranks: List of ranks in the context parallel group
        cp_group: Process group for context parallel communication
        
    Returns:
        Unshuffled tensor (local portion for each rank)
    """
    # reverse operation of shuffle_varlen
    cp_size = torch.distributed.get_world_size(group=cp_group)
    assert cp_size > 1, "cp_size should be greater than 1"
    cp_rank = torch.distributed.get_rank(group=cp_group)
    total_slices_of_any_sequence = 2 * cp_size
    slice_sizes = (
        cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    ) // total_slices_of_any_sequence
    sum_len = cu_seqlens_padded[-1].item()
    
    def process_tensor(val):
        if val is None:
            return val
        if isinstance(cu_seqlens_padded[-1], torch.Tensor):
            seq_len_val = cu_seqlens_padded[-1].item()
        else:
            seq_len_val = cu_seqlens_padded[-1]

        if val.ndim == 1:
            if val.shape[0] == seq_len_val:
                current_seq_dim = 0
            else:
                raise ValueError(
                    "1D tensor shape doesn't match expected sequence length. Make sure the"
                    " inputs are in THD format and padded correctly."
                )
        elif val.ndim >= 2:
            if val.shape[1] == seq_len_val:
                current_seq_dim = 1
            elif val.shape[0] == seq_len_val:
                current_seq_dim = 0
            else:
                raise ValueError(
                    f"Make sure the inputs are in THD format and padded correctly. cu_seqlens_padded: {cu_seqlens_padded}, val.shape: {val.shape}."
                )
        else:
            raise ValueError("Tensor must be at least 1D")

        cp_rank_slices = []
        for rank in range(cp_size):
            for slice_size, seq_start in zip(slice_sizes, cu_seqlens_padded[:-1]):
                # 1st segment
                cp_rank_slices.append(
                    torch.arange(
                        seq_start + (rank * slice_size),
                        seq_start + ((rank + 1) * slice_size),
                        device=val.device,
                    )
                )

                # 2nd segment
                cp_rank_slices.append(
                    torch.arange(
                        seq_start + ((total_slices_of_any_sequence - rank - 1) * slice_size),
                        seq_start + ((total_slices_of_any_sequence - rank) * slice_size),
                        device=val.device,
                    )
                )
        perm = torch.cat(cp_rank_slices)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(sum_len, device=val.device)
        
        # Create a tensor to hold the unshuffled result
        unshuffled = val.index_select(current_seq_dim, inv_perm)
        local_tensor = torch.chunk(unshuffled, cp_size, dim=current_seq_dim)[cp_rank]
        return local_tensor

    full_tensor = allgather_reducescatter(t, 0, cp_ranks)
    return process_tensor(full_tensor)
