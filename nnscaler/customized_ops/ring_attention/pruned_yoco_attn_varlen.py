# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact query-pruned YOCO attention with dense NNScaler-visible shapes."""

from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
from torch import Tensor

try:
    from flash_attn.cute import (
        flash_attn_varlen_func as flash_attn_cute_varlen_func,
    )
except ImportError:
    flash_attn_cute_varlen_func = None

from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.parser.register import register_op
from nnscaler.ir import IRTensor
from nnscaler.runtime.adapter.nn import allgather_reducescatter

from .core.utils import gen_head_anno
from .zigzag_allgather_attn_varlen import emit_ring


_QUERY_METADATA_CACHE_MAXSIZE = 8
_QUERY_METADATA_CACHE = OrderedDict()


def _local_sequence_lengths(cu_seqlens_q: Tensor, cp_size: int) -> Tensor:
    sequence_lengths = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    if cp_size == 1:
        return sequence_lengths
    if torch.any(sequence_lengths % (2 * cp_size) != 0):
        raise ValueError(
            "query-pruned YOCO attention requires every sequence length to "
            f"be divisible by 2 * cp_size ({2 * cp_size})"
        )
    return sequence_lengths // cp_size


def _compact_query_metadata(
    query_mask: Tensor,
    query_positions: Tensor,
    cu_seqlens_q: Tensor,
    cp_size: int,
):
    local_sequence_lengths = _local_sequence_lengths(cu_seqlens_q, cp_size)
    local_cu = torch.zeros(
        local_sequence_lengths.numel() + 1,
        dtype=cu_seqlens_q.dtype,
        device=cu_seqlens_q.device,
    )
    local_cu[1:] = torch.cumsum(local_sequence_lengths, dim=0)
    if local_cu[-1].item() != query_mask.numel():
        raise ValueError(
            "query mask length does not match the local zigzag token layout: "
            f"mask={query_mask.numel()}, expected={local_cu[-1].item()}"
        )

    active_idx = torch.nonzero(query_mask.bool(), as_tuple=False).flatten()
    active_counts = torch.stack([
        query_mask[local_cu[index]:local_cu[index + 1]].sum(dtype=torch.int32)
        for index in range(local_sequence_lengths.numel())
    ])
    compact_cu = torch.zeros_like(local_cu)
    compact_cu[1:] = torch.cumsum(active_counts, dim=0)
    compact_positions = query_positions.index_select(0, active_idx).to(torch.int32)
    return active_idx, compact_positions.contiguous(), compact_cu.contiguous()


def _cached_compact_query_metadata(
    query_mask: Tensor,
    query_positions: Tensor,
    cu_seqlens_q: Tensor,
    cp_size: int,
):
    key = (
        id(query_mask),
        query_mask._version,
        id(query_positions),
        query_positions._version,
        id(cu_seqlens_q),
        cu_seqlens_q._version,
        cp_size,
    )
    cached = _QUERY_METADATA_CACHE.get(key)
    if cached is not None:
        cached_mask, cached_positions, cached_cu, metadata = cached
        if (
            cached_mask is query_mask
            and cached_positions is query_positions
            and cached_cu is cu_seqlens_q
        ):
            _QUERY_METADATA_CACHE.move_to_end(key)
            return metadata
    metadata = _compact_query_metadata(
        query_mask, query_positions, cu_seqlens_q, cp_size)
    _QUERY_METADATA_CACHE[key] = (
        query_mask, query_positions, cu_seqlens_q, metadata)
    if len(_QUERY_METADATA_CACHE) > _QUERY_METADATA_CACHE_MAXSIZE:
        _QUERY_METADATA_CACHE.popitem(last=False)
    return metadata


def _consecutive_position_runs(
    positions: Tensor,
) -> List[Tuple[int, int, int]]:
    """Return compact-row runs whose original positions are consecutive.

    Each tuple contains ``(row_start, row_end, last_position)``.  A standard
    causal attention call over a run and the K/V prefix ending at
    ``last_position`` has exactly the same mask as the corresponding rows in
    dense self attention, including FlashAttention's bottom-right alignment.
    """
    if positions.numel() == 0:
        return []
    host_positions = positions.tolist()
    runs = []
    row_start = 0
    for row in range(1, len(host_positions)):
        if host_positions[row] != host_positions[row - 1] + 1:
            runs.append((row_start, row, host_positions[row - 1]))
            row_start = row
    runs.append((row_start, len(host_positions), host_positions[-1]))
    return runs


def wrap_pruned_yoco_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    query_mask: Tensor,
    query_positions: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    alibi_slopes: Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Tensor = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    return_attn_probs: bool = False,
    enable_ring: bool = True,
    use_cute: bool = True,
    process_group: Tuple[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    return_lse: bool = True,
    cp_size: Optional[int] = None,
    cp_sharded_kv: bool = True,
    kv_is_gathered: bool = False,
    require_full_plan_sequence_partition: bool = False,
):
    """Run attention only for selected Q rows and scatter into a dense output."""
    del require_full_plan_sequence_partition
    if not use_cute:
        raise ValueError("query-pruned YOCO attention requires use_cute=True")
    if flash_attn_cute_varlen_func is None:
        raise RuntimeError("flash-attn CuTe support is unavailable")
    if not causal:
        raise ValueError("query-pruned YOCO attention requires causal=True")
    if return_attn_probs:
        raise ValueError("return_attn_probs is not supported")
    if dropout_p != 0.0:
        raise ValueError("query-pruned YOCO attention does not support dropout")
    if alibi_slopes is not None:
        raise ValueError("query-pruned YOCO attention does not support ALiBi")
    if query_mask.dim() != 1 or query_mask.numel() != q.size(0):
        raise ValueError("query mask must match the dense local Q rows")
    if query_positions.shape != query_mask.shape:
        raise ValueError("query positions must match the query mask")

    # During graph tracing the wrapper receives the full tensor and no runtime
    # process group, even when the logical op carries an explicit CP size.  The
    # emitter supplies the real group after sequence partitioning.
    resolved_cp_size = (
        len(process_group) if process_group is not None else 1
    )
    if resolved_cp_size < 1:
        raise ValueError("cp_size must be positive")

    if cp_sharded_kv and not kv_is_gathered and resolved_cp_size > 1:
        if process_group is None:
            raise ValueError("CP-sharded K/V requires an explicit process group")
        cp_ranks = tuple(process_group)
        k = allgather_reducescatter(k, 0, cp_ranks)
        v = allgather_reducescatter(v, 0, cp_ranks)

    active_idx, compact_positions, compact_cu_q = _cached_compact_query_metadata(
        query_mask,
        query_positions,
        cu_seqlens_q,
        resolved_cp_size,
    )
    if active_idx.numel() == 0:
        # Keep every projection and the shared K/V gather in the autograd
        # graph.  In particular, all CP ranks must enter the gather's
        # reduce-scatter backward even when one rank owns no active queries.
        # Touch one scalar per tensor to avoid a full reduction over the large
        # gathered K/V while still producing correctly shaped zero gradients.
        zero_dependency = q.reshape(-1)[0] * 0
        zero_dependency = zero_dependency + k.reshape(-1)[0] * 0
        zero_dependency = zero_dependency + v.reshape(-1)[0] * 0
        dense_output = q.new_zeros(
            q.size(0), q.size(1), v.size(-1)) + zero_dependency
        return (dense_output, None) if return_lse else dense_output

    compact_q = q.index_select(0, active_idx)
    if max_seqlen_q is None:
        max_seqlen_q = int(
            (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item())
    if max_seqlen_k is None:
        max_seqlen_k = int(
            (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item())

    # FA4 4.0.0b13 produces incorrect gradients for mask_mod + aux_tensors on
    # SM100 even though its forward result is exact.  Avoid that backward path:
    # split selected Q rows into consecutive original-position runs and use an
    # ordinary causal varlen call against the K/V prefix ending at each run.
    # For a run [start, end], bottom-right causal alignment against K[:end+1]
    # maps query row zero back to original position ``start``.  Sliding-window
    # attention follows the same alignment.
    compact_outputs = []
    compact_lses = []
    cute_window_size = tuple(None if item == -1 else item for item in window_size)
    query_offsets = compact_cu_q.tolist()
    key_offsets = cu_seqlens_k.tolist()
    for sequence_idx in range(len(query_offsets) - 1):
        query_start = query_offsets[sequence_idx]
        query_end = query_offsets[sequence_idx + 1]
        if query_start == query_end:
            continue
        key_start = key_offsets[sequence_idx]
        sequence_positions = compact_positions[query_start:query_end]
        sequence_key_length = key_offsets[sequence_idx + 1] - key_start
        for run_start, run_end, last_position in _consecutive_position_runs(
            sequence_positions
        ):
            if last_position < 0 or last_position >= sequence_key_length:
                raise ValueError(
                    "query position falls outside its packed K/V sequence: "
                    f"position={last_position}, length={sequence_key_length}"
                )
            run_q_start = query_start + run_start
            run_q_end = query_start + run_end
            run_q = compact_q[run_q_start:run_q_end]
            run_kv_end = key_start + last_position + 1
            run_k = k[key_start:run_kv_end]
            run_v = v[key_start:run_kv_end]
            run_cu_q = torch.tensor(
                [0, run_q.size(0)],
                dtype=cu_seqlens_q.dtype,
                device=cu_seqlens_q.device,
            )
            run_cu_k = torch.tensor(
                [0, run_k.size(0)],
                dtype=cu_seqlens_k.dtype,
                device=cu_seqlens_k.device,
            )
            run_output, run_lse = flash_attn_cute_varlen_func(
                run_q,
                run_k,
                run_v,
                cu_seqlens_q=run_cu_q,
                cu_seqlens_k=run_cu_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=cute_window_size,
                deterministic=deterministic,
                return_lse=True,
            )
            compact_outputs.append(run_output)
            compact_lses.append(run_lse)

    compact_output = torch.cat(compact_outputs, dim=0)
    softmax_lse = torch.cat(compact_lses, dim=-1)
    dense_output = compact_output.new_zeros(
        q.size(0), compact_output.size(1), compact_output.size(2))
    dense_output = torch.index_copy(
        dense_output, 0, active_idx, compact_output)
    return (dense_output, softmax_lse) if return_lse else dense_output


def pruned_attention_anno(
    query_states,
    key_states,
    value_states,
    query_mask,
    query_positions,
    cu_seqlens_q,
    cu_seqlens_k,
    alibi_slopes,
    *args,
    **kwargs,
) -> str:
    q_anno, kv_anno = gen_head_anno(
        query_states, key_states, value_states, head_pos=1)
    alibi_anno = f'{q_anno}' if isinstance(alibi_slopes, IRTensor) else '?'
    cp_sharded_kv = kwargs.get('cp_sharded_kv', True)
    kv_sequence_anno = 'l' if cp_sharded_kv else 'al^'
    return (
        f"l {q_anno} hd^, {kv_sequence_anno} {kv_anno} hd^, "
        f"{kv_sequence_anno} {kv_anno} vd^, l, l, e^, e^, {alibi_anno} "
        f"-> l {q_anno} vd^, ?"
    )


def input_gen_fn(node: IRDimops):
    device = torch.cuda.current_device()
    inputs = []
    for index, tensor in enumerate(node.inputs()):
        if index < 3:
            inputs.append(torch.randn(
                tensor.shape,
                dtype=tensor.dtype,
                device=device,
                requires_grad=tensor.requires_grad,
            ))
        elif index == 3:
            inputs.append(torch.ones(tensor.shape, dtype=torch.bool, device=device))
        elif index == 4:
            inputs.append(torch.arange(
                tensor.shape[0], dtype=torch.int32, device=device))
        elif index in (5, 6):
            inputs.append(torch.tensor(
                [0, node.inputs()[0].shape[0]], dtype=torch.int32, device=device))
        elif index == 7:
            inputs.append(None if not isinstance(tensor, IRTensor) else torch.randn(
                tensor.shape, dtype=tensor.dtype, device=device))
        else:
            break
    return tuple(inputs)


register_op(
    pruned_attention_anno,
    emit_fn=emit_ring,
    input_gen_fn=input_gen_fn,
)(wrap_pruned_yoco_attn_varlen_func)
