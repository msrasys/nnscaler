# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Exact query-pruned YOCO attention with dense NNScaler-visible shapes."""

from bisect import bisect_left, bisect_right
from collections import OrderedDict
from dataclasses import dataclass
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


@dataclass
class _QueryRunBatch:
    # None means the batch already covers Q in its original dense order.
    query_indices: Optional[Tensor]
    cu_seqlens_q: Tensor
    cu_seqlens_k: Tensor
    key_start: int
    key_end: int


@dataclass
class _QueryRunMetadata:
    batches: List[_QueryRunBatch]
    output_indices: Optional[Tensor]
    lse_indices: Optional[Tensor]
    max_seqlen_q: int
    max_seqlen_k: int


@torch.inference_mode(False)
@torch.no_grad()
def _query_run_metadata(
    query_mask: Tensor,
    query_positions: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    cp_size: int,
) -> _QueryRunMetadata:
    # Copy metadata once per mask/layout, rather than synchronizing for every
    # sequence and every layer. Each round takes at most one run per sequence,
    # so its K/V prefixes never overlap and can share the original K/V storage.
    query_offsets = cu_seqlens_q.tolist()
    key_offsets = cu_seqlens_k.tolist()
    if len(query_offsets) != len(key_offsets):
        raise ValueError("Q and K/V must describe the same packed sequences")
    sequence_lengths = [
        end - start for start, end in zip(query_offsets, query_offsets[1:])
    ]
    if cp_size > 1 and any(length % (2 * cp_size) for length in sequence_lengths):
        raise ValueError(
            "query-pruned YOCO attention requires every sequence length to "
            f"be divisible by 2 * cp_size ({2 * cp_size})"
        )
    local_tokens = sum(sequence_lengths) // cp_size
    if local_tokens != query_mask.numel():
        raise ValueError(
            "query mask length does not match the local zigzag token layout: "
            f"mask={query_mask.numel()}, expected={local_tokens}"
        )

    active_idx = torch.nonzero(query_mask.bool(), as_tuple=False).flatten()
    active_rows = active_idx.tolist()
    positions = query_positions.index_select(0, active_idx).tolist()
    rounds = []
    local_end = query_start = 0
    for sequence_idx, length in enumerate(sequence_lengths):
        local_end += length // cp_size
        query_end = bisect_left(active_rows, local_end, lo=query_start)
        key_start = key_offsets[sequence_idx]
        key_length = key_offsets[sequence_idx + 1] - key_start
        for round_idx, (start, end, last_position) in enumerate(
            _consecutive_position_runs(positions[query_start:query_end])
        ):
            if positions[query_start + start] < 0 or last_position >= key_length:
                raise ValueError(
                    "query position falls outside its packed K/V sequence: "
                    f"position={last_position}, length={key_length}"
                )
            if round_idx == len(rounds):
                rounds.append([])
            rounds[round_idx].append((
                query_start + start, query_start + end,
                key_start, key_start + last_position + 1,
            ))
        query_start = query_end

    batches = []
    output_rows = []
    for runs in rounds:
        rows = []
        cu_q, cu_k = [0], [0]
        key_start = key_end = runs[0][2]
        for start, end, run_key_start, run_key_end in runs:
            if run_key_start > key_end:
                # An empty-Q sequence consumes the unused K/V gap, preserving
                # the next prefix's offset without copying or duplicating K/V.
                # Keep original sequence boundaries inside the gap so no
                # dummy K sequence exceeds the caller's max_seqlen_k bound.
                for gap_end in key_offsets[
                    bisect_right(key_offsets, key_end):
                    bisect_right(key_offsets, run_key_start)
                ]:
                    cu_q.append(cu_q[-1])
                    cu_k.append(gap_end - key_start)
            rows.extend(active_rows[start:end])
            cu_q.append(cu_q[-1] + end - start)
            cu_k.append(run_key_end - key_start)
            key_end = run_key_end
        indices = (
            None if len(rows) == query_mask.numel()
            else torch.tensor(rows, dtype=torch.long, device=query_mask.device)
        )
        batches.append(_QueryRunBatch(
            indices,
            torch.tensor(cu_q, dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device),
            torch.tensor(cu_k, dtype=cu_seqlens_k.dtype, device=cu_seqlens_k.device),
            key_start,
            key_end,
        ))
        output_rows.extend(rows)

    output_indices = (
        None if len(batches) == 1 and batches[0].query_indices is None
        else torch.tensor(output_rows, dtype=torch.long, device=query_mask.device)
    )
    # Outputs are batched by run number; the public LSE remains in compact Q
    # order, just as it was when runs were executed sequence by sequence.
    lse_indices = output_indices.argsort() if len(batches) > 1 else None
    return _QueryRunMetadata(
        batches, output_indices, lse_indices,
        max(sequence_lengths, default=0),
        max((end - start for start, end in zip(
            key_offsets, key_offsets[1:])), default=0),
    )


def _cached_query_run_metadata(
    query_mask: Tensor,
    query_positions: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    cp_size: int,
):
    tensors = (query_mask, query_positions, cu_seqlens_q, cu_seqlens_k)
    key = tuple((id(tensor), tensor._version) for tensor in tensors) + (cp_size,)
    cached = _QUERY_METADATA_CACHE.get(key)
    if cached is not None:
        cached_tensors, metadata = cached
        if all(old is new for old, new in zip(cached_tensors, tensors)):
            _QUERY_METADATA_CACHE.move_to_end(key)
            return metadata
    metadata = _query_run_metadata(*tensors, cp_size)
    _QUERY_METADATA_CACHE[key] = (tensors, metadata)
    if len(_QUERY_METADATA_CACHE) > _QUERY_METADATA_CACHE_MAXSIZE:
        _QUERY_METADATA_CACHE.popitem(last=False)
    return metadata


def _consecutive_position_runs(
    positions: List[int],
) -> List[Tuple[int, int, int]]:
    """Return compact-row runs whose original positions are consecutive.

    Each tuple contains ``(row_start, row_end, last_position)``.  A standard
    causal attention call over a run and the K/V prefix ending at
    ``last_position`` has exactly the same mask as the corresponding rows in
    dense self attention, including FlashAttention's bottom-right alignment.
    """
    if not positions:
        return []
    runs = []
    row_start = 0
    for row in range(1, len(positions)):
        if positions[row] != positions[row - 1] + 1:
            runs.append((row_start, row, positions[row - 1]))
            row_start = row
    runs.append((row_start, len(positions), positions[-1]))
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

    metadata = _cached_query_run_metadata(
        query_mask,
        query_positions,
        cu_seqlens_q,
        cu_seqlens_k,
        resolved_cp_size,
    )
    if not metadata.batches:
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

    if max_seqlen_q is None:
        max_seqlen_q = metadata.max_seqlen_q
    if max_seqlen_k is None:
        max_seqlen_k = metadata.max_seqlen_k

    # FA4 4.0.0b13's varlen autograd wrapper drops mask_mod in backward.
    # Fixed in 4.0.0b17: https://github.com/Dao-AILab/flash-attention/pull/2616.
    # Support older runtimes by splitting Q into consecutive original-position
    # runs and batching runs from different sequences against their K/V prefixes.
    # For a run [start, end], bottom-right causal alignment against K[:end+1]
    # maps query row zero back to original position ``start``.  Sliding-window
    # attention follows the same alignment.
    compact_outputs = []
    compact_lses = []
    cute_window_size = tuple(None if item == -1 else item for item in window_size)
    for batch in metadata.batches:
        batch_q = (
            q if batch.query_indices is None
            else q.index_select(0, batch.query_indices)
        )
        full_kv = batch.key_start == 0 and batch.key_end == k.size(0)
        batch_k = k if full_kv else k[batch.key_start:batch.key_end]
        batch_v = v if full_kv else v[batch.key_start:batch.key_end]
        batch_output, batch_lse = flash_attn_cute_varlen_func(
            batch_q,
            batch_k,
            batch_v,
            cu_seqlens_q=batch.cu_seqlens_q,
            cu_seqlens_k=batch.cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=True,
            window_size=cute_window_size,
            deterministic=deterministic,
            return_lse=True,
        )
        compact_outputs.append(batch_output)
        compact_lses.append(batch_lse)

    compact_output = (
        compact_outputs[0] if len(compact_outputs) == 1
        else torch.cat(compact_outputs, dim=0)
    )
    softmax_lse = (
        compact_lses[0] if len(compact_lses) == 1
        else torch.cat(compact_lses, dim=-1)
    )
    if metadata.lse_indices is not None:
        softmax_lse = softmax_lse.index_select(-1, metadata.lse_indices)
    if metadata.output_indices is None:
        return (compact_output, softmax_lse) if return_lse else compact_output
    dense_output = compact_output.new_zeros(
        q.size(0), compact_output.size(1), compact_output.size(2))
    dense_output = torch.index_copy(
        dense_output, 0, metadata.output_indices, compact_output)
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
