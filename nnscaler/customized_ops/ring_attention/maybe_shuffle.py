#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.parser.register import register_op
from nnscaler.runtime.device import DeviceGroup

from .varlen_utils import shuffle_varlen, unshuffle_varlen


def _get_cp_ranks(process_group) -> List[int]:
    world_size = dist.get_world_size(process_group)
    return [dist.get_global_rank(process_group, rank) for rank in range(world_size)]


def _resolve_local_process_group(process_group: Tuple[int]):
    local_process_group = DeviceGroup().get_group(process_group)
    if local_process_group is None:
        local_process_group = dist.group.WORLD
    return local_process_group


def wrap_maybe_shuffle(
    hidden_states: Tensor,
    cu_seqlens: Tensor,
    enable_ring: bool = True,
    process_group: Tuple[int] = None,
    cp_size: Optional[int] = None,
    require_full_plan_sequence_partition: bool = False,
):
    if not isinstance(require_full_plan_sequence_partition, bool):
        raise ValueError(
            "require_full_plan_sequence_partition must be a bool, got "
            f"{require_full_plan_sequence_partition!r}"
        )
    if cp_size is not None:
        if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size < 1:
            raise ValueError(f"cp_size must be a positive int or None, got {cp_size!r}")
        if process_group is not None and len(process_group) != cp_size:
            raise ValueError(
                f"process_group size ({len(process_group)}) must match cp_size ({cp_size})"
            )
        if cp_size == 1:
            enable_ring = False
    if process_group is None or len(process_group) == 1 or not enable_ring:
        return hidden_states
    local_process_group = _resolve_local_process_group(process_group)
    cp_ranks = _get_cp_ranks(local_process_group)
    return shuffle_varlen(hidden_states, cu_seqlens, cp_ranks, local_process_group)


def wrap_maybe_unshuffle(
    hidden_states: Tensor,
    cu_seqlens: Tensor,
    enable_ring: bool = True,
    process_group: Tuple[int] = None,
    cp_size: Optional[int] = None,
    require_full_plan_sequence_partition: bool = False,
):
    if not isinstance(require_full_plan_sequence_partition, bool):
        raise ValueError(
            "require_full_plan_sequence_partition must be a bool, got "
            f"{require_full_plan_sequence_partition!r}"
        )
    if cp_size is not None:
        if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size < 1:
            raise ValueError(f"cp_size must be a positive int or None, got {cp_size!r}")
        if process_group is not None and len(process_group) != cp_size:
            raise ValueError(
                f"process_group size ({len(process_group)}) must match cp_size ({cp_size})"
            )
        if cp_size == 1:
            enable_ring = False
    if process_group is None or len(process_group) == 1 or not enable_ring:
        return hidden_states
    local_process_group = _resolve_local_process_group(process_group)
    cp_ranks = _get_cp_ranks(local_process_group)
    return unshuffle_varlen(hidden_states, cu_seqlens, cp_ranks, local_process_group)


def wrap_maybe_shuffle_with_query_metadata(
    hidden_states: Tensor,
    query_mask: Tensor,
    query_positions: Tensor,
    cu_seqlens: Tensor,
    enable_ring: bool = True,
    process_group: Tuple[int] = None,
    cp_size: Optional[int] = None,
    require_full_plan_sequence_partition: bool = False,
):
    """Move hidden state and pruning metadata to identical zigzag ownership."""
    if cp_size is not None:
        if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size < 1:
            raise ValueError(f"cp_size must be a positive int or None, got {cp_size!r}")
        if process_group is not None and len(process_group) != cp_size:
            raise ValueError(
                f"process_group size ({len(process_group)}) must match cp_size ({cp_size})")
        if cp_size == 1:
            enable_ring = False
    if process_group is None or len(process_group) == 1 or not enable_ring:
        return hidden_states, query_mask, query_positions
    local_process_group = _resolve_local_process_group(process_group)
    cp_ranks = _get_cp_ranks(local_process_group)
    return (
        shuffle_varlen(hidden_states, cu_seqlens, cp_ranks, local_process_group),
        shuffle_varlen(query_mask, cu_seqlens, cp_ranks, local_process_group),
        shuffle_varlen(query_positions, cu_seqlens, cp_ranks, local_process_group),
    )


def wrap_maybe_shuffle_with_query_and_mtp_metadata(
    hidden_states: Tensor,
    query_mask: Tensor,
    query_positions: Tensor,
    mtp_input_ids: Tensor,
    mtp_query_masks: Tensor,
    cu_seqlens: Tensor,
    enable_ring: bool = True,
    process_group: Tuple[int] = None,
    cp_size: Optional[int] = None,
    require_full_plan_sequence_partition: bool = False,
):
    """Shuffle every tensor needed by pruned cross and MTP execution."""
    if cp_size is not None:
        if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size < 1:
            raise ValueError(f"cp_size must be a positive int or None, got {cp_size!r}")
        if process_group is not None and len(process_group) != cp_size:
            raise ValueError(
                f"process_group size ({len(process_group)}) must match cp_size ({cp_size})")
        if cp_size == 1:
            enable_ring = False
    if process_group is None or len(process_group) == 1 or not enable_ring:
        return (
            hidden_states,
            query_mask,
            query_positions,
            mtp_input_ids,
            mtp_query_masks,
        )
    local_process_group = _resolve_local_process_group(process_group)
    cp_ranks = _get_cp_ranks(local_process_group)
    return tuple(
        shuffle_varlen(tensor, cu_seqlens, cp_ranks, local_process_group)
        for tensor in (
            hidden_states,
            query_mask,
            query_positions,
            mtp_input_ids,
            mtp_query_masks,
        )
    )


def emit_ring(
    node: IRDimops,
    args: List[str],
    kwargs: Dict[str, str],
    runtime_devid: int,
    plan_ndevs: int,
    runtime_ndevs: int,
) -> str:
    signature = node.signature

    offset = (runtime_devid // plan_ndevs) * plan_ndevs
    remainder = runtime_devid % plan_ndevs

    kw_pairs = []
    for key, val in kwargs.items():
        if key == 'process_group':
            continue
        kw_pairs.append(f"{key}={val}")

    sub_input = node.inputs()[0]
    full_input = sub_input.parent
    partition_dims = [(i, f // s) for i, (s, f) in enumerate(zip(sub_input.shape, full_input.shape)) if s != f]
    assert len(partition_dims) <= 1, f"support no more than one partition dim, but got {partition_dims}"
    explicit_cp_size = node.kwargs.get('cp_size')
    if explicit_cp_size is not None and (
        not isinstance(explicit_cp_size, int)
        or isinstance(explicit_cp_size, bool)
        or explicit_cp_size < 1
    ):
        raise ValueError(
            f"cp_size must be a positive int or None, got {explicit_cp_size!r}"
        )
    require_full_partition = node.kwargs.get(
        'require_full_plan_sequence_partition', False
    )
    if not isinstance(require_full_partition, bool):
        raise ValueError(
            "require_full_plan_sequence_partition must be a bool, got "
            f"{require_full_partition!r}"
        )
    strict_sequence_partition = require_full_partition or (
        explicit_cp_size is not None and explicit_cp_size > 1
    )
    strict_reason = (
        f"data-lane layout with cp_size={explicit_cp_size}"
        if require_full_partition
        else f"explicit cp_size={explicit_cp_size}"
    )
    if not partition_dims:
        if strict_sequence_partition:
            raise ValueError(
                f"{strict_reason} requires partitioning "
                "the sequence dimension across the full plan"
            )
        kw_pairs.append("process_group=None")
    else:
        if partition_dims[0][0] == 0:
            partition_degree = partition_dims[0][1]
            cp_size = explicit_cp_size
            if cp_size is None:
                cp_size = partition_degree
            if plan_ndevs % cp_size != 0:
                raise ValueError(
                    f"cp_size ({cp_size}) must divide plan size ({plan_ndevs})"
                )
            if strict_sequence_partition and partition_degree != plan_ndevs:
                raise ValueError(
                    f"sequence partition degree ({partition_degree}) must equal "
                    f"plan size ({plan_ndevs}) for {strict_reason}"
                )
            if partition_degree % cp_size != 0:
                raise ValueError(
                    f"sequence partition degree ({partition_degree}) must be a "
                    f"multiple of cp_size ({cp_size})"
                )
            if cp_size == 1:
                kw_pairs.append("process_group=None")
            else:
                group_start = remainder // cp_size * cp_size
                scale_unit_dev_ids = [
                    local_rank + offset
                    for local_rank in range(group_start, group_start + cp_size)
                ]
                kw_pairs.append(f"process_group={scale_unit_dev_ids}")
        elif partition_dims[0][0] == 1:
            if strict_sequence_partition:
                raise ValueError(
                    f"{strict_reason} requires sequence-dimension "
                    "partitioning, not head-dimension partitioning"
                )
            kw_pairs.append("process_group=None")
        else:
            raise ValueError(f"unsupported partition dim: {partition_dims[0]}")

    args = ", ".join(list(args) + kw_pairs)
    return f"{signature}({args})"


def maybe_anno(hidden_states, cu_seqlens, *args, **kwargs) -> str:
    return "l h, e^ -> l h"


def shuffle_with_query_metadata_anno(
    hidden_states, query_mask, query_positions, cu_seqlens, *args, **kwargs,
) -> str:
    return "l h, l, l, e^ -> l h, l, l"


def shuffle_with_query_and_mtp_metadata_anno(
    hidden_states,
    query_mask,
    query_positions,
    mtp_input_ids,
    mtp_query_masks,
    cu_seqlens,
    *args,
    **kwargs,
) -> str:
    return (
        "l h, l, l, l m, l m, e^ -> "
        "l h, l, l, l m, l m"
    )


def _profile_tensor(shape, dtype, device, requires_grad=False):
    if dtype.is_floating_point or dtype.is_complex:
        return torch.randn(
            shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
    if requires_grad:
        raise ValueError(
            f"non-floating profile tensor with dtype {dtype} cannot require "
            "gradients"
        )
    return torch.zeros(shape, dtype=dtype, device=device)


def input_gen_fn(node: IRDimops):
    hidden_states = node.inputs()[0]
    device = torch.cuda.current_device()
    seqlen = hidden_states.shape[0]
    return (
        _profile_tensor(
            hidden_states.shape,
            hidden_states.dtype,
            device,
            hidden_states.requires_grad,
        ),
        torch.tensor([0, seqlen], dtype=torch.int32, device=device),
    )


def query_metadata_input_gen_fn(node: IRDimops):
    hidden_states = node.inputs()[0]
    device = torch.cuda.current_device()
    seqlen = hidden_states.shape[0]
    return (
        _profile_tensor(
            hidden_states.shape,
            hidden_states.dtype,
            device,
            hidden_states.requires_grad,
        ),
        torch.ones(seqlen, dtype=torch.bool, device=device),
        torch.arange(seqlen, dtype=torch.int32, device=device),
        torch.tensor([0, seqlen], dtype=torch.int32, device=device),
    )


def query_and_mtp_metadata_input_gen_fn(node: IRDimops):
    hidden_states = node.inputs()[0]
    mtp_depth = node.inputs()[3].shape[1]
    device = torch.cuda.current_device()
    seqlen = hidden_states.shape[0]
    return (
        _profile_tensor(
            hidden_states.shape,
            hidden_states.dtype,
            device,
            hidden_states.requires_grad,
        ),
        torch.ones(seqlen, dtype=torch.bool, device=device),
        torch.arange(seqlen, dtype=torch.int32, device=device),
        torch.zeros(seqlen, mtp_depth, dtype=torch.long, device=device),
        torch.ones(seqlen, mtp_depth, dtype=torch.bool, device=device),
        torch.tensor([0, seqlen], dtype=torch.int32, device=device),
    )


register_op(maybe_anno, emit_fn=emit_ring, input_gen_fn=input_gen_fn)(wrap_maybe_shuffle)
register_op(maybe_anno, emit_fn=emit_ring, input_gen_fn=input_gen_fn)(wrap_maybe_unshuffle)
register_op(
    shuffle_with_query_metadata_anno,
    emit_fn=emit_ring,
    input_gen_fn=query_metadata_input_gen_fn,
)(wrap_maybe_shuffle_with_query_metadata)
register_op(
    shuffle_with_query_and_mtp_metadata_anno,
    emit_fn=emit_ring,
    input_gen_fn=query_and_mtp_metadata_input_gen_fn,
)(wrap_maybe_shuffle_with_query_and_mtp_metadata)
