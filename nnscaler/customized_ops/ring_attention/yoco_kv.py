#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Optional, Tuple

import torch
from torch import Tensor

from nnscaler.graph.parser.register import register_op
from nnscaler.runtime.adapter.nn import allgather_reducescatter

from .maybe_shuffle import emit_ring


def wrap_yoco_kv_allgather(
    key: Tensor,
    value: Tensor,
    enable_ring: bool = True,
    process_group: Tuple[int] = None,
    cp_size: Optional[int] = None,
    require_full_plan_sequence_partition: bool = False,
):
    """Gather shared YOCO K/V once for all cross-attention layers.

    K and V are fused along the head dimension so the forward performs one
    all-gather.  The adapter's backward performs the matching reduce-scatter
    after gradients from every consumer of the shared tensors have accumulated.

    The registered graph annotation intentionally preserves the local
    sequence layout.  At runtime the returned tensors contain the complete CP
    lane; consumers must set ``kv_is_gathered=True`` to skip their internal
    K/V gather.
    """
    if not isinstance(require_full_plan_sequence_partition, bool):
        raise ValueError(
            "require_full_plan_sequence_partition must be a bool, got "
            f"{require_full_plan_sequence_partition!r}"
        )
    if cp_size is not None:
        if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size < 1:
            raise ValueError(
                f"cp_size must be a positive int or None, got {cp_size!r}"
            )
        if process_group is not None and len(process_group) != cp_size:
            raise ValueError(
                f"process_group size ({len(process_group)}) must match "
                f"cp_size ({cp_size})"
            )
        if cp_size == 1:
            enable_ring = False

    if key.dim() != 3 or value.dim() != 3:
        raise ValueError(
            "YOCO K/V must be 3D [tokens, heads, head_dim], got "
            f"{tuple(key.shape)} and {tuple(value.shape)}"
        )
    if key.shape[0] != value.shape[0] or key.shape[2] != value.shape[2]:
        raise ValueError(
            "YOCO K/V must have matching token and head dimensions, got "
            f"{tuple(key.shape)} and {tuple(value.shape)}"
        )

    if process_group is None or len(process_group) == 1 or not enable_ring:
        return key, value

    key_heads = key.size(1)
    value_heads = value.size(1)
    fused_kv = torch.cat((key, value), dim=1).contiguous()
    gathered_kv = allgather_reducescatter(
        fused_kv, 0, tuple(process_group)
    )
    return gathered_kv.split((key_heads, value_heads), dim=1)


def yoco_kv_anno(key, value, *args, **kwargs) -> str:
    # The runtime outputs are CP-complete, but the logical graph layout stays
    # sequence-sharded so all cross-attention consumers share one materialized
    # tensor without introducing graph-level replication adapters.
    return "l kh d^, l vh d^ -> l kh d^, l vh d^"


register_op(yoco_kv_anno, emit_fn=emit_ring)(wrap_yoco_kv_allgather)
