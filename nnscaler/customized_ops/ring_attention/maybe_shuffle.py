#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Dict, List, Tuple

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
):
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
):
    if process_group is None or len(process_group) == 1 or not enable_ring:
        return hidden_states
    local_process_group = _resolve_local_process_group(process_group)
    cp_ranks = _get_cp_ranks(local_process_group)
    return unshuffle_varlen(hidden_states, cu_seqlens, cp_ranks, local_process_group)


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
        kw_pairs.append(f"{key}={val}")

    sub_input = node.inputs()[0]
    full_input = sub_input.parent
    partition_dims = [(i, f // s) for i, (s, f) in enumerate(zip(sub_input.shape, full_input.shape)) if s != f]
    assert len(partition_dims) <= 1, f"support no more than one partition dim, but got {partition_dims}"
    if not partition_dims:
        kw_pairs.append("process_group=None")
    else:
        if partition_dims[0][0] == 0:
            num = partition_dims[0][1]
            scale_unit_dev_ids = [
                local_rank + offset
                for local_rank in range(remainder // num * num, (remainder // num + 1) * num)
            ]
            kw_pairs.append(f"process_group={scale_unit_dev_ids}")
        elif partition_dims[0][0] == 1:
            kw_pairs.append("process_group=None")
        else:
            raise ValueError(f"unsupported partition dim: {partition_dims[0]}")

    args = ", ".join(list(args) + kw_pairs)
    return f"{signature}({args})"


def maybe_anno(hidden_states, cu_seqlens, *args, **kwargs) -> str:
    return "l h, e^ -> l h"



def input_gen_fn(node: IRDimops):
    hidden_states = node.inputs()[0]
    device = torch.cuda.current_device()
    seqlen = hidden_states.shape[0]
    return (
        torch.randn(hidden_states.shape, dtype=hidden_states.dtype, device=device, requires_grad=hidden_states.requires_grad),
        torch.tensor([0, seqlen], dtype=torch.int32, device=device),
    )


register_op(maybe_anno, emit_fn=emit_ring, input_gen_fn=input_gen_fn)(wrap_maybe_shuffle)
register_op(maybe_anno, emit_fn=emit_ring, input_gen_fn=input_gen_fn)(wrap_maybe_unshuffle)
