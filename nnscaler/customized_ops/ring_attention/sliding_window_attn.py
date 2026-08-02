#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from dataclasses import replace
from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
import torch.distributed as dist

from nnscaler.graph.parser.register import register_op
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir import IRTensor
from nnscaler.runtime.device import DeviceGroup
from flash_attn import flash_attn_varlen_func
from .core.sliding_window_attn_implementation import (
    prepare_sliding_window_metadata,
    sliding_window_attn_func,
)
from .core.utils import gen_head_anno

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_cute_varlen_func
except ImportError:
    flash_attn_cute_varlen_func = None


def wrap_sliding_window_attn_func(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        alibi_slopes: Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Tensor = None,
        causal: bool = True,
        window_size: Tuple[int] = (-1, -1),
        deterministic: bool = False,
        return_attn_probs: bool = False,
        enable_ring: bool = True,
        use_cute: bool = False,
        process_group: Tuple[int] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        return_lse: bool = True,
        cp_size: Optional[int] = None,
        require_full_plan_sequence_partition: bool = False,
):
    '''
    Context parallel sliding window attention using single-hop A2A communication.

    Only fetches min(offset_in_seq, window_size_left) KV tokens from the
    previous rank instead of all_gather, then performs a single flash_attn
    computation (no ring loop).

    Constraints:
    - causal must be True
    - window_size[0] > 0 (must have a left sliding window)
    - window_size[0] <= length_per_rank (single-hop communication)
    '''
    assert not return_attn_probs, "return_attn_probs is not supported"
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
    if max_seqlen_q is None:
        max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    if max_seqlen_k is None:
        max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()

    if process_group is None or len(process_group) == 1 or not enable_ring:
        if use_cute:
            assert flash_attn_cute_varlen_func is not None, "flash_attn.cute is not available"
            cute_window_size = tuple(None if w == -1 else w for w in window_size)
            output, softmax_lse = flash_attn_cute_varlen_func(
                q, k, v,
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
            return (output, softmax_lse) if return_lse else output
        else:
            output, softmax_lse, _ = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
            )
            return (output, softmax_lse) if return_lse else output

    assert causal, "sliding window CP attention requires causal=True"
    assert window_size[0] > 0, (
        f"window_size[0] must be > 0 for sliding window CP, got {window_size}"
    )

    assert len(q.shape) == 3, "q must have shape [total_q, qh, dim]"
    assert len(k.shape) == 3, "k must have shape [total_k, kh, dim]"
    assert len(v.shape) == 3, "v must have shape [total_k, vh, dim]"
    total_q, qheads, qdim = q.shape
    total_k, kheads, kdim = k.shape
    total_v, vheads, vdim = v.shape
    assert total_q == total_k == total_v, "total_q, total_k and total_v must be the same"
    assert kheads == vheads, "number of k and v heads must be the same"
    assert qheads % kheads == 0, "number of q heads must be a multiple of k heads"
    assert qdim == kdim == vdim, "dimension must be the same"

    local_process_group = DeviceGroup().get_group(process_group)
    local_rank = dist.get_rank(local_process_group)
    local_world_size = dist.get_world_size(local_process_group)
    assert local_world_size == len(process_group)

    # Prepare metadata (cached globally across layers)
    metadata = prepare_sliding_window_metadata(
        cu_seqlens_q,
        window_size_left=window_size[0],
        rank=local_rank,
        world_size=local_world_size,
    )

    # Only the maxima consumed by FA4 are replaced; communication metadata is
    # still derived from the exact packed batch.
    if use_cute:
        metadata = replace(
            metadata,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
        )

    out, softmax_lse = sliding_window_attn_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        metadata,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        group=local_process_group,
        use_cute=use_cute,
    )

    return (out, softmax_lse) if return_lse else out


def emit_ring(node: IRDimops, args: List[str], kwargs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
    """Special rule to generate sliding_window_attn node"""
    signature = node.signature

    offset = (runtime_devid // plan_ndevs) * plan_ndevs
    remainder = runtime_devid % plan_ndevs

    kw_pairs = list()
    for key, val in kwargs.items():
        if key == 'process_group':
            continue
        code = f'{key}={val}'
        kw_pairs.append(code)

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
        if partition_dims[0][0] == 0:  # partition on sequence dim
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
            raise ValueError(f'unsupported partition dim: {partition_dims[0]}')

    args = ", ".join(list(args) + kw_pairs)
    return f"{signature}({args})"


def flash_attention_anno(query_states, key_states, value_states, cu_seqlens_q, cu_seqlens_k, alibi_slopes, *args, **kwargs) -> str:
    q_anno, kv_anno = gen_head_anno(query_states, key_states, value_states, head_pos=1)
    alibi_anno = f'{q_anno}' if isinstance(alibi_slopes, IRTensor) else '?'
    return f'l {q_anno} hd^, l {kv_anno} hd^, l {kv_anno} vd^, e^, e^, {alibi_anno} -> l {q_anno} vd^, ?'


def input_gen_fn(node: IRDimops):
    inputs = []
    device = torch.cuda.current_device()
    seqlen = node.inputs()[0].shape[0]
    for i, t in enumerate(node.inputs()):
        if i < 3:  # query, key, value
            inputs.append(torch.randn(t.shape, dtype=t.dtype, device=device, requires_grad=t.requires_grad))
        elif i in [3, 4]:  # cu_seqlens
            inputs.append(torch.Tensor([0, seqlen]).to(torch.int32).to(device))
        elif i in [5]:  # optional alibi_slopes
            if isinstance(t, IRTensor):
                inputs.append(torch.randn(t.shape, dtype=t.dtype, device=device, requires_grad=t.requires_grad))
            else:
                inputs.append(None)
        else:
            break
    return tuple(inputs)


register_op(flash_attention_anno, emit_fn=emit_ring, input_gen_fn=input_gen_fn)(wrap_sliding_window_attn_func)
