#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from flash_attn import flash_attn_varlen_func
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.graph.parser.register import register_op
from nnscaler.ir import IRTensor
from nnscaler.runtime.device import DeviceGroup

from .core.utils import gen_head_anno
from .core.utils import apply_sink_gate
from .core.zigzag_allgather_attn_varlen_implementation import (
    zigzag_allgather_attn_varlen_func,
)

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_cute_varlen_func
except ImportError:
    flash_attn_cute_varlen_func = None


def wrap_zigzag_allgather_attn_varlen_func(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    alibi_slopes: Tensor,
    learnable_sink: Tensor = None,
    dropout_p: float = 0.0,
    softmax_scale: Tensor = None,
    causal: bool = False,
    window_size: Tuple[int] = (-1, -1),
    deterministic: bool = False,
    return_attn_probs: bool = False,
    enable_ring: bool = True,
    use_cute: bool = False,
    process_group: Tuple[int] = None,
):
    assert not return_attn_probs, "return_attn_probs is not supported"
    if learnable_sink is not None:
        assert use_cute, "learnable_sink requires use_cute=True"

    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()

    if process_group is None or len(process_group) == 1 or not enable_ring:
        if use_cute:
            assert flash_attn_cute_varlen_func is not None, "flash_attn.cute is not available"
            cute_window_size = tuple(None if w == -1 else w for w in window_size)
            output, lse = flash_attn_cute_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=cute_window_size,
                deterministic=deterministic,
                return_lse=True,
            )
            if learnable_sink is not None:
                output = apply_sink_gate(output, lse, learnable_sink)
            return output

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

    local_process_group = DeviceGroup().get_group(process_group)
    if local_process_group is None:
        local_process_group = dist.group.WORLD

    return zigzag_allgather_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        process_group=local_process_group,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        use_cute=use_cute,
        learnable_sink=learnable_sink,
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


def flash_attention_anno(query_states, key_states, value_states, cu_seqlens_q, cu_seqlens_k, alibi_slopes, learnable_sink=None, *args, **kwargs) -> str:
    q_anno, kv_anno = gen_head_anno(query_states, key_states, value_states, head_pos=1)
    alibi_anno = f'{q_anno}' if isinstance(alibi_slopes, IRTensor) else '?'
    sink_anno = f'{q_anno}' if isinstance(learnable_sink, IRTensor) else '?'
    return f"l {q_anno} hd^, al^ {kv_anno} hd^, al^ {kv_anno} vd^, e^, e^, {alibi_anno}, {sink_anno} -> l {q_anno} vd^"


def input_gen_fn(node: IRDimops):
    inputs = []
    device = torch.cuda.current_device()
    seqlen = node.inputs()[0].shape[0]
    for i, t in enumerate(node.inputs()):
        if i < 3:
            inputs.append(torch.randn(t.shape, dtype=t.dtype, device=device, requires_grad=t.requires_grad))
        elif i in [3, 4]:
            inputs.append(torch.Tensor([0, seqlen]).to(torch.int32).to(device))
        elif i in [5, 6]:  # optional alibi_slopes, learnable_sink
            if isinstance(t, IRTensor):
                inputs.append(torch.randn(t.shape, dtype=t.dtype, device=device, requires_grad=t.requires_grad))
            else:
                inputs.append(None)
        else:
            break
    return tuple(inputs)


register_op(flash_attention_anno, emit_fn=emit_ring, input_gen_fn=input_gen_fn)(wrap_zigzag_allgather_attn_varlen_func)
