#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Tuple, List, Dict, Optional
import torch
import os
from torch import Tensor
import torch.distributed as dist
import warnings

from nnscaler.graph.parser.register import register_op
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir import IRTensor
from nnscaler.runtime.device import DeviceGroup
from flash_attn import flash_attn_varlen_func
from .core.ring_attn_varlen_implementation import llama3_flash_attn_prepare_cu_seqlens, llama3_flash_attn_varlen_func
from .core.utils import gen_head_anno
from .varlen_utils import shuffle_varlen, unshuffle_varlen

try:
    from flash_attn.cute import flash_attn_varlen_func as flash_attn_cute_varlen_func
except ImportError as e:
    print(f"flash_attn.cute not available: {e}")
    flash_attn_cute_varlen_func = None

# Try to import TransformerEngine with version check and optional CP enable via env var.
# Usage control:
#   Set environment variable ENABLE_TE_CP=1 to enable TransformerEngine context-parallel (CP) attention.
#   Default (unset or 0) will disable CP usage even if TE is installed.
_HAS_TRANSFORMER_ENGINE = False
_TE_VERSION_OK = False
attn_forward_func_with_cp = None

# Read environment variable switch (string compare to '1').
_ENABLE_TE_CP = os.getenv("ENABLE_TE_CP", "0") == "1"

try:
    import transformer_engine
    _HAS_TRANSFORMER_ENGINE = True
    
    # Check version - require 2.2.0+
    try:
        from packaging import version
        te_version = version.parse(transformer_engine.__version__)
        required_version = version.parse("2.2.0")
        _TE_VERSION_OK = te_version >= required_version
        
        if _TE_VERSION_OK:
            # Try different import paths for different versions
            try:
                # For v2.5.0+
                from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import attn_forward_func_with_cp
            except ImportError:
                try:
                    # For v2.2.0-v2.4.x
                    from transformer_engine.pytorch.attention import attn_forward_func_with_cp
                except ImportError:
                    warnings.warn(
                        "TransformerEngine attention module not available or incompatible. "
                        "Falling back to basic ring attention implementation."
                    )
        else:
            warnings.warn(
                f"TransformerEngine version {transformer_engine.__version__} is too old. "
                f"Require 2.2.0+. Falling back to basic ring attention implementation."
            )
    except ImportError:
        # packaging not available, try to import anyway
        try:
            # Try different import paths for different versions
            try:
                # For v2.5.0+
                from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import attn_forward_func_with_cp
            except ImportError:
                # For v2.2.0-v2.4.x
                from transformer_engine.pytorch.attention import attn_forward_func_with_cp
            _TE_VERSION_OK = True
        except (ImportError, AttributeError):
            warnings.warn(
                "TransformerEngine attention module not available or incompatible. "
                "Falling back to basic ring attention implementation."
            )
            
except ImportError:
    warnings.warn(
        "TransformerEngine not found. Falling back to basic ring attention implementation. "
        "For better performance with context parallelism, install TransformerEngine 2.2.0+."
    )


def get_transformer_engine_info() -> Dict[str, any]:
    """Get information about TransformerEngine availability and version."""
    return {
        "has_transformer_engine": _HAS_TRANSFORMER_ENGINE,
        "version_ok": _TE_VERSION_OK,
    "has_cp_function": attn_forward_func_with_cp is not None,
    "env_enable_cp": _ENABLE_TE_CP,
        "version": getattr(transformer_engine, "__version__", None) if _HAS_TRANSFORMER_ENGINE else None,
        "required_version": "2.2.0+",
    }


def print_transformer_engine_status():
    """Print TransformerEngine status for debugging."""
    info = get_transformer_engine_info()
    print("TransformerEngine Status:")
    print(f"  - Available: {info['has_transformer_engine']}")
    if info['has_transformer_engine']:
        print(f"  - Version: {info['version']}")
        print(f"  - Version OK (>= 2.2.0): {info['version_ok']}")
        print(f"  - CP Function Available: {info['has_cp_function']}")
    else:
        print(f"  - Required Version: {info['required_version']}")
    print(f"  - Env ENABLE_TE_CP=1: {info['env_enable_cp']}")
    print(f"  - Will use TE CP: {info['has_transformer_engine'] and info['version_ok'] and info['has_cp_function'] and info['env_enable_cp']}")


def wrap_ring_attn_varlen_func(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_seqlens_q: Tensor,
        cu_seqlens_k: Tensor,
        alibi_slopes: Tensor,
        dropout_p: float = 0.0,
        softmax_scale: Tensor = None,
        causal: bool = False,
        window_size: Tuple[int] = (-1, -1),
        deterministic: bool = False,
        return_attn_probs: bool = False,
        enable_ring: bool = True,
        use_cute:  bool = False,
        process_group: Tuple[int] = None,
):
    '''
    wrap the ring_attn_varlen_func to support the distributed training in nnScaler.
    most of the arguments are the same as the original flash_attn_varlen_func.
    `process_group` should be none in the user code since nnScaler accepts the
    program defined for the single device and will automatically generate the
    required communications.
    '''
    assert not return_attn_probs, "return_attn_probs is not supported in ring-attention"
    max_seqlen_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    max_seqlen_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]).max().item()

    if process_group is None or len(process_group) == 1 or not enable_ring:
        if use_cute:
            assert flash_attn_cute_varlen_func is not None, "flash_attn.cute is not available"
            output = flash_attn_cute_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic,
            )
            return output
        else:
            output = flash_attn_varlen_func(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=False,
            )
        return output

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
    assert local_world_size == len(process_group), "local_world_size should be the same with process_group size"

    if local_process_group is None:
        local_process_group = dist.group.WORLD

    if window_size == (-1, -1):
        # Use TransformerEngine with context parallelism if available and version is OK
        # Only use TransformerEngine CP path if env flag is enabled
        if _ENABLE_TE_CP and _HAS_TRANSFORMER_ENGINE and _TE_VERSION_OK and attn_forward_func_with_cp is not None:
            shuffled_q = shuffle_varlen(q, cu_seqlens_q, process_group, local_process_group)
            shuffled_k = shuffle_varlen(k, cu_seqlens_k, process_group, local_process_group)
            shuffled_v = shuffle_varlen(v, cu_seqlens_k, process_group, local_process_group)

            te_cu_seqlens_q = cu_seqlens_q.clone()
            te_cu_seqlens_k = cu_seqlens_k.clone()
            te_cu_seqlens_q = torch.cat(
                [
                    te_cu_seqlens_q,
                    torch.tensor([cu_seqlens_q[-1].item()], dtype=te_cu_seqlens_q.dtype, device=te_cu_seqlens_q.device)
                ]
            )
            te_cu_seqlens_k = torch.cat(
                [
                    te_cu_seqlens_k,
                    torch.tensor([cu_seqlens_k[-1].item()], dtype=te_cu_seqlens_k.dtype, device=te_cu_seqlens_k.device)
                ]
            )
            shuffled_output = attn_forward_func_with_cp(
                True,
                shuffled_q,
                shuffled_k,
                shuffled_v,
                te_cu_seqlens_q,
                te_cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                te_cu_seqlens_q,
                te_cu_seqlens_k,
                dropout_p,
                local_process_group,
                process_group,
                # TODO: optimize the stream usage
                torch.cuda.current_stream(),
                "p2p", # "all_gather" version cannot work with thd format
                qkv_format="thd",
                attn_mask_type="padding_causal" if causal else "padding",
            )
            output = unshuffle_varlen(shuffled_output, cu_seqlens_q, process_group, local_process_group)
            return output
        else:
            # Fallback to basic ring attention implementation
            if _ENABLE_TE_CP:
                # User requested CP but TE unavailable/incompatible
                warnings.warn(
                    "ENABLE_TE_CP=1 set but TransformerEngine CP attention unavailable (missing or incompatible). "
                    "Falling back to basic ring attention implementation."
                )
            # If not enabled, remain silent (no warning spam) unless TE missing earlier already warned.

    (
        local_cu_seqlens_q,
        local_cu_seqlens_k,
        local_max_seqlen_q,
        local_max_seqlen_k,
        local_k_slice,
    ) = llama3_flash_attn_prepare_cu_seqlens(
        cu_seqlens_q,
        causal=causal,
        rank=local_rank,
        world_size=local_world_size,
    )

    output = llama3_flash_attn_varlen_func(
        q,
        k,
        v,
        local_cu_seqlens_q,
        local_cu_seqlens_k,
        local_max_seqlen_q,
        local_max_seqlen_k,
        heads_k_stride=1,
        local_k_slice=local_k_slice,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=False,
        group=local_process_group,
    )

    return output


def emit_ring(node: IRDimops, args: List[str], kwargs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
    """Special rule to generate ring_attn node"""

    signature = node.signature

    offset = (runtime_devid // plan_ndevs) * plan_ndevs
    remainder = runtime_devid % plan_ndevs

    kw_pairs = list()
    for key, val in kwargs.items():
        code = f'{key}={val}'
        kw_pairs.append(code)

    sub_input = node.inputs()[0]
    full_input = sub_input.parent
    partition_dims = [(i, f // s) for i, (s, f) in enumerate(zip(sub_input.shape, full_input.shape)) if s != f]
    assert len(partition_dims) <= 1, f"support no more than one partition dim, but got {partition_dims}"
    if not partition_dims:
        kw_pairs.append("process_group=None")
    else:
        if partition_dims[0][0] == 0: # partition on sequence dim
            # the synchronization should occur across scaleunits
            num = partition_dims[0][1]
            scale_unit_dev_ids = [local_rank + offset for local_rank in range(remainder // num * num, (remainder // num + 1) * num)]
            kw_pairs.append(f"process_group={scale_unit_dev_ids}")
        elif partition_dims[0][0] == 1:
            # partition the head dim, use local flash_attn_func
            kw_pairs.append("process_group=None")
        else:
            raise ValueError(f'unsupported partition dim: {partition_dims[0]}')
                
    args = ", ".join(list(args) + kw_pairs)
    return f"{signature}({args})"


def flash_attention_anno(query_states, key_states, value_states, cu_seqlens_q, cu_seqlens_k, alibi_slopes, *args, **kwargs) -> str:
    q_anno, kv_anno = gen_head_anno(query_states, key_states, value_states, head_pos=1)
    if isinstance(alibi_slopes, IRTensor):
        return f'l {q_anno} hd^, l {kv_anno} hd^, l {kv_anno} vd^, e^, e^, {q_anno} -> l {q_anno} vd^'
    else:
        return f'l {q_anno} hd^, l {kv_anno} hd^, l {kv_anno} vd^, e^, e^, ? -> l {q_anno} vd^'


def input_gen_fn(node: IRDimops):
    inputs = []
    device = torch.cuda.current_device()
    seqlen = node.inputs()[0].shape[0]
    for i, t in enumerate(node.inputs()):
        if i < 3:  # query, key, value
            inputs.append(torch.randn(t.shape, dtype=t.dtype, device=device, requires_grad=t.requires_grad))
        elif i in [3, 4]: # cu_seqlens
            inputs.append(torch.Tensor([0, seqlen]).to(torch.int32).to(device))
        elif i == 5: # optional alibi_slopes
            if isinstance(t, IRTensor):
                inputs.append(torch.randn(t.shape, dtype=t.dtype, device=device, requires_grad=t.requires_grad))
            else:
                inputs.append(None)
        else:  # other kwargs, use defaults
            break
    return tuple(inputs)


register_op(flash_attention_anno, emit_fn=emit_ring, input_gen_fn=input_gen_fn)(wrap_ring_attn_varlen_func)
