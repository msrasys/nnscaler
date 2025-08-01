#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging
from typing import Optional, Tuple

from nnscaler.graph import parser
from nnscaler.ir import IRTensor

_logger = logging.getLogger(__name__)

try:

    import torch

    from transformers.modeling_flash_attention_utils import _flash_attention_forward, flash_attn_supports_top_left_mask


    _use_top_left_mask = flash_attn_supports_top_left_mask()


    def flash_attention_forward(
        # module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        sliding_window: Optional[int] = None,
        softcap: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        # This is before the transpose
        seq_len = query.shape[2]

        # FA2 uses non-transposed inputs
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (usually our RMSNorm modules handle it correctly)
        target_dtype = None
        if query.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # # Handle the case where the model is quantized
            # elif hasattr(module.config, "_pre_quantization_dtype"):
            #     target_dtype = module.config._pre_quantization_dtype
            # else:
            #     target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

        # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
        kwargs.pop("is_causal", None)

        kwargs["position_ids"] = position_ids
        attn_output = _flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length=seq_len,
            # is_causal=module.is_causal,
            is_causal=True,
            dropout=dropout,
            softmax_scale=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            use_top_left_mask=_use_top_left_mask,
            target_dtype=target_dtype,
            **kwargs,
        )

        return attn_output


    def flash_attention_anno(query_states, key_states, value_states, attention_mask, position_ids, *args, **kwargs) -> str:
        from nnscaler.ir import IRTensor
        if query_states.shape[1] != key_states.shape[1]:
            assert query_states.shape[1] % key_states.shape[1] == 0
            group_size = query_states.shape[1] // key_states.shape[1]
            assert query_states.shape[1] == value_states.shape[1] * group_size
            q_anno = f'(group_num {group_size})'
            kv_anno = 'group_num'
        else:
            q_anno = kv_anno = 'num_heads'
        if isinstance(position_ids, IRTensor):
            if isinstance(attention_mask, IRTensor):
                return f'b {q_anno} l^ hd^, b {kv_anno} s^ hd^, b {kv_anno} s^ vd^, b l^, b l^ -> b l^ {q_anno} vd^'
            else:
                return f'b {q_anno} l^ hd^, b {kv_anno} s^ hd^, b {kv_anno} s^ vd^, ?, b l^ -> b l^ {q_anno} vd^'
        else:
            if isinstance(attention_mask, IRTensor):
                return f'b {q_anno} l^ hd^, b {kv_anno} s^ hd^, b {kv_anno} s^ vd^, b l^ -> b l^ {q_anno} vd^'
            else:
                return f'b {q_anno} l^ hd^, b {kv_anno} s^ hd^, b {kv_anno} s^ vd^ -> b l^ {q_anno} vd^'

    parser.register(flash_attention_anno)(flash_attention_forward)

    _logger.info("HF flash attention registered successfully.")

except Exception as e:
    _logger.warning(f"HF flash attention will not be registered: {e}")
