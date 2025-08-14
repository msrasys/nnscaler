#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging
from typing import Optional, Tuple, List, Dict

from nnscaler.graph import parser
from nnscaler.ir import IRTensor
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.customized_ops.ring_attention.ring_attn import wrap_ring_attn_func
from nnscaler.customized_ops.ring_attention.ring_attn_varlen import wrap_ring_attn_varlen_func

_logger = logging.getLogger(__name__)

try:

    import torch

    from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask, flash_241, deterministic_g, \
        _flash_supports_window_size, fa_peft_integration_check, prepare_fa2_from_position_ids


    _use_top_left_mask = flash_attn_supports_top_left_mask()


    # we slightly modify the original _flash_attention_forward to pass in the process_group
    def _flash_attention_forward(
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_length: int,
        is_causal: bool,
        dropout: float = 0.0,
        position_ids: Optional[torch.Tensor] = None,
        softmax_scale: Optional[float] = None,
        sliding_window: Optional[int] = None,
        use_top_left_mask: bool = False,
        softcap: Optional[float] = None,
        deterministic: Optional[bool] = None,
        cu_seq_lens_q: Optional[torch.LongTensor] = None,
        cu_seq_lens_k: Optional[torch.LongTensor] = None,
        max_length_q: Optional[int] = None,
        max_length_k: Optional[int] = None,
        target_dtype: Optional[torch.dtype] = None,
        process_group: Optional[Tuple[int]] = None,
        **kwargs,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`, *optional*):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_top_left_mask (`bool`, defaults to `False`):
                flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignment, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
            softcap (`float`, *optional*):
                Softcap for the attention logits, used e.g. in gemma2.
            deterministic (`bool`, *optional*):
                Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
        """
        if not use_top_left_mask:
            causal = is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
            causal = is_causal and query_length != 1

        # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
        use_sliding_windows = (
            _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
        )
        flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

        if flash_241:
            if deterministic is None:
                deterministic = deterministic_g
            flash_kwargs["deterministic"] = deterministic

        if softcap is not None:
            flash_kwargs["softcap"] = softcap

        # PEFT possibly silently casts tensors to fp32, this potentially reconverts to correct dtype or is a no op
        query_states, key_states, value_states = fa_peft_integration_check(
            query_states, key_states, value_states, target_dtype
        )

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            # check attention_mask is all ones
            if attention_mask.sum().item() != attention_mask.numel():
                error_msg = (
                    "The attention mask should be all ones, but got a mask with padding tokens. "
                    "Please set `attention_mask=None` and use `position_ids` to handle padding tokens."
                )
                raise ValueError(error_msg)
            # use the original version of flash_attn
            attn_output = wrap_ring_attn_func(
                query_states,
                key_states,
                value_states,
                softmax_scale=softmax_scale,
                causal=causal,
                dropout_p=dropout,
                process_group=process_group,
                **flash_kwargs,
            )

        # If position_ids is provided and check all examples do not contain only 1 sequence, If tensor in increasing
        # then we probably have one sequence, otherwise it is packed. Additionally check we are in pre-fill/training stage.
        # Use `flash_attn_varlen_func` to prevent cross-example attention and also allow padding free approach
        elif position_ids is not None and (
            max_length_q is not None or (query_length != 1 and not (torch.diff(position_ids, dim=-1) >= 0).all())
        ):
            batch_size = query_states.size(0)

            if cu_seq_lens_q is None or cu_seq_lens_k is None:
                query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
                    prepare_fa2_from_position_ids(query_states, key_states, value_states, position_ids)
                )

                cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
                max_length_q, max_length_k = max_seq_lens

            else:
                query_states = query_states.reshape(-1, query_states.size(-2), query_states.size(-1))
                key_states = key_states.reshape(-1, key_states.size(-2), key_states.size(-1))
                value_states = value_states.reshape(-1, value_states.size(-2), value_states.size(-1))

            attn_output = wrap_ring_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seq_lens_q,
                cu_seq_lens_k,
                None,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
                process_group=process_group,
                **flash_kwargs,
            )

            attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

        else:
            attn_output = wrap_ring_attn_func(
                query_states,
                key_states,
                value_states,
                softmax_scale=softmax_scale,
                causal=causal,
                dropout_p=dropout,
                process_group=process_group,
                **flash_kwargs,
            )

        return attn_output


    def flash_attention_forward_ring(
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
        process_group: Optional[Tuple[int]] = None,
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
            process_group=process_group,
            **kwargs,
        )

        return attn_output


    def flash_attention_ring_anno(query_states, key_states, value_states, attention_mask, position_ids, *args, **kwargs) -> str:
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
                return f'b {q_anno} l hd^, b {kv_anno} l hd^, b {kv_anno} l vd^, b e^, b e^ -> b l {q_anno} vd^'
            else:
                return f'b {q_anno} l hd^, b {kv_anno} l hd^, b {kv_anno} l vd^, ?, b e^ -> b l {q_anno} vd^'
        else:
            if isinstance(attention_mask, IRTensor):
                return f'b {q_anno} l hd^, b {kv_anno} l hd^, b {kv_anno} l vd^, b e^ -> b l {q_anno} vd^'
            else:
                return f'b {q_anno} l hd^, b {kv_anno} l hd^, b {kv_anno} l vd^ -> b l {q_anno} vd^'


    def emit_ring(node: IRDimops, args: List[str], kwargs: Dict[str, str], runtime_devid: int, plan_ndevs: int, runtime_ndevs: int) -> str:
        """Special rule to generate ring_attn node"""

        signature = node.signature

        offset = (runtime_devid // plan_ndevs) * plan_ndevs
        scale_unit_dev_ids = [local_rank + offset for local_rank in range(plan_ndevs)]

        kw_pairs = list()
        for key, val in kwargs.items():
            code = f'{key}={val}'
            kw_pairs.append(code)

        sub_input = node.inputs()[0]
        full_input = sub_input.parent
        partition_dims = [i for i, (s, f) in enumerate(zip(sub_input.shape, full_input.shape)) if s != f]
        assert len(partition_dims) <= 1, f"support no more than one partition dim, but got {partition_dims}"
        if not partition_dims:
            kw_pairs.append("process_group=None")
        else:
            # if the 'process_group' is None, we will use the local attention (flash_attn_func)
            if partition_dims[0] == 0: # partition on batch dim
                # partition the bsz dim, use local flash_attn_func
                kw_pairs.append("process_group=None")
            elif partition_dims[0] == 1:
                # partition on num_head dim
                kw_pairs.append("process_group=None")
            elif partition_dims[0] == 2: # partition on sequence dim
                # the synchronization should occur across scaleunits
                kw_pairs.append(f"process_group={scale_unit_dev_ids}")
            else:
                raise ValueError(f'unsupported partition dim: {partition_dims[0]}')
                
        args = ", ".join(list(args) + kw_pairs)
        return f"{signature}({args})"


    def input_gen_fn(node: IRDimops):
        inputs = []
        device = torch.cuda.current_device()
        bsz = node.inputs()[0].shape[0]
        seqlen = node.inputs()[0].shape[1]
        for i, t in enumerate(node.inputs()):
            if i < 3:  # query, key, value
                inputs.append(torch.randn(t.shape, dtype=t.dtype, device=device, requires_grad=t.requires_grad))
            elif i == 3: # attention_mask
                if isinstance(t, IRTensor):
                    inputs.append(torch.ones(t.shape, dtype=t.dtype, device=device, requires_grad=t.requires_grad))
                else:
                    inputs.append(None)
            elif i == 4:  # position_ids
                if isinstance(t, IRTensor):
                    inputs.append(torch.arange(seqlen, dtype=torch.int64, device=device, requires_grad=t.requires_grad).expand(bsz, -1))
                else:
                    inputs.append(None)
            else:  # other kwargs, use defaults
                break
        return tuple(inputs)

    parser.register(flash_attention_ring_anno, emit_fn=emit_ring, input_gen_fn=input_gen_fn)(flash_attention_forward_ring)

    _logger.info("HF flash attention registered successfully.")

except Exception as e:
    _logger.warning(f"HF flash attention will not be registered: {e}")
