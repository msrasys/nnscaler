# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Surgical in-memory HuggingFace rewrites for Kimi-Linear and Kimi-K2.5.

Patches live HF `nn.Module` instances in-memory before `nnscaler` tracing:
  1. `KimiLinearAttention` (NoPE MLA): replaces `expand_kv` inplace `.copy_` + `.expand()`
     with 3D `kv_b_proj` + `kimi_mla_concat_kv` so `h` shards cleanly across `tp_size`.
  2. `KimiLinearDeltaAttention` (KDA): replaces `causal_conv1d_fn` + `chunk_kimi_delta_attention`
     with head-partitionable `kimi_causal_conv1d` (per Q/K/V) and `kimi_chunk_kda`, keeping
     `q/k/v_proj`, `conv1d`, `forget_gate.f_b_proj/dt_bias/A_log`, `b_proj`, `g_b_proj`,
     `o_norm`, and `o_proj` sharded along `linear_num_heads` with a single `AllReduce` at `o_proj`.
  3. `KimiLinearMoE` & `DeepseekV3MoE`: replaces `@use_experts_implementation` + router with
     `kimi_moe_route` and either fused `kimi_moe_grouped_gemm` (`expand_moe_ops=False`) or
     expanded `permute -> grouped_linear -> unpermute` (`expand_moe_ops=True`).
  4. `Kimi_K25ForConditionalGeneration`:
     - `Kimi_K25VisionAttention`: replaces `lengths.tolist()` + `torch.split` with
       `kimi_k25_vision_varlen_sdpa`.
     - `Kimi_K25Model`: replaces inplace mask indexing & `.tolist()` split+cat with
       `torch.where` + `kimi_k25_embed_scatter`.
     - `language_model` (`DeepseekV3Model`): applies head-parallel MLA + MoE rewrites.
  5. `KimiCausalLMTaskWrapper`: `reduction='sum' / valid_token_count` CE loss.
"""

from __future__ import annotations

from dataclasses import dataclass
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3MoE,
    apply_rotary_pos_emb_interleave as v3_apply_rotary_pos_emb_interleave,
    eager_attention_forward as v3_eager_attention_forward,
)
from transformers.models.kimi_k25.modeling_kimi_k25 import (
    Kimi_K25Model,
    Kimi_K25VisionAttention,
    Kimi_K25VisionMLP,
    Kimi_K25VisionModel,
    Kimi_K25VisionPositionEmbeddings,
    Kimi_K25VisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)
from transformers.models.kimi_linear.modeling_kimi_linear import (
    KimiLinearAttention,
    KimiLinearDeltaAttention,
    KimiLinearMoE,
    apply_mask_to_padding_states,
    eager_attention_forward as kl_eager_attention_forward,
)

from ..ops import (
    kimi_causal_conv1d,
    kimi_chunk_kda,
    kimi_grouped_linear,
    kimi_k25_embed_scatter,
    kimi_k25_vision_meta_indices,
    kimi_k25_vision_pos_emb,
    kimi_k25_vision_varlen_sdpa,
    kimi_mla_concat_kv,
    kimi_moe_grouped_gemm,
    kimi_moe_permute,
    kimi_moe_route,
    kimi_moe_unpermute,
    register_kimi_ops,
)


@dataclass
class RewriteConfig:
    """Declarative toggles for Kimi graph rewrites."""

    expand_moe_ops: bool = False
    sum_ce_loss: bool = True
    decouple_row_tp_bias: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "expand_moe_ops": self.expand_moe_ops,
            "sum_ce_loss": self.sum_ce_loss,
            "decouple_row_tp_bias": self.decouple_row_tp_bias,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RewriteConfig":
        if not data:
            return cls()
        return cls(
            expand_moe_ops=bool(data.get("expand_moe_ops", False)),
            sum_ce_loss=bool(data.get("sum_ce_loss", True)),
            decouple_row_tp_bias=bool(data.get("decouple_row_tp_bias", True)),
        )


def _patch_row_parallel_linear_bias(linear: nn.Linear) -> bool:
    if linear.bias is None:
        return False

    def _decoupled_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, None)
        return out + self.bias

    linear.forward = types.MethodType(_decoupled_linear_forward, linear)
    return True


# ============================================================================
# 1. KimiLinearAttention (NoPE MLA)
# ============================================================================

def _kl_attention_forward(
    self: KimiLinearAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

    if self.q_lora_rank is None:
        q_states = self.q_proj(hidden_states)
    else:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    query_states = q_states.view(query_shape).transpose(1, 2)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_nope = self.kv_a_layernorm(kv_nope)
    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    kv_proj = self.kv_b_proj(kv_nope).view(
        batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim
    ).transpose(1, 2)
    k_nope, value_states = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    key_states = kimi_mla_concat_kv(k_nope, k_rot)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, kl_eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ============================================================================
# 2. KimiLinearDeltaAttention (Head-Parallel KDA)
# ============================================================================

def _kl_kda_forward(
    self: KimiLinearDeltaAttention,
    hidden_states: torch.Tensor,
    cache_params: Any = None,
    attention_mask: torch.Tensor | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len = hidden_states.shape[:2]
    hidden_shape = (batch_size, seq_len, -1, self.head_dim)

    q = self.q_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).permute(0, 2, 3, 1)
    k = self.k_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).permute(0, 2, 3, 1)
    v = self.v_proj(hidden_states).view(batch_size, seq_len, -1, self.head_dim).permute(0, 2, 3, 1)

    conv_w = self.conv1d.weight.squeeze(1)
    wq, wk, wv = torch.split(conv_w, [self.qkv_dim, self.qkv_dim, self.qkv_dim], dim=0)
    wq = wq.view(-1, self.head_dim, self.conv_kernel_size)
    wk = wk.view(-1, self.head_dim, self.conv_kernel_size)
    wv = wv.view(-1, self.head_dim, self.conv_kernel_size)

    q = kimi_causal_conv1d(q, wq, activation=self.activation).permute(0, 3, 1, 2)
    k = kimi_causal_conv1d(k, wk, activation=self.activation).permute(0, 3, 1, 2)
    v = kimi_causal_conv1d(v, wv, activation=self.activation).permute(0, 3, 1, 2)

    g = self.forget_gate(hidden_states)
    beta = torch.sigmoid(self.b_proj(hidden_states))

    core_attn_out = kimi_chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
    output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
    return self.o_proj(output)


# ============================================================================
# 3. Kimi_K25 Language Model (DeepSeek-V3 MLA)
# ============================================================================

def _v3_attention_forward(
    self: DeepseekV3Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

    if self.q_lora_rank is None:
        q_states = self.q_proj(hidden_states)
    else:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_pass = self.kv_a_layernorm(k_pass)

    cos, sin = position_embeddings
    q_rot, k_rot = v3_apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    kv_proj = self.kv_b_proj(k_pass).view(
        batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim
    ).transpose(1, 2)
    k_nope, value_states = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    key_states = kimi_mla_concat_kv(k_nope, k_rot)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, v3_eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ============================================================================
# 4. Sparse MoE Rewrites (`KimiLinearMoE` & `DeepseekV3MoE`)
# ============================================================================

def _make_moe_forward(expand_moe_ops: bool):
    def _moe_forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
        num_tokens = flat_hidden.shape[0]

        gate = self.gate
        topk_indices, topk_weights = kimi_moe_route(
            flat_hidden,
            gate.weight,
            gate.e_score_correction_bias,
            top_k=gate.top_k,
            n_group=gate.num_group,
            topk_group=gate.topk_group,
            norm_topk_prob=gate.norm_topk_prob,
            routed_scaling_factor=gate.routed_scaling_factor,
        )

        experts = self.experts
        num_experts = experts.num_experts
        hidden_act = getattr(self.config, "hidden_act", "silu")

        if not expand_moe_ops:
            routed_out = kimi_moe_grouped_gemm(
                flat_hidden,
                topk_indices,
                topk_weights,
                experts.gate_up_proj,
                experts.down_proj,
                hidden_act=hidden_act,
                local_expert_start=0,
                local_expert_end=num_experts,
                _total_experts=num_experts,
            )
        else:
            permuted, expert_counts, token_indices, slot_indices = kimi_moe_permute(
                flat_hidden,
                topk_indices,
                num_experts=num_experts,
                local_expert_start=0,
                local_expert_end=num_experts,
            )
            gate_up_out = kimi_grouped_linear(
                permuted, expert_counts, experts.gate_up_proj,
                local_expert_start=0, local_expert_end=num_experts,
            )
            g_part, u_part = gate_up_out.chunk(2, dim=-1)
            act_out = (F.silu(g_part) if hidden_act == "silu" else F.gelu(g_part)) * u_part
            down_out = kimi_grouped_linear(
                act_out, expert_counts, experts.down_proj,
                local_expert_start=0, local_expert_end=num_experts,
            )
            routed_out = kimi_moe_unpermute(down_out, topk_weights, token_indices, slot_indices, num_tokens)

        return routed_out.view(*orig_shape) + self.shared_experts(residuals)

    return _moe_forward


# ============================================================================
# 5. Kimi_K25 Vision & Multimodal Rewrites
# ============================================================================

def _k25_vision_attn_forward(
    self: Kimi_K25VisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    max_seqlen: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states = self.q_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
    key_states = self.k_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)
    value_states = self.v_proj(hidden_states).reshape(1, seq_length, -1, self.head_dim)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(2, 1)
    key_states = key_states.transpose(2, 1)
    value_states = value_states.transpose(2, 1)

    attn_output = kimi_k25_vision_varlen_sdpa(
        query_states, key_states, value_states, cu_seqlens, self.scaling
    )
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return self.proj(attn_output)


def _k25_vision_pos_emb_forward(
    self: Kimi_K25VisionPositionEmbeddings,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    return kimi_k25_vision_pos_emb(
        hidden_states,
        grid_thw,
        self.position_embeddings,
        self.time_position_embeddings,
        num_grid_per_side=self.num_grid_per_side,
        interpolation_mode=self.interpolation_mode,
        interpolation_align_corners=self.interpolation_align_corners,
    )


def _k25_vision_recomposition_frequencies(
    self: Kimi_K25VisionRotaryEmbedding,
    freq: torch.Tensor,
) -> torch.Tensor:
    freq_wh = freq.transpose(1, 2).flip((-1,)).flatten(1)
    return torch.cat([freq_wh, freq_wh], dim=-1)


def _k25_vision_model_forward(
    self: Kimi_K25VisionModel,
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs: Any,
) -> Any:
    hidden_states = self.patch_embed(pixel_values, grid_thw=grid_thw, **kwargs)
    mh, mw = self.merge_kernel_size
    position_ids, cu_seqlens, merge_index = kimi_k25_vision_meta_indices(
        hidden_states, grid_thw, merge_h=mh, merge_w=mw
    )
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for block in self.layers:
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.final_layernorm(hidden_states)
    pooled_hidden_states = hidden_states[merge_index].mean(dim=1)
    return (hidden_states, pooled_hidden_states)


def _k25_model_forward(
    self: Kimi_K25Model,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    **kwargs: Any,
) -> Any:
    if inputs_embeds is None:
        multimodal_mask = (input_ids == self.config.image_token_id) | (input_ids == self.config.video_token_id)
        llm_input_ids = torch.where(multimodal_mask, 0, input_ids)
        inputs_embeds = self.get_input_embeddings()(llm_input_ids)

    if pixel_values is not None and image_grid_thw is not None:
        vision_outputs = self.vision_tower(pixel_values, grid_thw=image_grid_thw, **kwargs)
        pooled = vision_outputs[1] if isinstance(vision_outputs, tuple) else vision_outputs.pooler_output
        image_embeds = self.mm_projector(pooled).squeeze(1)
        inputs_embeds = kimi_k25_embed_scatter(
            inputs_embeds, input_ids, image_embeds, self.config.image_token_id
        )

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )
    return outputs


# ============================================================================
# 6. End-to-End Compile Wrapper (`sum / valid_count` CE)
# ============================================================================

class KimiCausalLMTaskWrapper(nn.Module):
    """Thin training wrapper accepting `samples: dict[str, Tensor]` for `nnscaler.parallelize`."""

    def __init__(self, model: nn.Module, *, sum_ce_loss: bool = True) -> None:
        super().__init__()
        self.model = model
        self.sum_ce_loss = sum_ce_loss

    def forward(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = samples["input_ids"]
        attention_mask = samples.get("attention_mask")
        labels = samples["labels"]

        model_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if "pixel_values" in samples and "image_grid_thw" in samples:
            model_kwargs["pixel_values"] = samples["pixel_values"]
            model_kwargs["image_grid_thw"] = samples["image_grid_thw"]

        outputs = self.model(**model_kwargs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab_size = shift_logits.shape[-1]

        if self.sum_ce_loss:
            loss_sum = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            valid_count = samples.get("valid_token_count")
            if valid_count is None:
                valid_count = (shift_labels != -100).sum()
            else:
                valid_count = valid_count.sum()
            return loss_sum / valid_count.clamp(min=1).to(loss_sum.dtype)

        return F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="mean",
        )


def apply_kimi_rewrites(
    model: nn.Module,
    config: RewriteConfig | None = None,
) -> dict[str, int]:
    """Register custom ops and apply in-memory surgical rewrites to Kimi-Linear / Kimi-K2.5."""
    cfg = config or RewriteConfig()
    register_kimi_ops()

    counts: dict[str, int] = {
        "mla_rewrites": 0,
        "kda_rewrites": 0,
        "moe_rewrites": 0,
        "vision_attn_rewrites": 0,
        "k25_model_rewrites": 0,
        "row_tp_bias_decoupled": 0,
    }
    moe_fn = _make_moe_forward(cfg.expand_moe_ops)

    for module in model.modules():
        if isinstance(module, KimiLinearAttention):
            module.forward = types.MethodType(_kl_attention_forward, module)
            counts["mla_rewrites"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["row_tp_bias_decoupled"] += 1
        elif isinstance(module, KimiLinearDeltaAttention):
            module.forward = types.MethodType(_kl_kda_forward, module)
            counts["kda_rewrites"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["row_tp_bias_decoupled"] += 1
        elif isinstance(module, DeepseekV3Attention):
            module.forward = types.MethodType(_v3_attention_forward, module)
            counts["mla_rewrites"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["row_tp_bias_decoupled"] += 1
        elif isinstance(module, (KimiLinearMoE, DeepseekV3MoE)):
            module.forward = types.MethodType(moe_fn, module)
            counts["moe_rewrites"] += 1
        elif isinstance(module, Kimi_K25VisionAttention):
            module.forward = types.MethodType(_k25_vision_attn_forward, module)
            counts["vision_attn_rewrites"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.proj):
                counts["row_tp_bias_decoupled"] += 1
        elif isinstance(module, Kimi_K25VisionMLP):
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.fc2):
                counts["row_tp_bias_decoupled"] += 1
        elif isinstance(module, Kimi_K25VisionPositionEmbeddings):
            module.forward = types.MethodType(_k25_vision_pos_emb_forward, module)
        elif isinstance(module, Kimi_K25VisionRotaryEmbedding):
            module.recomposition_frequencies = types.MethodType(_k25_vision_recomposition_frequencies, module)
        elif isinstance(module, Kimi_K25VisionModel):
            module.forward = types.MethodType(_k25_vision_model_forward, module)
        elif isinstance(module, Kimi_K25Model):
            module.forward = types.MethodType(_k25_model_forward, module)
            counts["k25_model_rewrites"] += 1

    return counts
