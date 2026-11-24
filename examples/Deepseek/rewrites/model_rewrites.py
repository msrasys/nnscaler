# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Surgical in-memory HuggingFace module rewrites for DeepSeek-V2, V2-Lite, V3, R1, and V3.2-DSA.

Every rewrite preserves 100% of HuggingFace checkpoint parameter names and shapes while making
the traced graph SSA-pure, static-shape, and maximally partitionable across TP, EP, ETP, PP, and DP:
  1. MLA `expand_kv` (`V2`, `V3`, `V3.2`): Replaces broken inplace slice-`.copy_` and `Expand(1 -> H^)`
     head-freezing with 3D `kv_b_proj` + `deepseek_mla_concat_kv(k_nope, k_rot)`.
  2. V2 Complex RoPE (`DeepseekV2RotaryEmbedding` + `DeepseekV2Attention`): Replaces `torch.polar` +
     `view_as_complex` / `view_as_real` with bit-identical real interleaved `(cos, sin)` `IRDimops`.
  3. V3.2 Sparse Attention (`DeepseekV32Attention`): Wraps `@torch.no_grad()` `DeepseekV32Indexer` +
     scatter mask in `deepseek_v32_dsa_index_mask` and marks frozen indexer weights `requires_grad_(False)`.
  4. Sparse MoE (`DeepseekV2Moe`, `DeepseekV3MoE`, `DeepseekV32MoE`): Supports both `expand_moe_ops=False`
     (default fused `deepseek_moe_grouped_gemm`) and `expand_moe_ops=True` (`permute -> chunk -> grouped_linear`).
  5. Biased Row-TP Linears: Decouples additive bias from `o_proj` / `down_proj` when bias is enabled.
  6. `DeepseekCausalLMTaskWrapper`: Replaces `reduction='mean'` CrossEntropy (`N^ C^`) with sharded
     `reduction='sum' / valid_token_count` (`N C^`), eliminating the multi-GiB FP32 logits AllGather.
"""

from __future__ import annotations

from dataclasses import dataclass
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v2.modeling_deepseek_v2 import (
    DeepseekV2Attention,
    DeepseekV2MLP,
    DeepseekV2Moe,
    DeepseekV2RotaryEmbedding,
    eager_attention_forward as v2_eager_attention_forward,
)
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3MLP,
    DeepseekV3MoE,
    apply_rotary_pos_emb as v3_apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave as v3_apply_rotary_pos_emb_interleave,
    eager_attention_forward as v3_eager_attention_forward,
)
from transformers.models.deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32Attention,
    DeepseekV32MLP,
    DeepseekV32MoE,
    apply_rotary_pos_emb_interleave as v32_apply_rotary_pos_emb_interleave,
    eager_attention_forward as v32_eager_attention_forward,
)

from examples.Deepseek.ops.mla_ops import (
    deepseek_mla_concat_kv,
    deepseek_v32_dsa_index_mask,
)
from examples.Deepseek.ops.moe_ops import (
    _apply_act,
    deepseek_grouped_linear,
    deepseek_moe_grouped_gemm,
    deepseek_moe_permute,
    deepseek_moe_unpermute,
    deepseek_v2_moe_route,
    deepseek_v3_moe_route,
)


@dataclass
class RewriteConfig:
    """Declarative toggles for DeepSeek graph rewrites."""

    expand_moe_ops: bool = False  # Default False: fused `deepseek_moe_grouped_gemm`
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


# ============================================================================
# 1. Biased Row-Parallel Linear Bias Decoupling
# ============================================================================

def _patch_row_parallel_linear_bias(linear: nn.Linear) -> bool:
    if linear.bias is None:
        return False

    def _decoupled_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, None)
        return out + self.bias

    linear.forward = types.MethodType(_decoupled_linear_forward, linear)
    return True


# ============================================================================
# 2. DeepSeek-V2 Real Interleaved RoPE + Head-Parallel MLA
# ============================================================================

@torch.no_grad()
def _v2_rotary_forward(
    self: DeepseekV2RotaryEmbedding,
    x: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs = position_ids[..., None].float() * self.inv_freq.to(dtype=torch.float32)
    cos = torch.cos(freqs) * self.attention_scaling
    sin = torch.sin(freqs) * self.attention_scaling
    return cos, sin


def _v2_apply_real_interleaved_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Matches `view_as_real(view_as_complex(x.float().reshape(..., -1, 2)) * freqs_cis.unsqueeze(1)).flatten(3)`
    cos_u = cos.unsqueeze(1)
    sin_u = sin.unsqueeze(1)

    qf = xq.float()
    kf = xk.float()
    q_even, q_odd = qf[..., 0::2], qf[..., 1::2]
    k_even, k_odd = kf[..., 0::2], kf[..., 1::2]

    xq_out = torch.stack((q_even * cos_u - q_odd * sin_u, q_even * sin_u + q_odd * cos_u), dim=-1).flatten(3)
    xk_out = torch.stack((k_even * cos_u - k_odd * sin_u, k_even * sin_u + k_odd * cos_u), dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def _v2_attention_forward(
    self: DeepseekV2Attention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Any = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

    if self.q_lora_rank is None:
        q = self.q_proj(hidden_states)
    else:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q = q.view(query_shape).transpose(1, 2)
    q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_nope, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_nope = self.kv_a_layernorm(kv_nope)
    k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    assert position_embeddings is not None
    cos, sin = position_embeddings
    q_pe, k_pe = _v2_apply_real_interleaved_rope(q_pe, k_pe, cos, sin)

    query_states = torch.cat((q_nope, q_pe), dim=-1)

    # 3D `kv_b_proj` + `deepseek_mla_concat_kv` keeps head dimension cleanly sharded across TP
    kv_proj = self.kv_b_proj(kv_nope).view(
        batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim
    ).transpose(1, 2)
    k_nope_heads, value_states = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    key_states = deepseek_mla_concat_kv(k_nope_heads, k_pe)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, v2_eager_attention_forward
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
# 3. DeepSeek-V3 / R1 Head-Parallel MLA & DeepSeek-V3.2 DSA MLA
# ============================================================================

def _v3_attention_forward(
    self: DeepseekV3Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
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
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_nope, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_nope = self.kv_a_layernorm(kv_nope)
    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    cos, sin = position_embeddings
    if self.config.rope_interleave:
        q_rot, k_rot = v3_apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        q_rot, k_rot = v3_apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

    query_states = torch.cat((q_pass, q_rot), dim=-1)

    kv_proj = self.kv_b_proj(kv_nope).view(
        batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim
    ).transpose(1, 2)
    k_nope_heads, value_states = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    key_states = deepseek_mla_concat_kv(k_nope_heads, k_rot)

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


def _v32_attention_forward(
    self: DeepseekV32Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values: Any = None,
    position_ids: torch.Tensor | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_pass = self.kv_a_layernorm(kv_pass)
    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    cos, sin = position_embeddings
    q_rot, k_rot = v32_apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    query_states = torch.cat((q_pass, q_rot), dim=-1)

    kv_proj = self.kv_b_proj(kv_pass).view(
        batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim
    ).transpose(1, 2)
    k_nope_heads, value_states = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    key_states = deepseek_mla_concat_kv(k_nope_heads, k_rot)

    idx = self.indexer
    index_mask = deepseek_v32_dsa_index_mask(
        hidden_states,
        q_resid,
        cos,
        sin,
        attention_mask[:, 0, :, :],
        idx.wq_b.weight,
        idx.wk.weight,
        idx.k_norm.weight,
        idx.k_norm.bias,
        idx.weights_proj.weight,
        n_heads=idx.n_heads,
        head_dim=idx.head_dim,
        qk_rope_head_dim=idx.qk_rope_head_dim,
        index_topk=idx.index_topk,
        k_norm_eps=idx.k_norm.eps,
    )

    if attention_mask.dtype == torch.bool:
        attention_mask = attention_mask & ~index_mask
    else:
        attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, v32_eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        indices=None,
        **kwargs,
    )
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ============================================================================
# 4. Sparse MoE Blocks (`DeepseekV2Moe`, `DeepseekV3MoE`, `DeepseekV32MoE`)
# ============================================================================

def _dispatch_routed_experts(
    flat_hidden: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    experts_mod: nn.Module,
    expand_moe_ops: bool,
) -> torch.Tensor:
    num_experts = int(experts_mod.num_experts)
    act_fn_name = str(experts_mod.config.hidden_act)
    if not expand_moe_ops:
        return deepseek_moe_grouped_gemm(
            flat_hidden,
            topk_weights,
            topk_indices,
            experts_mod.gate_up_proj,
            experts_mod.down_proj,
            act_fn_name=act_fn_name,
            local_expert_start=0,
            local_expert_end=num_experts,
        )

    num_tokens = flat_hidden.shape[0]
    permuted_x, expert_counts, sorted_tokens, sorted_probs = deepseek_moe_permute(
        flat_hidden,
        topk_weights,
        topk_indices,
        num_experts=num_experts,
    )
    gate_weight, up_weight = torch.chunk(experts_mod.gate_up_proj, 2, dim=1)
    gate_out = deepseek_grouped_linear(
        permuted_x, gate_weight, expert_counts, local_expert_start=0, local_expert_end=num_experts
    )
    up_out = deepseek_grouped_linear(
        permuted_x, up_weight, expert_counts, local_expert_start=0, local_expert_end=num_experts
    )
    inter = _apply_act(gate_out, act_fn_name) * up_out
    down_out = deepseek_grouped_linear(
        inter, experts_mod.down_proj, expert_counts, local_expert_start=0, local_expert_end=num_experts
    )
    return deepseek_moe_unpermute(down_out, sorted_tokens, sorted_probs, num_tokens=num_tokens)


def _make_v2_moe_forward(expand_moe_ops: bool):
    def _v2_moe_forward(self: DeepseekV2Moe, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])

        gate = self.gate
        n_group = int(gate.num_group) if gate.num_group is not None else 1
        topk_group = int(gate.topk_group) if gate.topk_group is not None else 1
        topk_method = str(gate.topk_method or "greedy")

        topk_weights, topk_indices = deepseek_v2_moe_route(
            flat_hidden,
            gate.weight,
            top_k=int(gate.top_k),
            n_group=n_group,
            topk_group=topk_group,
            topk_method=topk_method,
            routed_scaling_factor=float(gate.routed_scaling_factor),
        )
        routed_out = _dispatch_routed_experts(
            flat_hidden, topk_weights, topk_indices, self.experts, expand_moe_ops=expand_moe_ops
        ).view(*orig_shape)
        return routed_out + self.shared_experts(residuals)

    return _v2_moe_forward


def _make_v3_moe_forward(expand_moe_ops: bool):
    def _v3_moe_forward(self: DeepseekV3MoE | DeepseekV32MoE, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])

        gate = self.gate
        topk_weights, topk_indices = deepseek_v3_moe_route(
            flat_hidden,
            gate.weight,
            gate.e_score_correction_bias,
            top_k=int(gate.top_k),
            n_group=int(gate.num_group),
            topk_group=int(gate.topk_group),
            norm_topk_prob=bool(gate.norm_topk_prob),
            routed_scaling_factor=float(gate.routed_scaling_factor),
        )
        routed_out = _dispatch_routed_experts(
            flat_hidden, topk_weights, topk_indices, self.experts, expand_moe_ops=expand_moe_ops
        ).view(*orig_shape)
        return routed_out + self.shared_experts(residuals)

    return _v3_moe_forward


# ============================================================================
# 5. End-to-End CausalLM Wrapper (`sum / valid_count` Vocabulary-Parallel Loss)
# ============================================================================

class DeepseekCausalLMTaskWrapper(nn.Module):
    """Wraps any DeepSeek `*ForCausalLM` so end-to-end training uses sharded `sum / valid_count` CE."""

    def __init__(self, model: nn.Module, *, sum_ce_loss: bool = True, ignore_index: int = -100):
        super().__init__()
        self.model = model
        self.sum_ce_loss = sum_ce_loss
        self.ignore_index = ignore_index

    def forward(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.model(
            input_ids=samples["input_ids"],
            attention_mask=samples.get("attention_mask"),
            use_cache=False,
        )
        logits = outputs.logits
        labels = samples["labels"]

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        vocab_size = shift_logits.shape[-1]
        flat_logits = shift_logits.view(-1, vocab_size).float()
        flat_labels = shift_labels.view(-1)

        if not self.sum_ce_loss:
            return F.cross_entropy(flat_logits, flat_labels, ignore_index=self.ignore_index, reduction="mean")

        loss_sum = F.cross_entropy(flat_logits, flat_labels, ignore_index=self.ignore_index, reduction="sum")
        if "valid_token_count" in samples:
            denom = samples["valid_token_count"].to(dtype=loss_sum.dtype).sum()
        else:
            valid_mask = (flat_labels != self.ignore_index).to(dtype=loss_sum.dtype)
            denom = torch.clamp(valid_mask.sum(), min=1.0)
        return loss_sum / denom


# ============================================================================
# 6. Top-Level In-Memory Patch Entrypoint
# ============================================================================

def apply_deepseek_rewrites(
    model: nn.Module,
    config: RewriteConfig | None = None,
) -> dict[str, int]:
    """Patch a HuggingFace DeepSeek model (`V2`, `V2-Lite`, `V3`, `R1`, `V3.2`) in-memory."""
    cfg = config or RewriteConfig()
    counts = {
        "v2_rotary_modules": 0,
        "mla_attn_modules": 0,
        "dsa_indexer_modules": 0,
        "moe_blocks": 0,
        "decoupled_bias_linears": 0,
    }

    v2_moe_fn = _make_v2_moe_forward(cfg.expand_moe_ops)
    v3_moe_fn = _make_v3_moe_forward(cfg.expand_moe_ops)

    for module in model.modules():
        if isinstance(module, DeepseekV2RotaryEmbedding):
            module.forward = types.MethodType(_v2_rotary_forward, module)
            counts["v2_rotary_modules"] += 1
        elif isinstance(module, DeepseekV2Attention):
            module.forward = types.MethodType(_v2_attention_forward, module)
            counts["mla_attn_modules"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["decoupled_bias_linears"] += 1
        elif isinstance(module, DeepseekV3Attention):
            module.forward = types.MethodType(_v3_attention_forward, module)
            counts["mla_attn_modules"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["decoupled_bias_linears"] += 1
        elif isinstance(module, DeepseekV32Attention):
            module.forward = types.MethodType(_v32_attention_forward, module)
            # Match HF `@torch.no_grad()` semantics on `DeepseekV32Indexer`
            module.indexer.requires_grad_(False)
            counts["mla_attn_modules"] += 1
            counts["dsa_indexer_modules"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["decoupled_bias_linears"] += 1
        elif isinstance(module, DeepseekV2Moe):
            module.forward = types.MethodType(v2_moe_fn, module)
            counts["moe_blocks"] += 1
        elif isinstance(module, (DeepseekV3MoE, DeepseekV32MoE)):
            module.forward = types.MethodType(v3_moe_fn, module)
            counts["moe_blocks"] += 1
        elif isinstance(module, (DeepseekV2MLP, DeepseekV3MLP, DeepseekV32MLP)):
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.down_proj):
                counts["decoupled_bias_linears"] += 1

    return counts
