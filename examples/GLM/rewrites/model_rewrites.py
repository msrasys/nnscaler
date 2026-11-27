# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Surgical in-memory HuggingFace rewrites for the GLM model family.

Supports:
  - `Glm4ForCausalLM` (Dense GQA + interleaved `Glm4MLP.gate_up_proj`)
  - `Glm4MoeForCausalLM` (GQA + Sparse MoE + `shared_experts`)
  - `Glm4MoeLiteForCausalLM` (MLA + Sparse MoE)
  - `GlmMoeDsaForCausalLM` (GLM-5: MLA + DSA Sparse Indexer + Sparse MoE)
  - `Glm5NextForConditionalGeneration` (GLM-5-Next: Hybrid KDA + MLA/DSA + MoE)
"""

from __future__ import annotations

from dataclasses import dataclass
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.glm4.modeling_glm4 import Glm4Attention, Glm4MLP
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeAttention, Glm4MoeMoE
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteMoE,
    apply_rotary_pos_emb_interleave as lite_apply_rotary_interleave,
    eager_attention_forward as lite_eager_attention_forward,
)
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextAttention,
    Glm5NextTextHyperConnection,
    Glm5NextTextLinearAttention,
    Glm5NextTextMoE,
    apply_mask_to_padding_states,
    eager_attention_forward as glm5_next_eager_attention_forward,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (
    GlmMoeDsaAttention,
    GlmMoeDsaMoE,
    apply_rotary_pos_emb_interleave as dsa_apply_rotary_interleave,
    eager_attention_forward as dsa_eager_attention_forward,
)

from ..ops import (
    glm_causal_conv1d,
    glm_chunk_kda,
    glm_dsa_index_mask,
    glm_grouped_linear,
    glm_mla_concat_kv,
    glm_moe_grouped_gemm,
    glm_moe_permute,
    glm_moe_route,
    glm_moe_unpermute,
    register_glm_ops,
)


@dataclass
class RewriteConfig:
    """Declarative toggles for GLM graph rewrites."""

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
# 1. Glm4MLP Interleaved `gate_up_proj` for TP-Col Sharding
# ============================================================================

def _patch_glm4_mlp(mlp: Glm4MLP) -> None:
    if not getattr(mlp, "_glm_interleaved_gate_up", False):
        w = mlp.gate_up_proj.weight
        if not w.is_meta:
            with torch.no_grad():
                g, u = w.chunk(2, dim=0)
                interleaved = torch.stack((g, u), dim=1).reshape_as(w)
                w.copy_(interleaved)
        mlp._glm_interleaved_gate_up = True

    def _glm4_mlp_forward(self: Glm4MLP, hidden_states: torch.Tensor) -> torch.Tensor:
        up_states = self.gate_up_proj(hidden_states)
        paired = up_states.view(*up_states.shape[:-1], -1, 2)
        gate, up = paired[..., 0], paired[..., 1]
        return self.down_proj(up * self.activation_fn(gate))

    mlp.forward = types.MethodType(_glm4_mlp_forward, mlp)


# ============================================================================
# 2. GLM MLA + Optional DSA Sparse Indexer (`Glm4MoeLite`, `GlmMoeDsa`, `Glm5Next`)
# ============================================================================

def _make_mla_forward(rope_fn: Any, eager_fn: Any, *, return_three_tuple: bool = False):
    def _mla_forward(
        self: Any,
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
            q_resid = hidden_states
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
            q_states = self.q_b_proj(q_resid)

        q_states = q_states.view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(k_pass)

        cos, sin = position_embeddings
        q_rot, k_rot = rope_fn(q_rot, k_rot, cos, sin)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        kv_proj = self.kv_b_proj(k_pass).view(
            batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim
        ).transpose(1, 2)
        k_nope, value_states = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        key_states = glm_mla_concat_kv(k_nope, k_rot)

        effective_mask = attention_mask
        indexer = getattr(self, "indexer", None)
        if indexer is not None:
            mask_3d = attention_mask.squeeze(1) if (attention_mask is not None and attention_mask.ndim == 4) else attention_mask
            effective_mask = glm_dsa_index_mask(
                hidden_states,
                q_resid,
                cos,
                sin,
                mask_3d,
                indexer.wq_b.weight,
                indexer.wk.weight,
                indexer.k_norm.weight,
                indexer.k_norm.bias,
                indexer.weights_proj.weight,
                n_heads=indexer.n_heads,
                head_dim=indexer.head_dim,
                qk_rope_head_dim=indexer.qk_rope_head_dim,
                index_topk=indexer.index_topk,
                k_norm_eps=indexer.k_norm.eps,
            )

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_fn
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            effective_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        if return_three_tuple:
            return attn_output, attn_weights, None
        return attn_output, attn_weights

    return _mla_forward


def _glm5_next_mla_forward(
    self: Glm5NextTextAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    query_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    if self.qk_rope_head_dim > 0:
        k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass = self.kv_a_layernorm(k_pass)
        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
    else:
        k_pass = self.kv_a_layernorm(compressed_kv)
        k_rot = None

    kv_proj = self.kv_b_proj(k_pass).view(
        batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim
    ).transpose(1, 2)
    k_nope, value_states = torch.split(kv_proj, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    key_states = glm_mla_concat_kv(k_nope, k_rot) if k_rot is not None else k_nope

    effective_mask = attention_mask
    indexer = getattr(self, "indexer", None)
    if indexer is not None:
        mask_3d = attention_mask.squeeze(1) if (attention_mask is not None and attention_mask.ndim == 4) else attention_mask
        dr = max(1, indexer.qk_rope_head_dim)
        cos_dummy = hidden_states.new_ones((1, seq_length, dr))
        sin_dummy = hidden_states.new_zeros((1, seq_length, dr))
        effective_mask = glm_dsa_index_mask(
            hidden_states,
            q_resid,
            cos_dummy,
            sin_dummy,
            mask_3d,
            indexer.wq_b.weight,
            indexer.wk.weight,
            indexer.k_norm.weight,
            indexer.k_norm.bias,
            indexer.weights_proj.weight,
            n_heads=indexer.n_heads,
            head_dim=indexer.head_dim,
            qk_rope_head_dim=indexer.qk_rope_head_dim,
            index_topk=indexer.index_topk,
            k_norm_eps=indexer.k_norm.eps,
        )

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, glm5_next_eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        effective_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, None


# ============================================================================
# 3. GLM-5-Next Linear Attention (KDA)
# ============================================================================

def _glm5_next_kda_forward(
    self: Glm5NextTextLinearAttention,
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

    q = glm_causal_conv1d(q, wq, activation=self.activation).permute(0, 3, 1, 2)
    k = glm_causal_conv1d(k, wk, activation=self.activation).permute(0, 3, 1, 2)
    v = glm_causal_conv1d(v, wv, activation=self.activation).permute(0, 3, 1, 2)

    g = self.forget_gate(hidden_states)
    beta = torch.sigmoid(self.b_proj(hidden_states))

    core_attn_out = glm_chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(hidden_shape)
    output = self.o_norm(core_attn_out, gate).reshape(batch_size, seq_len, -1)
    return self.o_proj(output)


# ============================================================================
# 4. GLM-5-Next HyperConnections and GLM Sparse MoE
# ============================================================================

def _glm5_next_hyperconnection_forward(
    self: Glm5NextTextHyperConnection,
    hidden_streams: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seq_len = hidden_streams.shape[:2]
    hc = self.hc_mult
    flattened = self.input_norm(hidden_streams.view(batch_size, seq_len, -1).float())
    flattened = F.linear(flattened, self.fn.float())
    pre_w, post_w, comb_w = flattened.split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = self.scale[0], self.scale[1], self.scale[2]
    comb_w = comb_w.view(*comb_w.shape[:-1], hc, hc)
    comb_b = comb_b.view(hc, hc)
    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb = F.softmax(comb_w * comb_scale + comb_b, dim=-1) + self.hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    for _ in range(self.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
    return post, comb, collapsed


def _make_moe_forward(expand_moe_ops: bool):
    def _moe_forward(self: Any, hidden_states: torch.Tensor) -> torch.Tensor:
        residuals = hidden_states
        orig_shape = hidden_states.shape
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
        num_tokens = flat_hidden.shape[0]

        gate = self.gate
        topk_indices, topk_weights = glm_moe_route(
            flat_hidden,
            gate.weight,
            gate.e_score_correction_bias,
            top_k=gate.top_k,
            n_group=getattr(gate, "n_group", getattr(gate, "num_group", 1)),
            topk_group=gate.topk_group,
            norm_topk_prob=gate.norm_topk_prob,
            routed_scaling_factor=gate.routed_scaling_factor,
        )

        experts = self.experts
        num_experts = experts.num_experts
        hidden_act = getattr(self.config, "hidden_act", "silu")

        if not expand_moe_ops:
            routed_out = glm_moe_grouped_gemm(
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
            permuted, expert_counts, token_indices, slot_indices = glm_moe_permute(
                flat_hidden,
                topk_indices,
                num_experts=num_experts,
                local_expert_start=0,
                local_expert_end=num_experts,
            )
            gate_up_out = glm_grouped_linear(
                permuted, expert_counts, experts.gate_up_proj,
                local_expert_start=0, local_expert_end=num_experts,
            )
            g_part, u_part = gate_up_out.chunk(2, dim=-1)
            act_out = (F.silu(g_part) if hidden_act == "silu" else F.gelu(g_part)) * u_part
            down_out = glm_grouped_linear(
                act_out, expert_counts, experts.down_proj,
                local_expert_start=0, local_expert_end=num_experts,
            )
            routed_out = glm_moe_unpermute(down_out, topk_weights, token_indices, slot_indices, num_tokens)

        return routed_out.view(*orig_shape) + self.shared_experts(residuals)

    return _moe_forward


# ============================================================================
# 5. End-to-End Compile Wrapper (`sum / valid_count` CE)
# ============================================================================

class GlmCausalLMTaskWrapper(nn.Module):
    """Thin training wrapper accepting `samples: dict[str, Tensor]` for `nnscaler.parallelize`."""

    def __init__(self, model: nn.Module, *, sum_ce_loss: bool = True) -> None:
        super().__init__()
        self.model = model
        self.sum_ce_loss = sum_ce_loss

    def forward(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = samples["input_ids"]
        attention_mask = samples.get("attention_mask")
        labels = samples["labels"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
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


def apply_glm_rewrites(
    model: nn.Module,
    config: RewriteConfig | None = None,
) -> dict[str, int]:
    """Register custom ops and apply in-memory surgical rewrites to GLM models."""
    cfg = config or RewriteConfig()
    register_glm_ops()

    counts: dict[str, int] = {
        "glm4_mlp_rewrites": 0,
        "mla_rewrites": 0,
        "dsa_indexer_frozen": 0,
        "kda_rewrites": 0,
        "hyperconnection_rewrites": 0,
        "moe_rewrites": 0,
        "row_tp_bias_decoupled": 0,
    }
    moe_fn = _make_moe_forward(cfg.expand_moe_ops)
    lite_mla_fn = _make_mla_forward(lite_apply_rotary_interleave, lite_eager_attention_forward)
    dsa_mla_fn = _make_mla_forward(dsa_apply_rotary_interleave, dsa_eager_attention_forward, return_three_tuple=True)

    for module in model.modules():
        if isinstance(module, Glm4MLP):
            _patch_glm4_mlp(module)
            counts["glm4_mlp_rewrites"] += 1
        elif isinstance(module, (Glm4Attention, Glm4MoeAttention)):
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["row_tp_bias_decoupled"] += 1
        elif isinstance(module, Glm4MoeLiteAttention):
            module.forward = types.MethodType(lite_mla_fn, module)
            counts["mla_rewrites"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["row_tp_bias_decoupled"] += 1
        elif isinstance(module, GlmMoeDsaAttention):
            module.forward = types.MethodType(dsa_mla_fn, module)
            counts["mla_rewrites"] += 1
            if getattr(module, "indexer", None) is not None:
                module.indexer.requires_grad_(False)
                counts["dsa_indexer_frozen"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["row_tp_bias_decoupled"] += 1
        elif isinstance(module, Glm5NextTextAttention):
            module.forward = types.MethodType(_glm5_next_mla_forward, module)
            counts["mla_rewrites"] += 1
            if getattr(module, "indexer", None) is not None:
                module.indexer.requires_grad_(False)
                counts["dsa_indexer_frozen"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["row_tp_bias_decoupled"] += 1
        elif isinstance(module, Glm5NextTextLinearAttention):
            module.forward = types.MethodType(_glm5_next_kda_forward, module)
            counts["kda_rewrites"] += 1
            if cfg.decouple_row_tp_bias and _patch_row_parallel_linear_bias(module.o_proj):
                counts["row_tp_bias_decoupled"] += 1
        elif isinstance(module, Glm5NextTextHyperConnection):
            module.forward = types.MethodType(_glm5_next_hyperconnection_forward, module)
            counts["hyperconnection_rewrites"] += 1
        elif isinstance(module, (Glm4MoeMoE, Glm4MoeLiteMoE, GlmMoeDsaMoE, Glm5NextTextMoE)):
            module.forward = types.MethodType(moe_fn, module)
            counts["moe_rewrites"] += 1

    return counts
