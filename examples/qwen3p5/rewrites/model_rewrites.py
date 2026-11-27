"""Surgical instance-level rewrites and weightless model construction for Qwen3.5.

Supports all three stages of Qwen3.5 adaptation without mutating global HuggingFace classes:
  - Stage 1 (Dense `0.8B` .. `27B`): `Qwen3_5ForCausalLM`
  - Stage 2 (Sparse MoE `35B-A3B` .. `397B-A17B`): `Qwen3_5MoeForCausalLM`
    with configurable `expand_moe_ops: False` (default) or `True`
  - Stage 3 (Multimodal VL): `Qwen3_5ForConditionalGeneration` & `Qwen3_5MoeForConditionalGeneration`
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import types
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    Qwen3_5VisionAttention,
    Qwen3_5VisionModel,
    Qwen3_5VisionPatchEmbed,
    apply_mask_to_padding_states,
    apply_rotary_pos_emb_vision,
    get_vision_attention_seqlens,
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeGatedDeltaNet,
    Qwen3_5MoeModel,
    Qwen3_5MoeSparseMoeBlock,
)

from examples.qwen3p5.ops.gdn_ops import (
    qwen3_5_causal_conv1d,
    qwen3_5_chunk_gated_delta_rule,
)
from examples.qwen3p5.ops.moe_ops import (
    qwen3_5_grouped_linear,
    qwen3_5_moe_grouped_gemm,
    qwen3_5_moe_permute,
    qwen3_5_moe_route,
    qwen3_5_moe_unpermute,
)
from examples.qwen3p5.ops.vision_ops import (
    qwen3_5_masked_scatter,
    qwen3_5_vision_varlen_attn,
)


@dataclass
class RewriteConfig:
    """Declarative rewrite switches decoupled from HuggingFace globals."""

    loss_reduction: str = "sum-count"  # "sum-count" | "mean"
    wrap_gdn_kernels: bool = True
    head_grouped_gdn_qkv: bool = True
    static_moe_grouped_gemm: bool = True
    expand_moe_ops: bool = False  # Default False per design: fused route + grouped_gemm
    wrap_vision_ops: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RewriteConfig":
        if not data:
            return cls()
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# ============================================================================
# 1. Gated DeltaNet (`Qwen3_5GatedDeltaNet` & `Qwen3_5MoeGatedDeltaNet`)
# ============================================================================

def _gdn_head_group_perm(num_k_heads: int, head_k_dim: int, num_v_heads: int, head_v_dim: int, device: torch.device) -> torch.Tensor:
    """Index permutation from HF `[Q_all, K_all, V_all]` to head-grouped `[Hk, (Q_g, K_g, V_g)]`."""
    key_dim = num_k_heads * head_k_dim
    v_per_k = num_v_heads // num_k_heads
    v_group_dim = v_per_k * head_v_dim
    idx_chunks: list[torch.Tensor] = []
    for hk in range(num_k_heads):
        q_idx = torch.arange(hk * head_k_dim, (hk + 1) * head_k_dim, device=device)
        k_idx = torch.arange(key_dim + hk * head_k_dim, key_dim + (hk + 1) * head_k_dim, device=device)
        v_idx = torch.arange(2 * key_dim + hk * v_group_dim, 2 * key_dim + (hk + 1) * v_group_dim, device=device)
        idx_chunks.extend((q_idx, k_idx, v_idx))
    return torch.cat(idx_chunks, dim=0)


@torch.no_grad()
def permute_gdn_qkv_weights_(module: nn.Module, inverse: bool = False) -> None:
    """Lossless bijection on `in_proj_qkv.weight` & `conv1d.weight` (and optional bias) for real checkpoints."""
    for sub in module.modules():
        if isinstance(sub, (Qwen3_5GatedDeltaNet, Qwen3_5MoeGatedDeltaNet)):
            if sub.in_proj_qkv.weight.is_meta:
                continue
            perm = _gdn_head_group_perm(
                sub.num_k_heads, sub.head_k_dim, sub.num_v_heads, sub.head_v_dim, sub.in_proj_qkv.weight.device
            )
            if inverse:
                inv = torch.empty_like(perm)
                inv[perm] = torch.arange(perm.numel(), device=perm.device)
                perm = inv
            sub.in_proj_qkv.weight.copy_(sub.in_proj_qkv.weight.index_select(0, perm))
            sub.conv1d.weight.copy_(sub.conv1d.weight.index_select(0, perm))
            if sub.conv1d.bias is not None:
                sub.conv1d.bias.copy_(sub.conv1d.bias.index_select(0, perm))


def _make_gdn_forward(head_grouped_qkv: bool):
    def _gdn_forward(
        self: Qwen3_5GatedDeltaNet | Qwen3_5MoeGatedDeltaNet,
        hidden_states: torch.Tensor,
        cache_params: Any = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        mixed_qkv = qwen3_5_causal_conv1d(mixed_qkv, self.conv1d.weight, self.conv1d.bias, self.activation)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if head_grouped_qkv:
            v_per_k = self.num_v_heads // self.num_k_heads
            v_group_dim = v_per_k * self.head_v_dim
            group_width = 2 * self.head_k_dim + v_group_dim
            grouped = mixed_qkv.reshape(batch_size, seq_len, self.num_k_heads, group_width)
            query, key, value = torch.split(grouped, [self.head_k_dim, self.head_k_dim, v_group_dim], dim=-1)
            value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [self.key_dim, self.key_dim, self.value_dim],
                dim=-1,
            )
            query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
            key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
            value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # NOTE: We intentionally keep GQA head ratio inside `qwen3_5_chunk_gated_delta_rule`
        # (both FLA Triton kernel and our CPU fallback support `Hv % Hk == 0` natively!)
        # without `repeat_interleave`, avoiding `repeat_interleave` graph bloat.
        core_attn_out = qwen3_5_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            scale=None,
            use_qk_l2norm_in_kernel=True,
        )

        # Keep 4D `[B, S, Hv, Dv]` into `self.norm` instead of flattening `(B, S, Hv)` to `[-1, Dv]`,
        # preserving `Hv` (`dim=2`) sharding straight into `out_proj` (`dim=1`) with zero extra collectives!
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    return _gdn_forward


# ============================================================================
# 2. Sparse MoE (`Qwen3_5MoeSparseMoeBlock`)
# ============================================================================

def _make_moe_block_forward(expand_moe_ops: bool):
    def _moe_forward_fused(
        self: Qwen3_5MoeSparseMoeBlock, hidden_states: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states_reshaped)
        routing_weights, selected_experts = qwen3_5_moe_route(
            hidden_states_reshaped,
            self.gate.weight,
            top_k=self.gate.top_k,
        )
        expert_output = qwen3_5_moe_grouped_gemm(
            hidden_states_reshaped,
            routing_weights,
            selected_experts,
            self.experts.gate_up_proj,
            self.experts.down_proj,
            act_fn_name=self.experts.config.hidden_act,
            local_expert_start=0,
            local_expert_end=self.experts.num_experts,
        )

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)

    def _moe_forward_expanded(
        self: Qwen3_5MoeSparseMoeBlock, hidden_states: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        num_tokens = batch_size * sequence_length
        hidden_states_reshaped = hidden_states.view(num_tokens, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states_reshaped)

        # Native IRDimops router!
        router_logits = F.linear(hidden_states_reshaped, self.gate.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        router_top_value, selected_experts = torch.topk(router_probs, self.gate.top_k, dim=-1)
        router_top_value = router_top_value / router_top_value.sum(dim=-1, keepdim=True)
        routing_weights = router_top_value.to(hidden_states_reshaped.dtype)

        num_experts = self.experts.num_experts
        permuted_x, expert_counts, sorted_tokens, sorted_probs = qwen3_5_moe_permute(
            hidden_states_reshaped,
            routing_weights,
            selected_experts,
            num_experts=num_experts,
        )

        # Native Chunk along dim=1 (`2*I -> I, I`) unlocks both EP (`dim=0`) and ETP (`dim=1`)!
        gate_weight, up_weight = torch.chunk(self.experts.gate_up_proj, 2, dim=1)
        gate_out = qwen3_5_grouped_linear(
            permuted_x,
            gate_weight,
            expert_counts,
            local_expert_start=0,
            local_expert_end=num_experts,
        )
        up_out = qwen3_5_grouped_linear(
            permuted_x,
            up_weight,
            expert_counts,
            local_expert_start=0,
            local_expert_end=num_experts,
        )
        inter = self.experts.act_fn(gate_out) * up_out
        down_out = qwen3_5_grouped_linear(
            inter,
            self.experts.down_proj,
            expert_counts,
            local_expert_start=0,
            local_expert_end=num_experts,
        )
        expert_output = qwen3_5_moe_unpermute(
            down_out,
            sorted_tokens,
            sorted_probs,
            num_tokens=num_tokens,
        )

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        return expert_output.reshape(batch_size, sequence_length, hidden_dim)

    return _moe_forward_expanded if expand_moe_ops else _moe_forward_fused


# ============================================================================
# 3. Multimodal Vision & Scatter (`Qwen3_5Vision*` & `Qwen3_5Model`)
# ============================================================================

def _vision_patch_embed_forward(self: Qwen3_5VisionPatchEmbed, hidden_states: torch.Tensor) -> torch.Tensor:
    target_dtype = self.proj.weight.dtype
    flat_x = hidden_states.reshape(-1, self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size)
    flat_w = self.proj.weight.reshape(self.embed_dim, -1)
    return F.linear(flat_x.to(dtype=target_dtype), flat_w, self.proj.bias)


def _vision_attn_forward(
    self: Qwen3_5VisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    max_seqlen: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    qkv_states = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3)
    query_states, key_states, value_states = qkv_states[0], qkv_states[1], qkv_states[2]
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
    attn_output = qwen3_5_vision_varlen_attn(
        query_states,
        key_states,
        value_states,
        cu_seqlens,
        scaling=float(self.scaling),
    )
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return self.proj(attn_output)


def _vision_model_forward(
    self: Qwen3_5VisionModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    vision_interp_indices: torch.Tensor | None = None,
    vision_interp_weights: torch.Tensor | None = None,
    vision_pos_ids: torch.Tensor | None = None,
    vision_cu_seqlens: torch.Tensor | None = None,
    **kwargs: Any,
) -> BaseModelOutputWithPooling:
    if vision_interp_indices is None or vision_interp_weights is None:
        vision_interp_indices, vision_interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            mode=self.interpolation_mode,
            align_corners=self.interpolation_align_corners,
            spatial_merge_size=self.config.spatial_merge_size,
            kwargs=kwargs,
        )
    if vision_pos_ids is None:
        vision_pos_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size, kwargs=kwargs)
    if vision_cu_seqlens is None:
        vision_cu_seqlens, _ = get_vision_attention_seqlens(grid_thw, self.config, kwargs=kwargs)

    hidden_states = self.patch_embed(hidden_states)
    pos_embeds = (self.pos_embed(vision_interp_indices) * vision_interp_weights[:, :, None]).sum(1)
    hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)
    position_embeddings = self.rotary_pos_emb(hidden_states, vision_pos_ids)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    for blk in self.blocks:
        hidden_states = blk(
            hidden_states,
            cu_seqlens=vision_cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    merged_hidden_states = self.merger(hidden_states)
    return BaseModelOutputWithPooling(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
    )


def _multimodal_model_forward(
    self: Qwen3_5Model | Qwen3_5MoeModel,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Any = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    pixel_values: torch.Tensor | None = None,
    image_grid_thw: torch.LongTensor | None = None,
    vision_interp_indices: torch.Tensor | None = None,
    vision_interp_weights: torch.Tensor | None = None,
    vision_pos_ids: torch.Tensor | None = None,
    vision_cu_seqlens: torch.Tensor | None = None,
    **kwargs: Any,
):
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        vis_out = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            vision_interp_indices=vision_interp_indices,
            vision_interp_weights=vision_interp_weights,
            vision_pos_ids=vision_pos_ids,
            vision_cu_seqlens=vision_cu_seqlens,
            **kwargs,
        )
        image_embeds = vis_out.pooler_output.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
        inputs_embeds = qwen3_5_masked_scatter(inputs_embeds, image_mask, image_embeds)

    if position_ids is None and input_ids is not None and not input_ids.is_meta:
        position_ids = self.compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )

    return self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )


def _row_tp_safe_linear_forward(self: nn.Linear, input: torch.Tensor) -> torch.Tensor:
    out = F.linear(input, self.weight, None)
    return out + self.bias if self.bias is not None else out


def apply_qwen3_5_rewrites(model: nn.Module, rewrite_cfg: RewriteConfig) -> dict[str, int]:
    """Bind surgical instance-level forward rewrites onto `model` submodules."""
    counts = {
        "gdn": 0,
        "moe": 0,
        "vision_patch": 0,
        "vision_attn": 0,
        "vision_model": 0,
        "vision_biased_linear": 0,
        "mm_model": 0,
    }
    gdn_fn = _make_gdn_forward(rewrite_cfg.head_grouped_gdn_qkv)
    moe_fn = _make_moe_block_forward(rewrite_cfg.expand_moe_ops)

    if rewrite_cfg.wrap_gdn_kernels and rewrite_cfg.head_grouped_gdn_qkv:
        permute_gdn_qkv_weights_(model, inverse=False)

    for fqn, sub in model.named_modules():
        if rewrite_cfg.wrap_gdn_kernels and isinstance(sub, (Qwen3_5GatedDeltaNet, Qwen3_5MoeGatedDeltaNet)):
            sub.forward = types.MethodType(gdn_fn, sub)
            counts["gdn"] += 1
        elif rewrite_cfg.static_moe_grouped_gemm and isinstance(sub, Qwen3_5MoeSparseMoeBlock):
            sub.forward = types.MethodType(moe_fn, sub)
            counts["moe"] += 1
        elif rewrite_cfg.wrap_vision_ops and isinstance(sub, Qwen3_5VisionPatchEmbed):
            sub.forward = types.MethodType(_vision_patch_embed_forward, sub)
            counts["vision_patch"] += 1
        elif rewrite_cfg.wrap_vision_ops and isinstance(sub, Qwen3_5VisionAttention):
            sub.forward = types.MethodType(_vision_attn_forward, sub)
            counts["vision_attn"] += 1
        elif rewrite_cfg.wrap_vision_ops and isinstance(sub, Qwen3_5VisionModel):
            sub.forward = types.MethodType(_vision_model_forward, sub)
            counts["vision_model"] += 1
        elif rewrite_cfg.wrap_vision_ops and isinstance(sub, (Qwen3_5Model, Qwen3_5MoeModel)):
            sub.forward = types.MethodType(_multimodal_model_forward, sub)
            counts["mm_model"] += 1
        elif rewrite_cfg.wrap_vision_ops and isinstance(sub, nn.Linear) and sub.bias is not None and "visual" in fqn:
            sub.forward = types.MethodType(_row_tp_safe_linear_forward, sub)
            counts["vision_biased_linear"] += 1
    return counts


# ============================================================================
# 4. End-to-End Training Wrapper & Weightless Model Builder
# ============================================================================

class Qwen3_5CompileWrapper(nn.Module):
    """End-to-end training wrapper with configurable `sum-count` vs `mean` cross-entropy loss."""

    def __init__(self, model: nn.Module, loss_reduction: str = "sum-count") -> None:
        super().__init__()
        if loss_reduction not in ("sum-count", "mean"):
            raise ValueError(f"Unsupported loss_reduction={loss_reduction!r}; expected 'sum-count' or 'mean'")
        self.model = model
        self.loss_reduction = loss_reduction

    def forward(self, samples: dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = samples["input_ids"]
        attention_mask = samples["attention_mask"] if "attention_mask" in samples else None
        labels = samples["labels"]

        extra_kwargs: dict[str, Any] = {}
        for opt_key in (
            "position_ids",
            "pixel_values",
            "image_grid_thw",
            "vision_interp_indices",
            "vision_interp_weights",
            "vision_pos_ids",
            "vision_cu_seqlens",
        ):
            if opt_key in samples:
                extra_kwargs[opt_key] = samples[opt_key]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **extra_kwargs,
        )
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        vocab_size = shift_logits.size(-1)

        if self.loss_reduction == "mean":
            return F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )

        # `sum-count` loss rewrite: unlocks `N+ C^, N+ -> 1+` token-sharded logits/CE without AllGather!
        loss_sum = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        valid_count = (shift_labels != -100).sum().clamp(min=1).to(loss_sum.dtype)
        return loss_sum / valid_count


def _fix_meta_rotary_buffers_(model: nn.Module) -> None:
    """Materialize tiny non-persistent RoPE `inv_freq` buffers if needed, or ensure dtype consistency."""
    # Note: Under `MetaStrategy` with our `trace_strategy.py` patch, meta buffers trace cleanly on meta!
    # Nothing needs to be materialized in RAM even for 397B-A17B.
    return None


def build_qwen3_5_config(config_dict: dict[str, Any], multimodal: bool = False):
    """Instantiate the right HuggingFace Qwen3.5 config class from a JSON/preset dict."""
    cfg_data = dict(config_dict)
    model_type = cfg_data.get("model_type", "")
    is_moe = "moe" in model_type or "num_experts" in cfg_data or (
        "text_config" in cfg_data and "num_experts" in cfg_data["text_config"]
    )

    if multimodal or "vision_config" in cfg_data:
        if "text_config" not in cfg_data:
            text_fields = {k: v for k, v in cfg_data.items() if k not in ("vision_config", "model_type")}
            cfg_data = {"text_config": text_fields, "vision_config": cfg_data.get("vision_config", {})}
        cfg_cls = Qwen3_5MoeConfig if is_moe else Qwen3_5Config
        cfg = cfg_cls(**cfg_data)
        cfg._attn_implementation = "sdpa"
        cfg.text_config._attn_implementation = "sdpa"
        cfg.vision_config._attn_implementation = "sdpa"
        return cfg, True, is_moe

    text_dict = cfg_data.get("text_config", cfg_data)
    text_cls = Qwen3_5MoeTextConfig if is_moe else Qwen3_5TextConfig
    cfg = text_cls(**text_dict)
    cfg._attn_implementation = "sdpa"
    return cfg, False, is_moe


def build_qwen3_5_model(
    config_dict: dict[str, Any],
    rewrite_cfg: RewriteConfig | None = None,
    *,
    multimodal: bool = False,
    weightless: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
) -> tuple[Qwen3_5CompileWrapper, dict[str, Any]]:
    """Construct a weightless (`meta`) or eager (`cpu`) Qwen3.5 training wrapper."""
    rewrite_cfg = rewrite_cfg or RewriteConfig()
    cfg, is_mm, is_moe = build_qwen3_5_config(config_dict, multimodal=multimodal)

    if is_mm:
        model_cls = Qwen3_5MoeForConditionalGeneration if is_moe else Qwen3_5ForConditionalGeneration
    else:
        model_cls = Qwen3_5MoeForCausalLM if is_moe else Qwen3_5ForCausalLM

    if weightless:
        with torch.device("meta"):
            inner_model = model_cls(cfg)
            inner_model.to(dtype=dtype)
        _fix_meta_rotary_buffers_(inner_model)
    else:
        torch.manual_seed(seed)
        inner_model = model_cls(cfg)
        inner_model.to(dtype=dtype)

    inner_model.train()
    rewrite_counts = apply_qwen3_5_rewrites(inner_model, rewrite_cfg)
    wrapper = Qwen3_5CompileWrapper(inner_model, loss_reduction=rewrite_cfg.loss_reduction)
    wrapper.train()

    meta_info = {
        "model_class": model_cls.__name__,
        "is_moe": is_moe,
        "is_multimodal": is_mm,
        "weightless": weightless,
        "dtype": str(dtype),
        "rewrite_counts": rewrite_counts,
        "rewrite_config": rewrite_cfg.to_dict(),
    }
    return wrapper, meta_info


def build_dummy_samples(
    config_dict: dict[str, Any],
    *,
    batch_size: int = 2,
    seq_len: int = 128,
    multimodal: bool = False,
    image_grid: tuple[int, int, int] = (1, 4, 4),
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Build deterministic dummy input samples (text-only or multimodal VL) on CPU for tracing/testing."""
    cfg, is_mm, _ = build_qwen3_5_config(config_dict, multimodal=multimodal)
    text_cfg = cfg.text_config if is_mm else cfg
    vocab_size = text_cfg.vocab_size

    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234)
    input_ids = torch.randint(0, min(vocab_size, 1024), (batch_size, seq_len), dtype=torch.long, generator=gen)
    labels = input_ids.clone()
    labels[:, -1] = -100
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    samples: dict[str, torch.Tensor] = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }

    if is_mm:
        vis_cfg = cfg.vision_config
        t_grid, h_grid, w_grid = image_grid
        merge = vis_cfg.spatial_merge_size
        if h_grid % merge != 0 or w_grid % merge != 0:
            raise ValueError(f"image_grid (H={h_grid}, W={w_grid}) must be divisible by spatial_merge_size={merge}")
        patches_per_img = t_grid * h_grid * w_grid
        merged_tokens_per_img = patches_per_img // (merge * merge)
        total_patches = batch_size * patches_per_img
        patch_dim = vis_cfg.in_channels * vis_cfg.temporal_patch_size * vis_cfg.patch_size * vis_cfg.patch_size

        if merged_tokens_per_img + 4 >= seq_len:
            raise ValueError(
                f"seq_len={seq_len} too short for {merged_tokens_per_img} merged image tokens per batch item"
            )

        # Place exact `merged_tokens_per_img` image placeholder tokens in each sequence!
        input_ids_cpu = input_ids.clone()
        input_ids_cpu[:, 2 : 2 + merged_tokens_per_img] = cfg.image_token_id
        samples["input_ids"] = input_ids_cpu.to(device)

        grid_thw = torch.tensor([list(image_grid) for _ in range(batch_size)], dtype=torch.long)
        pixel_values = torch.randn((total_patches, patch_dim), generator=gen, dtype=dtype)

        num_grid_per_side = int(vis_cfg.num_position_embeddings**0.5)
        interp_indices, interp_weights = get_vision_interpolation_indices_and_weights(
            grid_thw,
            num_grid_per_side=num_grid_per_side,
            mode="bilinear",
            align_corners=True,
            spatial_merge_size=merge,
            kwargs={},
        )
        vis_pos_ids = get_vision_position_ids(grid_thw, merge, kwargs={})
        vis_cu_seqlens, _ = get_vision_attention_seqlens(grid_thw, vis_cfg, kwargs={})
        pos_ids_4d = (
            torch.arange(seq_len, dtype=torch.long)
            .view(1, 1, seq_len)
            .expand(4, batch_size, seq_len)
            .contiguous()
        )

        samples.update(
            {
                "pixel_values": pixel_values.to(device),
                "image_grid_thw": grid_thw.to(device),
                "vision_interp_indices": interp_indices.to(device),
                "vision_interp_weights": interp_weights.to(dtype=dtype, device=device),
                "vision_pos_ids": vis_pos_ids.to(device),
                "vision_cu_seqlens": vis_cu_seqlens.to(device),
                "position_ids": pos_ids_4d.to(device),
            }
        )

    return samples
