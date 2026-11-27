# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Custom nnScaler operators for Kimi-Linear and Kimi-K2.5 Sparse MoE.

Supports both:
  - Mode A (`expand_moe_ops=False`, default): `kimi_moe_route` + fused `kimi_moe_grouped_gemm`
    with `TransformRule` for Expert Parallelism (`EP`).
  - Mode B (`expand_moe_ops=True`, opt-in): `kimi_moe_permute` -> `kimi_grouped_linear`
    (EP + ETP) -> `kimi_moe_unpermute`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from nnscaler.graph.function.dimops import DimopSplit, TransformRule
from nnscaler.graph.parser.register import CustomizedOps, register_op
from nnscaler.ir.cten import IRObject


def _unwrap_int(val: Any) -> int:
    return int(val.value) if isinstance(val, IRObject) else int(val)


def _apply_act(x: torch.Tensor, hidden_act: str) -> torch.Tensor:
    if hidden_act == "silu":
        return F.silu(x)
    if hidden_act == "gelu":
        return F.gelu(x)
    raise ValueError(f"Unsupported MoE hidden_act: {hidden_act}")


def kimi_moe_route(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Kimi-Linear / Kimi-K2.5 sigmoid + group-limited top-k router."""
    num_experts = router_weight.shape[0]
    hidden_dim = router_weight.shape[1]
    flat = hidden_states.view(-1, hidden_dim)
    router_logits = F.linear(flat.type(torch.float32), router_weight.type(torch.float32))
    scores = router_logits.sigmoid()
    scores_for_choice = scores + e_score_correction_bias
    group_scores = (
        scores_for_choice.view(-1, n_group, num_experts // n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, num_experts // n_group)
        .reshape(-1, num_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor
    return topk_indices, topk_weights


def kimi_moe_route_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = hidden_states.numel() // hidden_states.shape[-1]
    return (
        torch.empty((t, top_k), dtype=torch.int64, device=hidden_states.device),
        torch.empty(
            (t, top_k), dtype=torch.float32, device=hidden_states.device,
            requires_grad=torch.is_grad_enabled() and (hidden_states.requires_grad or router_weight.requires_grad),
        ),
    )


def kimi_moe_route_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 3:
        raise ValueError(f"kimi_moe_route expects 3 tensor inputs, got {len(args)}")
    top_k = _unwrap_int(kwargs.get("top_k", 8))
    return f"t^ h^, E^ h^, E^ -> t^ {top_k}^, t^ {top_k}^"


def kimi_moe_grouped_gemm(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    hidden_act: str = "silu",
    local_expert_start: int = 0,
    local_expert_end: int = -1,
    _total_experts: int = -1,
) -> torch.Tensor:
    """Fused 3D routed MoE experts matching `KimiLinearExperts` / `DeepseekV3Experts`."""
    num_local_experts = gate_up_proj.shape[0]
    if local_expert_end < 0:
        local_expert_end = local_expert_start + num_local_experts

    final_hidden_states = torch.zeros_like(hidden_states)
    num_global_experts = max(local_expert_end, int(topk_indices.max().item()) + 1) if topk_indices.numel() > 0 else local_expert_end
    expert_mask = F.one_hot(topk_indices.long(), num_classes=num_global_experts).permute(2, 1, 0)

    for local_idx in range(num_local_experts):
        global_idx = local_expert_start + local_idx
        top_x, token_idx = torch.where(expert_mask[global_idx])
        current_state = hidden_states[token_idx]
        gate_up = F.linear(current_state, gate_up_proj[local_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        current_hidden = _apply_act(gate, hidden_act) * up
        current_hidden = F.linear(current_hidden, down_proj[local_idx])
        current_hidden = current_hidden * topk_weights[token_idx, top_x, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))
    return final_hidden_states


def kimi_moe_grouped_gemm_fake(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    hidden_act: str = "silu",
    local_expert_start: int = 0,
    local_expert_end: int = -1,
    _total_experts: int = -1,
) -> torch.Tensor:
    return torch.empty_like(
        hidden_states,
        requires_grad=torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (hidden_states, topk_weights, gate_up_proj, down_proj)
        ),
    )


def kimi_moe_grouped_gemm_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 5:
        raise ValueError(f"kimi_moe_grouped_gemm expects 5 tensor inputs, got {len(args)}")
    return "t^ h^, t^ k^, t^ k^, E+ m^ h^, E+ h^ n^ -> t^ h^"


def build_fused_ep_transform_rule() -> TransformRule:
    itransform = [
        DimopSplit.R(),
        DimopSplit.R(),
        DimopSplit.R(),
        DimopSplit.D(0),
        DimopSplit.D(0),
    ]
    otransform = [DimopSplit.V()]

    def modifier(kwargs: dict[str, Any], idx: int, dim: int, num: int, pos: int) -> dict[str, Any]:
        new_kwargs = dict(kwargs)
        orig_start = _unwrap_int(new_kwargs.get("local_expert_start", 0))
        orig_end = _unwrap_int(new_kwargs.get("local_expert_end", -1))
        total_local = orig_end - orig_start if orig_end > orig_start else _unwrap_int(new_kwargs.get("_total_experts", -1))
        if total_local > 0:
            per = total_local // num
            new_kwargs["local_expert_start"] = orig_start + pos * per
            new_kwargs["local_expert_end"] = orig_start + (pos + 1) * per
        return new_kwargs

    return TransformRule(itransform, otransform, modifier)


def kimi_moe_permute(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int = 8,
    local_expert_start: int = 0,
    local_expert_end: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if local_expert_end < 0:
        local_expert_end = num_experts
    num_local = local_expert_end - local_expert_start
    flat_idx = topk_indices.reshape(-1)
    mask = (flat_idx >= local_expert_start) & (flat_idx < local_expert_end)
    valid_pos = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    token_indices = valid_pos // topk_indices.shape[1]
    slot_indices = valid_pos % topk_indices.shape[1]
    local_experts = flat_idx[valid_pos] - local_expert_start
    order = torch.argsort(local_experts, stable=True)
    token_indices = token_indices[order]
    slot_indices = slot_indices[order]
    sorted_local_experts = local_experts[order]
    expert_counts = torch.bincount(sorted_local_experts, minlength=num_local)
    permuted = hidden_states.index_select(0, token_indices)
    return permuted, expert_counts, token_indices, slot_indices


def kimi_moe_permute_fake(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int = 8,
    local_expert_start: int = 0,
    local_expert_end: int = -1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if local_expert_end < 0:
        local_expert_end = num_experts
    num_local = local_expert_end - local_expert_start
    t, k = topk_indices.shape
    h = hidden_states.shape[-1]
    dev = hidden_states.device
    return (
        torch.empty(
            (t * k, h), dtype=hidden_states.dtype, device=dev,
            requires_grad=torch.is_grad_enabled() and hidden_states.requires_grad,
        ),
        torch.empty((num_local,), dtype=torch.int64, device=dev),
        torch.empty((t * k,), dtype=torch.int64, device=dev),
        torch.empty((t * k,), dtype=torch.int64, device=dev),
    )


def kimi_moe_permute_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 2:
        raise ValueError(f"kimi_moe_permute expects 2 tensor inputs, got {len(args)}")
    tk_len = args[0].shape[0] * args[1].shape[1]
    num_experts = _unwrap_int(kwargs.get("num_experts", 8))
    local_start = _unwrap_int(kwargs.get("local_expert_start", 0))
    local_end = _unwrap_int(kwargs.get("local_expert_end", -1))
    num_local = (num_experts - local_start) if local_end < 0 else (local_end - local_start)
    return f"t^ h^, t^ k^ -> {tk_len}^ h^, {num_local}^, {tk_len}^, {tk_len}^"


def kimi_grouped_linear(
    permuted_tokens: torch.Tensor,
    expert_counts: torch.Tensor,
    weight: torch.Tensor,
    local_expert_start: int = 0,
    local_expert_end: int = -1,
) -> torch.Tensor:
    num_local = weight.shape[0]
    if local_expert_end < 0:
        local_expert_end = local_expert_start + num_local
    counts = expert_counts.tolist()
    all_chunks = torch.split(permuted_tokens, counts, dim=0)
    out_features = weight.shape[1]
    outs = []
    for g_idx, c in enumerate(counts):
        if local_expert_start <= g_idx < local_expert_end:
            outs.append(F.linear(all_chunks[g_idx], weight[g_idx - local_expert_start]))
        else:
            outs.append(permuted_tokens.new_zeros((c, out_features)))
    return torch.cat(outs, dim=0)


def kimi_grouped_linear_fake(
    permuted_tokens: torch.Tensor,
    expert_counts: torch.Tensor,
    weight: torch.Tensor,
    local_expert_start: int = 0,
    local_expert_end: int = -1,
) -> torch.Tensor:
    return permuted_tokens.new_empty(
        (permuted_tokens.shape[0], weight.shape[1]),
        requires_grad=torch.is_grad_enabled() and (permuted_tokens.requires_grad or weight.requires_grad),
    )


def kimi_grouped_linear_anno(*args: Any, **kwargs: Any) -> str:
    return "tk^ k+, g^, E+ m k+ -> tk^ m"


def build_grouped_linear_ep_rule() -> TransformRule:
    itransform = [DimopSplit.R(), DimopSplit.R(), DimopSplit.D(0)]
    otransform = [DimopSplit.V()]

    def modifier(kwargs: dict[str, Any], idx: int, dim: int, num: int, pos: int) -> dict[str, Any]:
        new_kwargs = dict(kwargs)
        orig_start = _unwrap_int(new_kwargs.get("local_expert_start", 0))
        orig_end = _unwrap_int(new_kwargs.get("local_expert_end", -1))
        total_local = orig_end - orig_start
        if total_local > 0:
            per = total_local // num
            new_kwargs["local_expert_start"] = orig_start + pos * per
            new_kwargs["local_expert_end"] = orig_start + (pos + 1) * per
        return new_kwargs

    return TransformRule(itransform, otransform, modifier)


def kimi_moe_unpermute(
    expert_outputs: torch.Tensor,
    topk_weights: torch.Tensor,
    token_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    weights = topk_weights[token_indices, slot_indices].unsqueeze(-1)
    weighted = expert_outputs * weights
    out = expert_outputs.new_zeros((num_tokens, expert_outputs.shape[-1]))
    return out.index_add(0, token_indices, weighted.to(out.dtype))


def kimi_moe_unpermute_fake(
    expert_outputs: torch.Tensor,
    topk_weights: torch.Tensor,
    token_indices: torch.Tensor,
    slot_indices: torch.Tensor,
    num_tokens: int,
) -> torch.Tensor:
    return expert_outputs.new_empty(
        (num_tokens, expert_outputs.shape[-1]),
        requires_grad=torch.is_grad_enabled() and (expert_outputs.requires_grad or topk_weights.requires_grad),
    )


def kimi_moe_unpermute_anno(*args: Any, **kwargs: Any) -> str:
    return "tk^ h^, t^ k^, tk^, tk^ -> t^ h^"


def register_moe_ops() -> dict[str, str]:
    """Idempotently register all Kimi MoE operators."""
    status: dict[str, str] = {}
    specs = (
        (kimi_moe_route, kimi_moe_route_anno, kimi_moe_route_fake, None),
        (kimi_moe_grouped_gemm, kimi_moe_grouped_gemm_anno, kimi_moe_grouped_gemm_fake, (build_fused_ep_transform_rule(),)),
        (kimi_moe_permute, kimi_moe_permute_anno, kimi_moe_permute_fake, None),
        (kimi_grouped_linear, kimi_grouped_linear_anno, kimi_grouped_linear_fake, (build_grouped_linear_ep_rule(),)),
        (kimi_moe_unpermute, kimi_moe_unpermute_anno, kimi_moe_unpermute_fake, None),
    )
    for fn, anno, fake_fn, rules in specs:
        sig = f"{fn.__module__}.{fn.__qualname__}"
        if sig in CustomizedOps.kOpMap:
            status[sig] = "already_registered"
            continue
        if rules is not None:
            register_op(anno, fake_fn=fake_fn, transform_rules=rules)(fn)
        else:
            register_op(anno, fake_fn=fake_fn)(fn)
        status[sig] = "registered"
    return status
