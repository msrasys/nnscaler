# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""DeepSeek Sparse MoE custom operators supporting:
  - Both DeepSeek-V2 (`softmax` greedy / `group_limited_greedy`) and DeepSeek-V3 / R1 / V3.2
    (`sigmoid` + `e_score_correction_bias` auxiliary-loss-free group TopK) routers;
  - Mode A (Default: `expand_moe_ops=False`): Fused `deepseek_moe_grouped_gemm` with `TransformRule`
    for Expert Parallelism (`E+` -> `DimopSplit.R()` inputs + `DimopSplit.V()` partial-sum output);
  - Mode B (Opt-in: `expand_moe_ops=True`): Expanded `deepseek_moe_permute`, `deepseek_grouped_linear`,
    and `deepseek_moe_unpermute` exposing `EP` (`E+`), Column-ETP (`m`), and Row-ETP (`k+`).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from nnscaler.graph.function.dimops import DimopSplit, TransformRule
from nnscaler.graph.parser.register import CustomizedOps, register_op
from nnscaler.ir.cten import IRObject, IRTensor

try:
    from grouped_gemm.ops import gmm as _cutlass_gmm
except Exception:  # pragma: no cover - optional CUTLASS package
    _cutlass_gmm = None


def _unwrap_int(val: Any) -> int:
    while isinstance(val, IRObject):
        val = val.value
    return int(val)


def _apply_act(x: torch.Tensor, act_fn_name: str) -> torch.Tensor:
    if act_fn_name in ("silu", "swish"):
        return F.silu(x)
    if act_fn_name == "gelu":
        return F.gelu(x)
    if act_fn_name == "relu":
        return F.relu(x)
    raise ValueError(f"Unsupported expert activation: {act_fn_name}")


# ============================================================================
# Routers (Shared by both Mode A and Mode B)
# ============================================================================

def deepseek_v2_moe_route(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int = 6,
    n_group: int = 1,
    topk_group: int = 1,
    topk_method: str = "greedy",
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static-shape DeepSeek-V2 / V2-Lite Softmax TopK router (`[N, H] -> ([N, K] fp32, [N, K] int64)`)."""
    router_logits = F.linear(hidden_states.to(torch.float32), router_weight.to(torch.float32))
    scores = router_logits.softmax(dim=-1, dtype=torch.float32)
    num_experts = router_weight.shape[0]

    if topk_method == "greedy":
        topk_weights, topk_indices = torch.topk(scores, k=top_k, dim=-1, sorted=False)
    elif topk_method == "group_limited_greedy":
        group_scores = scores.view(-1, n_group, num_experts // n_group).max(dim=-1).values
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores).scatter(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, n_group, num_experts // n_group)
            .reshape(-1, num_experts)
        )
        scores = scores.masked_fill(~score_mask.bool(), 0.0)
        topk_weights, topk_indices = torch.topk(scores, k=top_k, dim=-1, sorted=False)
    else:
        raise ValueError(f"Unsupported DeepSeek-V2 topk_method: {topk_method!r}")

    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_indices


def deepseek_v2_moe_route_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int = 6,
    n_group: int = 1,
    topk_group: int = 1,
    topk_method: str = "greedy",
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.shape[0]
    weights = hidden_states.new_empty(
        (num_tokens, top_k), dtype=torch.float32,
        requires_grad=torch.is_grad_enabled() and (hidden_states.requires_grad or router_weight.requires_grad),
    )
    indices = hidden_states.new_empty((num_tokens, top_k), dtype=torch.long)
    return weights, indices


def v2_moe_route_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 2:
        raise ValueError(f"deepseek_v2_moe_route expects (hidden_states, router_weight), got {len(args)}")
    hidden_states, router_weight = args[0], args[1]
    top_k = _unwrap_int(args[2] if len(args) >= 3 else kwargs.get("top_k", 6))
    if not isinstance(hidden_states, IRTensor) or len(hidden_states.shape) != 2:
        raise ValueError(f"Expected 2D hidden_states [N, H], got {hidden_states}")
    if not isinstance(router_weight, IRTensor) or len(router_weight.shape) != 2:
        raise ValueError(f"Expected 2D router_weight [E, H], got {router_weight}")
    return f"n h^, e^ h^ -> n {top_k}^, n {top_k}^"


def deepseek_v3_moe_route(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static-shape DeepSeek-V3 / R1 / V3.2 Sigmoid + Bias-Corrected Group TopK router."""
    router_logits = F.linear(hidden_states.to(torch.float32), router_weight.to(torch.float32))
    scores = router_logits.sigmoid()
    num_experts = router_weight.shape[0]

    scores_for_choice = scores + e_score_correction_bias
    group_scores = (
        scores_for_choice.view(-1, n_group, num_experts // n_group)
        .topk(2, dim=-1)[0]
        .sum(dim=-1)
    )
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores).scatter(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, num_experts // n_group)
        .reshape(-1, num_experts)
    )
    scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    topk_indices = torch.topk(scores_for_choice, k=top_k, dim=-1, sorted=False)[1]
    topk_weights = scores.gather(1, topk_indices)
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    topk_weights = topk_weights * routed_scaling_factor
    return topk_weights, topk_indices


def deepseek_v3_moe_route_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    top_k: int = 8,
    n_group: int = 8,
    topk_group: int = 4,
    norm_topk_prob: bool = True,
    routed_scaling_factor: float = 2.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.shape[0]
    weights = hidden_states.new_empty(
        (num_tokens, top_k), dtype=torch.float32,
        requires_grad=torch.is_grad_enabled() and (hidden_states.requires_grad or router_weight.requires_grad),
    )
    indices = hidden_states.new_empty((num_tokens, top_k), dtype=torch.long)
    return weights, indices


def v3_moe_route_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 3:
        raise ValueError(
            f"deepseek_v3_moe_route expects (hidden_states, router_weight, e_score_correction_bias), got {len(args)}"
        )
    hidden_states, router_weight, bias = args[0], args[1], args[2]
    top_k = _unwrap_int(args[3] if len(args) >= 4 else kwargs.get("top_k", 8))
    if not isinstance(hidden_states, IRTensor) or len(hidden_states.shape) != 2:
        raise ValueError(f"Expected 2D hidden_states [N, H], got {hidden_states}")
    if not isinstance(router_weight, IRTensor) or len(router_weight.shape) != 2:
        raise ValueError(f"Expected 2D router_weight [E, H], got {router_weight}")
    if not isinstance(bias, IRTensor) or len(bias.shape) != 1:
        raise ValueError(f"Expected 1D e_score_correction_bias [E], got {bias}")
    return f"n h^, e^ h^, e^ -> n {top_k}^, n {top_k}^"


# ============================================================================
# Mode A (Default: expand_moe_ops = False) — Fused Grouped GEMM
# ============================================================================

def deepseek_moe_grouped_gemm(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn_name: str = "silu",
    local_expert_start: int = 0,
    local_expert_end: int = 256,
) -> torch.Tensor:
    """Fused routed-expert execution over local slice `[local_expert_start, local_expert_end)`.

    Preserves HF DeepSeek 3D parameter tensors (`gate_up_proj [E_local, 2*I, H]`,
    `down_proj [E_local, H, I]`) and exact `float32` `routing_weights` scaling before casting back
    to `hidden_states.dtype`.
    """
    num_local_experts = local_expert_end - local_expert_start
    if gate_up_proj.shape[0] != num_local_experts or down_proj.shape[0] != num_local_experts:
        raise ValueError(
            f"Local expert weight slice mismatch: expected {num_local_experts} experts "
            f"for [{local_expert_start}, {local_expert_end}), got {gate_up_proj.shape[0]} and {down_proj.shape[0]}"
        )

    if (
        hidden_states.is_cuda and _cutlass_gmm is not None
        and hidden_states.dtype == gate_up_proj.dtype == down_proj.dtype == torch.bfloat16
    ):
        local_mask = (selected_experts >= local_expert_start) & (selected_experts < local_expert_end)
        if not local_mask.any():
            return hidden_states.new_zeros(hidden_states.shape)
        token_idx, slot_idx = torch.where(local_mask)
        local_expert_ids = selected_experts[token_idx, slot_idx] - local_expert_start
        order = torch.argsort(local_expert_ids, stable=True)
        sorted_tokens = token_idx[order]
        sorted_probs = routing_weights[token_idx[order], slot_idx[order]].unsqueeze(-1)
        sorted_experts = local_expert_ids[order]
        permuted_x = hidden_states.index_select(0, sorted_tokens)
        counts = torch.bincount(sorted_experts, minlength=num_local_experts).cpu().to(torch.long)

        gate_up = _cutlass_gmm(permuted_x, gate_up_proj, counts, trans_b=True)
        gate, up = gate_up.chunk(2, dim=-1)
        inter = _apply_act(gate, act_fn_name) * up
        expert_out = _cutlass_gmm(inter, down_proj, counts, trans_b=True)
        weighted = (expert_out * sorted_probs).to(hidden_states.dtype)
        out = hidden_states.new_zeros(hidden_states.shape)
        return out.index_add(0, sorted_tokens, weighted)

    accum = hidden_states.new_zeros(hidden_states.shape)
    for local_idx in range(num_local_experts):
        global_expert_id = local_expert_start + local_idx
        mask = selected_experts == global_expert_id
        if not mask.any():
            accum = accum + (gate_up_proj[local_idx].sum() * 0.0 + down_proj[local_idx].sum() * 0.0).to(accum.dtype)
            continue
        # Match HF `torch.where(expert_mask[expert_idx])` where `expert_mask` is `[E, K, N]`
        slot_idx, token_idx = torch.where(mask.transpose(0, 1))
        cur_x = hidden_states.index_select(0, token_idx)
        gate, up = F.linear(cur_x, gate_up_proj[local_idx]).chunk(2, dim=-1)
        inter = _apply_act(gate, act_fn_name) * up
        cur_out = F.linear(inter, down_proj[local_idx])
        weights = routing_weights[token_idx, slot_idx].unsqueeze(-1)
        accum = accum.index_add(0, token_idx, (cur_out * weights).to(accum.dtype))
    return accum


def deepseek_moe_grouped_gemm_fake(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    act_fn_name: str = "silu",
    local_expert_start: int = 0,
    local_expert_end: int = 256,
) -> torch.Tensor:
    return torch.empty_like(
        hidden_states,
        requires_grad=torch.is_grad_enabled() and any(
            tensor.requires_grad for tensor in (hidden_states, routing_weights, gate_up_proj, down_proj)
        ),
    )


def moe_grouped_gemm_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 5:
        raise ValueError(f"deepseek_moe_grouped_gemm expects 5 tensor inputs, got {len(args)}")
    hidden_states, routing_weights, selected_experts, gate_up_proj, down_proj = args[:5]
    if len(hidden_states.shape) != 2 or len(routing_weights.shape) != 2 or len(selected_experts.shape) != 2:
        raise ValueError("Expected 2D tensors for hidden_states, routing_weights, selected_experts")
    if len(gate_up_proj.shape) != 3 or len(down_proj.shape) != 3:
        raise ValueError("Expected 3D tensors for gate_up_proj [E, 2I, H] and down_proj [E, H, I]")
    top_k = routing_weights.shape[1]
    return f"n h^, n {top_k}^, n {top_k}^, E+ g^ h^, E+ h^ d^ -> n h^"


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
        updated = dict(kwargs)
        start = _unwrap_int(updated["local_expert_start"])
        end = _unwrap_int(updated["local_expert_end"])
        total = end - start
        if total % num != 0:
            raise ValueError(f"Cannot partition {total} experts evenly across {num} EP ranks")
        per_rank = total // num
        updated["local_expert_start"] = start + per_rank * pos
        updated["local_expert_end"] = start + per_rank * (pos + 1)
        return updated

    return TransformRule(itransform, otransform, modifier)


# ============================================================================
# Mode B (Opt-in: expand_moe_ops = True) — Static-Capacity Atomic Primitives
# ============================================================================

def deepseek_moe_permute(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Static-capacity `[N * K, H]` permutation sorted by global expert ID `[0 .. E-1]`."""
    num_tokens, _ = hidden_states.shape
    top_k = selected_experts.shape[1]

    flat_experts = selected_experts.reshape(-1)
    flat_probs = routing_weights.reshape(-1)
    flat_tokens = torch.arange(num_tokens, device=hidden_states.device).unsqueeze(1).expand(-1, top_k).reshape(-1)

    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    sorted_tokens = flat_tokens[order]
    sorted_probs = flat_probs[order].unsqueeze(-1)
    permuted_x = hidden_states.index_select(0, sorted_tokens)
    counts = torch.bincount(sorted_experts, minlength=num_experts).to(dtype=torch.long, device=hidden_states.device)
    return permuted_x, counts, sorted_tokens, sorted_probs


def deepseek_moe_permute_fake(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    capacity = hidden_states.shape[0] * selected_experts.shape[1]
    return (
        hidden_states.new_empty(
            (capacity, hidden_states.shape[1]),
            requires_grad=torch.is_grad_enabled() and hidden_states.requires_grad,
        ),
        hidden_states.new_empty((num_experts,), dtype=torch.long),
        hidden_states.new_empty((capacity,), dtype=torch.long),
        routing_weights.new_empty(
            (capacity, 1), requires_grad=torch.is_grad_enabled() and routing_weights.requires_grad,
        ),
    )


def moe_permute_anno(*args: Any, **kwargs: Any) -> str:
    _, routing_weights, _ = args[:3]
    top_k = routing_weights.shape[1]
    num_experts = _unwrap_int(args[3] if len(args) >= 4 else kwargs.get("num_experts", 256))
    return f"n^ h, n^ {top_k}^, n^ {top_k}^ -> (n^ {top_k}^) h, {num_experts}^, (n^ {top_k}^), (n^ {top_k}^) 1"


def deepseek_grouped_linear(
    permuted_x: torch.Tensor,
    expert_weight: torch.Tensor,
    expert_counts: torch.Tensor,
    local_expert_start: int = 0,
    local_expert_end: int = 256,
) -> torch.Tensor:
    """Atomic 3D grouped linear exposing `E+` (EP), `m` (Column-ETP), and `k+` (Row-ETP)."""
    capacity = permuted_x.shape[0]
    out_features = expert_weight.shape[1]
    num_local = local_expert_end - local_expert_start
    if expert_weight.shape[0] != num_local:
        raise ValueError(
            f"expert_weight shape[0]={expert_weight.shape[0]} != expected {num_local} "
            f"for [{local_expert_start}, {local_expert_end})"
        )

    counts_list = [int(c) for c in expert_counts.detach().cpu().tolist()]
    if (
        permuted_x.is_cuda
        and _cutlass_gmm is not None
        and permuted_x.dtype == expert_weight.dtype == torch.bfloat16
        and local_expert_start == 0
        and local_expert_end == len(counts_list)
    ):
        return _cutlass_gmm(permuted_x, expert_weight, expert_counts.cpu().to(torch.long), trans_b=True)

    zero_anchor = (expert_weight.sum() * 0.0).to(permuted_x.dtype)
    prefix_before = sum(counts_list[:local_expert_start])
    chunks = [permuted_x.new_zeros((prefix_before, out_features)) + zero_anchor]

    offset = prefix_before
    for local_idx in range(num_local):
        cnt = counts_list[local_expert_start + local_idx]
        seg = permuted_x[offset : offset + cnt]
        chunks.append(F.linear(seg, expert_weight[local_idx]))
        offset += cnt

    suffix_after = capacity - offset
    chunks.append(permuted_x.new_zeros((suffix_after, out_features)) + zero_anchor)
    return torch.cat(chunks, dim=0)


def deepseek_grouped_linear_fake(
    permuted_x: torch.Tensor,
    expert_weight: torch.Tensor,
    expert_counts: torch.Tensor,
    local_expert_start: int = 0,
    local_expert_end: int = 256,
) -> torch.Tensor:
    return permuted_x.new_empty(
        (permuted_x.shape[0], expert_weight.shape[1]),
        requires_grad=torch.is_grad_enabled() and (permuted_x.requires_grad or expert_weight.requires_grad),
    )


def grouped_linear_anno(*args: Any, **kwargs: Any) -> str:
    return "c^ k+, E+ m k+, g^ -> c^ m"


def build_grouped_linear_ep_rule() -> TransformRule:
    itransform = [DimopSplit.R(), DimopSplit.D(0), DimopSplit.R()]
    otransform = [DimopSplit.V()]

    def modifier(kwargs: dict[str, Any], idx: int, dim: int, num: int, pos: int) -> dict[str, Any]:
        updated = dict(kwargs)
        start = _unwrap_int(updated["local_expert_start"])
        end = _unwrap_int(updated["local_expert_end"])
        total = end - start
        if total % num != 0:
            raise ValueError(f"Cannot partition {total} experts evenly across {num} EP ranks")
        per_rank = total // num
        updated["local_expert_start"] = start + per_rank * pos
        updated["local_expert_end"] = start + per_rank * (pos + 1)
        return updated

    return TransformRule(itransform, otransform, modifier)


def deepseek_moe_unpermute(
    expert_outs: torch.Tensor,
    sorted_tokens: torch.Tensor,
    sorted_probs: torch.Tensor,
    num_tokens: int = 1,
) -> torch.Tensor:
    weighted = (expert_outs * sorted_probs).to(expert_outs.dtype)
    out = expert_outs.new_zeros((num_tokens, expert_outs.shape[1]))
    return out.index_add(0, sorted_tokens, weighted)


def deepseek_moe_unpermute_fake(
    expert_outs: torch.Tensor,
    sorted_tokens: torch.Tensor,
    sorted_probs: torch.Tensor,
    num_tokens: int = 1,
) -> torch.Tensor:
    return expert_outs.new_empty(
        (num_tokens, expert_outs.shape[1]),
        requires_grad=torch.is_grad_enabled() and (expert_outs.requires_grad or sorted_probs.requires_grad),
    )


def moe_unpermute_anno(*args: Any, **kwargs: Any) -> str:
    num_tokens = _unwrap_int(args[3] if len(args) >= 4 else kwargs.get("num_tokens", 1))
    return f"c^ h, c^, c^ 1 -> {num_tokens}^ h"


def register_moe_ops() -> dict[str, str]:
    """Idempotently register both fused and expanded DeepSeek MoE operators."""
    status: dict[str, str] = {}
    specs = (
        (deepseek_v2_moe_route, v2_moe_route_anno, deepseek_v2_moe_route_fake, None),
        (deepseek_v3_moe_route, v3_moe_route_anno, deepseek_v3_moe_route_fake, None),
        (
            deepseek_moe_grouped_gemm,
            moe_grouped_gemm_anno,
            deepseek_moe_grouped_gemm_fake,
            (build_fused_ep_transform_rule(),),
        ),
        (deepseek_moe_permute, moe_permute_anno, deepseek_moe_permute_fake, None),
        (
            deepseek_grouped_linear,
            grouped_linear_anno,
            deepseek_grouped_linear_fake,
            (build_grouped_linear_ep_rule(),),
        ),
        (deepseek_moe_unpermute, moe_unpermute_anno, deepseek_moe_unpermute_fake, None),
    )
    for fn, anno, fake_fn, rules in specs:
        sig = f"{fn.__module__}.{fn.__qualname__}"
        if sig in CustomizedOps.kOpMap:
            status[sig] = "already_registered"
            continue
        register_op(anno, transform_rules=rules, fake_fn=fake_fn)(fn)
        status[sig] = "registered"
    return status
