# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Qwen3.5-MoE custom operators supporting both fused (`expand_moe_ops=False`, default)
and expanded atomic (`expand_moe_ops=True`) modes, all with CPU/meta `fake_fn`s.
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
# Mode A (Default: expand_moe_ops = False) — Fused Mega-Ops
# ============================================================================

def qwen3_5_moe_route(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Static-shape Qwen3.5 Softmax-TopK router (`[N, H] -> ([N, K], [N, K])`)."""
    router_logits = F.linear(hidden_states, router_weight)
    router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
    top_vals, top_indices = torch.topk(router_probs, k=top_k, dim=-1)
    top_vals = (top_vals / top_vals.sum(dim=-1, keepdim=True)).to(router_logits.dtype)
    return top_vals, top_indices


def qwen3_5_moe_route_fake(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    top_k: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = hidden_states.shape[0]
    scores = hidden_states.new_empty(
        (num_tokens, top_k),
        requires_grad=torch.is_grad_enabled() and (hidden_states.requires_grad or router_weight.requires_grad),
    )
    indices = hidden_states.new_empty((num_tokens, top_k), dtype=torch.long)
    return scores, indices


def moe_route_anno(*args: Any, **kwargs: Any) -> str:
    if len(args) < 2:
        raise ValueError(f"qwen3_5_moe_route expects (hidden_states, router_weight), got {len(args)}")
    hidden_states, router_weight = args[0], args[1]
    top_k = _unwrap_int(args[2] if len(args) >= 3 else kwargs.get("top_k", 8))
    if not isinstance(hidden_states, IRTensor) or len(hidden_states.shape) != 2:
        raise ValueError(f"Expected 2D hidden_states [N, H], got {hidden_states}")
    if not isinstance(router_weight, IRTensor) or len(router_weight.shape) != 2:
        raise ValueError(f"Expected 2D router_weight [E, H], got {router_weight}")
    return f"n h^, e^ h^ -> n {top_k}^, n {top_k}^"


def qwen3_5_moe_grouped_gemm(
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

    `gate_up_proj` has shape `[E_local, 2 * I, H]` and `down_proj` has shape `[E_local, H, I]`,
    matching `Qwen3_5MoeExperts` 3D parameter tensors directly without checkpoint key changes.
    """
    num_local_experts = local_expert_end - local_expert_start
    if gate_up_proj.shape[0] != num_local_experts or down_proj.shape[0] != num_local_experts:
        raise ValueError(
            f"Local expert weight slice mismatch: expected {num_local_experts} experts "
            f"for [{local_expert_start}, {local_expert_end}), got {gate_up_proj.shape[0]} and {down_proj.shape[0]}"
        )

    # Fast CUTLASS path when on CUDA and grouped_gemm is installed
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

    # Differentiable out-of-place PyTorch reference (CPU & fallback CUDA)
    accum = hidden_states.new_zeros(hidden_states.shape)
    for local_idx in range(num_local_experts):
        global_expert_id = local_expert_start + local_idx
        mask = selected_experts == global_expert_id
        if not mask.any():
            # Touch slice with zero contribution so autograd populates zero grad for unhit experts
            accum = accum + (gate_up_proj[local_idx].sum() * 0.0 + down_proj[local_idx].sum() * 0.0).to(accum.dtype)
            continue
        token_idx, slot_idx = torch.where(mask)
        cur_x = hidden_states.index_select(0, token_idx)
        gate_up = F.linear(cur_x, gate_up_proj[local_idx])
        gate, up = gate_up.chunk(2, dim=-1)
        inter = _apply_act(gate, act_fn_name) * up
        cur_out = F.linear(inter, down_proj[local_idx])
        weights = routing_weights[token_idx, slot_idx].unsqueeze(-1).to(cur_out.dtype)
        accum = accum.index_add(0, token_idx, (cur_out * weights).to(accum.dtype))
    return accum


def qwen3_5_moe_grouped_gemm_fake(
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
        raise ValueError(f"qwen3_5_moe_grouped_gemm expects 5 tensor inputs, got {len(args)}")
    hidden_states, routing_weights, selected_experts, gate_up_proj, down_proj = args[:5]
    if len(hidden_states.shape) != 2 or len(routing_weights.shape) != 2 or len(selected_experts.shape) != 2:
        raise ValueError("Expected 2D tensors for hidden_states, routing_weights, selected_experts")
    if len(gate_up_proj.shape) != 3 or len(down_proj.shape) != 3:
        raise ValueError("Expected 3D tensors for gate_up_proj [E, 2I, H] and down_proj [E, H, I]")
    top_k = routing_weights.shape[1]
    # `n` supports Token-DP automatically; `E+` uses `build_fused_ep_transform_rule` for Expert Parallelism.
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

def qwen3_5_moe_permute(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Static-capacity `[N * K, H]` permutation sorted by global expert ID `[0 .. E-1]`.

    Produces deterministic `[N * K, ...]` tensors and global `expert_counts [E]` so downstream
    `qwen3_5_grouped_linear` supports EP (`E+`), Column-ETP (`m`), and Row-ETP (`k+`) cleanly.
    """
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


def qwen3_5_moe_permute_fake(
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


def qwen3_5_grouped_linear(
    permuted_x: torch.Tensor,
    expert_weight: torch.Tensor,
    expert_counts: torch.Tensor,
    local_expert_start: int = 0,
    local_expert_end: int = 256,
) -> torch.Tensor:
    """Atomic 3D grouped linear (`permuted_x [C, K], expert_weight [E_local, Out, K], counts [E_global] -> [C, Out]`).

    Exposes `E+` (Expert Parallelism), `m` (column-wise Expert-TP), and `k+` (row-wise Expert-TP)!
    """
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


def qwen3_5_grouped_linear_fake(
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
    # Exposes E+ (`idx=1, dim=0`), Out `m` (`idx=1, dim=1`), and K+ (`idx=0, dim=1` / `idx=1, dim=2`)!
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


def qwen3_5_moe_unpermute(
    expert_outs: torch.Tensor,
    sorted_tokens: torch.Tensor,
    sorted_probs: torch.Tensor,
    num_tokens: int = 1,
) -> torch.Tensor:
    weighted = (expert_outs * sorted_probs.to(expert_outs.dtype)).to(expert_outs.dtype)
    out = expert_outs.new_zeros((num_tokens, expert_outs.shape[1]))
    return out.index_add(0, sorted_tokens, weighted)


def qwen3_5_moe_unpermute_fake(
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
    """Idempotently register both fused and expanded Qwen3.5 MoE ops."""
    status: dict[str, str] = {}
    specs = (
        (qwen3_5_moe_route, moe_route_anno, qwen3_5_moe_route_fake, None),
        (
            qwen3_5_moe_grouped_gemm,
            moe_grouped_gemm_anno,
            qwen3_5_moe_grouped_gemm_fake,
            (build_fused_ep_transform_rule(),),
        ),
        (qwen3_5_moe_permute, moe_permute_anno, qwen3_5_moe_permute_fake, None),
        (
            qwen3_5_grouped_linear,
            grouped_linear_anno,
            qwen3_5_grouped_linear_fake,
            (build_grouped_linear_ep_rule(),),
        ),
        (qwen3_5_moe_unpermute, moe_unpermute_anno, qwen3_5_moe_unpermute_fake, None),
    )
    for fn, anno, fake_fn, rules in specs:
        sig = f"{fn.__module__}.{fn.__name__}"
        if CustomizedOps.exist(sig):
            status[sig] = "already_registered"
        else:
            register_op(anno, transform_rules=rules, fake_fn=fake_fn)(fn)
            status[sig] = "registered"
    return status
