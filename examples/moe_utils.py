# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from nnscaler.graph.function.dimops import DimopSplit, TransformRule
from nnscaler.graph.parser.register import register_op
from nnscaler.ir.operator import IRFwOperation


def _expert_parallel_modifier(kwargs, idx, dim, num, pos):
    updated_kwargs = dict(kwargs)
    num_experts = kwargs["num_experts"]
    if num_experts % num != 0:
        raise ValueError("num_experts must be divisible by the expert parallel degree")
    experts_per_rank = num_experts // num
    updated_kwargs["local_expert_start"] = experts_per_rank * pos
    updated_kwargs["local_expert_end"] = experts_per_rank * (pos + 1)
    return updated_kwargs


def _build_expert_parallel_rule(num_weight_inputs: int) -> TransformRule:
    input_transforms = [
        DimopSplit.R(),
        DimopSplit.R(),
        DimopSplit.R(),
    ] + [DimopSplit.D(0)] * num_weight_inputs
    output_transforms = [DimopSplit.V()]
    return TransformRule(input_transforms, output_transforms, _expert_parallel_modifier)


def _expert_input_gen(node: IRFwOperation):
    inputs = []
    device = torch.cuda.current_device()
    for index, tensor in enumerate(node.inputs()):
        if index == 1:
            inputs.append(
                torch.randint(
                    0,
                    node.kwargs["num_experts"],
                    tensor.shape,
                    dtype=torch.int64,
                    device=device,
                )
            )
        else:
            inputs.append(
                torch.rand(
                    tensor.shape,
                    dtype=tensor.dtype,
                    device=device,
                    requires_grad=tensor.requires_grad,
                )
            )
    return tuple(inputs)


@register_op(
    "n^ h^, n^ k, n^ k, E+ (2 d+) h^, E+ h^ d+ -> n^ h^",
    transform_rules=(_build_expert_parallel_rule(2),),
    input_gen_fn=_expert_input_gen,
)
def tensorized_moe(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
) -> torch.Tensor:
    local_indices = topk_indices.long() - local_expert_start
    valid_routes = (local_indices >= 0) & (local_indices < (local_expert_end - local_expert_start))
    safe_indices = local_indices.clamp(0, gate_up_proj.shape[0] - 1).reshape(-1)
    top_k = topk_indices.shape[-1]
    routed_states = (
        hidden_states.unsqueeze(1)
        .expand(-1, top_k, -1)
        .reshape(-1, hidden_states.shape[-1])
    )

    gate_up = torch.bmm(
        gate_up_proj[safe_indices],
        routed_states.unsqueeze(-1),
    ).squeeze(-1)
    gate, up = gate_up.chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    routed_output = torch.bmm(
        down_proj[safe_indices],
        activated.unsqueeze(-1),
    ).squeeze(-1)
    route_weights = topk_weights * valid_routes.to(topk_weights.dtype)
    routed_output = routed_output * route_weights.reshape(-1, 1)
    return routed_output.view(hidden_states.shape[0], top_k, -1).sum(dim=1).to(hidden_states.dtype)


@register_op(
    "n^ h^, n^ k, n^ k, E+ d+ h^, E+ h^ d+, E+ d+ h^ -> n^ h^",
    transform_rules=(_build_expert_parallel_rule(3),),
    input_gen_fn=_expert_input_gen,
)
def tensorized_situ_moe(
    hidden_states: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    gate_proj: torch.Tensor,
    down_proj: torch.Tensor,
    up_proj: torch.Tensor,
    *,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
    activation_beta: float,
    activation_linear_beta: float | None,
) -> torch.Tensor:
    local_indices = topk_indices.long() - local_expert_start
    valid_routes = (local_indices >= 0) & (local_indices < (local_expert_end - local_expert_start))
    safe_indices = local_indices.clamp(0, gate_proj.shape[0] - 1).reshape(-1)
    top_k = topk_indices.shape[-1]
    routed_states = (
        hidden_states.unsqueeze(1)
        .expand(-1, top_k, -1)
        .reshape(-1, hidden_states.shape[-1])
    )

    gate = torch.bmm(gate_proj[safe_indices], routed_states.unsqueeze(-1)).squeeze(-1)
    up = torch.bmm(up_proj[safe_indices], routed_states.unsqueeze(-1)).squeeze(-1)
    activated_gate = activation_beta * torch.tanh(gate / activation_beta) * torch.sigmoid(gate)
    if activation_linear_beta is not None:
        up = activation_linear_beta * torch.tanh(up / activation_linear_beta)
    activated = activated_gate * up
    routed_output = torch.bmm(
        down_proj[safe_indices],
        activated.unsqueeze(-1),
    ).squeeze(-1)
    route_weights = topk_weights * valid_routes.to(topk_weights.dtype)
    routed_output = routed_output * route_weights.reshape(-1, 1)
    return routed_output.view(hidden_states.shape[0], top_k, -1).sum(dim=1).to(hidden_states.dtype)


def select_experts(
    scores: torch.Tensor,
    correction_bias: torch.Tensor,
    num_groups: int,
    num_experts: int,
    topk_groups: int,
    top_k: int,
) -> torch.Tensor:
    scores_for_choice = scores.detach() + correction_bias.detach()
    grouped_scores = scores_for_choice.view(
        -1,
        num_groups,
        num_experts // num_groups,
    )
    top_two_indices = torch.argsort(grouped_scores, dim=-1)[..., -2:]
    group_scores = grouped_scores.gather(-1, top_two_indices).sum(dim=-1)
    group_indices = torch.argsort(group_scores, dim=-1)[..., -topk_groups:]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_indices, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, num_groups, num_experts // num_groups)
        .reshape(-1, num_experts)
    )
    masked_scores = scores_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
    return torch.argsort(masked_scores, dim=-1)[..., -top_k:]
