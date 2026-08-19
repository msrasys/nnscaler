# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


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
