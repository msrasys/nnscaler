# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from examples.moe_utils import tensorized_moe, tensorized_situ_moe


def test_tensorized_moe_matches_route_reference():
    torch.manual_seed(0)
    hidden_states = torch.randn(5, 8, requires_grad=True)
    topk_indices = torch.tensor(
        [
            [0, 2],
            [1, 3],
            [0, 1],
            [2, 3],
            [3, 0],
        ]
    )
    topk_weights = torch.rand(5, 2, requires_grad=True)
    gate_up_proj = torch.randn(4, 12, 8, requires_grad=True)
    down_proj = torch.randn(4, 8, 6, requires_grad=True)

    actual = tensorized_moe(
        hidden_states,
        topk_indices,
        topk_weights,
        gate_up_proj,
        down_proj,
        num_experts=4,
        local_expert_start=0,
        local_expert_end=4,
    )

    expected = torch.zeros_like(actual)
    for token in range(hidden_states.shape[0]):
        for route in range(topk_indices.shape[1]):
            expert = topk_indices[token, route]
            gate, up = torch.nn.functional.linear(
                hidden_states[token],
                gate_up_proj[expert],
            ).chunk(2, dim=-1)
            expert_output = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up,
                down_proj[expert],
            )
            expected[token] += expert_output * topk_weights[token, route]

    torch.testing.assert_close(actual, expected)

    first_half = tensorized_moe(
        hidden_states,
        topk_indices,
        topk_weights,
        gate_up_proj[:2],
        down_proj[:2],
        num_experts=4,
        local_expert_start=0,
        local_expert_end=2,
    )
    second_half = tensorized_moe(
        hidden_states,
        topk_indices,
        topk_weights,
        gate_up_proj[2:],
        down_proj[2:],
        num_experts=4,
        local_expert_start=2,
        local_expert_end=4,
    )
    torch.testing.assert_close(first_half + second_half, actual)


def test_tensorized_situ_moe_sums_expert_shards():
    torch.manual_seed(1)
    hidden_states = torch.randn(4, 8)
    topk_indices = torch.tensor([[0, 2], [1, 3], [3, 0], [2, 1]])
    topk_weights = torch.rand(4, 2)
    gate_proj = torch.randn(4, 6, 8)
    down_proj = torch.randn(4, 8, 6)
    up_proj = torch.randn(4, 6, 8)
    kwargs = {
        "num_experts": 4,
        "activation_beta": 4.0,
        "activation_linear_beta": 25.0,
    }

    full = tensorized_situ_moe(
        hidden_states,
        topk_indices,
        topk_weights,
        gate_proj,
        down_proj,
        up_proj,
        local_expert_start=0,
        local_expert_end=4,
        **kwargs,
    )
    shards = []
    for start in (0, 2):
        shards.append(
            tensorized_situ_moe(
                hidden_states,
                topk_indices,
                topk_weights,
                gate_proj[start : start + 2],
                down_proj[start : start + 2],
                up_proj[start : start + 2],
                local_expert_start=start,
                local_expert_end=start + 2,
                **kwargs,
            )
        )
    torch.testing.assert_close(sum(shards), full)
