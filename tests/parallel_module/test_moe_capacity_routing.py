#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CPU-only, pure-tensor unit tests for
``tests.parallel_module.phase_moe_common``'s capacity-based MoE routing
helpers (``_capacity_scatter``/``_capacity_gather``, ``MoEFFN._capacity``):
real, non-uniform, per-token-varying expert assignment, overflow-drop,
underflow-zero-pad, and gradient flow -- decoupled from any nnScaler
compilation or distributed context (no GPUs, no ``torch.distributed``
needed), complementing ``tests/parallel_module/test_phase_moe_e2e.py``'s
real-communication/real-schedule proof (whose specific data happens to be
identical across ranks -- see ``phase_moe_common``'s "Honest scoping note")
with a direct check that the routing/capacity *logic itself* is correct for
genuinely varied per-token expert assignment, including the communication
-volume/shape/split edge cases (Task category 6: illegal/edge
volume-shape-split inputs)."""
import pytest
import torch

from tests.parallel_module.phase_moe_common import _capacity_scatter, _capacity_gather, MoEFFN


def test_capacity_scatter_gather_roundtrip_no_overflow():
    """With generous capacity (no drops), scattering then gathering must
    recover exactly the original per-token values (weighted by 1.0)."""
    torch.manual_seed(0)
    T, H, E = 6, 4, 3
    x = torch.randn(T, H, requires_grad=True)
    expert_idx = torch.tensor([0, 1, 2, 0, 1, 2])  # perfectly balanced, 2 tokens/expert
    capacity = 2  # exactly enough, no drops
    buffer, dest = _capacity_scatter(x, expert_idx, E, capacity)
    assert buffer.shape == (E, capacity, H)
    # "combine" with identity pass-through (skip expert compute) and uniform
    # gate_weight=1 must reconstruct x exactly.
    gathered = _capacity_gather(buffer, dest, torch.ones(T))
    assert torch.allclose(gathered, x)


def test_capacity_scatter_respects_non_uniform_routing():
    """Genuinely varied (not round-robin/uniform) per-token expert
    assignment must still route each token to its OWN assigned expert's
    slot, distinctly from other tokens -- i.e. this is real routing, not a
    fixed permutation."""
    torch.manual_seed(1)
    T, H, E = 8, 4, 2
    x = torch.randn(T, H)
    # skewed: 6 tokens to expert 0, 2 to expert 1 (not balanced/round-robin)
    expert_idx = torch.tensor([0, 0, 0, 1, 0, 0, 1, 0])
    capacity = 6  # exactly enough for expert 0's 6 tokens
    buffer, dest = _capacity_scatter(x, expert_idx, E, capacity)
    # expert 0's slots (buffer[0]) must contain exactly the 6 tokens routed
    # to it, in original relative order; expert 1's slots (buffer[1]) must
    # contain exactly the 2 tokens routed to it.
    expert0_tokens = x[expert_idx == 0]
    expert1_tokens = x[expert_idx == 1]
    assert torch.allclose(buffer[0, :6], expert0_tokens)
    assert torch.allclose(buffer[1, :2], expert1_tokens)


def test_capacity_scatter_drops_overflow_tokens():
    """When more tokens are routed to one expert than its capacity, the
    overflowing tokens must be dropped (contribute zero after combine), not
    corrupt other tokens' slots or crash."""
    torch.manual_seed(2)
    T, H, E = 5, 4, 1  # single expert, all tokens compete for its capacity
    x = torch.randn(T, H, requires_grad=True)
    expert_idx = torch.zeros(T, dtype=torch.long)
    capacity = 3  # only 3 of 5 tokens fit
    buffer, dest = _capacity_scatter(x, expert_idx, E, capacity)
    assert buffer.shape == (E, capacity, H)
    # exactly the first `capacity` tokens (stable "first come, first served"
    # order) are kept; the rest are dropped (dest points at the trash row).
    assert torch.allclose(buffer[0], x[:capacity])
    dropped = dest[capacity:]
    assert (dropped == E * capacity).all(), dropped

    gathered = _capacity_gather(buffer, dest, torch.ones(T))
    assert torch.allclose(gathered[:capacity], x[:capacity])
    assert torch.allclose(gathered[capacity:], torch.zeros(T - capacity, H))


def test_capacity_scatter_gather_gradient_flows_only_to_kept_tokens():
    """Real differentiability: gradient must flow back to exactly the
    kept (not dropped) input tokens, and be exactly zero for dropped ones."""
    torch.manual_seed(3)
    T, H, E = 5, 3, 1
    x = torch.randn(T, H, requires_grad=True)
    expert_idx = torch.zeros(T, dtype=torch.long)
    capacity = 2
    buffer, dest = _capacity_scatter(x, expert_idx, E, capacity)
    gathered = _capacity_gather(buffer, dest, torch.ones(T))
    gathered.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad[:capacity], torch.ones(capacity, H))
    assert torch.allclose(x.grad[capacity:], torch.zeros(T - capacity, H))


def test_capacity_gather_applies_gate_weight():
    torch.manual_seed(4)
    T, H, E = 3, 2, 1
    x = torch.randn(T, H)
    expert_idx = torch.zeros(T, dtype=torch.long)
    capacity = 3
    buffer, dest = _capacity_scatter(x, expert_idx, E, capacity)
    gate_weight = torch.tensor([0.1, 0.5, 1.0])
    gathered = _capacity_gather(buffer, dest, gate_weight)
    assert torch.allclose(gathered, x * gate_weight.unsqueeze(-1))


@pytest.mark.parametrize('num_local_tokens,num_experts,capacity_factor,expected', [
    (8, 2, 1.0, 4),
    (8, 3, 1.0, 3),   # ceil(8/3) = 3
    (7, 2, 1.0, 4),   # ceil(7/2) = 4
    (8, 2, 0.5, 2),   # ceil(4/2) = 2
    (1, 4, 1.0, 1),   # max(1, ceil(1/4)) = 1, never zero
    (0, 4, 1.0, 1),   # degenerate zero-token batch still >= 1
])
def test_moe_ffn_capacity_formula(num_local_tokens, num_experts, capacity_factor, expected):
    """`MoEFFN._capacity`'s pure-integer ceiling-division formula (chosen to
    avoid `math.ceil`/`round` on a *traced* token-count value -- see its own
    docstring) must match the mathematical ceil(num_local_tokens *
    capacity_factor / num_experts), clamped to >= 1."""
    ffn = MoEFFN(dim=8, ffn_hidden=16, ep_ranks=list(range(num_experts)),
                 layer_id=0, capacity_factor=capacity_factor)
    assert ffn._capacity(num_local_tokens) == expected


def test_capacity_scatter_rejects_mismatched_expert_idx_length():
    x = torch.randn(4, 3)
    expert_idx = torch.zeros(3, dtype=torch.long)  # wrong length (3 != 4)
    with pytest.raises((RuntimeError, IndexError, ValueError)):
        _capacity_scatter(x, expert_idx, num_experts=2, capacity=4)
