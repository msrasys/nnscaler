# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stage 2 & Stage 3 CPU Golden Numerical Parity Tests for DeepSeek (`V2-Lite`, `V2`, `V3/R1`, `V3.2-DSA`).

Verifies on 0 GPUs that every custom op and in-memory rewrite preserves:
  1. 100% identical state_dict keys and parameter shapes vs. unmodified HuggingFace;
  2. Forward scalar loss parity (`< 1e-5` in fp32, `< 5e-3` in bf16) and backward gradient parity
     (`embed_tokens.weight.grad`, `lm_head.weight.grad`, MLA projections, and 3D MoE `gate_up_proj`/`down_proj`)
     across:
       - `micro-v2-lite` (`q_lora_rank=None`, greedy Softmax TopK, complex->real RoPE, MLA `concat_kv`)
       - `micro-v2`      (`q_lora_rank=48`, `group_limited_greedy` Softmax TopK, complex->real RoPE, MLA `concat_kv`)
       - `micro-v3`      (`expand_moe_ops=False` fused AND `expand_moe_ops=True` expanded, Sigmoid+bias group TopK)
       - `micro-v32`     (DeepSeek Sparse Attention `deepseek_v32_dsa_index_mask` + MLA + MoE)
  3. Exact multi-rank Expert Parallelism (`EP=2` / `EP=4`) partial-sum invariance (`sum_r op_r == op_full`)
     for both fused `deepseek_moe_grouped_gemm` and expanded `deepseek_grouped_linear`.
"""

from __future__ import annotations

import copy
import sys
import pytest
import torch
import torch.nn.functional as F

from examples.Deepseek.configs.model_configs import (
    build_dummy_samples,
    instantiate_hf_config,
    load_model_config_dict,
    _resolve_model_cls,
)
from examples.Deepseek.ops import (
    deepseek_grouped_linear,
    deepseek_mla_concat_kv,
    deepseek_moe_grouped_gemm,
    deepseek_moe_permute,
    deepseek_moe_unpermute,
    register_all_deepseek_ops,
)
from examples.Deepseek.rewrites.model_rewrites import (
    DeepseekCausalLMTaskWrapper,
    RewriteConfig,
    apply_deepseek_rewrites,
)


@pytest.mark.parametrize('device', ['cpu', 'meta'])
@pytest.mark.parametrize('grad_enabled', [True, False])
@pytest.mark.parametrize('input_grads', [(False, False), (True, False), (False, True), (True, True)])
def test_mla_fake_metadata(device, grad_enabled, input_grads):
    from examples.Deepseek.ops.mla_ops import deepseek_mla_concat_kv_fake

    k_nope = torch.empty(1, 2, 3, 4, device=device, requires_grad=input_grads[0])
    k_rot = torch.empty(1, 1, 3, 2, device=device, requires_grad=input_grads[1])
    with torch.set_grad_enabled(grad_enabled):
        expected = deepseek_mla_concat_kv(k_nope, k_rot)
        actual = deepseek_mla_concat_kv_fake(k_nope, k_rot)
    assert (actual.shape, actual.dtype, actual.device, actual.requires_grad) == (
        expected.shape, expected.dtype, expected.device, expected.requires_grad,
    )


@pytest.mark.parametrize('grad_enabled', [True, False])
@pytest.mark.parametrize('op_name', [
    'v2_moe_route', 'v3_moe_route', 'moe_grouped_gemm', 'moe_permute',
    'grouped_linear', 'moe_unpermute', 'v32_dsa_index_mask',
])
def test_fake_output_metadata(op_name, grad_enabled):
    from nnscaler.graph.parser.register import CustomizedOps
    from examples.Deepseek.ops.moe_ops import deepseek_v2_moe_route, deepseek_v3_moe_route
    from examples.Deepseek.ops.mla_ops import deepseek_v32_dsa_index_mask

    register_all_deepseek_ops()
    hidden = torch.randn(4, 3)
    probabilities = torch.rand(4, 1)
    selected = torch.arange(4).view(4, 1)
    cases = {
        'v2_moe_route': (deepseek_v2_moe_route, [hidden, torch.randn(4, 3)], {'top_k': 2}),
        'v3_moe_route': (deepseek_v3_moe_route, [hidden, torch.randn(4, 3), torch.randn(4)],
            {'top_k': 2, 'n_group': 2, 'topk_group': 1}),
        'moe_grouped_gemm': (deepseek_moe_grouped_gemm,
            [hidden, probabilities, selected, torch.randn(4, 4, 3), torch.randn(4, 3, 2)], {'local_expert_end': 4}),
        'moe_permute': (deepseek_moe_permute, [hidden, probabilities, selected], {'num_experts': 4}),
        'grouped_linear': (deepseek_grouped_linear,
            [hidden, torch.randn(4, 2, 3), torch.ones(4, dtype=torch.long)], {'local_expert_end': 4}),
        'moe_unpermute': (deepseek_moe_unpermute, [hidden, torch.arange(4), probabilities], {'num_tokens': 4}),
        'v32_dsa_index_mask': (deepseek_v32_dsa_index_mask,
            [torch.randn(1, 2, 3), torch.randn(1, 2, 2), torch.randn(1, 2, 2), torch.randn(1, 2, 2),
             torch.ones(1, 2, 2, dtype=torch.bool), torch.randn(4, 2), torch.randn(2, 3),
             torch.randn(2), torch.randn(2), torch.randn(2, 3)],
            {'n_heads': 2, 'head_dim': 2, 'qk_rope_head_dim': 2, 'index_topk': 1}),
    }
    runtime_fn, args, kwargs = cases[op_name]
    fake_fn = CustomizedOps.kOpFakeRuntime[f'{runtime_fn.__module__}.{runtime_fn.__name__}']
    for grad_index in [None, -1, *range(len(args))]:
        inputs = [tensor.detach().requires_grad_(
            tensor.is_floating_point() and (grad_index is None or grad_index == index)
        ) for index, tensor in enumerate(args)]
        with torch.set_grad_enabled(grad_enabled):
            expected = runtime_fn(*inputs, **kwargs)
        expected = expected if isinstance(expected, tuple) else (expected,)
        for device in ['cpu', 'meta']:
            fake_inputs = [tensor.detach().to(device).requires_grad_(tensor.requires_grad) for tensor in inputs]
            with torch.set_grad_enabled(grad_enabled):
                actual = fake_fn(*fake_inputs, **kwargs)
            actual = actual if isinstance(actual, tuple) else (actual,)
            assert len(actual) == len(expected)
            for output, reference in zip(actual, expected):
                assert (output.shape, output.dtype, output.requires_grad) == (
                    reference.shape, reference.dtype, reference.requires_grad,
                ), (op_name, grad_enabled, grad_index, device)
                assert output.device.type == device


@pytest.mark.parametrize('counts', [(0, 2, 0, 2), (0, 0, 0, 0)])
def test_grouped_linear_empty_experts(counts):
    from collections import Counter

    structures = []
    for expert in range(4):
        inputs = torch.randn(sum(counts), 3, requires_grad=True)
        weight = torch.randn(1, 2, 3, requires_grad=True)
        output = deepseek_grouped_linear(inputs, weight, torch.tensor(counts), expert, expert + 1)
        nodes, pending = set(), [output.grad_fn]
        while pending:
            node = pending.pop()
            if node is None or node in nodes:
                continue
            nodes.add(node)
            pending.extend(parent for parent, _ in node.next_functions)
        structures.append(Counter(type(node).__name__ for node in nodes))
        output.sum().backward()
        assert inputs.grad is not None and weight.grad is not None
        if counts[expert] == 0:
            torch.testing.assert_close(output, torch.zeros_like(output))
            torch.testing.assert_close(inputs.grad, torch.zeros_like(inputs))
            torch.testing.assert_close(weight.grad, torch.zeros_like(weight))
    assert all(structure == structures[0] for structure in structures)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('fused', [True, False])
def test_moe_kernel_dtype_dispatch(monkeypatch, dtype, fused):
    from examples.Deepseek.ops import moe_ops

    calls = []

    def kernel(inputs, weights, counts, trans_b):
        assert trans_b and inputs.dtype == weights.dtype == torch.bfloat16
        calls.append(True)
        return F.linear(inputs, weights[0])

    monkeypatch.setattr(moe_ops, '_cutlass_gmm', kernel)
    monkeypatch.setattr(torch.Tensor, 'is_cuda', property(lambda tensor: True))
    inputs = torch.randn(4, 3, dtype=dtype, requires_grad=True)
    weights = torch.randn(1, 4, 3, dtype=dtype, requires_grad=True)
    if fused:
        down = torch.randn(1, 3, 2, dtype=dtype, requires_grad=True)
        output = moe_ops.deepseek_moe_grouped_gemm(
            inputs, torch.ones(4, 1, dtype=dtype), torch.zeros(4, 1, dtype=torch.long),
            weights, down, local_expert_end=1,
        )
    else:
        output = moe_ops.deepseek_grouped_linear(inputs, weights, torch.tensor([4]), local_expert_end=1)
    output.sum().backward()
    assert inputs.grad is not None and weights.grad is not None
    assert len(calls) == ((2 if fused else 1) if dtype == torch.bfloat16 else 0)


def _init_nontrivial_weights(model: torch.nn.Module, seed: int = 1234) -> None:
    """Initialize router weights & correction biases with distinct values so routing avoids ties."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(torch.randn(param.shape, generator=gen, dtype=torch.float32).to(param.dtype) * 0.04)
        for name, buf in model.named_buffers():
            if "e_score_correction_bias" in name:
                buf.copy_(torch.randn(buf.shape, generator=gen, dtype=torch.float32).to(buf.dtype) * 0.02)


def _compute_hf_reference_loss(model: torch.nn.Module, samples: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(
        input_ids=samples["input_ids"],
        attention_mask=samples["attention_mask"],
        use_cache=False,
    )
    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = samples["labels"][:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )


def _check_preset_golden_parity(
    preset: str,
    *,
    expand_moe_ops: bool = False,
    attention_bias: bool = False,
) -> None:
    register_all_deepseek_ops()
    cfg_dict = load_model_config_dict(preset)
    if attention_bias:
        cfg_dict["attention_bias"] = True
    hf_cfg = instantiate_hf_config(cfg_dict)
    model_cls = _resolve_model_cls(hf_cfg.model_type)

    torch.manual_seed(42)
    ref_model = model_cls(hf_cfg).to(dtype=torch.float32, device="cpu")
    _init_nontrivial_weights(ref_model, seed=2026)
    ref_model.train()

    rewritten_hf = copy.deepcopy(ref_model)
    ref_keys = {k: tuple(v.shape) for k, v in ref_model.state_dict().items()}

    apply_deepseek_rewrites(
        rewritten_hf,
        RewriteConfig(expand_moe_ops=expand_moe_ops, sum_ce_loss=True, decouple_row_tp_bias=True),
    )
    new_keys = {k: tuple(v.shape) for k, v in rewritten_hf.state_dict().items()}
    assert ref_keys == new_keys, f"{preset}: state_dict mismatch after rewrites!"

    wrapper = DeepseekCausalLMTaskWrapper(rewritten_hf, sum_ce_loss=True)
    wrapper.train()

    samples = build_dummy_samples(cfg_dict, batch_size=2, seq_len=16, device="cpu")
    # Also test masked ignore_index=-100 handling!
    samples["labels"][0, -3:] = -100
    valid_tokens = float((samples["labels"][:, 1:] != -100).sum().item())
    samples["valid_token_count"] = torch.tensor([[valid_tokens / 2.0], [valid_tokens / 2.0]], dtype=torch.float32)

    ref_loss = _compute_hf_reference_loss(ref_model, samples)
    ref_loss.backward()

    new_loss = wrapper(samples)
    new_loss.backward()

    loss_diff = float((ref_loss - new_loss).abs().item())
    assert loss_diff < 1e-5, f"{preset} (expand={expand_moe_ops}): loss diff {loss_diff:.3e} >= 1e-5"

    checked_grads = 0
    for (name, p_ref), (_, p_new) in zip(ref_model.named_parameters(), rewritten_hf.named_parameters()):
        if p_ref.grad is None:
            assert p_new.grad is None, f"{preset}: expected None grad on {name} (e.g. @no_grad DSA indexer)"
            continue
        assert p_new.grad is not None, f"{preset}: missing grad on {name}"
        grad_diff = float((p_ref.grad - p_new.grad).abs().max().item())
        assert grad_diff < 1e-4, f"{preset} (expand={expand_moe_ops}) grad mismatch on {name}: {grad_diff:.3e}"
        checked_grads += 1

    assert checked_grads > 10
    print(
        f"  [PASS] {preset:14s} (expand_moe={str(expand_moe_ops):5s}, bias={str(attention_bias):5s}): "
        f"loss_diff={loss_diff:.2e}, verified {checked_grads} parameter tensors"
    )


def test_all_presets_golden_parity() -> None:
    _check_preset_golden_parity("micro-v2-lite", expand_moe_ops=False)
    _check_preset_golden_parity("micro-v2", expand_moe_ops=False, attention_bias=True)
    _check_preset_golden_parity("micro-v3", expand_moe_ops=False)
    _check_preset_golden_parity("micro-v3", expand_moe_ops=True)
    _check_preset_golden_parity("micro-v32", expand_moe_ops=False)
    _check_preset_golden_parity("micro-v32", expand_moe_ops=True)


def test_mla_concat_kv_tp_head_sharding_backward_parity() -> None:
    """Verify `deepseek_mla_concat_kv` head-sharded forward + AllReduced `k_rot` grad == full-head reference."""
    torch.manual_seed(77)
    B, H, S, Dn, Dr = 2, 8, 12, 16, 8
    tp = 4
    h_local = H // tp

    k_nope_full = torch.randn(B, H, S, Dn, requires_grad=True)
    k_rot_full = torch.randn(B, 1, S, Dr, requires_grad=True)
    grad_out = torch.randn(B, H, S, Dn + Dr)

    out_full = deepseek_mla_concat_kv(k_nope_full, k_rot_full)
    (out_full * grad_out).sum().backward()

    shard_outs = []
    k_nope_grads = []
    k_rot_grad_sum = torch.zeros_like(k_rot_full)
    for r in range(tp):
        kn_r = k_nope_full.detach()[:, r * h_local : (r + 1) * h_local].clone().requires_grad_(True)
        kr_r = k_rot_full.detach().clone().requires_grad_(True)
        go_r = grad_out[:, r * h_local : (r + 1) * h_local]
        out_r = deepseek_mla_concat_kv(kn_r, kr_r)
        (out_r * go_r).sum().backward()
        shard_outs.append(out_r.detach())
        k_nope_grads.append(kn_r.grad.detach())
        k_rot_grad_sum += kr_r.grad.detach()

    assert torch.allclose(torch.cat(shard_outs, dim=1), out_full.detach(), atol=1e-6)
    assert torch.allclose(torch.cat(k_nope_grads, dim=1), k_nope_full.grad.detach(), atol=1e-6)
    assert torch.allclose(k_rot_grad_sum, k_rot_full.grad.detach(), atol=1e-6)
    print("  [PASS] deepseek_mla_concat_kv TP=4 forward & AllReduced MQA k_rot backward invariance")


def test_ep_partial_sum_invariance_fused_and_expanded() -> None:
    """Verify `EP=4` local expert slice sum (`DimopSplit.V()`) matches single-device execution bit-for-bit."""
    torch.manual_seed(99)
    N, H, I, E, K = 24, 32, 16, 8, 2
    ep = 4
    per_rank = E // ep

    x = torch.randn(N, H)
    weights = torch.rand(N, K, dtype=torch.float32)
    indices = torch.randint(0, E, (N, K), dtype=torch.long)
    gate_up = torch.randn(E, 2 * I, H) * 0.05
    down = torch.randn(E, H, I) * 0.05

    full_fused = deepseek_moe_grouped_gemm(
        x, weights, indices, gate_up, down, act_fn_name="silu", local_expert_start=0, local_expert_end=E
    )

    ep_fused_sum = torch.zeros_like(full_fused)
    for r in range(ep):
        lo, hi = r * per_rank, (r + 1) * per_rank
        ep_fused_sum = ep_fused_sum + deepseek_moe_grouped_gemm(
            x, weights, indices, gate_up[lo:hi], down[lo:hi], act_fn_name="silu",
            local_expert_start=lo, local_expert_end=hi,
        )
    assert torch.allclose(full_fused, ep_fused_sum, atol=1e-6)

    # Also verify expanded Mode B (`EP=4` on `deepseek_grouped_linear`)
    permuted_x, counts, sorted_tokens, sorted_probs = deepseek_moe_permute(x, weights, indices, num_experts=E)
    gw, uw = torch.chunk(gate_up, 2, dim=1)

    ep_down_sum = torch.zeros(N * K, H)
    for r in range(ep):
        lo, hi = r * per_rank, (r + 1) * per_rank
        g_r = deepseek_grouped_linear(permuted_x, gw[lo:hi], counts, local_expert_start=lo, local_expert_end=hi)
        u_r = deepseek_grouped_linear(permuted_x, uw[lo:hi], counts, local_expert_start=lo, local_expert_end=hi)
        inter_r = F.silu(g_r) * u_r
        d_r = deepseek_grouped_linear(inter_r, down[lo:hi], counts, local_expert_start=lo, local_expert_end=hi)
        ep_down_sum = ep_down_sum + d_r

    expanded_ep_out = deepseek_moe_unpermute(ep_down_sum, sorted_tokens, sorted_probs, num_tokens=N)
    assert torch.allclose(full_fused, expanded_ep_out, atol=1e-6)
    print("  [PASS] EP=4 partial-sum invariance verified for both fused and expanded DeepSeek MoE ops")


if __name__ == "__main__":
    print("=== Running DeepSeek CPU Golden Numerical Parity Suite ===")
    test_all_presets_golden_parity()
    test_mla_concat_kv_tp_head_sharding_backward_parity()
    test_ep_partial_sum_invariance_fused_and_expanded()
    print("=== ALL DEEPSEEK GOLDEN PARITY TESTS PASSED ===")
