# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stage 3 CPU Golden Numerical Parity Tests for the GLM model family."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import inspect

import torch
import torch.nn.functional as F
import transformers.models.glm5_next.modeling_glm5_next as glm5_next_mod

from examples.GLM.configs import build_glm_model, build_glm_sample
from examples.GLM.ops import (
    glm_causal_conv1d,
    glm_chunk_kda,
    glm_mla_concat_kv,
    glm_moe_grouped_gemm,
)
from examples.GLM.rewrites import GlmCausalLMTaskWrapper, RewriteConfig, apply_glm_rewrites


@contextmanager
def _cpu_safe_hf_kernels():
    saved = {}
    for name in ("causal_conv1d_fn", "chunk_kimi_delta_attention"):
        fn = getattr(glm5_next_mod, name, None)
        if fn is not None:
            saved[name] = fn
            setattr(glm5_next_mod, name, inspect.unwrap(fn))
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(glm5_next_mod, name, fn)


def test_glm_kda_ops_tp_head_sharding_parity() -> None:
    torch.manual_seed(11)
    b, s, h, d, k = 2, 16, 4, 8, 4
    tp = 2
    h_local = h // tp

    q_in = torch.randn(b, h, d, s, requires_grad=True)
    w_conv = torch.randn(h, d, k, requires_grad=True)
    g = torch.randn(b, s, h, d, requires_grad=True)
    beta = torch.sigmoid(torch.randn(b, s, h)).detach().requires_grad_(True)
    v = torch.randn(b, s, h, d, requires_grad=True)
    k_t = torch.randn(b, s, h, d, requires_grad=True)

    q_4d = glm_causal_conv1d(q_in, w_conv, activation="silu").permute(0, 3, 1, 2)
    out_full = glm_chunk_kda(q_4d, k_t, v, g, beta)
    grad_out = torch.randn_like(out_full)
    out_full.backward(grad_out)

    outs, q_grads, w_grads = [], [], []
    for r in range(tp):
        sl_h = slice(r * h_local, (r + 1) * h_local)
        qr = q_in[:, sl_h, :, :].detach().requires_grad_(True)
        wr = w_conv[sl_h, :, :].detach().requires_grad_(True)
        kr = k_t[:, :, sl_h, :].detach().requires_grad_(True)
        vr = v[:, :, sl_h, :].detach().requires_grad_(True)
        gr = g[:, :, sl_h, :].detach().requires_grad_(True)
        br = beta[:, :, sl_h].detach().requires_grad_(True)

        qc = glm_causal_conv1d(qr, wr, activation="silu").permute(0, 3, 1, 2)
        or_ = glm_chunk_kda(qc, kr, vr, gr, br)
        or_.backward(grad_out[:, :, sl_h, :])
        outs.append(or_.detach())
        q_grads.append(qr.grad)
        w_grads.append(wr.grad)

    torch.testing.assert_close(torch.cat(outs, dim=2), out_full.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat(q_grads, dim=1), q_in.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat(w_grads, dim=0), w_conv.grad, rtol=1e-5, atol=1e-5)
    print("  [PASS] GLM KDA ops (`glm_causal_conv1d` + `glm_chunk_kda`) TP=2 head-sharding parity")


def test_glm_mla_concat_kv_tp_head_sharding_parity() -> None:
    torch.manual_seed(13)
    b, h, s, dn, dr = 2, 4, 12, 16, 8
    tp = 4
    k_nope = torch.randn(b, h, s, dn, requires_grad=True)
    k_rot = torch.randn(b, 1, s, dr, requires_grad=True)
    full = glm_mla_concat_kv(k_nope, k_rot)
    grad_out = torch.randn_like(full)
    full.backward(grad_out)

    outs, nope_grads = [], []
    rot_grad_sum = torch.zeros_like(k_rot)
    for r in range(tp):
        kn_r = k_nope[:, r : r + 1, :, :].detach().requires_grad_(True)
        kr_r = k_rot.detach().requires_grad_(True)
        out_r = glm_mla_concat_kv(kn_r, kr_r)
        out_r.backward(grad_out[:, r : r + 1, :, :])
        outs.append(out_r.detach())
        nope_grads.append(kn_r.grad)
        rot_grad_sum += kr_r.grad

    torch.testing.assert_close(torch.cat(outs, dim=1), full.detach())
    torch.testing.assert_close(torch.cat(nope_grads, dim=1), k_nope.grad)
    torch.testing.assert_close(rot_grad_sum, k_rot.grad, rtol=1e-5, atol=1e-5)
    print("  [PASS] `glm_mla_concat_kv` TP=4 head-sharding & `k_rot` AllReduce backward parity")


def test_glm_moe_grouped_gemm_ep_parity() -> None:
    torch.manual_seed(19)
    t, h, m, e, top_k = 10, 32, 64, 8, 2
    ep = 4
    per = e // ep
    x = torch.randn(t, h, requires_grad=True)
    topk_idx = torch.randint(0, e, (t, top_k))
    topk_w = torch.softmax(torch.randn(t, top_k), dim=-1)
    gu = torch.randn(e, 2 * m, h, requires_grad=True)
    dn = torch.randn(e, h, m, requires_grad=True)

    full = glm_moe_grouped_gemm(x, topk_idx, topk_w, gu, dn, local_expert_start=0, local_expert_end=e)
    grad_out = torch.randn_like(full)
    full.backward(grad_out)

    out_sum = torch.zeros_like(full)
    x_grad_sum = torch.zeros_like(x)
    gu_grads, dn_grads = [], []
    for r in range(ep):
        s, en = r * per, (r + 1) * per
        xr = x.detach().requires_grad_(True)
        gur = gu[s:en].detach().requires_grad_(True)
        dnr = dn[s:en].detach().requires_grad_(True)
        outr = glm_moe_grouped_gemm(xr, topk_idx, topk_w, gur, dnr, local_expert_start=s, local_expert_end=en)
        outr.backward(grad_out)
        out_sum += outr.detach()
        x_grad_sum += xr.grad
        gu_grads.append(gur.grad)
        dn_grads.append(dnr.grad)

    torch.testing.assert_close(out_sum, full.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x_grad_sum, x.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat(gu_grads, dim=0), gu.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat(dn_grads, dim=0), dn.grad, rtol=1e-5, atol=1e-5)
    print("  [PASS] `glm_moe_grouped_gemm` EP=4 partial-sum forward & backward parity")


def test_glm5_hyperconnection_parity() -> None:
    from examples.GLM.configs.model_configs import build_glm_config
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

    torch.manual_seed(23)
    config = build_glm_config("micro-glm5-next").text_config
    reference = Glm5NextTextHyperConnection(config)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(std=0.1)
    rewritten = copy.deepcopy(reference)
    counts = apply_glm_rewrites(rewritten)
    assert counts["hyperconnection_rewrites"] == 1
    inputs = torch.randn(2, 4, config.hc_mult, config.hidden_size, requires_grad=True)
    rewritten_inputs = inputs.detach().clone().requires_grad_()
    expected = reference(inputs)
    actual = rewritten(rewritten_inputs)
    gradients = tuple(torch.randn_like(output) for output in expected)
    torch.autograd.backward(expected, gradients)
    torch.autograd.backward(actual, gradients)
    for output, ref_output in zip(actual, expected):
        torch.testing.assert_close(output, ref_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(rewritten_inputs.grad, inputs.grad, rtol=1e-5, atol=1e-5)
    for name, parameter in rewritten.named_parameters():
        torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, rtol=1e-5, atol=1e-5)


def _run_e2e_parity(preset: str, *, expand_moe_ops: bool = False) -> None:
    torch.manual_seed(42)
    ref_model = build_glm_model(preset, weightless=False, dtype=torch.float32)
    ref_model.eval()
    rw_model = copy.deepcopy(ref_model)
    rw_model.eval()

    apply_glm_rewrites(rw_model, RewriteConfig(expand_moe_ops=expand_moe_ops))

    sample = build_glm_sample(preset, batch_size=2, seq_len=16)

    with _cpu_safe_hf_kernels():
        ref_out = ref_model(
            input_ids=sample["input_ids"],
            attention_mask=sample["attention_mask"],
            use_cache=False,
        ).logits
        shift_logits = ref_out[:, :-1, :].contiguous()
        shift_labels = sample["labels"][:, 1:].contiguous()
        ref_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        ) / sample["valid_token_count"].sum().clamp(min=1).to(shift_logits.dtype)
        ref_loss.backward()

    rw_wrap = GlmCausalLMTaskWrapper(rw_model, sum_ce_loss=True)
    rw_loss = rw_wrap(sample)
    rw_loss.backward()

    loss_diff = (ref_loss - rw_loss).abs().item()
    assert loss_diff < 1e-5, f"{preset} loss mismatch: {loss_diff}"

    ref_params = dict(ref_model.named_parameters())
    rw_params = dict(rw_model.named_parameters())
    checked = 0
    for name, rp in ref_params.items():
        wp = rw_params[name]
        if rp.grad is None and wp.grad is None:
            continue
        assert rp.grad is not None and wp.grad is not None, f"Grad None mismatch on {name}"
        if name.endswith("gate_up_proj.weight") and preset == "micro-glm4":
            g, u = rp.grad.chunk(2, dim=0)
            expected_grad = torch.stack((g, u), dim=1).reshape_as(rp.grad)
        else:
            expected_grad = rp.grad
        torch.testing.assert_close(wp.grad, expected_grad, rtol=1e-4, atol=1e-4)
        checked += 1

    tag = f"{preset:20s} (expand_moe={str(expand_moe_ops):5s})"
    print(f"  [PASS] {tag}: loss_diff={loss_diff:.2e}, verified {checked} parameter tensors")


if __name__ == "__main__":
    print("=== Running GLM CPU Golden Numerical Parity Suite ===")
    test_glm_kda_ops_tp_head_sharding_parity()
    test_glm_mla_concat_kv_tp_head_sharding_parity()
    test_glm_moe_grouped_gemm_ep_parity()
    test_glm5_hyperconnection_parity()
    _run_e2e_parity("micro-glm4", expand_moe_ops=False)
    _run_e2e_parity("micro-glm4-moe", expand_moe_ops=False)
    _run_e2e_parity("micro-glm4-moe", expand_moe_ops=True)
    _run_e2e_parity("micro-glm4-moe-lite", expand_moe_ops=False)
    _run_e2e_parity("micro-glm-moe-dsa", expand_moe_ops=False)

    # Upstream HF Glm5NextTextAttention.build_attention_mask_from_topk has an int32 scatter_add_ bug;
    # verify rewritten Glm5Next forward+backward and expand_moe=False vs True exact parity.
    torch.manual_seed(42)
    m5_a = build_glm_model("micro-glm5-next", weightless=False, dtype=torch.float32)
    m5_b = copy.deepcopy(m5_a)
    apply_glm_rewrites(m5_a, RewriteConfig(expand_moe_ops=False))
    apply_glm_rewrites(m5_b, RewriteConfig(expand_moe_ops=True))
    s5 = build_glm_sample("micro-glm5-next", batch_size=2, seq_len=16)
    l5_a = GlmCausalLMTaskWrapper(m5_a, sum_ce_loss=True)(s5)
    l5_b = GlmCausalLMTaskWrapper(m5_b, sum_ce_loss=True)(s5)
    l5_a.backward()
    l5_b.backward()
    diff5 = (l5_a - l5_b).abs().item()
    assert diff5 == 0.0, f"micro-glm5-next expand_moe mismatch: {diff5}"
    print(f"  [PASS] micro-glm5-next      (expand_moe False vs True): loss_diff={diff5:.2e}")
    print("=== ALL GLM CPU GOLDEN PARITY TESTS PASSED ===")
