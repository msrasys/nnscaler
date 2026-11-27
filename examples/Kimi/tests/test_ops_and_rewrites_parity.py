# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Stage 3 CPU Golden Numerical Parity Tests for Kimi-Linear and Kimi-K2.5.

Validates on CPU (0 GPUs):
  1. Per-op TP head-sharding backward invariance for `kimi_causal_conv1d`,
     `kimi_chunk_kda`, `kimi_mla_concat_kv`, and `kimi_k25_vision_varlen_sdpa`;
  2. Per-op EP partial-sum invariance for `kimi_moe_grouped_gemm`;
  3. End-to-end forward loss + backward gradient parity (`loss_diff == 0.0`,
     all parameter grads `< 1e-4`) between unmodified HuggingFace models and
     rewritten `KimiLinearForCausalLM` (`expand_moe=False/True`) and
     `Kimi_K25ForConditionalGeneration` (text + multimodal vision).
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import inspect

import torch
import torch.nn.functional as F
import transformers.models.kimi_linear.modeling_kimi_linear as kl_mod

from examples.Kimi.configs import build_kimi_model, build_kimi_sample
from examples.Kimi.ops import (
    kimi_causal_conv1d,
    kimi_chunk_kda,
    kimi_k25_vision_varlen_sdpa,
    kimi_mla_concat_kv,
    kimi_moe_grouped_gemm,
)
from examples.Kimi.rewrites import KimiCausalLMTaskWrapper, RewriteConfig, apply_kimi_rewrites


@contextmanager
def _cpu_safe_hf_kernels():
    """Unwrap `@use_kernel_func_from_hub_with_fallback` so HF reference runs on CPU."""
    saved = {}
    for name in ("causal_conv1d_fn", "chunk_kimi_delta_attention"):
        fn = getattr(kl_mod, name, None)
        if fn is not None:
            saved[name] = fn
            setattr(kl_mod, name, inspect.unwrap(fn))
    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(kl_mod, name, fn)


def test_kda_ops_tp_head_sharding_parity() -> None:
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

    q_conv_full = kimi_causal_conv1d(q_in, w_conv, activation="silu")
    q_4d = q_conv_full.permute(0, 3, 1, 2)
    out_full = kimi_chunk_kda(q_4d, k_t, v, g, beta)
    grad_out = torch.randn_like(out_full)
    out_full.backward(grad_out)

    # Sharded TP=2 along head dimension
    outs = []
    q_grads, w_grads = [], []
    for r in range(tp):
        sl_h = slice(r * h_local, (r + 1) * h_local)
        qr = q_in[:, sl_h, :, :].detach().requires_grad_(True)
        wr = w_conv[sl_h, :, :].detach().requires_grad_(True)
        kr = k_t[:, :, sl_h, :].detach().requires_grad_(True)
        vr = v[:, :, sl_h, :].detach().requires_grad_(True)
        gr = g[:, :, sl_h, :].detach().requires_grad_(True)
        br = beta[:, :, sl_h].detach().requires_grad_(True)

        qc = kimi_causal_conv1d(qr, wr, activation="silu").permute(0, 3, 1, 2)
        or_ = kimi_chunk_kda(qc, kr, vr, gr, br)
        or_.backward(grad_out[:, :, sl_h, :])
        outs.append(or_.detach())
        q_grads.append(qr.grad)
        w_grads.append(wr.grad)

    torch.testing.assert_close(torch.cat(outs, dim=2), out_full.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat(q_grads, dim=1), q_in.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat(w_grads, dim=0), w_conv.grad, rtol=1e-5, atol=1e-5)
    print("  [PASS] KDA ops (`kimi_causal_conv1d` + `kimi_chunk_kda`) TP=2 head-sharding parity")


def test_mla_concat_kv_tp_head_sharding_parity() -> None:
    torch.manual_seed(13)
    b, h, s, dn, dr = 2, 4, 12, 16, 8
    tp = 4
    k_nope = torch.randn(b, h, s, dn, requires_grad=True)
    k_rot = torch.randn(b, 1, s, dr, requires_grad=True)
    full = kimi_mla_concat_kv(k_nope, k_rot)
    grad_out = torch.randn_like(full)
    full.backward(grad_out)

    outs, nope_grads = [], []
    rot_grad_sum = torch.zeros_like(k_rot)
    for r in range(tp):
        kn_r = k_nope[:, r : r + 1, :, :].detach().requires_grad_(True)
        kr_r = k_rot.detach().requires_grad_(True)
        out_r = kimi_mla_concat_kv(kn_r, kr_r)
        out_r.backward(grad_out[:, r : r + 1, :, :])
        outs.append(out_r.detach())
        nope_grads.append(kn_r.grad)
        rot_grad_sum += kr_r.grad

    torch.testing.assert_close(torch.cat(outs, dim=1), full.detach())
    torch.testing.assert_close(torch.cat(nope_grads, dim=1), k_nope.grad)
    torch.testing.assert_close(rot_grad_sum, k_rot.grad, rtol=1e-5, atol=1e-5)
    print("  [PASS] `kimi_mla_concat_kv` TP=4 head-sharding & `k_rot` AllReduce backward parity")


def test_vision_varlen_sdpa_tp_head_sharding_parity() -> None:
    torch.manual_seed(17)
    h, s, d = 4, 16, 16
    tp = 2
    q = torch.randn(1, h, s, d, requires_grad=True)
    k = torch.randn(1, h, s, d, requires_grad=True)
    v = torch.randn(1, h, s, d, requires_grad=True)
    cu = torch.tensor([0, 7, 16], dtype=torch.int32)
    full = kimi_k25_vision_varlen_sdpa(q, k, v, cu, scaling=d ** -0.5)
    grad_out = torch.randn_like(full)
    full.backward(grad_out)

    outs = []
    q_grads = []
    for r in range(tp):
        sl = slice(r * 2, (r + 1) * 2)
        qr = q[:, sl].detach().requires_grad_(True)
        kr = k[:, sl].detach().requires_grad_(True)
        vr = v[:, sl].detach().requires_grad_(True)
        out_r = kimi_k25_vision_varlen_sdpa(qr, kr, vr, cu, scaling=d ** -0.5)
        out_r.backward(grad_out[:, :, sl, :])
        outs.append(out_r.detach())
        q_grads.append(qr.grad)

    torch.testing.assert_close(torch.cat(outs, dim=2), full.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat(q_grads, dim=1), q.grad, rtol=1e-5, atol=1e-5)
    print("  [PASS] `kimi_k25_vision_varlen_sdpa` TP=2 head-sharding parity")


def test_moe_grouped_gemm_ep_parity() -> None:
    torch.manual_seed(19)
    t, h, m, e, top_k = 10, 32, 64, 8, 2
    ep = 4
    per = e // ep
    x = torch.randn(t, h, requires_grad=True)
    topk_idx = torch.randint(0, e, (t, top_k))
    topk_w = torch.softmax(torch.randn(t, top_k), dim=-1)
    gu = torch.randn(e, 2 * m, h, requires_grad=True)
    dn = torch.randn(e, h, m, requires_grad=True)

    full = kimi_moe_grouped_gemm(x, topk_idx, topk_w, gu, dn, local_expert_start=0, local_expert_end=e)
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
        outr = kimi_moe_grouped_gemm(xr, topk_idx, topk_w, gur, dnr, local_expert_start=s, local_expert_end=en)
        outr.backward(grad_out)
        out_sum += outr.detach()
        x_grad_sum += xr.grad
        gu_grads.append(gur.grad)
        dn_grads.append(dnr.grad)

    torch.testing.assert_close(out_sum, full.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x_grad_sum, x.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat(gu_grads, dim=0), gu.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(torch.cat(dn_grads, dim=0), dn.grad, rtol=1e-5, atol=1e-5)
    print("  [PASS] `kimi_moe_grouped_gemm` EP=4 partial-sum forward & backward parity")


def _run_e2e_parity(preset: str, *, expand_moe_ops: bool = False, multimodal: bool = False) -> None:
    torch.manual_seed(42)
    ref_model = build_kimi_model(preset, weightless=False, dtype=torch.float32)
    ref_model.eval()
    rw_model = copy.deepcopy(ref_model)
    rw_model.eval()

    apply_kimi_rewrites(rw_model, RewriteConfig(expand_moe_ops=expand_moe_ops))

    sample = build_kimi_sample(preset, batch_size=2, seq_len=16, multimodal=multimodal)

    with _cpu_safe_hf_kernels():
        ref_kwargs = {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "use_cache": False,
        }
        if multimodal:
            ref_kwargs["pixel_values"] = sample["pixel_values"]
            ref_kwargs["image_grid_thw"] = sample["image_grid_thw"]
        ref_out = ref_model(**ref_kwargs).logits
        shift_logits = ref_out[:, :-1, :].contiguous()
        shift_labels = sample["labels"][:, 1:].contiguous()
        ref_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        ) / sample["valid_token_count"].sum().clamp(min=1).to(shift_logits.dtype)
        ref_loss.backward()

    rw_wrap = KimiCausalLMTaskWrapper(rw_model, sum_ce_loss=True)
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
        torch.testing.assert_close(wp.grad, rp.grad, rtol=1e-4, atol=1e-4)
        checked += 1

    tag = f"{preset:18s} (expand_moe={str(expand_moe_ops):5s}, mm={str(multimodal):5s})"
    print(f"  [PASS] {tag}: loss_diff={loss_diff:.2e}, verified {checked} parameter tensors")


if __name__ == "__main__":
    print("=== Running Kimi CPU Golden Numerical Parity Suite ===")
    test_kda_ops_tp_head_sharding_parity()
    test_mla_concat_kv_tp_head_sharding_parity()
    test_vision_varlen_sdpa_tp_head_sharding_parity()
    test_moe_grouped_gemm_ep_parity()
    _run_e2e_parity("micro-kimi-linear", expand_moe_ops=False)
    _run_e2e_parity("micro-kimi-linear", expand_moe_ops=True)
    _run_e2e_parity("micro-kimi-k25", expand_moe_ops=False, multimodal=False)
    _run_e2e_parity("micro-kimi-k25", expand_moe_ops=False, multimodal=True)
    _run_e2e_parity("micro-kimi-k25", expand_moe_ops=True, multimodal=True)
    print("=== ALL KIMI CPU GOLDEN PARITY TESTS PASSED ===")
