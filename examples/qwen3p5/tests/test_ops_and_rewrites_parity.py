"""Mathematical split-equivalence and micro-model parity tests on CPU."""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from examples.qwen3p5.configs.model_presets import load_model_config_dict
from examples.qwen3p5.ops import register_all_qwen3_5_ops
from examples.qwen3p5.ops.gdn_ops import (
    qwen3_5_causal_conv1d,
    qwen3_5_chunk_gated_delta_rule,
)
from examples.qwen3p5.ops.moe_ops import (
    qwen3_5_grouped_linear,
    qwen3_5_moe_grouped_gemm,
    qwen3_5_moe_permute,
    qwen3_5_moe_route,
    qwen3_5_moe_unpermute,
)
from examples.qwen3p5.ops.vision_ops import (
    qwen3_5_masked_scatter,
    qwen3_5_vision_varlen_attn,
)
from examples.qwen3p5.rewrites.model_rewrites import (
    Qwen3_5CompileWrapper,
    RewriteConfig,
    build_dummy_samples,
    build_qwen3_5_model,
)


def test_gdn_ops_tp_split_forward_backward_parity() -> None:
    register_all_qwen3_5_ops()
    torch.manual_seed(7)

    # 1. Depthwise causal conv1d channel split (TP=2)
    x = torch.randn(2, 8, 12, requires_grad=True)
    w = torch.randn(8, 1, 4, requires_grad=True)
    b = torch.randn(8, requires_grad=True)
    full_out = qwen3_5_causal_conv1d(x, w, b, "silu")
    grad_out = torch.randn_like(full_out)
    full_out.backward(grad_out)

    x0 = x.detach()[:, :4, :].clone().requires_grad_(True)
    x1 = x.detach()[:, 4:, :].clone().requires_grad_(True)
    w0 = w.detach()[:4].clone().requires_grad_(True)
    w1 = w.detach()[4:].clone().requires_grad_(True)
    b0 = b.detach()[:4].clone().requires_grad_(True)
    b1 = b.detach()[4:].clone().requires_grad_(True)
    sharded_out = torch.cat(
        [
            qwen3_5_causal_conv1d(x0, w0, b0, "silu"),
            qwen3_5_causal_conv1d(x1, w1, b1, "silu"),
        ],
        dim=1,
    )
    sharded_out.backward(grad_out)

    torch.testing.assert_close(full_out, sharded_out, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, torch.cat([x0.grad, x1.grad], dim=1), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(w.grad, torch.cat([w0.grad, w1.grad], dim=0), rtol=1e-5, atol=1e-5)

    # 2. Chunk Gated Delta Rule head split with GQA ratio Hv/Hk = 2 (Hk=2, Hv=4 -> TP=2)
    q = torch.randn(2, 10, 2, 16, requires_grad=True)
    k = torch.randn(2, 10, 2, 16, requires_grad=True)
    v = torch.randn(2, 10, 4, 16, requires_grad=True)
    g = -torch.rand(2, 10, 4, requires_grad=True)
    beta = torch.sigmoid(torch.randn(2, 10, 4)).detach().requires_grad_(True)

    y_full = qwen3_5_chunk_gated_delta_rule(q, k, v, g, beta)
    gy = torch.randn_like(y_full)
    y_full.backward(gy)

    parts = []
    q_s = [t.detach().clone().requires_grad_(True) for t in q.chunk(2, dim=2)]
    k_s = [t.detach().clone().requires_grad_(True) for t in k.chunk(2, dim=2)]
    v_s = [t.detach().clone().requires_grad_(True) for t in v.chunk(2, dim=2)]
    g_s = [t.detach().clone().requires_grad_(True) for t in g.chunk(2, dim=2)]
    b_s = [t.detach().clone().requires_grad_(True) for t in beta.chunk(2, dim=2)]
    for i in range(2):
        parts.append(qwen3_5_chunk_gated_delta_rule(q_s[i], k_s[i], v_s[i], g_s[i], b_s[i]))
    y_tp = torch.cat(parts, dim=2)
    y_tp.backward(gy)

    torch.testing.assert_close(y_full, y_tp, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(q.grad, torch.cat([t.grad for t in q_s], dim=2), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(v.grad, torch.cat([t.grad for t in v_s], dim=2), rtol=1e-5, atol=1e-5)


def test_moe_fused_and_expanded_ep_etp_parity() -> None:
    register_all_qwen3_5_ops()
    torch.manual_seed(11)

    num_tokens, hidden, inter, num_experts, top_k = 12, 16, 24, 8, 2
    x = torch.randn(num_tokens, hidden, requires_grad=True)
    router_w = torch.randn(num_experts, hidden, requires_grad=True)
    gate_up = torch.randn(num_experts, 2 * inter, hidden, requires_grad=True)
    down = torch.randn(num_experts, hidden, inter, requires_grad=True)

    # Mode A (default): Fused `qwen3_5_moe_route` + `qwen3_5_moe_grouped_gemm`
    rw, sel = qwen3_5_moe_route(x, router_w, top_k=top_k)
    out_fused = qwen3_5_moe_grouped_gemm(x, rw, sel, gate_up, down, "silu", 0, num_experts)
    g_out = torch.randn_like(out_fused)
    out_fused.backward(g_out)

    # Mode A under EP=2 (`[0..4)` + `[4..8)` summed via AllReduce)
    x_ep = x.detach().clone().requires_grad_(True)
    rw_ep = router_w.detach().clone().requires_grad_(True)
    gu0 = gate_up.detach()[:4].clone().requires_grad_(True)
    gu1 = gate_up.detach()[4:].clone().requires_grad_(True)
    dn0 = down.detach()[:4].clone().requires_grad_(True)
    dn1 = down.detach()[4:].clone().requires_grad_(True)
    rw2, sel2 = qwen3_5_moe_route(x_ep, rw_ep, top_k=top_k)
    out_ep = (
        qwen3_5_moe_grouped_gemm(x_ep, rw2, sel2, gu0, dn0, "silu", 0, 4)
        + qwen3_5_moe_grouped_gemm(x_ep, rw2, sel2, gu1, dn1, "silu", 4, 8)
    )
    out_ep.backward(g_out)

    torch.testing.assert_close(out_fused, out_ep, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, x_ep.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gate_up.grad, torch.cat([gu0.grad, gu1.grad], dim=0), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(down.grad, torch.cat([dn0.grad, dn1.grad], dim=0), rtol=1e-5, atol=1e-5)

    # Mode B (opt-in `expand_moe_ops=True`): permute + grouped_linear (EP=2 & ETP=2) + unpermute
    x_exp = x.detach().clone().requires_grad_(True)
    rw_exp = router_w.detach().clone().requires_grad_(True)
    gu_exp = gate_up.detach().clone().requires_grad_(True)
    dn_exp = down.detach().clone().requires_grad_(True)

    logits = F.linear(x_exp, rw_exp)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    top_v, top_i = torch.topk(probs, top_k, dim=-1)
    top_v = (top_v / top_v.sum(dim=-1, keepdim=True)).to(x_exp.dtype)

    perm_x, counts, s_tok, s_prob = qwen3_5_moe_permute(x_exp, top_v, top_i, num_experts=num_experts)
    gw, uw = torch.chunk(gu_exp, 2, dim=1)
    # Test EP=2 on `qwen3_5_grouped_linear` (`[0..4)` + `[4..8)`) + ETP=2 (`m` / `k+` split)
    g_out_ep = (
        qwen3_5_grouped_linear(perm_x, gw[:4], counts, 0, 4)
        + qwen3_5_grouped_linear(perm_x, gw[4:], counts, 4, 8)
    )
    u_out_etp = torch.cat(
        [
            qwen3_5_grouped_linear(perm_x, uw[:, : inter // 2, :], counts, 0, num_experts),
            qwen3_5_grouped_linear(perm_x, uw[:, inter // 2 :, :], counts, 0, num_experts),
        ],
        dim=1,
    )
    act = F.silu(g_out_ep) * u_out_etp
    d_out_row_etp = (
        qwen3_5_grouped_linear(act[:, : inter // 2], dn_exp[:, :, : inter // 2], counts, 0, num_experts)
        + qwen3_5_grouped_linear(act[:, inter // 2 :], dn_exp[:, :, inter // 2 :], counts, 0, num_experts)
    )
    out_expanded = qwen3_5_moe_unpermute(d_out_row_etp, s_tok, s_prob, num_tokens=num_tokens)
    out_expanded.backward(g_out)

    torch.testing.assert_close(out_fused, out_expanded, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad, x_exp.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(gate_up.grad, gu_exp.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(down.grad, dn_exp.grad, rtol=1e-5, atol=1e-5)


def test_vision_ops_split_parity() -> None:
    register_all_qwen3_5_ops()
    torch.manual_seed(19)

    # 1. Varlen vision attention head split (TP=2)
    q = torch.randn(12, 4, 16, requires_grad=True)
    k = torch.randn(12, 4, 16, requires_grad=True)
    v = torch.randn(12, 4, 16, requires_grad=True)
    cu = torch.tensor([0, 4, 12], dtype=torch.int32)
    out = qwen3_5_vision_varlen_attn(q, k, v, cu, scaling=0.25)
    out_tp = torch.cat(
        [
            qwen3_5_vision_varlen_attn(q[:, :2], k[:, :2], v[:, :2], cu, scaling=0.25),
            qwen3_5_vision_varlen_attn(q[:, 2:], k[:, 2:], v[:, 2:], cu, scaling=0.25),
        ],
        dim=1,
    )
    torch.testing.assert_close(out, out_tp, rtol=1e-5, atol=1e-5)

    # 2. Masked scatter hidden split (TP=2)
    inp = torch.randn(2, 6, 8, requires_grad=True)
    mask = torch.zeros(2, 6, 1, dtype=torch.bool)
    mask[:, 1:3, :] = True
    src = torch.randn(4, 8, requires_grad=True)
    ref = inp.masked_scatter(mask.expand_as(inp), src)
    got = torch.cat(
        [
            qwen3_5_masked_scatter(inp[:, :, :4], mask, src[:, :4]),
            qwen3_5_masked_scatter(inp[:, :, 4:], mask, src[:, 4:]),
        ],
        dim=2,
    )
    torch.testing.assert_close(ref, got, rtol=1e-6, atol=1e-6)


def test_rewritten_micro_models_match_hf_eager_golden() -> None:
    import contextlib
    import inspect
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf_dense
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf_moe

    @contextlib.contextmanager
    def _force_hf_cpu_kernels():
        # HF `@use_kernel_func_from_hub_with_fallback` dispatches to CUDA extensions even on CPU tensors;
        # `inspect.unwrap(...)` exposes HF's own pure-PyTorch reference implementation for CPU golden comparison!
        saved = [
            (hf_dense, "causal_conv1d_fn", hf_dense.causal_conv1d_fn),
            (hf_dense, "torch_chunk_gated_delta_rule", hf_dense.torch_chunk_gated_delta_rule),
            (hf_moe, "causal_conv1d_fn", hf_moe.causal_conv1d_fn),
            (hf_moe, "torch_chunk_gated_delta_rule", hf_moe.torch_chunk_gated_delta_rule),
        ]
        try:
            for mod, attr, fn in saved:
                setattr(mod, attr, inspect.unwrap(fn))
            yield
        finally:
            for mod, attr, fn in saved:
                setattr(mod, attr, fn)

    register_all_qwen3_5_ops()

    for preset, expand_flag, is_mm in [
        ("micro-dense", False, False),
        ("micro-moe", False, False),
        ("micro-moe", True, False),
        ("micro-vl", False, True),
    ]:
        cfg_dict = load_model_config_dict(preset)
        samples = build_dummy_samples(
            cfg_dict, batch_size=2, seq_len=16, multimodal=is_mm, dtype=torch.float32
        )

        no_rewrite = RewriteConfig(
            loss_reduction="sum-count",
            wrap_gdn_kernels=False,
            head_grouped_gdn_qkv=False,
            static_moe_grouped_gemm=False,
            expand_moe_ops=False,
            wrap_vision_ops=False,
        )
        golden_wrap, _ = build_qwen3_5_model(
            cfg_dict, no_rewrite, multimodal=is_mm, weightless=False, dtype=torch.float32, seed=99
        )

        rewrite_cfg = RewriteConfig(
            loss_reduction="sum-count",
            wrap_gdn_kernels=True,
            head_grouped_gdn_qkv=True,
            static_moe_grouped_gemm=True,
            expand_moe_ops=expand_flag,
            wrap_vision_ops=True,
        )
        rewritten_wrap, _ = build_qwen3_5_model(
            cfg_dict, rewrite_cfg, multimodal=is_mm, weightless=False, dtype=torch.float32, seed=99
        )

        golden_samples = copy.deepcopy(samples)
        if is_mm:
            for k in ("vision_interp_indices", "vision_interp_weights", "vision_pos_ids", "vision_cu_seqlens"):
                golden_samples.pop(k, None)

        with _force_hf_cpu_kernels():
            loss_golden = golden_wrap(golden_samples)
            loss_golden.backward()

        loss_rewritten = rewritten_wrap(samples)
        loss_rewritten.backward()

        torch.testing.assert_close(loss_rewritten, loss_golden, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            rewritten_wrap.model.get_input_embeddings().weight.grad,
            golden_wrap.model.get_input_embeddings().weight.grad,
            rtol=1e-4,
            atol=1e-4,
        )
