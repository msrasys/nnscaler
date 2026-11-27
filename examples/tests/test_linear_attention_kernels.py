# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optional-kernel dispatch regressions and real CUDA forward/backward parity."""

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from examples import _kernel_utils
from examples.qwen3p5.ops import gdn_ops


def _attention_case(family):
    if family == "qwen":
        return gdn_ops, gdn_ops.qwen3_5_chunk_gated_delta_rule, gdn_ops._reference_gated_delta_rule
    module = importlib.import_module(f"examples.{family}.ops.kda_ops")
    return module, getattr(module, f"{family.lower()}_chunk_kda"), module._reference_chunk_kda


def _attention_inputs(family, device="cpu", dtype=torch.bfloat16, dim=16):
    heads = 2 if family == "qwen" else 4
    q, k = [
        (torch.randn(2, 7, heads * 2, dim, device=device, dtype=dtype) * 0.1)[:, :, ::2]
        .detach().requires_grad_()
        for _ in range(2)
    ]
    v = torch.randn(2, 7, 8, 16, device=device, dtype=dtype)[:, :, ::2].detach().requires_grad_()
    gate_shape = v.shape[:3] if family == "qwen" else q.shape
    g = (-torch.rand(gate_shape, device=device) * 0.2).requires_grad_()
    beta = torch.rand(v.shape[:3], device=device, requires_grad=True)
    return [q, k, v, g, beta]


def _reference(reference, inputs):
    output = reference(*inputs, use_qk_l2norm_in_kernel=True)
    return output[0] if isinstance(output, tuple) else output


def _mock_cuda(monkeypatch, capability=8):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda tensor: True))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (capability, 0))


@pytest.mark.parametrize("family", ["qwen", "GLM", "Kimi"])
def test_cpu_attention_never_calls_optional_or_hf_dispatch(monkeypatch, family):
    module, operation, reference = _attention_case(family)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU inputs must not enter optional kernel dispatch")

    monkeypatch.setattr(_kernel_utils, "load_optional_kernel", forbidden)
    if family != "qwen":
        monkeypatch.setattr(module, "chunk_kimi_delta_attention", forbidden)
    inputs = _attention_inputs(family, dtype=torch.float32)
    expected = _reference(reference, inputs)
    actual = operation(*inputs)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert all(tensor.grad is not None for tensor in inputs)


@pytest.mark.parametrize("family", ["qwen", "GLM", "Kimi"])
def test_fla_adapts_grouped_heads_and_strides(monkeypatch, family):
    _, operation, reference = _attention_case(family)
    inputs = _attention_inputs(family)
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    calls = []

    def kernel(q, k, v, g, beta, **kwargs):
        assert q.shape[2] == k.shape[2] == v.shape[2]
        assert all(tensor.is_contiguous() for tensor in (q, k, v, g, beta))
        assert g.dtype == beta.dtype == torch.float32
        assert kwargs["initial_state"] is None and not kwargs["output_final_state"]
        calls.append(True)
        if family == "qwen":
            return reference(q, k, v, g, beta, **{
                key: value for key, value in kwargs.items()
                if key not in ("initial_state", "output_final_state")
            }), None
        return reference(q, k, v, g, beta, **kwargs)

    def load(module, name):
        op = "gated_delta_rule" if family == "qwen" else "kda"
        assert (module, name) == (f"fla.ops.{op}", f"chunk_{op}")
        return kernel

    _mock_cuda(monkeypatch)
    monkeypatch.setattr(_kernel_utils, "load_optional_kernel", load)
    actual = operation(*inputs)
    expected = _reference(reference, expected_inputs)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert calls == [True]
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    for tensor, expected_tensor in zip(inputs, expected_inputs):
        torch.testing.assert_close(tensor.grad, expected_tensor.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("family", ["qwen", "GLM", "Kimi"])
@pytest.mark.parametrize("reason", ["fp32", "head_dim", "architecture", "mixed_dtype"])
def test_unsupported_fla_inputs_use_reference(monkeypatch, family, reason):
    _, operation, reference = _attention_case(family)
    inputs = _attention_inputs(family, dtype=torch.float32 if reason == "fp32" else torch.bfloat16,
                               dim=257 if reason == "head_dim" else 16)
    if reason == "mixed_dtype":
        inputs[1] = inputs[1].detach().float().requires_grad_()
    _mock_cuda(monkeypatch, capability=7 if reason == "architecture" else 8)

    def forbidden(*args):
        raise AssertionError("Unsupported inputs must not load a FLA kernel")

    monkeypatch.setattr(_kernel_utils, "load_optional_kernel", forbidden)
    output = operation(*inputs)
    torch.testing.assert_close(output, _reference(reference, inputs))
    output.sum().backward()
    assert all(tensor.grad is not None for tensor in inputs)


def test_optional_import_failure_is_reported_once(monkeypatch, caplog):
    attempts = []

    def unavailable(module):
        attempts.append(module)
        raise RuntimeError("no active CUDA driver")

    _kernel_utils.load_optional_kernel.cache_clear()
    monkeypatch.setattr(_kernel_utils.importlib, "import_module", unavailable)
    try:
        for _ in range(2):
            assert _kernel_utils.load_optional_kernel("fla.ops.kda", "chunk_kda") is None
        assert attempts == ["fla.ops.kda"]
        assert caplog.text.count("no active CUDA driver") == 1
        assert "using the PyTorch reference" in caplog.text
    finally:
        _kernel_utils.load_optional_kernel.cache_clear()


def test_legacy_fla_is_explicitly_sequence_first(monkeypatch):
    _, operation, reference = _attention_case("GLM")
    inputs = _attention_inputs("GLM")
    layouts = []

    def legacy_kernel(q, k, v, g, beta, head_first=True, **kwargs):
        layouts.append(head_first)
        assert not head_first
        return reference(q, k, v, g, beta, **kwargs)

    _mock_cuda(monkeypatch)
    _kernel_utils.load_optional_kernel.cache_clear()
    monkeypatch.setattr(_kernel_utils.importlib, "import_module",
                        lambda name: SimpleNamespace(chunk_kda=legacy_kernel))
    try:
        torch.testing.assert_close(operation(*inputs), _reference(reference, inputs))
        assert layouts == [False]
    finally:
        _kernel_utils.load_optional_kernel.cache_clear()


@pytest.mark.parametrize("family", ["qwen", "GLM", "Kimi", "conv"])
def test_missing_optional_kernel_preserves_gradients(monkeypatch, family):
    _mock_cuda(monkeypatch)
    if family == "conv":
        operation = gdn_ops.qwen3_5_causal_conv1d
        reference = _conv_reference
        inputs = [torch.randn(1, 6, 8, requires_grad=True), torch.randn(6, 4, requires_grad=True)]
        monkeypatch.setattr(gdn_ops, "load_optional_kernel", lambda *args: None)
    else:
        _, operation, reference = _attention_case(family)
        inputs = _attention_inputs(family)
        monkeypatch.setattr(_kernel_utils, "load_optional_kernel", lambda *args: None)
    actual = operation(*inputs)
    expected = reference(*inputs) if family == "conv" else _reference(reference, inputs)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert all(tensor.grad is not None for tensor in inputs)


def _conv_reference(x, weight, bias=None, activation="silu"):
    w = weight.squeeze(1) if weight.ndim == 3 else weight
    out = F.conv1d(x.to(w.dtype), w.unsqueeze(1), bias.to(w.dtype) if bias is not None else None,
                   padding=w.shape[-1] - 1, groups=x.shape[1])[..., :x.shape[-1]]
    return (out if activation is None else F.silu(out)).to(x.dtype)


@pytest.mark.parametrize("width,weight_dtype", [(2, torch.float32), (4, torch.bfloat16),
                                               (5, torch.float32), (3, torch.float64)])
def test_causal_conv_dispatch_and_tp_layout(monkeypatch, width, weight_dtype):
    x = torch.randn(2, 9, 6).transpose(1, 2).detach().requires_grad_()
    weight = torch.randn(12, 1, width, dtype=weight_dtype)[::2].detach().requires_grad_()
    bias = torch.randn(12, dtype=weight_dtype)[::2].detach().requires_grad_()
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in (x, weight, bias)]
    calls = []

    def kernel(x, weight, bias, activation):
        assert all(tensor.is_contiguous() for tensor in (x, weight, bias))
        assert x.dtype == weight.dtype == bias.dtype
        calls.append(True)
        return _conv_reference(x, weight, bias, activation)

    _mock_cuda(monkeypatch)
    monkeypatch.setattr(gdn_ops, "load_optional_kernel", lambda *args: kernel)
    actual = gdn_ops.qwen3_5_causal_conv1d(x, weight, bias, "swish")
    expected = _conv_reference(*expected_inputs, "swish")
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    assert len(calls) == int(width in (2, 3, 4) and weight_dtype != torch.float64)
    torch.testing.assert_close(actual, expected)
    for tensor, expected_tensor in zip((x, weight, bias), expected_inputs):
        torch.testing.assert_close(tensor.grad, expected_tensor.grad)


@pytest.mark.parametrize("family", ["qwen", "GLM", "Kimi", "conv"])
def test_kernel_execution_errors_are_not_silenced(monkeypatch, family):
    def broken(*args, **kwargs):
        raise RuntimeError("kernel execution failed")

    _mock_cuda(monkeypatch)
    if family == "conv":
        monkeypatch.setattr(gdn_ops, "load_optional_kernel", lambda *args: broken)
        operation = gdn_ops.qwen3_5_causal_conv1d
        inputs = [torch.randn(1, 6, 8), torch.randn(6, 4)]
    else:
        _, operation, _ = _attention_case(family)
        inputs = _attention_inputs(family)
        monkeypatch.setattr(_kernel_utils, "load_optional_kernel", lambda *args: broken)
    with pytest.raises(RuntimeError, match="kernel execution failed"):
        operation(*inputs)


@pytest.fixture
def _disable_tf32():
    matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    with torch.backends.cudnn.flags(allow_tf32=False):
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_allow_tf32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("family,dtype", [
    (family, dtype) for family in ["qwen", "GLM", "Kimi", "conv"]
    for dtype in ([torch.float16, torch.bfloat16, torch.float32] if family == "conv"
                  else [torch.float16, torch.bfloat16])
])
def test_real_cuda_kernel_forward_backward_and_tp(monkeypatch, _disable_tf32, family, dtype):
    if family != "conv" and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("FLA path requires an Ampere or newer GPU")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16 support")
    torch.manual_seed(7)
    if family == "conv":
        module = gdn_ops
        operation = gdn_ops.qwen3_5_causal_conv1d
        inputs = [
            torch.randn(2, 17, 12, device="cuda", dtype=dtype).transpose(1, 2).detach().requires_grad_(),
            torch.randn(12, 1, 4, device="cuda", dtype=dtype, requires_grad=True),
            torch.randn(12, device="cuda", dtype=dtype, requires_grad=True),
        ]
        reference = _conv_reference
        kernel = _kernel_utils.load_optional_kernel("causal_conv1d", "causal_conv1d_fn")
    else:
        module, operation, reference = _attention_case(family)
        inputs = _attention_inputs(family, device="cuda", dtype=dtype)
        kernel = _kernel_utils.fla_kernel("gated_delta_rule" if family == "qwen" else "kda", *inputs[:3])
    if kernel is None:
        pytest.skip("optional CUDA kernel is unavailable; reference execution is not kernel validation")
    calls = []

    def tracked_kernel(*args, **kwargs):
        calls.append(True)
        return kernel(*args, **kwargs)

    if family == "conv":
        monkeypatch.setattr(module, "load_optional_kernel", lambda *args: tracked_kernel)
    else:
        monkeypatch.setattr(_kernel_utils, "load_optional_kernel", lambda *args: tracked_kernel)
    expected_inputs = [tensor.detach().clone().requires_grad_() for tensor in inputs]
    actual = operation(*inputs)
    expected = reference(*expected_inputs) if family == "conv" else _reference(reference, expected_inputs)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    rtol, atol = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (0.01, 0.002),
        torch.bfloat16: (0.05, 0.01),
    }[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    for tensor, expected_tensor in zip(inputs, expected_inputs):
        torch.testing.assert_close(tensor.grad, expected_tensor.grad, rtol=rtol, atol=atol)
    input_dims = [1, 0, 0] if family == "conv" else [2] * 5
    output_dim = 1 if family == "conv" else 2
    shards = [[chunk.detach().requires_grad_() for chunk in tensor.chunk(2, dim=dim)]
              for tensor, dim in zip(inputs, input_dims)]
    outputs = [operation(*(parts[rank] for parts in shards)) for rank in range(2)]
    sharded = torch.cat(outputs, dim=output_dim)
    sharded.backward(grad)
    torch.testing.assert_close(sharded, actual, rtol=rtol, atol=atol)
    for tensor, parts, dim in zip(inputs, shards, input_dims):
        torch.testing.assert_close(torch.cat([part.grad for part in parts], dim=dim), tensor.grad,
                                   rtol=rtol, atol=atol)
    assert len(calls) == 3, "full and both TP slices must execute the real CUDA kernel"
