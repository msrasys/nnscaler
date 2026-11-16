#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Tracing fakes of the ring-attention wrappers: metadata only, no FlashAttention kernel."""

import sys
import types
from typing import Optional, Tuple

import pytest
import torch

try:
    import flash_attn  # noqa: F401
except ImportError:
    # The wrapper modules import flash_attn at module level. The fakes never call it, so a stub
    # whose kernels raise is enough to import them and proves tracing takes the fake path.
    def _no_kernel(*args, **kwargs):
        raise AssertionError('FlashAttention kernel called during tracing')

    _stub = types.ModuleType('flash_attn')
    _stub.__path__ = []
    _interface = types.ModuleType('flash_attn.flash_attn_interface')
    for _name in ('flash_attn_func', 'flash_attn_varlen_func', 'flash_attn_with_kvcache'):
        setattr(_stub, _name, _no_kernel)
    for _name in ('_flash_attn_forward', '_flash_attn_backward',
                  '_flash_attn_varlen_forward', '_flash_attn_varlen_backward'):
        setattr(_interface, _name, _no_kernel)
    _stub.flash_attn_interface = _interface
    sys.modules['flash_attn'] = _stub
    sys.modules['flash_attn.flash_attn_interface'] = _interface

from nnscaler.flags import CompileFlag
from nnscaler.graph.parser.converter import to_fx_graph
from nnscaler.customized_ops.ring_attention import wrap_ring_attn_func, wrap_sliding_window_attn_func
from nnscaler.customized_ops.ring_attention.core.utils import fake_attention_fn


def dense_wrapper(q, k, v, softmax_scale=None, dropout_p: float = 0.0, causal: bool = True,
                  window_size: Tuple[int] = (-1, -1), alibi_slopes=None, deterministic: bool = False,
                  return_attn_probs: bool = False, process_group: Tuple[int] = None):
    raise AssertionError('the real wrapper must not run during tracing')


def varlen_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, dropout_p: float = 0.0,
                   softmax_scale=None, causal: bool = True, window_size: Tuple[int] = (-1, -1),
                   deterministic: bool = False, return_attn_probs: bool = False, enable_ring: bool = True,
                   use_cute: bool = False, process_group: Tuple[int] = None, return_lse: bool = False,
                   max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None):
    raise AssertionError('the real wrapper must not run during tracing')


@pytest.mark.parametrize('device', ['cpu', 'meta'])
def test_dense_fake_matches_output_metadata(device):
    fake = fake_attention_fn(dense_wrapper)
    q = torch.randn(2, 8, 8, 16, dtype=torch.bfloat16, device=device, requires_grad=True)
    k = torch.randn(2, 8, 2, 16, dtype=torch.bfloat16, device=device)
    v = torch.randn(2, 8, 2, 32, dtype=torch.bfloat16, device=device)
    out = fake(q, k, v, causal=True)
    assert out.shape == (2, 8, 8, 32) and out.dtype == torch.bfloat16 and out.device.type == device
    assert out.requires_grad and out.is_contiguous()


@pytest.mark.parametrize('device', ['cpu', 'meta'])
@pytest.mark.parametrize('positional', [False, True])
def test_varlen_fake_returns_lse_like_the_wrapper(device, positional):
    fake = fake_attention_fn(varlen_wrapper)
    q = torch.randn(10, 4, 16, dtype=torch.bfloat16, device=device)
    k = torch.randn(10, 2, 16, dtype=torch.bfloat16, device=device, requires_grad=True)
    v = torch.randn(10, 2, 16, dtype=torch.bfloat16, device=device)
    cu = torch.tensor([0, 10], dtype=torch.int32, device=device)
    if positional:
        out, lse = fake(q, k, v, cu, cu, None, 0.0, None, True, (-1, -1), False, False, True, False, None, True)
    else:
        out, lse = fake(q, k, v, cu, cu, None, return_lse=True)
    assert out.shape == (10, 4, 16) and out.dtype == torch.bfloat16
    assert lse.shape == (4, 10) and lse.dtype == torch.float32 and lse.device.type == device
    # both outputs of the real autograd function require grad when any of q/k/v does
    assert out.requires_grad and lse.requires_grad
    assert fake(q, k, v, cu, cu, None).shape == (10, 4, 16)


def test_fake_requires_grad_follows_inputs():
    fake = fake_attention_fn(varlen_wrapper)
    q = torch.randn(6, 2, 8, dtype=torch.bfloat16)
    k = torch.randn(6, 2, 8, dtype=torch.bfloat16)
    v = torch.randn(6, 2, 8, dtype=torch.bfloat16)
    cu = torch.tensor([0, 6], dtype=torch.int32)
    assert not fake(q, k, v, cu, cu, None).requires_grad
    q.requires_grad_()
    assert fake(q, k, v, cu, cu, None).requires_grad
    with torch.no_grad():
        assert not fake(q, k, v, cu, cu, None).requires_grad


class AttentionModel(torch.nn.Module):
    def forward(self, q, k, v, cu_seqlens):
        # no max_seqlen: the real wrapper would read cu_seqlens values here
        out, lse = wrap_sliding_window_attn_func(
            q, k, v, cu_seqlens, cu_seqlens, None, causal=True, window_size=(4, 0), return_lse=True)
        dense = wrap_ring_attn_func(q[None], k[None], v[None])
        return out.float().sum() + lse.sum() + dense.float().sum()


@pytest.mark.parametrize('strategy', ['cpu', 'meta'])
def test_registered_fakes_trace_without_kernels(monkeypatch, strategy):
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    inputs = {
        'q': torch.randn(12, 4, 16, dtype=torch.bfloat16, requires_grad=True),
        'k': torch.randn(12, 2, 16, dtype=torch.bfloat16, requires_grad=True),
        'v': torch.randn(12, 2, 16, dtype=torch.bfloat16, requires_grad=True),
        'cu_seqlens': torch.tensor([0, 5, 12], dtype=torch.int32),
    }
    graph = to_fx_graph(AttentionModel(), inputs)
    nodes = {n.target: n for n in graph.graph.nodes if n.op == 'call_function'}
    varlen = nodes[wrap_sliding_window_attn_func].meta['tensor_meta']
    assert tuple(varlen[0].shape) == (12, 4, 16) and varlen[0].dtype == torch.bfloat16 and varlen[0].requires_grad
    assert tuple(varlen[1].shape) == (4, 12) and varlen[1].dtype == torch.float32 and varlen[1].requires_grad
    dense = nodes[wrap_ring_attn_func].meta['tensor_meta']
    assert tuple(dense.shape) == (1, 12, 4, 16) and dense.requires_grad
