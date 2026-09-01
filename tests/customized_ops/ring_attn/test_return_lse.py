#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import importlib
import inspect

import pytest
import torch

# Skip all tests if flash_attn_func is not available
try:
    from flash_attn import flash_attn_func
except ImportError:
    pytest.skip("flash_attn_func not available", allow_module_level=True)


pytest.importorskip("flash_attn")

from nnscaler.customized_ops.ring_attention.core.utils import call_flash_attn_cute_varlen_func


def test_call_flash_attn_cute_varlen_func_passes_return_lse_by_name():
    calls = {}

    def fake_cute_func(x, return_lse=False):
        calls["return_lse"] = return_lse
        return x, x + 1

    x = torch.tensor([1.0])
    out, lse = call_flash_attn_cute_varlen_func(fake_cute_func, x, return_lse=True)

    assert calls["return_lse"] is True
    assert out is x
    assert torch.equal(lse, x + 1)


def test_call_flash_attn_cute_varlen_func_rejects_missing_lse():
    def fake_cute_func(x):
        return x

    with pytest.raises(RuntimeError, match="did not return softmax_lse"):
        call_flash_attn_cute_varlen_func(fake_cute_func, torch.tensor([1.0]), return_lse=True)


def _annotation_inputs():
    q = torch.empty(8, 4, 16)
    k = torch.empty(8, 4, 16)
    v = torch.empty(8, 4, 16)
    return q, k, v, None, None, None


def _return_lse_positional_args(func):
    parameters = list(inspect.signature(func).parameters.values())[6:]
    args = []
    for parameter in parameters:
        if parameter.name == "return_lse":
            args.append(True)
            break
        args.append(parameter.default)
    return tuple(args)


@pytest.mark.parametrize(
    "module_name",
    [
        "nnscaler.customized_ops.ring_attention.ring_attn_varlen",
        "nnscaler.customized_ops.ring_attention.sliding_window_attn",
        "nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen",
    ],
)
def test_flash_attention_anno_marks_lse_shape_by_keyword_and_position(module_name):
    module = importlib.import_module(module_name)
    wrapper_name = {
        "ring_attn_varlen": "wrap_ring_attn_varlen_func",
        "sliding_window_attn": "wrap_sliding_window_attn_func",
        "zigzag_allgather_attn_varlen": "wrap_zigzag_allgather_attn_varlen_func",
    }[module_name.rsplit(".", 1)[-1]]
    wrapper = getattr(module, wrapper_name)
    inputs = _annotation_inputs()

    default_anno = module.flash_attention_anno(*inputs)
    keyword_anno = module.flash_attention_anno(*inputs, return_lse=True)
    positional_anno = module.flash_attention_anno(
        *inputs, *_return_lse_positional_args(wrapper)
    )

    assert default_anno.endswith("-> l num_heads vd^")
    assert keyword_anno.endswith("-> l num_heads vd^, num_heads l")
    assert positional_anno.endswith("-> l num_heads vd^, num_heads l")
