#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import importlib

import pytest
import torch


try:
    implementation = importlib.import_module(
        "nnscaler.customized_ops.ring_attention.core."
        "zigzag_allgather_attn_varlen_implementation"
    )
except ImportError:
    pytest.skip(
        "zigzag_allgather_attn_varlen implementation is unavailable",
        allow_module_level=True,
    )


def test_metadata_cached_during_inference_is_safe_for_backward():
    implementation._METADATA_CACHE.clear()

    with torch.inference_mode():
        cu_seqlens_q = torch.tensor([0, 8], dtype=torch.int32)
        cu_seqlens_k = torch.tensor([0, 8], dtype=torch.int32)
        inference_metadata = (
            implementation.prepare_zigzag_allgather_attn_varlen_metadata(
                cu_seqlens_q,
                cu_seqlens_k,
                world_size=2,
                rank=0,
            )
        )

    cached_metadata = (
        implementation.prepare_zigzag_allgather_attn_varlen_metadata(
            cu_seqlens_q,
            cu_seqlens_k,
            world_size=2,
            rank=0,
        )
    )
    assert cached_metadata is inference_metadata
    for value in vars(cached_metadata).values():
        if isinstance(value, torch.Tensor):
            assert not value.is_inference()

    q = torch.randn(4, requires_grad=True)
    q[cached_metadata.q_front_idx].sum().backward()
    assert q.grad is not None
