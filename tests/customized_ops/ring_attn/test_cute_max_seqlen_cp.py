#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import importlib

import torch


def _patch_process_group(monkeypatch, module, world_size=2):
    process_group = object()

    class FakeDeviceGroup:
        def get_group(self, ranks):
            return process_group

    monkeypatch.setattr(module, "DeviceGroup", FakeDeviceGroup)
    monkeypatch.setattr(module.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: world_size)
    return process_group


def _inputs():
    q = torch.empty(8, 2, 4)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    cu_seqlens = torch.tensor([0, 8], dtype=torch.int32)
    return q, k, v, cu_seqlens


def test_ring_cute_cp_uses_caller_max_seqlen(monkeypatch):
    module = importlib.import_module(
        "nnscaler.customized_ops.ring_attention.ring_attn_varlen"
    )
    _patch_process_group(monkeypatch, module)
    q, k, v, cu_seqlens = _inputs()
    observed = {}

    monkeypatch.setattr(
        module,
        "llama3_flash_attn_prepare_cu_seqlens",
        lambda *args, **kwargs: (cu_seqlens, cu_seqlens, 4, 6, slice(0, 8)),
    )

    def fake_flash_attn(*args, **kwargs):
        observed["max_seqlen_q"] = args[5]
        observed["max_seqlen_k"] = args[6]
        return q, torch.empty(2, 8)

    monkeypatch.setattr(module, "llama3_flash_attn_varlen_func", fake_flash_attn)

    module.wrap_ring_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        None,
        causal=True,
        use_cute=True,
        process_group=(0, 1),
        max_seqlen_q=16,
        max_seqlen_k=24,
    )

    assert observed == {"max_seqlen_q": 16, "max_seqlen_k": 24}


def test_sliding_window_cute_cp_uses_caller_max_seqlen(monkeypatch):
    module = importlib.import_module(
        "nnscaler.customized_ops.ring_attention.sliding_window_attn"
    )
    implementation = importlib.import_module(
        "nnscaler.customized_ops.ring_attention.core.sliding_window_attn_implementation"
    )
    _patch_process_group(monkeypatch, module)
    q, k, v, cu_seqlens = _inputs()
    observed = {}
    metadata = implementation.SlidingWindowMetadata(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=4,
        max_seqlen_k=6,
        input_split_sizes=[0, 0],
        output_split_sizes=[0, 0],
    )

    monkeypatch.setattr(
        module, "prepare_sliding_window_metadata", lambda *args, **kwargs: metadata
    )

    def fake_sliding_window_attn(*args, **kwargs):
        observed["max_seqlen_q"] = args[5].max_seqlen_q
        observed["max_seqlen_k"] = args[5].max_seqlen_k
        return q, torch.empty(2, 8)

    monkeypatch.setattr(module, "sliding_window_attn_func", fake_sliding_window_attn)

    module.wrap_sliding_window_attn_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        None,
        causal=True,
        window_size=(2, 0),
        use_cute=True,
        process_group=(0, 1),
        max_seqlen_q=16,
        max_seqlen_k=24,
    )

    assert observed == {"max_seqlen_q": 16, "max_seqlen_k": 24}
    assert metadata.max_seqlen_q == 4
    assert metadata.max_seqlen_k == 6


def test_zigzag_cute_cp_uses_caller_max_seqlen(monkeypatch):
    module = importlib.import_module(
        "nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen"
    )
    implementation = importlib.import_module(
        "nnscaler.customized_ops.ring_attention.core.zigzag_allgather_attn_varlen_implementation"
    )
    _patch_process_group(monkeypatch, module)
    monkeypatch.setattr(implementation.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(implementation.dist, "get_world_size", lambda group: 2)
    q, k, v, cu_seqlens = _inputs()
    observed = []
    metadata = implementation.ZigZagAllGatherVarlenMetadata(
        cu_seqlens_q_front=cu_seqlens,
        cu_seqlens_k_front=cu_seqlens,
        cu_seqlens_q_end=cu_seqlens,
        cu_seqlens_k_end=cu_seqlens,
        max_seqlen_q=4,
        max_seqlen_k_front=6,
        max_seqlen_k_end=8,
        q_front_idx=torch.arange(4),
        q_end_idx=torch.arange(4, 8),
    )

    monkeypatch.setattr(
        implementation,
        "prepare_zigzag_allgather_attn_varlen_metadata",
        lambda *args, **kwargs: metadata,
    )

    def fake_flash_attn(q_arg, *args, **kwargs):
        observed.append((args[4], args[5]))
        return q_arg, torch.empty(2, q_arg.shape[0])

    monkeypatch.setattr(implementation, "_run_flash_attn_varlen", fake_flash_attn)
    monkeypatch.setattr(
        module,
        "zigzag_allgather_attn_varlen_func",
        implementation.zigzag_allgather_attn_varlen_func,
    )

    module.wrap_zigzag_allgather_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        None,
        causal=True,
        use_cute=True,
        process_group=(0, 1),
        max_seqlen_q=16,
        max_seqlen_k=24,
    )

    assert observed == [(16, 24), (16, 24)]
