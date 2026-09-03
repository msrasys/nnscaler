import tempfile
from functools import partial
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from flash_attn.cute import flash_attn_varlen_func

import nnscaler.customized_ops.ring_attention.pruned_yoco_attn_varlen as pruned_yoco_attn
from nnscaler.codegen.emit import FuncEmission
from nnscaler.customized_ops.ring_attention.pruned_yoco_attn_varlen import (
    _PRUNED_YOCO_CAUSAL_MASK_CUTE_HASH,
    _PRUNED_YOCO_WINDOW_MASK_CUTE_HASH,
    _original_position_causal_mask,
    _original_position_window_mask,
    wrap_pruned_yoco_attn_varlen_func,
)
from nnscaler.customized_ops.ring_attention.yoco_kv import (
    wrap_yoco_kv_allgather,
)
from nnscaler.runtime.device import DeviceGroup
from nnscaler.graph.parser.converter import convert_model
from nnscaler.ir.operator import IRFwOperation
from tests.launch_torchrun import torchrun


class _PrunedAttentionModule(nn.Module):

    def forward(self, q, k, v, query_mask, query_positions, cu_seqlens):
        output, _ = wrap_pruned_yoco_attn_varlen_func(
            q,
            k,
            v,
            query_mask,
            query_positions,
            cu_seqlens,
            cu_seqlens,
            None,
            cp_size=2,
            require_full_plan_sequence_partition=True,
        )
        return output


def test_pruned_attention_masks_have_stable_precompile_cache_keys():
    assert _original_position_causal_mask.__cute_hash__ == (
        _PRUNED_YOCO_CAUSAL_MASK_CUTE_HASH
    )
    assert _original_position_window_mask.__cute_hash__ == (
        _PRUNED_YOCO_WINDOW_MASK_CUTE_HASH
    )


def test_pruned_attention_uses_fixed_calls_and_skips_empty_packed_sequence(
    monkeypatch,
):
    calls = []

    def fake_fixed_attention(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, tuple(kwargs)))
        dependency = (k.reshape(-1)[0] + v.reshape(-1)[0]) * 0
        output = q + dependency
        lse = q.new_zeros(q.size(0), q.size(2), q.size(1)) + dependency
        return output, lse

    monkeypatch.setattr(
        pruned_yoco_attn,
        'flash_attn_cute_varlen_func',
        fake_fixed_attention,
    )
    q = torch.randn(10, 4, 8, requires_grad=True)
    k = torch.randn(10, 2, 8, requires_grad=True)
    v = torch.randn(10, 2, 8, requires_grad=True)
    query_mask = torch.tensor(
        [False, False, False, False, False, True, False, True, False, False]
    )
    positions = torch.tensor(
        [0, 1, 2, 3, 0, 1, 2, 3, 4, 5], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 4, 10], dtype=torch.int32)

    output, lse = wrap_pruned_yoco_attn_varlen_func(
        q,
        k,
        v,
        query_mask,
        positions,
        cu_seqlens,
        cu_seqlens,
        None,
        enable_ring=False,
        cp_size=1,
        kv_is_gathered=True,
    )

    assert len(calls) == 1
    q_shape, k_shape, kwarg_names = calls[0]
    assert q_shape == (1, 2, 4, 8)
    assert k_shape == (1, 6, 2, 8)
    assert 'cu_seqlens_q' not in kwarg_names
    assert 'cu_seqlens_k' not in kwarg_names
    assert lse.shape == (4, 2)
    assert torch.count_nonzero(output[~query_mask]) == 0
    output.sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


def test_pruned_attention_cp_group_survives_codegen():
    tokens = 64
    inputs = {
        'q': torch.randn(tokens, 4, 128, dtype=torch.bfloat16),
        'k': torch.randn(tokens, 2, 128, dtype=torch.bfloat16),
        'v': torch.randn(tokens, 2, 128, dtype=torch.bfloat16),
        'query_mask': torch.ones(tokens, dtype=torch.bool),
        'query_positions': torch.arange(tokens, dtype=torch.int32),
        'cu_seqlens': torch.tensor([0, tokens], dtype=torch.int32),
    }
    with tempfile.TemporaryDirectory() as savedir:
        graph = convert_model(
            _PrunedAttentionModule(), inputs, savedir, constant_folding=False)

    node = next(
        node for node in graph.select(ntype=IRFwOperation)
        if node.fn is wrap_pruned_yoco_attn_varlen_func
    )
    sub_nodes = graph.partition(
        node, node.algorithm('dim'), idx=0, dim=0, num=8)
    emitted = FuncEmission().emit_fnode(
        sub_nodes[5], runtime_devid=5, plan_ndevs=8, runtime_ndevs=8)[-1]
    assert 'cp_size=2' in emitted
    assert 'process_group=[4, 5]' in emitted
    assert 'require_full_plan_sequence_partition=True' in emitted


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 10,
    reason='FA4 query-position masks require Blackwell',
)
@pytest.mark.parametrize('window_size', [(-1, -1), (2, 0)])
def test_pruned_attention_matches_dense_reference(window_size):
    torch.manual_seed(19)
    device = torch.device('cuda')
    lengths = (5, 7)
    cu = torch.tensor([0, 5, 12], dtype=torch.int32, device=device)
    positions = torch.tensor(
        list(range(lengths[0])) + list(range(lengths[1])),
        dtype=torch.int32,
        device=device,
    )
    query_mask = torch.tensor(
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        dtype=torch.bool,
        device=device,
    )
    q = torch.randn(
        12, 4, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
    k = torch.randn(
        12, 2, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
    v = torch.randn(
        12, 2, 128, dtype=torch.bfloat16, device=device, requires_grad=True)

    actual, _ = wrap_pruned_yoco_attn_varlen_func(
        q,
        k,
        v,
        query_mask,
        positions,
        cu,
        cu,
        None,
        window_size=window_size,
        enable_ring=False,
        cp_size=1,
        kv_is_gathered=True,
    )
    expected, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        causal=True,
        window_size=tuple(None if item == -1 else item for item in window_size),
        return_lse=True,
    )
    torch.testing.assert_close(actual[query_mask], expected[query_mask])
    assert torch.count_nonzero(actual[~query_mask]) == 0

    grad = torch.randn_like(actual)
    actual.backward(grad, retain_graph=True)
    actual_grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())
    q.grad = k.grad = v.grad = None
    expected[query_mask].backward(grad[query_mask])
    for actual_grad, expected_grad in zip(actual_grads, (q.grad, k.grad, v.grad)):
        torch.testing.assert_close(actual_grad, expected_grad)


def _cp2_pruned_attention_worker():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    rank = dist.get_rank()
    assert dist.get_world_size() == 2

    torch.manual_seed(31)
    tokens = 32
    slice_size = tokens // 4
    global_q = torch.randn(tokens, 4, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(
        tokens, 2, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
    v = torch.randn(
        tokens, 2, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
    front = torch.arange(rank * slice_size, (rank + 1) * slice_size, device=device)
    end_start = (4 - rank - 1) * slice_size
    end = torch.arange(end_start, end_start + slice_size, device=device)
    local_idx = torch.cat((front, end))
    q = global_q.index_select(0, local_idx).detach().requires_grad_(True)
    global_mask = torch.tensor(
        [(index % 3) != 1 for index in range(tokens)],
        dtype=torch.bool,
        device=device,
    )
    local_mask = global_mask.index_select(0, local_idx)
    local_positions = local_idx.to(torch.int32)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)

    actual, _ = wrap_pruned_yoco_attn_varlen_func(
        q,
        k,
        v,
        local_mask,
        local_positions,
        cu,
        cu,
        None,
        process_group=(0, 1),
        cp_size=2,
        kv_is_gathered=True,
    )
    reference, _ = flash_attn_varlen_func(
        global_q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=tokens,
        max_seqlen_k=tokens,
        causal=True,
        return_lse=True,
    )
    torch.testing.assert_close(
        actual[local_mask], reference[local_idx][local_mask])
    assert torch.count_nonzero(actual[~local_mask]) == 0
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or torch.cuda.get_device_capability()[0] < 10,
    reason='requires two Blackwell GPUs',
)
def test_cp2_pruned_attention_uses_zigzag_query_positions():
    partial(torchrun, 2, _cp2_pruned_attention_worker)()


def _cp2_empty_query_rank_backward_worker():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    DeviceGroup().get_group([0, 1])

    torch.manual_seed(43)
    tokens = 32
    hidden_dim = 128
    slice_size = tokens // 4
    global_hidden = torch.randn(
        tokens, hidden_dim, dtype=torch.bfloat16, device=device)
    q_weight = torch.nn.Parameter(torch.randn(
        4 * 128, hidden_dim, dtype=torch.bfloat16, device=device))
    k_weight = torch.nn.Parameter(torch.randn(
        2 * 128, hidden_dim, dtype=torch.bfloat16, device=device))
    v_weight = torch.nn.Parameter(torch.randn(
        2 * 128, hidden_dim, dtype=torch.bfloat16, device=device))

    front = torch.arange(
        rank * slice_size, (rank + 1) * slice_size, device=device)
    end_start = (4 - rank - 1) * slice_size
    end = torch.arange(end_start, end_start + slice_size, device=device)
    local_query_idx = torch.cat((front, end))
    q = F.linear(
        global_hidden.index_select(0, local_query_idx), q_weight
    ).view(-1, 4, 128)

    contiguous_start = rank * (tokens // 2)
    contiguous_idx = torch.arange(
        contiguous_start, contiguous_start + tokens // 2, device=device)
    local_k = F.linear(
        global_hidden.index_select(0, contiguous_idx), k_weight
    ).view(-1, 2, 128)
    local_v = F.linear(
        global_hidden.index_select(0, contiguous_idx), v_weight
    ).view(-1, 2, 128)
    gathered_k, gathered_v = wrap_yoco_kv_allgather(
        local_k,
        local_v,
        process_group=(0, 1),
        cp_size=2,
    )

    # Rank 1 deliberately owns no active Q. Rank 0 keeps late queries so its
    # K/V gradient reaches both contiguous K/V shards.
    query_mask = torch.zeros(
        local_query_idx.numel(), dtype=torch.bool, device=device)
    if rank == 0:
        query_mask[slice_size:] = True
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    output, _ = wrap_pruned_yoco_attn_varlen_func(
        q,
        gathered_k,
        gathered_v,
        query_mask,
        local_query_idx.to(torch.int32),
        cu,
        cu,
        None,
        process_group=(0, 1),
        cp_size=2,
        kv_is_gathered=True,
    )
    output.float().sum().backward()

    assert q_weight.grad is not None
    assert k_weight.grad is not None
    assert v_weight.grad is not None
    assert torch.isfinite(q_weight.grad).all()
    assert torch.isfinite(k_weight.grad).all()
    assert torch.isfinite(v_weight.grad).all()
    if rank == 1:
        assert torch.count_nonzero(q_weight.grad) == 0
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or torch.cuda.get_device_capability()[0] < 10,
    reason='requires two Blackwell GPUs',
)
def test_cp2_empty_query_rank_preserves_qkv_backward_dependencies():
    partial(torchrun, 2, _cp2_empty_query_rank_backward_worker)()
