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


def test_pruned_attention_uses_standard_causal_runs_and_skips_empty_sequence(
    monkeypatch,
):
    calls = []

    def fake_varlen_attention(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, kwargs))
        dependency = (k.reshape(-1)[0] + v.reshape(-1)[0]) * 0
        output = q + dependency
        lse = q.new_zeros(q.size(1), q.size(0)) + dependency
        return output, lse

    monkeypatch.setattr(
        pruned_yoco_attn,
        'flash_attn_cute_varlen_func',
        fake_varlen_attention,
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

    assert len(calls) == 2
    assert [call[0] for call in calls] == [(1, 4, 8), (1, 4, 8)]
    assert [call[1] for call in calls] == [(2, 2, 8), (4, 2, 8)]
    for _, _, kwargs in calls:
        assert kwargs['causal'] is True
        assert 'cu_seqlens_q' in kwargs
        assert 'cu_seqlens_k' in kwargs
        assert 'mask_mod' not in kwargs
        assert 'aux_tensors' not in kwargs
    assert lse.shape == (4, 2)
    assert torch.count_nonzero(output[~query_mask]) == 0
    output.sum().backward()
    assert q.grad is not None
    assert k.grad is not None
    assert v.grad is not None


@pytest.mark.parametrize('pattern, expected_calls', [
    ('all', 1), ('suffix', 1), ('two_runs', 2),
])
def test_packed_queries_are_batched_without_copying_kv(monkeypatch, pattern, expected_calls):
    tokens, length = 256, 8
    q = torch.randn(tokens, 4, 8, requires_grad=True)
    k = torch.randn(tokens, 2, 8, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    positions = torch.arange(length, dtype=torch.int32).repeat(tokens // length)
    mask = torch.ones(tokens, dtype=torch.bool)
    if pattern == 'suffix':
        mask = positions >= 4
    elif pattern == 'two_runs':
        mask = ((positions >= 1) & (positions < 3)) | (positions >= 5)
        mask[8:16] = False  # Include a completely pruned packed sequence.
    cu = torch.arange(0, tokens + 1, length, dtype=torch.int32)
    calls = []

    def fake_attention(batch_q, batch_k, batch_v, **kwargs):
        calls.append(batch_q.size(0))
        assert batch_k.untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
        assert batch_v.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
        assert kwargs['causal'] is True
        assert 'mask_mod' not in kwargs
        # Fully pruned sequences between runs must still respect the original
        # maximum K length; merging them into a long dummy sequence is invalid.
        assert torch.diff(kwargs['cu_seqlens_k']).max() <= length
        if pattern == 'all':
            assert batch_q is q
            assert batch_k is k
            assert batch_v is v
            torch.testing.assert_close(kwargs['cu_seqlens_q'], cu)
            torch.testing.assert_close(kwargs['cu_seqlens_k'], cu)
        dependency = (batch_k[0, 0, 0] + batch_v[0, 0, 0]) * 0
        return batch_q + dependency, batch_q[:, :, 0].T + dependency

    monkeypatch.setattr(pruned_yoco_attn, 'flash_attn_cute_varlen_func', fake_attention)
    actual, lse = wrap_pruned_yoco_attn_varlen_func(
        q, k, v, mask, positions, cu, cu, None, enable_ring=False,
    )
    assert len(calls) == expected_calls
    assert sum(calls) == mask.sum().item()
    torch.testing.assert_close(actual[mask], q[mask])
    assert torch.count_nonzero(actual[~mask]) == 0
    torch.testing.assert_close(lse, q[mask, :, 0].T)
    actual.sum().backward()
    torch.testing.assert_close(q.grad, mask[:, None, None].expand_as(q).float())
    assert k.grad is not None and v.grad is not None


def test_query_batch_cache_tracks_inplace_metadata_changes(monkeypatch):
    q = torch.randn(8, 4, 8)
    k = torch.randn(12, 2, 8)
    v = torch.randn_like(k)
    mask = torch.ones(8, dtype=torch.bool)
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32)
    cu_q = torch.tensor([0, 4, 8], dtype=torch.int32)
    cu_k = torch.tensor([0, 6, 12], dtype=torch.int32)
    calls = []

    def fake_attention(batch_q, batch_k, batch_v, **kwargs):
        calls.append(kwargs['cu_seqlens_k'])
        return batch_q, batch_q[:, :, 0].T

    monkeypatch.setattr(pruned_yoco_attn, 'flash_attn_cute_varlen_func', fake_attention)

    def run():
        return wrap_pruned_yoco_attn_varlen_func(
            q, k, v, mask, positions, cu_q, cu_k, None, enable_ring=False,
        )[0]

    run()
    run()
    assert calls[0] is calls[1]
    cu_k[1] = 5
    run()
    assert calls[-1].tolist() == [0, 4, 5, 9]
    positions.add_(1)
    run()
    assert calls[-1].tolist() == [0, 5, 10]
    mask[:4] = False
    actual = run()
    assert torch.count_nonzero(actual[:4]) == 0
    torch.testing.assert_close(actual[4:], q[4:])


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
    reason='FA4 query-pruned attention requires Blackwell',
)
@pytest.mark.parametrize('window_size', [(-1, -1), (2, 0)])
@pytest.mark.parametrize('pattern', ['sparse', 'all', 'suffix', 'empty_sequence'])
def test_pruned_attention_matches_dense_reference(window_size, pattern):
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
    if pattern == 'all':
        query_mask.fill_(True)
    elif pattern == 'suffix':
        query_mask = positions >= 2
    elif pattern == 'empty_sequence':
        query_mask[:5] = False
    q = torch.randn(
        12, 4, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
    k = torch.randn(
        12, 2, 128, dtype=torch.bfloat16, device=device, requires_grad=True)
    v = torch.randn(
        12, 2, 128, dtype=torch.bfloat16, device=device, requires_grad=True)

    actual, actual_lse = wrap_pruned_yoco_attn_varlen_func(
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
    expected, expected_lse = flash_attn_varlen_func(
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
    torch.testing.assert_close(actual_lse, expected_lse[:, query_mask])
    assert torch.count_nonzero(actual[~query_mask]) == 0

    grad = torch.randn_like(actual)
    actual.backward(grad, retain_graph=True)
    actual_grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())
    q.grad = k.grad = v.grad = None
    expected[query_mask].backward(grad[query_mask])
    for actual_grad, expected_grad in zip(actual_grads, (q.grad, k.grad, v.grad)):
        # Splitting non-consecutive queries into standard causal calls changes
        # BF16 accumulation order slightly.  This still rejects the broken
        # mask_mod backward, whose max error is O(1e2) for this fixture.
        torch.testing.assert_close(
            actual_grad.float().norm(),
            expected_grad.float().norm(),
            atol=2e-3,
            rtol=2e-3,
        )
        torch.testing.assert_close(
            actual_grad, expected_grad, atol=2e-2, rtol=2e-2)

    # Reuse the same metadata for inference, including the tensor-only API.
    with torch.no_grad():
        inference_output = wrap_pruned_yoco_attn_varlen_func(
            q, k, v, query_mask, positions, cu, cu, None,
            window_size=window_size, enable_ring=False, return_lse=False,
        )
    torch.testing.assert_close(inference_output, actual)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 10,
    reason='FA4 query-pruned attention requires Blackwell',
)
def test_batched_mha_zeroes_kv_gradients_across_fully_pruned_sequences():
    torch.manual_seed(61)
    device = torch.device('cuda')
    lengths = (128, 256, 256, 128)
    tokens = sum(lengths)
    cu = torch.tensor([0, 128, 384, 640, tokens], dtype=torch.int32, device=device)
    positions = torch.cat([torch.arange(n, device=device) for n in lengths]).int()
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    mask[3:6] = True
    mask[698:701] = True
    q, k, v = [torch.randn(
        tokens, 2, 128, dtype=torch.bfloat16, device=device, requires_grad=True,
    ) for _ in range(3)]
    actual, _ = wrap_pruned_yoco_attn_varlen_func(
        q, k, v, mask, positions, cu, cu, None, enable_ring=False,
        max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
    )
    expected, _ = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
        causal=True, return_lse=True,
    )
    torch.testing.assert_close(actual[mask], expected[mask])
    grad = torch.randn_like(q)
    actual.backward(grad)
    actual_grads = [t.grad.clone() for t in (q, k, v)]
    q.grad = k.grad = v.grad = None
    expected[mask].backward(grad[mask])
    for actual_grad, tensor in zip(actual_grads, (q, k, v)):
        torch.testing.assert_close(actual_grad, tensor.grad, atol=2e-2, rtol=2e-2)
        assert torch.count_nonzero(actual_grad[128:640]) == 0


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


def _cp2_packed_query_batches_worker():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    rank = dist.get_rank()
    DeviceGroup().get_group([0, 1])
    lengths = (8, 12, 20)
    tokens = sum(lengths)
    cu = torch.tensor([0, 8, 20, tokens], dtype=torch.int32, device=device)
    positions = torch.cat([torch.arange(n, device=device) for n in lengths]).int()

    def query_indices(cp_rank):
        indices = []
        offset = 0
        for length in lengths:
            size = length // 4
            for chunk in (cp_rank, 3 - cp_rank):
                indices.append(torch.arange(
                    offset + chunk * size, offset + (chunk + 1) * size, device=device))
            offset += length
        return torch.cat(indices)

    local_idx = query_indices(rank)
    local_kv_slice = slice(rank * tokens // 2, (rank + 1) * tokens // 2)
    torch.manual_seed(57)
    global_q = torch.randn(tokens, 4, 128, dtype=torch.bfloat16, device=device)
    global_k = torch.randn(tokens, 2, 128, dtype=torch.bfloat16, device=device)
    global_v = torch.randn_like(global_k)
    global_grad = torch.randn_like(global_q)

    for gathered in (True, False):
        for window_size in ((-1, -1), (2, 0)):
            for pattern in ('all', 'suffix', 'empty_rank'):
                mask = torch.ones(tokens, dtype=torch.bool, device=device)
                if pattern == 'suffix':
                    mask = positions >= 2
                    mask[8:20] = False
                elif pattern == 'empty_rank':
                    mask.fill_(False)
                    mask[query_indices(0)] = True
                local_mask = mask[local_idx]
                q = global_q[local_idx].detach().requires_grad_(True)
                local_k = global_k[local_kv_slice].clone().requires_grad_(True)
                local_v = global_v[local_kv_slice].clone().requires_grad_(True)
                k, v = local_k, local_v
                if gathered:
                    k, v = wrap_yoco_kv_allgather(
                        k, v, process_group=(0, 1), cp_size=2)
                actual, actual_lse = wrap_pruned_yoco_attn_varlen_func(
                    q, k, v, local_mask, positions[local_idx], cu, cu, None,
                    process_group=(0, 1), cp_size=2, kv_is_gathered=gathered,
                    window_size=window_size,
                )
                rq = global_q.clone().requires_grad_(True)
                rk = global_k.clone().requires_grad_(True)
                rv = global_v.clone().requires_grad_(True)
                reference, ref_lse = flash_attn_varlen_func(
                    rq, rk, rv, cu_seqlens_q=cu, cu_seqlens_k=cu,
                    max_seqlen_q=max(lengths), max_seqlen_k=max(lengths),
                    causal=True, return_lse=True,
                    window_size=tuple(None if n == -1 else n for n in window_size),
                )
                torch.testing.assert_close(
                    actual[local_mask], reference[local_idx][local_mask])
                assert torch.count_nonzero(actual[~local_mask]) == 0
                if local_mask.any():
                    torch.testing.assert_close(actual_lse, ref_lse[:, local_idx][:, local_mask])
                else:
                    assert actual_lse is None
                actual.backward(global_grad[local_idx])
                reference[mask].backward(global_grad[mask])
                for actual_grad, ref_grad in (
                    (q.grad, rq.grad[local_idx]),
                    (local_k.grad, rk.grad[local_kv_slice]),
                    (local_v.grad, rv.grad[local_kv_slice]),
                ):
                    torch.testing.assert_close(actual_grad, ref_grad, atol=2e-2, rtol=2e-2)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2
    or torch.cuda.get_device_capability()[0] < 10,
    reason='requires two Blackwell GPUs',
)
def test_cp2_packed_query_batches_match_dense_backward():
    partial(torchrun, 2, _cp2_packed_query_batches_worker)()


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
