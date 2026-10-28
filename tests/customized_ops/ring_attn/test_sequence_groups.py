import os

import nnscaler
import pytest
import torch
import torch.distributed as dist

from nnscaler.customized_ops.ring_attention.ring_attn_varlen import (
    emit_ring as emit_varlen_ring,
    wrap_ring_attn_varlen_func,
)
from nnscaler.customized_ops.ring_attention.sliding_window_attn import (
    emit_ring as emit_sliding_window_ring,
    wrap_sliding_window_attn_func,
)
from nnscaler.customized_ops.ring_attention.varlen_utils import (
    select_sequence_group_cu_seqlens,
)
from nnscaler.runtime.device import DeviceGroup
from tests.launch_torchrun import launch_torchrun


@pytest.mark.parametrize(
    ('process_group', 'expected'),
    [
        ([0, 1], [0, 4, 16]),
        ([2, 3], [0, 8, 16]),
        ([4, 5], [0, 1, 16]),
        ([6, 7], [0, 8, 16]),
        ([8, 9], [0, 4, 16]),
    ],
)
def test_select_sequence_group_cu_seqlens(process_group, expected):
    combined = torch.tensor(
        [0, 4, 16, 24, 32, 33, 48, 56, 64], dtype=torch.int32)

    selected = select_sequence_group_cu_seqlens(
        combined, process_group, sequence_group_count=4)

    torch.testing.assert_close(selected, torch.tensor(expected, dtype=torch.int32))


def test_select_sequence_group_requires_packed_boundary_alignment():
    combined = torch.tensor([0, 7, 17, 32], dtype=torch.int32)

    with pytest.raises(ValueError, match='must align'):
        select_sequence_group_cu_seqlens(
            combined, [0, 1], sequence_group_count=2)


@pytest.mark.parametrize('count', [0, -1])
def test_sequence_group_count_must_be_positive(count):
    boundaries = torch.tensor([0, 16], dtype=torch.int32)
    with pytest.raises(ValueError, match='sequence_group_count must be >= 1'):
        select_sequence_group_cu_seqlens(boundaries, [0], count)
    q = torch.randn(16, 2, 8)
    for wrapper in (wrap_ring_attn_varlen_func, wrap_sliding_window_attn_func):
        with pytest.raises(ValueError, match='sequence_group_count must be >= 1'):
            wrapper(q, q, q, boundaries, boundaries, None, sequence_group_count=count)


@pytest.mark.parametrize(
    ('target', 'wrapper'),
    [
        (
            'nnscaler.customized_ops.ring_attention.ring_attn_varlen.'
            'flash_attn_varlen_func',
            wrap_ring_attn_varlen_func,
        ),
        (
            'nnscaler.customized_ops.ring_attention.sliding_window_attn.'
            'flash_attn_varlen_func',
            wrap_sliding_window_attn_func,
        ),
    ],
)
def test_single_rank_sequence_group_selects_local_cu_seqlens(
    monkeypatch, target, wrapper,
):
    captured = {}

    def flash_attn(q, _k, _v, cu_q, cu_k, max_q, max_k, **_kwargs):
        captured.update(cu_q=cu_q, cu_k=cu_k, max_q=max_q, max_k=max_k)
        return q

    monkeypatch.setattr(target, flash_attn)
    q = torch.randn(16, 2, 8)
    combined = torch.tensor([0, 4, 16, 24, 32], dtype=torch.int32)

    output = wrapper(
        q, q, q, combined, combined, None,
        process_group=[1],
        sequence_parallel_size=1,
        sequence_group_count=2,
        max_seqlen_q=32,
        max_seqlen_k=32,
    )

    assert output is q
    expected = torch.tensor([0, 8, 16], dtype=torch.int32)
    torch.testing.assert_close(captured['cu_q'], expected)
    torch.testing.assert_close(captured['cu_k'], expected)
    assert captured['max_q'] == 8
    assert captured['max_k'] == 8

    # Unsplit metadata keeps caller-supplied maximum bounds on the fast path.
    wrapper(q, q, q, expected, expected, None, process_group=[0],
            sequence_parallel_size=1, sequence_group_count=1,
            max_seqlen_q=16, max_seqlen_k=16)
    assert captured['max_q'] == captured['max_k'] == 16


class _FakeTensor:
    def __init__(self, shape, parent=None):
        self.shape = shape
        self.parent = parent


class _FakeNode:
    signature = 'attention'

    def __init__(self):
        full = _FakeTensor((16, 8, 64))
        self._input = _FakeTensor((4, 8, 64), parent=full)

    def inputs(self):
        return [self._input]


@pytest.mark.parametrize('emit', [emit_varlen_ring, emit_sliding_window_ring])
@pytest.mark.parametrize(
    ('runtime_devid', 'expected_group'),
    [(0, '[0, 1]'), (1, '[0, 1]'), (2, '[2, 3]'), (3, '[2, 3]'),
     (4, '[4, 5]'), (6, '[6, 7]')],
)
def test_emit_ring_uses_requested_sequence_subgroup(
    emit, runtime_devid, expected_group,
):
    code = emit(
        _FakeNode(),
        args=['q', 'k', 'v'],
        kwargs={'sequence_parallel_size': '2', 'sequence_group_count': '2'},
        runtime_devid=runtime_devid,
        plan_ndevs=4,
        runtime_ndevs=8,
    )

    assert f'process_group={expected_group}' in code


@pytest.mark.parametrize('emit', [emit_varlen_ring, emit_sliding_window_ring])
def test_emit_ring_treats_explicit_none_as_default_group_size(emit):
    code = emit(
        _FakeNode(),
        args=['q', 'k', 'v'],
        kwargs={'sequence_parallel_size': 'None'},
        runtime_devid=0,
        plan_ndevs=4,
        runtime_ndevs=4,
    )

    assert 'process_group=[0, 1, 2, 3]' in code


@pytest.mark.parametrize('emit', [emit_varlen_ring, emit_sliding_window_ring])
@pytest.mark.parametrize('count', ['0', '-1', '1', '3'])
def test_emit_ring_rejects_mismatched_metadata_group_count(emit, count):
    with pytest.raises(ValueError, match='sequence_group_count must equal'):
        emit(
            _FakeNode(), args=['q', 'k', 'v'],
            kwargs={'sequence_parallel_size': '2', 'sequence_group_count': count},
            runtime_devid=0, plan_ndevs=4, runtime_ndevs=4,
        )


def _sequence_group_attention_worker():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    nnscaler.init()

    groups = ([0, 1], [2, 3])
    for ranks in groups:
        DeviceGroup().get_group(ranks)

    rank = dist.get_rank()
    group_index = rank // 2
    rank_in_group = rank % 2
    process_group = list(groups[group_index])
    cu_seqlens_by_group = ([0, 4, 16], [0, 8, 16])
    combined_cu_seqlens = torch.tensor(
        [0, 4, 16, 24, 32], dtype=torch.int32, device=local_rank)

    torch.manual_seed(1234 + group_index)
    shape = (16, 2, 32)
    q_data = torch.randn(shape, dtype=torch.bfloat16, device=local_rank)
    k_data = torch.randn(shape, dtype=torch.bfloat16, device=local_rank)
    v_data = torch.randn(shape, dtype=torch.bfloat16, device=local_rank)
    q_ref = q_data.detach().clone().requires_grad_()
    k_ref = k_data.detach().clone().requires_grad_()
    v_ref = v_data.detach().clone().requires_grad_()

    local_slice = slice(rank_in_group * 8, (rank_in_group + 1) * 8)
    q_local = q_data[local_slice].detach().clone().requires_grad_()
    k_local = k_data[local_slice].detach().clone().requires_grad_()
    v_local = v_data[local_slice].detach().clone().requires_grad_()
    group_cu_seqlens = torch.tensor(
        cu_seqlens_by_group[group_index], dtype=torch.int32, device=local_rank)

    reference = wrap_ring_attn_varlen_func(
        q_ref, k_ref, v_ref, group_cu_seqlens, group_cu_seqlens, None,
        causal=True, deterministic=True,
    )
    actual = wrap_ring_attn_varlen_func(
        q_local, k_local, v_local,
        combined_cu_seqlens, combined_cu_seqlens, None,
        causal=True,
        deterministic=True,
        process_group=process_group,
        sequence_parallel_size=2,
        sequence_group_count=2,
    )
    torch.testing.assert_close(
        actual, reference[local_slice], atol=3e-2, rtol=3e-2)

    torch.manual_seed(4321 + group_index)
    output_grad = torch.randn_like(reference)
    reference.backward(output_grad)
    actual.backward(output_grad[local_slice])
    for local_grad, reference_grad in (
        (q_local.grad, q_ref.grad),
        (k_local.grad, k_ref.grad),
        (v_local.grad, v_ref.grad),
    ):
        torch.testing.assert_close(
            local_grad, reference_grad[local_slice], atol=3e-2, rtol=3e-2)

    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason='requires four CUDA devices',
)
def test_sequence_group_attention_forward_backward_4gpu():
    launch_torchrun(4, _sequence_group_attention_worker)


def _eager_packed_attention(q, k, v, bounds, window):
    outputs = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        query, key, value = q[start:end].float(), k[start:end].float(), v[start:end].float()
        scores = torch.einsum('qhd,khd->hqk', query, key) / query.shape[-1] ** 0.5
        position = torch.arange(end - start, device=q.device)
        distance = position[:, None] - position[None, :]
        allowed = distance >= 0
        if window >= 0:
            allowed = allowed & (distance <= window)
        scores = scores.masked_fill(~allowed[None], float('-inf'))
        outputs.append(torch.einsum('hqk,khd->qhd', scores.softmax(-1), value).to(q.dtype))
    return torch.cat(outputs)


def _singleton_sequence_group_worker():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    bounds = ([0, 4, 16], [0, 8, 16], [0, 2, 16], [0, 6, 16])[rank]
    combined = torch.tensor([0, 4, 16, 24, 32, 34, 48, 54, 64],
                            dtype=torch.int32, device=rank)
    for use_cute in [False, True]:
        for wrapper, window in [(wrap_ring_attn_varlen_func, -1),
                                (wrap_sliding_window_attn_func, 8)]:
            torch.manual_seed(631 + rank)
            inputs = [torch.randn(16, 2, 64, device=rank, dtype=torch.bfloat16,
                                  requires_grad=True) for _ in range(3)]
            refs = [x.detach().clone().requires_grad_() for x in inputs]
            expected = _eager_packed_attention(*refs, bounds, window)
            actual = wrapper(*inputs, combined, combined, None, causal=True,
                             window_size=(window, 0), deterministic=True,
                             use_cute=use_cute, process_group=[rank],
                             sequence_parallel_size=1, sequence_group_count=4,
                             max_seqlen_q=64, max_seqlen_k=64)
            output_grad = torch.randn_like(actual)
            actual.backward(output_grad)
            expected.backward(output_grad)
            errors = []
            for observed, reference in [(actual, expected),
                                        *[(x.grad, y.grad) for x, y in zip(inputs, refs)]]:
                errors.append((observed.float() - reference.float()).abs().max().item())
                torch.testing.assert_close(observed, reference, atol=3e-2, rtol=3e-2)
            print(f'singleton_attention rank={rank} cute={use_cute} window={window} max_abs={errors}')
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires four CUDA devices')
def test_singleton_sequence_group_attention_against_eager_4gpu():
    launch_torchrun(4, _singleton_sequence_group_worker)
