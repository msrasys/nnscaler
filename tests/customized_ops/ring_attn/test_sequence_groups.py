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
