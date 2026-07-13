import pytest
import torch

from nnscaler.customized_ops.ring_attention.ring_attn_varlen import (
    emit_ring as emit_varlen_ring,
)
from nnscaler.customized_ops.ring_attention.sliding_window_attn import (
    emit_ring as emit_sliding_window_ring,
)
from nnscaler.customized_ops.ring_attention.varlen_utils import (
    select_sequence_group_cu_seqlens,
)


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
