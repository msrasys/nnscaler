import tempfile

import pytest
import torch
from torch import nn

from nnscaler.codegen.emit import FuncEmission
from nnscaler.customized_ops.ring_attention.maybe_shuffle import (
    _profile_tensor,
    emit_ring as emit_shuffle,
    wrap_maybe_shuffle,
    wrap_maybe_unshuffle,
)
from nnscaler.customized_ops.ring_attention.ring_attn_varlen import (
    emit_ring as emit_attention,
    wrap_ring_attn_varlen_func,
)
from nnscaler.customized_ops.ring_attention.sliding_window_attn import (
    emit_ring as emit_sliding_window,
)
from nnscaler.customized_ops.ring_attention.zigzag_allgather_attn_varlen import (
    emit_ring as emit_zigzag_allgather,
)
from nnscaler.graph.parser.converter import convert_model
from nnscaler.ir.operator import IRFwOperation


EMITTERS = [
    emit_attention,
    emit_sliding_window,
    emit_zigzag_allgather,
    emit_shuffle,
]


@pytest.mark.parametrize(
    ('dtype', 'requires_grad'),
    [
        (torch.bool, False),
        (torch.int64, False),
        (torch.float32, True),
    ],
)
def test_maybe_shuffle_profile_tensor_supports_input_dtype(
    dtype, requires_grad
):
    tensor = _profile_tensor(
        (4, 8), dtype, torch.device('cpu'), requires_grad)

    assert tensor.shape == (4, 8)
    assert tensor.dtype == dtype
    assert tensor.requires_grad is requires_grad


class _FakeTensor:

    def __init__(self, shape, parent=None):
        self.shape = shape
        self.parent = parent


class _FakeNode:
    signature = 'test.ring'

    def __init__(
        self,
        local_shape,
        full_shape,
        cp_size_marker=...,
        require_full_partition_marker=...,
    ):
        full = _FakeTensor(full_shape)
        self._inputs = [_FakeTensor(local_shape, parent=full)]
        self.kwargs = {}
        if cp_size_marker is not ...:
            self.kwargs['cp_size'] = cp_size_marker
        if require_full_partition_marker is not ...:
            self.kwargs['require_full_plan_sequence_partition'] = (
                require_full_partition_marker
            )

    def inputs(self):
        return self._inputs


class _RingModule(nn.Module):

    def forward(self, q, k, v, cu_seqlens):
        return wrap_ring_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            None,
            causal=True,
            cp_size=2,
            require_full_plan_sequence_partition=True,
        )


@pytest.mark.parametrize('emitter', EMITTERS)
def test_explicit_cp_size_overrides_full_token_partition_degree(emitter):
    # The token tensor is sharded over the full 8-rank plan, while ring
    # communication must stay inside the explicit 2-rank CP subgroup.
    node = _FakeNode((16, 8), (128, 8), cp_size_marker=2)

    emitted = emitter(
        node,
        args=['x'],
        kwargs={'cp_size': '2'},
        runtime_devid=5,
        plan_ndevs=8,
        runtime_ndevs=8,
    )

    assert 'cp_size=2' in emitted
    assert 'process_group=[4, 5]' in emitted
    assert 'process_group=[0, 1, 2, 3, 4, 5, 6, 7]' not in emitted


@pytest.mark.parametrize('emitter', EMITTERS)
def test_explicit_cp_group_stays_in_runtime_scale_unit(emitter):
    node = _FakeNode((16, 8), (128, 8), cp_size_marker=2)

    emitted = emitter(
        node,
        args=['x'],
        kwargs={'cp_size': '2'},
        runtime_devid=10,
        plan_ndevs=8,
        runtime_ndevs=16,
    )

    assert 'process_group=[10, 11]' in emitted


@pytest.mark.parametrize('emitter', EMITTERS)
def test_cp_one_disables_codegen_group(emitter):
    node = _FakeNode((16, 8), (128, 8), cp_size_marker=1)

    emitted = emitter(
        node,
        args=['x'],
        kwargs={'cp_size': '1'},
        runtime_devid=5,
        plan_ndevs=8,
        runtime_ndevs=8,
    )

    assert 'process_group=None' in emitted
    assert 'process_group=[' not in emitted


@pytest.mark.parametrize('emitter', EMITTERS)
def test_missing_cp_size_preserves_legacy_shard_inference(emitter):
    node = _FakeNode((16, 8), (128, 8))

    emitted = emitter(
        node,
        args=['x'],
        kwargs={},
        runtime_devid=5,
        plan_ndevs=8,
        runtime_ndevs=8,
    )

    assert 'process_group=[0, 1, 2, 3, 4, 5, 6, 7]' in emitted


@pytest.mark.parametrize('emitter', EMITTERS)
def test_explicit_cp_size_must_divide_plan(emitter):
    node = _FakeNode((16, 8), (128, 8), cp_size_marker=3)

    with pytest.raises(ValueError, match='must divide plan size'):
        emitter(
            node,
            args=['x'],
            kwargs={'cp_size': '3'},
            runtime_devid=0,
            plan_ndevs=8,
            runtime_ndevs=8,
        )


@pytest.mark.parametrize('emitter', EMITTERS)
@pytest.mark.parametrize(
    ('local_shape', 'full_shape', 'error'),
    [
        ((128, 8), (128, 8), 'requires partitioning the sequence dimension'),
        ((128, 4), (128, 8), 'not head-dimension partitioning'),
        ((64, 8), (128, 8), 'must equal plan size'),
    ],
)
def test_explicit_cp_fails_closed_for_incompatible_partition(
    emitter, local_shape, full_shape, error
):
    node = _FakeNode(local_shape, full_shape, cp_size_marker=2)

    with pytest.raises(ValueError, match=error):
        emitter(
            node,
            args=['x'],
            kwargs={'cp_size': '2'},
            runtime_devid=0,
            plan_ndevs=8,
            runtime_ndevs=8,
        )


@pytest.mark.parametrize('emitter', EMITTERS)
@pytest.mark.parametrize(
    ('local_shape', 'full_shape'),
    [
        ((128, 8), (128, 8)),
        ((128, 4), (128, 8)),
    ],
)
def test_cp_one_preserves_local_or_head_partition_codegen(
    emitter, local_shape, full_shape
):
    node = _FakeNode(local_shape, full_shape, cp_size_marker=1)

    emitted = emitter(
        node,
        args=['x'],
        kwargs={'cp_size': '1'},
        runtime_devid=0,
        plan_ndevs=8,
        runtime_ndevs=8,
    )

    assert 'process_group=None' in emitted


@pytest.mark.parametrize('emitter', EMITTERS)
def test_cp_one_data_lane_layout_accepts_full_plan_token_partition(emitter):
    node = _FakeNode(
        (16, 8),
        (128, 8),
        cp_size_marker=1,
        require_full_partition_marker=True,
    )

    emitted = emitter(
        node,
        args=['x'],
        kwargs={
            'cp_size': '1',
            'require_full_plan_sequence_partition': 'True',
        },
        runtime_devid=5,
        plan_ndevs=8,
        runtime_ndevs=8,
    )

    assert 'require_full_plan_sequence_partition=True' in emitted
    assert 'process_group=None' in emitted


@pytest.mark.parametrize('emitter', EMITTERS)
@pytest.mark.parametrize(
    ('local_shape', 'full_shape', 'error'),
    [
        ((128, 8), (128, 8), 'requires partitioning the sequence dimension'),
        ((128, 4), (128, 8), 'not head-dimension partitioning'),
        ((64, 8), (128, 8), 'must equal plan size'),
    ],
)
def test_cp_one_data_lane_layout_fails_closed_for_incompatible_partition(
    emitter, local_shape, full_shape, error
):
    node = _FakeNode(
        local_shape,
        full_shape,
        cp_size_marker=1,
        require_full_partition_marker=True,
    )

    with pytest.raises(ValueError, match=error):
        emitter(
            node,
            args=['x'],
            kwargs={
                'cp_size': '1',
                'require_full_plan_sequence_partition': 'True',
            },
            runtime_devid=0,
            plan_ndevs=8,
            runtime_ndevs=8,
        )


@pytest.mark.parametrize('wrapper', [wrap_maybe_shuffle, wrap_maybe_unshuffle])
@pytest.mark.parametrize('process_group', [(0,), [0]])
def test_cp_one_disables_shuffle_runtime(wrapper, process_group):
    hidden = torch.randn(4, 8)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)

    result = wrapper(
        hidden,
        cu_seqlens,
        enable_ring=True,
        process_group=process_group,
        cp_size=1,
    )

    assert result is hidden


@pytest.mark.skipif(not torch.cuda.is_available(), reason='parser executes FlashAttention')
def test_cp_size_survives_parser_partition_and_codegen():
    q = torch.randn(16, 2, 8, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 16], dtype=torch.int32)
    with tempfile.TemporaryDirectory() as savedir:
        graph = convert_model(
            _RingModule(),
            {'q': q, 'k': q, 'v': q, 'cu_seqlens': cu_seqlens},
            savedir,
            constant_folding=False,
        )

    node = next(
        node
        for node in graph.select(ntype=IRFwOperation)
        if node.fn is wrap_ring_attn_varlen_func
    )
    assert node.kwargs['cp_size'] == 2
    assert node.kwargs['require_full_plan_sequence_partition'] is True

    sub_nodes = graph.partition(
        node,
        node.algorithm('dim'),
        idx=0,
        dim=0,
        num=8,
    )
    assert all(sub_node.kwargs['cp_size'] == 2 for sub_node in sub_nodes)
    assert all(
        sub_node.kwargs['require_full_plan_sequence_partition'] is True
        for sub_node in sub_nodes
    )

    emitted = FuncEmission().emit_fnode(
        sub_nodes[5],
        runtime_devid=5,
        plan_ndevs=8,
        runtime_ndevs=8,
    )[-1]
    assert 'cp_size=2' in emitted
    assert 'require_full_plan_sequence_partition=True' in emitted
    assert 'process_group=[4, 5]' in emitted
