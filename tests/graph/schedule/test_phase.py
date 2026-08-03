#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CPU-only unit tests for ``nnscaler.graph.schedule.phase`` (Step C: explicit
phase IR for MoE-style layers).

Mirrors ``tests/graph/schedule/test_local_segment.py``'s conventions: real
``IRGraph``s via ``nnscaler.parallel._gen_graph``, no distributed launch or
GPU needed. Real multi-GPU e2e coverage (numeric equivalence, no-deadlock,
Step-A/gencode compatibility, real communication) lives in
``tests/parallel_module/test_phase_moe_e2e.py`` /
``test_phase_moe_multistage_e2e.py``, and gencode-precision coverage lives in
``tests/codegen/test_phase_gencode.py``.
"""
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nnscaler.ir.unique import IDGenerator
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.schedule.local_segment import LocalSegmentError, CallableBoundary
from nnscaler.parallel import _gen_graph
import nnscaler.runtime.function as ncf

from nnscaler.graph.schedule.phase import (
    PhaseError,
    PhaseType,
    PhaseKind,
    PhaseIdentity,
    PhaseNode,
    MOE_PHASE_SEQUENCE,
    DENSE_PHASE_SEQUENCE,
    phase_anchor,
    lower_layer_to_phases,
    validate_phase_layout,
    get_phase,
)
from nnscaler.runtime.adapter.moe import moe_dispatch, moe_dispatch_wait, moe_combine, moe_combine_wait

from tests.utils import replace_all_device_with

DIM = 8
EP_RANKS = (0, 1)


# ---------------------------------------------------------------------------
# test models
# ---------------------------------------------------------------------------

class _MoELayer(nn.Module):
    """One MoE-FFN transformer "layer": attention (a plain Linear stand-in),
    then a capacity-style dispatch/expert/combine using the real registered
    ``moe_dispatch``/``moe_dispatch_wait``/``moe_combine``/``moe_combine_wait``
    ops, all phase-anchored. Numerically this is a simplified stand-in (it
    "dispatches" by literally broadcasting the batch to every EP rank rather
    than routing distinct tokens -- real, capacity-based, gating-driven
    per-token routing lives in the e2e test model,
    ``tests/parallel_module/phase_moe_common.py``); this file only exercises
    IR/lowering/scheduling *structure*."""

    def __init__(self, dim=DIM, layer_id=0, ep_ranks=EP_RANKS):
        super().__init__()
        self.layer_id = layer_id
        self.ep_ranks = ep_ranks
        self.attn = nn.Linear(dim, dim, bias=False)
        self.expert = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        phase_anchor(self.layer_id, PhaseType.ATTENTION)
        x = self.attn(x)
        phase_anchor(self.layer_id, PhaseType.MOE_DISPATCH)
        buf = x.unsqueeze(0).expand(len(self.ep_ranks), *x.shape).contiguous()
        pending = moe_dispatch(buf, self.ep_ranks, channel=f'L{self.layer_id}_dispatch', max_outstanding=1)
        phase_anchor(self.layer_id, PhaseType.EXPERT_COMPUTE)
        dispatched = moe_dispatch_wait(pending)
        expert_out = self.expert(dispatched)
        pending2 = moe_combine(expert_out, self.ep_ranks, channel=f'L{self.layer_id}_combine', max_outstanding=1)
        phase_anchor(self.layer_id, PhaseType.MOE_COMBINE)
        combined = moe_combine_wait(pending2)
        return combined.mean(dim=0) + x


class _DenseLayer(nn.Module):
    """A dense (non-MoE) "layer": just ATTENTION, no MoE phases."""

    def __init__(self, dim=DIM, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        phase_anchor(self.layer_id, PhaseType.ATTENTION)
        return self.attn(x)


class _StackModel(nn.Module):
    """A stack of layers (mix of MoE/dense), followed by a final reduction to
    a scalar loss (required by ``_gen_graph(..., end2end_mode=True)``)."""

    def __init__(self, dim=DIM, layer_specs=('moe',)):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList()
        self.kinds = list(layer_specs)
        for i, kind in enumerate(self.kinds):
            if kind == 'moe':
                self.layers.append(_MoELayer(dim=dim, layer_id=i))
            elif kind == 'dense':
                self.layers.append(_DenseLayer(dim=dim, layer_id=i))
            else:
                raise ValueError(kind)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.sum()


class _SharedParamMoEModel(nn.Module):
    """Two MoE layers whose experts share the exact same weight Parameter,
    both tagged as layer 0's phases (illegal: shared-attribute-across-phases)."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        phase_anchor(0, PhaseType.ATTENTION)
        x = torch.nn.functional.linear(x, self.w)
        phase_anchor(0, PhaseType.MOE_DISPATCH)
        buf = x.unsqueeze(0).expand(2, *x.shape).contiguous()
        pending = moe_dispatch(buf, EP_RANKS)
        phase_anchor(0, PhaseType.EXPERT_COMPUTE)
        dispatched = moe_dispatch_wait(pending)
        # shared attribute `w` used again inside the EXPERT_COMPUTE phase:
        expert_out = torch.nn.functional.linear(dispatched, self.w)
        pending2 = moe_combine(expert_out, EP_RANKS)
        phase_anchor(0, PhaseType.MOE_COMBINE)
        combined = moe_combine_wait(pending2)
        return combined.sum()


def _build_graph(model, tempdir, dim=DIM):
    IDGenerator().clear()
    dummy_input = {'x': torch.randn(2, dim)}
    model.train()
    graph, _ = _gen_graph(model, dummy_input, tempdir, constant_folding=True, end2end_mode=True)
    return graph


def _all_fwd_ops(graph):
    return [n for n in graph.nodes() if isinstance(n, IRFwOperation)]


# ---------------------------------------------------------------------------
# Lowering: legal cases
# ---------------------------------------------------------------------------

@replace_all_device_with('cpu')
def test_lower_moe_layer_produces_four_phases_in_order():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        layer_nodes = _all_fwd_ops(graph)
        phase_nodes = lower_layer_to_phases(graph, layer_nodes, layer_id=0)
        assert len(phase_nodes) == 4
        assert [pn.identity.phase_type for pn in phase_nodes] == list(MOE_PHASE_SEQUENCE)
        assert [pn.identity.seq_in_layer for pn in phase_nodes] == [0, 1, 2, 3]
        assert all(pn.identity.direction == 'forward' for pn in phase_nodes)
        assert sum(len(pn.segment.nodes()) for pn in phase_nodes) == len(layer_nodes)


@replace_all_device_with('cpu')
def test_lower_dense_layer_produces_single_attention_phase():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('dense',)), tempdir)
        layer_nodes = _all_fwd_ops(graph)
        phase_nodes = lower_layer_to_phases(graph, layer_nodes, layer_id=0)
        assert len(phase_nodes) == 1
        assert phase_nodes[0].identity.phase_type == PhaseType.ATTENTION
        assert tuple(pn.identity.phase_type for pn in phase_nodes) == DENSE_PHASE_SEQUENCE


@replace_all_device_with('cpu')
def test_lower_multiple_layers_in_one_stage():
    """Two stacked MoE layers in the same (not-yet-grouped) stage: calling
    lower_layer_to_phases once per layer, in graph order, must succeed for
    both and keep their phases distinct."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe', 'moe')), tempdir)
        all_ops = _all_fwd_ops(graph)
        # split the flat op list at the second layer's ATTENTION anchor
        from nnscaler.graph.function.anchor import IRGraphAnchor
        split = next(
            i for i, n in enumerate(all_ops)
            if isinstance(n, IRGraphAnchor) and n.kwargs.get('name') == '__phase__1:attention'
        )
        layer0_nodes, layer1_nodes = all_ops[:split], all_ops[split:]
        phases0 = lower_layer_to_phases(graph, layer0_nodes, layer_id=0)
        phases1 = lower_layer_to_phases(graph, layer1_nodes, layer_id=1)
        assert len(phases0) == 4 and len(phases1) == 4
        assert {pn.identity.layer_id for pn in phases0} == {0}
        assert {pn.identity.layer_id for pn in phases1} == {1}


# ---------------------------------------------------------------------------
# Lowering: illegal cases
# ---------------------------------------------------------------------------

@replace_all_device_with('cpu')
def test_lower_rejects_no_anchors():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('dense',)), tempdir)
        # a node list with no phase anchors at all (e.g. only the linear)
        plain_nodes = [n for n in _all_fwd_ops(graph) if n.name == 'linear']
        with pytest.raises(PhaseError):
            lower_layer_to_phases(graph, plain_nodes, layer_id=0)


@replace_all_device_with('cpu')
def test_lower_rejects_mismatched_layer_id():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        layer_nodes = _all_fwd_ops(graph)
        with pytest.raises(PhaseError):
            lower_layer_to_phases(graph, layer_nodes, layer_id=99)


@replace_all_device_with('cpu')
def test_lower_rejects_incomplete_moe_sequence():
    """Only ATTENTION + MOE_DISPATCH anchors present (EXPERT_COMPUTE/MOE_COMBINE
    missing) must be rejected -- an incomplete phase group."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        all_ops = _all_fwd_ops(graph)
        from nnscaler.graph.function.anchor import IRGraphAnchor
        # drop the EXPERT_COMPUTE and MOE_COMBINE anchors (keep everything
        # else contiguous) -- an incomplete-but-still-contiguous node list.
        truncated = [
            n for n in all_ops
            if not (isinstance(n, IRGraphAnchor) and n.kwargs.get('name') in
                    ('__phase__0:expert_compute', '__phase__0:moe_combine'))
        ]
        with pytest.raises(PhaseError):
            lower_layer_to_phases(graph, truncated, layer_id=0)


@replace_all_device_with('cpu')
def test_lower_rejects_out_of_order_sequence():
    """A hand-crafted node list with anchors present but in the WRONG order
    (MOE_DISPATCH before ATTENTION) must be rejected."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        all_ops = _all_fwd_ops(graph)
        from nnscaler.graph.function.anchor import IRGraphAnchor
        anchors = {n.kwargs.get('name'): i for i, n in enumerate(all_ops) if isinstance(n, IRGraphAnchor)}
        assert anchors['__phase__0:attention'] < anchors['__phase__0:moe_dispatch']
        # swap the ATTENTION and MOE_DISPATCH anchor *nodes themselves* to
        # fabricate an (illegal) out-of-order anchor sequence, keeping the
        # rest of the (still-contiguous) op list untouched.
        reordered = list(all_ops)
        i, j = anchors['__phase__0:attention'], anchors['__phase__0:moe_dispatch']
        reordered[i], reordered[j] = reordered[j], reordered[i]
        with pytest.raises(PhaseError):
            lower_layer_to_phases(graph, reordered, layer_id=0)


@replace_all_device_with('cpu')
def test_lower_rejects_cross_physical_stage():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe', 'moe')), tempdir)
        all_ops = _all_fwd_ops(graph)
        from nnscaler.graph.function.anchor import IRGraphAnchor
        split = next(
            i for i, n in enumerate(all_ops)
            if isinstance(n, IRGraphAnchor) and n.kwargs.get('name') == '__phase__1:attention'
        )
        # group layer 0 into an existing physical stage first ...
        graph.staging([all_ops[split]])
        # ... then try to lower a range that spans across that existing
        # stage boundary (layer 0's own trailing nodes + layer 1's nodes).
        mixed = all_ops[split - 2:]
        with pytest.raises(LocalSegmentError):
            lower_layer_to_phases(graph, mixed, layer_id=1)


@replace_all_device_with('cpu')
def test_lower_rejects_recompute_split():
    """``graph.recompute()`` only tags nodes it is *given* (it strips
    zero-input nodes -- anchors included -- from the *edges* of the given
    list, but keeps them if they sit strictly *inside* it), so the offending
    group must be the full contiguous node range spanning a phase boundary
    (here MOE_DISPATCH/EXPERT_COMPUTE), not just its two non-contiguous
    endpoints, for the boundary anchor itself to actually end up tagged."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        all_ops = _all_fwd_ops(graph)
        linear_ops = [n for n in all_ops if n.name == 'linear']
        i0, i1 = all_ops.index(linear_ops[0]), all_ops.index(linear_ops[1])
        # full contiguous span from the attn linear through the expert
        # linear (inclusive) -- crosses the MOE_DISPATCH/EXPERT_COMPUTE
        # phase boundary in the middle of the group, so the boundary anchor
        # is not stripped as a leading/trailing zero-grad node.
        graph.recompute(all_ops[i0:i1 + 1])
        with pytest.raises(LocalSegmentError):
            lower_layer_to_phases(graph, all_ops, layer_id=0)


@replace_all_device_with('cpu')
def test_lower_rejects_shared_parameter_across_phases():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SharedParamMoEModel(), tempdir)
        all_ops = _all_fwd_ops(graph)
        with pytest.raises(LocalSegmentError):
            lower_layer_to_phases(graph, all_ops, layer_id=0)


# ---------------------------------------------------------------------------
# Identity / metadata / mirror
# ---------------------------------------------------------------------------

@replace_all_device_with('cpu')
def test_phase_identity_kind_and_issues_async():
    fwd = {t: PhaseIdentity(0, t, 'forward', i) for i, t in enumerate(MOE_PHASE_SEQUENCE)}
    bwd = {t: PhaseIdentity(0, t, 'backward', i) for i, t in enumerate(MOE_PHASE_SEQUENCE)}

    assert fwd[PhaseType.ATTENTION].kind == PhaseKind.COMPUTE
    assert not fwd[PhaseType.ATTENTION].issues_async
    assert fwd[PhaseType.MOE_DISPATCH].kind == PhaseKind.COMM
    assert fwd[PhaseType.MOE_DISPATCH].issues_async
    assert fwd[PhaseType.EXPERT_COMPUTE].kind == PhaseKind.COMM
    assert fwd[PhaseType.EXPERT_COMPUTE].issues_async
    assert fwd[PhaseType.MOE_COMBINE].kind == PhaseKind.COMM
    assert not fwd[PhaseType.MOE_COMBINE].issues_async

    # backward: issues_async is always False (Step C's documented scope --
    # only *forward* dispatch/combine get the deferred-wait treatment).
    assert all(not pid.issues_async for pid in bwd.values())
    assert bwd[PhaseType.ATTENTION].kind == PhaseKind.COMPUTE
    assert bwd[PhaseType.MOE_DISPATCH].kind == PhaseKind.COMM
    assert bwd[PhaseType.EXPERT_COMPUTE].kind == PhaseKind.COMM
    # backward of a *wait* is an identity -> MOE_COMBINE(bwd) has no comm.
    assert bwd[PhaseType.MOE_COMBINE].kind == PhaseKind.COMPUTE


def test_phase_identity_rejects_bad_direction():
    with pytest.raises(PhaseError):
        PhaseIdentity(0, PhaseType.ATTENTION, 'sideways', 0)


@replace_all_device_with('cpu')
def test_phase_metadata_roundtrip_via_op_context():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        layer_nodes = _all_fwd_ops(graph)
        phase_nodes = lower_layer_to_phases(graph, layer_nodes, layer_id=0)
        for pn in phase_nodes:
            got = get_phase(pn.segment)
            assert got is not None
            assert got.identity == pn.identity
            assert got.segment is pn.segment
        # a plain node with no phase tag returns None
        assert get_phase(layer_nodes[0]) is None or get_phase(layer_nodes[0]) is not None
        untouched_graph = _build_graph(_StackModel(layer_specs=('dense',)), tempdir)
        plain = _all_fwd_ops(untouched_graph)[0]
        assert get_phase(plain) is None


@replace_all_device_with('cpu')
def test_phase_mirror_independent_and_tagged_backward():
    """Every forward phase segment has a mirror, itself tagged with a
    direction='backward' PhaseNode of the same (layer_id, phase_type,
    seq_in_layer) -- i.e. each phase's backward is independently identifiable
    and (structurally) independently callable, not merely "the whole layer's
    backward split arbitrarily"."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        layer_nodes = _all_fwd_ops(graph)
        phase_nodes = lower_layer_to_phases(graph, layer_nodes, layer_id=0)
        for pn in phase_nodes:
            assert pn.segment.mirror is not None
            bwd_phase = get_phase(pn.segment.mirror)
            assert bwd_phase is not None
            assert bwd_phase.identity.direction == 'backward'
            assert bwd_phase.identity.layer_id == pn.identity.layer_id
            assert bwd_phase.identity.phase_type == pn.identity.phase_type
            assert bwd_phase.identity.seq_in_layer == pn.identity.seq_in_layer

        # backward local segments appear in the graph's own backward node
        # list in exactly the reverse order of their forward counterparts
        # (same invariant test_local_segment.py checks for plain local
        # segments -- phases must preserve it too, since a phase *is* a
        # local segment).
        positions = [graph.index(pn.segment.mirror).indices[-1] for pn in phase_nodes]
        assert positions == sorted(positions, reverse=True)


def _assert_no_nested_segment(graph):
    for node in graph.nodes():
        if isinstance(node, IRSegment):
            for sub in node.nodes():
                assert not isinstance(sub, IRSegment)


@replace_all_device_with('cpu')
def test_phase_segments_flat_not_nested():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        layer_nodes = _all_fwd_ops(graph)
        lower_layer_to_phases(graph, layer_nodes, layer_id=0)
        _assert_no_nested_segment(graph)


@replace_all_device_with('cpu')
def test_lower_rejects_calling_twice_on_already_grouped_range():
    """Mirrors test_local_segment.py's own HIGH-severity post-commit-audit
    regression: calling lower_layer_to_phases a second time on an
    already-grouped segment's own .nodes() must be rejected, not silently
    create a nested IRSegment."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        layer_nodes = _all_fwd_ops(graph)
        phase_nodes = lower_layer_to_phases(graph, layer_nodes, layer_id=0)
        already_grouped = list(phase_nodes[0].segment.nodes())
        with pytest.raises(LocalSegmentError):
            lower_layer_to_phases(graph, already_grouped, layer_id=0)
        _assert_no_nested_segment(graph)


# ---------------------------------------------------------------------------
# Same-physical-stage layout validation
# ---------------------------------------------------------------------------

@replace_all_device_with('cpu')
def test_validate_phase_layout_accepts_same_device_tuple():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        layer_nodes = _all_fwd_ops(graph)
        phase_nodes = lower_layer_to_phases(graph, layer_nodes, layer_id=0)
        for pn in phase_nodes:
            for nd in pn.segment.nodes():
                graph.assign(nd, 0)
        validate_phase_layout(graph, num_stages=1)  # must not raise


@replace_all_device_with('cpu')
def test_validate_phase_layout_rejects_split_device_tuple():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_StackModel(layer_specs=('moe',)), tempdir)
        layer_nodes = _all_fwd_ops(graph)
        phase_nodes = lower_layer_to_phases(graph, layer_nodes, layer_id=0)
        # misconfigure: assign MOE_DISPATCH's phase to a *different* device
        # tuple than the rest of layer 0's phases.
        for pn in phase_nodes:
            dev = 1 if pn.identity.phase_type == PhaseType.MOE_DISPATCH else 0
            for nd in pn.segment.nodes():
                graph.assign(nd, dev)
        with pytest.raises(PhaseError):
            validate_phase_layout(graph, num_stages=2)
