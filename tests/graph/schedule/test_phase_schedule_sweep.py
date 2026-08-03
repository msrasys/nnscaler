#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Schedule property-sweep tests for ``nnscaler.graph.schedule.phase``
(Step C), CPU-only (real ``IRGraph``/``SchedulePlan`` construction, no
distributed launch or GPU -- mirrors ``tests/graph/schedule/test_local_segment.py``'s
own scheduling tests, which likewise inspect
``SchedulePlan.all_blocks()``/``.start()`` directly rather than a full
codegen'd ``ExecutionPlan``; gencode-text-level precision is
``tests/codegen/test_phase_gencode.py``'s job instead).

The central, precisely-checked property (see
:func:`_dispatch_and_combine_windows`): for the pipeline's first stage
steady-state window, ``F(m+1)``'s ``MOE_DISPATCH`` issue must be placed
*before* ``F(m+1)``'s ``EXPERT_COMPUTE`` (the deferred consumer/wait), with
at least one *other* micro-batch's phase in between -- and, in the stronger
form checked by :func:`_dispatch_window_is_communication_free`, that
in-between phase is genuinely communication-free (``B(m)`` independent
compute) for the dispatch window specifically (see phase.py's module
docstring "Known limitations" for why only the dispatch window -- not the
combine window -- is guaranteed communication-free filler). Swept across
multiple micro-batch counts, stage counts, and (via ``layers_per_stage``)
multiple MoE layers stacked in one physical stage.
"""
import tempfile
from typing import List, Tuple

import pytest
import torch
import torch.nn as nn

from nnscaler.ir.unique import IDGenerator
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.schedule.predefined import PredefinedSched
from nnscaler.parallel import _gen_graph

from nnscaler.graph.schedule.phase import (
    PhaseType,
    PhaseIdentity,
    PhaseNode,
    PhaseAwareSched,
    PhaseError,
    PhaseExecutionIdentity,
    phase_execution_identity,
    lower_layer_to_phases,
    validate_phase_layout,
    get_phase,
    phase_anchor,
)
from nnscaler.runtime.adapter.moe import moe_dispatch, moe_dispatch_wait, moe_combine, moe_combine_wait

from tests.utils import replace_all_device_with

DIM = 8
EP_RANKS = (0, 1)


class _MoELayer(nn.Module):
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


class _IllegalSequenceLayer(nn.Module):
    """A ``_MoELayer`` variant that emits an EXPLICITLY-CONTROLLED,
    hard-coded (not looped -- looping over a list of ``PhaseType`` members
    stored as a module attribute was found, in this session, to interact
    badly with nnScaler's tracer, turning ``phase_type.value`` into the
    enum's ``str()`` repr instead of its real ``.value`` by the time
    ``phase_anchor`` runs; hard-coded literal calls, matching
    ``_MoELayer``'s own working pattern exactly, avoid that) sequence of
    phase anchors per named ``case`` -- so a caller can construct a MISSING
    (e.g. omit ``MOE_COMBINE``), DUPLICATE (e.g. ``ATTENTION`` twice), or
    OUT-OF-ORDER (e.g. ``EXPERT_COMPUTE`` before ``MOE_DISPATCH``) phase
    sequence for :func:`lower_layer_to_phases`'s ``_validate_sequence`` to
    reject. Always still performs the real dispatch/combine communication
    ops (so the graph itself is well-formed up to the anchor sequence
    issue), regardless of anchor order, so the ONLY thing under test is the
    anchor-sequence legality check itself, not some incidental graph
    -construction failure."""

    def __init__(self, case: str, dim=DIM, layer_id=0, ep_ranks=EP_RANKS):
        super().__init__()
        self.layer_id = layer_id
        self.ep_ranks = ep_ranks
        self.case = case
        self.attn = nn.Linear(dim, dim, bias=False)
        self.expert = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        lid = self.layer_id
        case = self.case
        if case == 'missing-combine':
            phase_anchor(lid, PhaseType.ATTENTION)
            phase_anchor(lid, PhaseType.MOE_DISPATCH)
            phase_anchor(lid, PhaseType.EXPERT_COMPUTE)
        elif case == 'missing-dispatch':
            phase_anchor(lid, PhaseType.ATTENTION)
            phase_anchor(lid, PhaseType.EXPERT_COMPUTE)
            phase_anchor(lid, PhaseType.MOE_COMBINE)
        elif case == 'duplicate-attention':
            phase_anchor(lid, PhaseType.ATTENTION)
            phase_anchor(lid, PhaseType.ATTENTION)
            phase_anchor(lid, PhaseType.MOE_DISPATCH)
            phase_anchor(lid, PhaseType.EXPERT_COMPUTE)
            phase_anchor(lid, PhaseType.MOE_COMBINE)
        elif case == 'duplicate-dispatch':
            phase_anchor(lid, PhaseType.ATTENTION)
            phase_anchor(lid, PhaseType.MOE_DISPATCH)
            phase_anchor(lid, PhaseType.MOE_DISPATCH)
            phase_anchor(lid, PhaseType.EXPERT_COMPUTE)
            phase_anchor(lid, PhaseType.MOE_COMBINE)
        elif case == 'out-of-order-expert-before-dispatch':
            phase_anchor(lid, PhaseType.ATTENTION)
            phase_anchor(lid, PhaseType.EXPERT_COMPUTE)
            phase_anchor(lid, PhaseType.MOE_DISPATCH)
            phase_anchor(lid, PhaseType.MOE_COMBINE)
        elif case == 'out-of-order-combine-before-expert':
            phase_anchor(lid, PhaseType.ATTENTION)
            phase_anchor(lid, PhaseType.MOE_DISPATCH)
            phase_anchor(lid, PhaseType.MOE_COMBINE)
            phase_anchor(lid, PhaseType.EXPERT_COMPUTE)
        elif case == 'out-of-order-attention-last':
            phase_anchor(lid, PhaseType.MOE_DISPATCH)
            phase_anchor(lid, PhaseType.EXPERT_COMPUTE)
            phase_anchor(lid, PhaseType.MOE_COMBINE)
            phase_anchor(lid, PhaseType.ATTENTION)
        else:
            raise ValueError(f'unknown illegal-sequence case {case!r}')
        x = self.attn(x)
        buf = x.unsqueeze(0).expand(len(self.ep_ranks), *x.shape).contiguous()
        pending = moe_dispatch(buf, self.ep_ranks, channel=f'L{self.layer_id}_dispatch', max_outstanding=1)
        dispatched = moe_dispatch_wait(pending)
        expert_out = self.expert(dispatched)
        pending2 = moe_combine(expert_out, self.ep_ranks, channel=f'L{self.layer_id}_combine', max_outstanding=1)
        combined = moe_combine_wait(pending2)
        return combined.mean(dim=0) + x



class _MultiStageMoEModel(nn.Module):
    """``num_stages`` physical stages, each with ``layers_per_stage`` stacked
    MoE layers (global, unique ``layer_id``s), followed by a scalar loss."""

    def __init__(self, num_stages: int, layers_per_stage: int, dim=DIM):
        super().__init__()
        self.num_stages = num_stages
        self.layers_per_stage = layers_per_stage
        total = num_stages * layers_per_stage
        self.layers = nn.ModuleList(_MoELayer(dim=dim, layer_id=i) for i in range(total))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.sum()


def _build_graph(model, tempdir, dim=DIM):
    IDGenerator().clear()
    dummy_input = {'x': torch.randn(2, dim)}
    model.train()
    graph, _ = _gen_graph(model, dummy_input, tempdir, constant_folding=True, end2end_mode=True)
    return graph


def _pas_phase_pipeline(graph, num_stages: int, layers_per_stage: int, nmicros: int):
    """Stage the graph into ``num_stages`` physical stages, each
    ``layers_per_stage`` MoE layers deep, lowering every layer to its 4
    phases (:func:`lower_layer_to_phases`) before assigning devices, then
    schedule with :meth:`PhaseAwareSched.sched_1f1b_phase_aware`.

    Returns ``(graph, phase_nodes_per_stage)`` where
    ``phase_nodes_per_stage[sid]`` is the flat, in-order list of that stage's
    ``PhaseNode``\\ s (``layers_per_stage * 4`` of them).
    """
    all_ops = [n for n in graph.nodes() if isinstance(n, IRFwOperation)]
    anchor_positions = {
        n.kwargs.get('name'): i for i, n in enumerate(all_ops) if isinstance(n, IRGraphAnchor)
    }

    total_layers = num_stages * layers_per_stage
    # global layer start positions: the position of layer L's ATTENTION anchor
    layer_starts = [anchor_positions[f'__phase__{lid}:attention'] for lid in range(total_layers)]

    dataloader = graph.nodes()[0]

    phase_nodes_per_stage: List[list] = []
    for sid in range(num_stages):
        stage_layer_ids = list(range(sid * layers_per_stage, (sid + 1) * layers_per_stage))
        stage_start = layer_starts[stage_layer_ids[0]]
        if sid == 0:
            stage_start = 0  # absorb any leading ops before the very first anchor
        stage_end = (
            layer_starts[stage_layer_ids[-1] + 1] if stage_layer_ids[-1] + 1 < total_layers else len(all_ops)
        )
        stage_phase_nodes = []
        cursor = stage_start
        for pos, lid in enumerate(stage_layer_ids):
            is_last_in_stage = (pos == len(stage_layer_ids) - 1)
            layer_end = stage_end if is_last_in_stage else layer_starts[lid + 1]
            layer_nodes = all_ops[cursor:layer_end]
            stage_phase_nodes += lower_layer_to_phases(graph, layer_nodes, layer_id=lid)
            cursor = layer_end
        phase_nodes_per_stage.append(stage_phase_nodes)

    sub_nodes = graph.replicate(dataloader, num_stages)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)
    for sid in range(num_stages):
        for pn in phase_nodes_per_stage[sid]:
            for node in pn.segment.nodes():
                graph.assign(node, sid)

    validate_phase_layout(graph, num_stages)
    sched = PhaseAwareSched.sched_1f1b_phase_aware(graph, nmicros, num_stages)
    return graph, phase_nodes_per_stage, sched


def _device0_phase_sequence(sched, device=0) -> List[Tuple[int, PhaseNode]]:
    """Device 0's (first stage's) phase blocks -- BOTH forward and backward
    (the steady-state interleave specifically alternates F(m+1)'s *forward*
    phases with B(m)'s *backward* ones, so backward phases must stay in this
    sequence for the issue/wait window checks below to see them) -- in
    schedule (start step) order, as ``(mid, PhaseNode)`` pairs."""
    blocks = [blk for blk in sched.all_blocks() if device in blk.device]
    blocks = [blk for blk in blocks if get_phase(blk.content) is not None]
    blocks.sort(key=lambda blk: sched.start(blk))
    return [(blk.mid, get_phase(blk.content)) for blk in blocks]


def _forward_only(seq):
    return [(mid, phase) for mid, phase in seq if phase.identity.direction == 'forward']


def _dispatch_and_combine_windows(seq):
    """Weaker property (always expected to hold whenever >= 2 micro-batches
    exist): for at least one (micro-batch, layer), MOE_DISPATCH's issue is
    placed strictly before EXPERT_COMPUTE (dispatch's consumer), with at
    least one *other* micro-batch's phase in between; and, independently,
    EXPERT_COMPUTE (combine's issuer) strictly before MOE_COMBINE (combine's
    consumer), likewise with an other-micro-batch phase in between.

    Only ever examines *forward* MOE_DISPATCH/EXPERT_COMPUTE/MOE_COMBINE
    (the ones that actually issue/wait -- see phase.py's ``issues_async``);
    ``seq`` itself still contains both directions, since it is exactly
    ``B(m)``'s *backward* phases that are expected to fill the window.

    Returns (found_dispatch_window, found_combine_window).
    """
    by_key = {}
    for i, (mid, phase) in enumerate(seq):
        key = (mid, phase.identity.direction, phase.identity.phase_type, phase.identity.layer_id)
        by_key.setdefault(key, []).append(i)

    found_dispatch_window = False
    found_combine_window = False
    for (mid, direction, ptype, lid), idxs in by_key.items():
        if direction != 'forward' or ptype != PhaseType.MOE_DISPATCH:
            continue
        dispatch_idx = idxs[0]
        expert_idxs = by_key.get((mid, 'forward', PhaseType.EXPERT_COMPUTE, lid))
        if not expert_idxs:
            continue
        expert_idx = expert_idxs[0]
        assert dispatch_idx < expert_idx, (mid, lid, seq)
        between = seq[dispatch_idx + 1:expert_idx]
        if any(m != mid for m, _ in between):
            found_dispatch_window = True

        combine_idxs = by_key.get((mid, 'forward', PhaseType.MOE_COMBINE, lid))
        if combine_idxs:
            combine_idx = combine_idxs[0]
            assert expert_idx < combine_idx, (mid, lid, seq)
            between2 = seq[expert_idx + 1:combine_idx]
            if any(m != mid for m, _ in between2):
                found_combine_window = True

    return found_dispatch_window, found_combine_window


def _dispatch_window_is_communication_free(seq) -> bool:
    """The STRONGER, honestly-scoped property (see phase.py's module
    docstring "Known limitations"): the dispatch window specifically
    contains at least one OTHER micro-batch's communication-FREE compute
    phase (not merely *some* other-microbatch phase, which could in
    principle itself be a communication phase)."""
    by_key = {}
    for i, (mid, phase) in enumerate(seq):
        key = (mid, phase.identity.direction, phase.identity.phase_type, phase.identity.layer_id)
        by_key.setdefault(key, []).append(i)


    for (mid, direction, ptype, lid), idxs in by_key.items():
        if direction != 'forward' or ptype != PhaseType.MOE_DISPATCH:
            continue
        dispatch_idx = idxs[0]
        expert_idxs = by_key.get((mid, 'forward', PhaseType.EXPERT_COMPUTE, lid))
        if not expert_idxs:
            continue
        between = seq[dispatch_idx + 1:expert_idxs[0]]
        if any(m != mid and not p.identity.has_communication for m, p in between):
            return True
    return False


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

_SWEEP_CONFIGS = [
    (num_stages, layers_per_stage, nmicros)
    for num_stages in (1, 2, 3)
    for layers_per_stage in (1, 2)
    for nmicros in (2, 3, 4, 6)
]

# K>1 (multiple MoE layers stacked in one physical stage, K=layers_per_stage
# here up to 4) explicitly COMBINED with multistage (num_stages > 1) --
# broader than _SWEEP_CONFIGS's own (1,2) x (1,2,3) grid specifically to
# stress the "K>1 AND multistage at once" interaction the remediation
# instructions call out by name (K=3/4 layers per stage were never swept
# together with num_stages>1 before this).
_K_MULTISTAGE_CONFIGS = [
    (num_stages, layers_per_stage, nmicros)
    for num_stages in (2, 3)
    for layers_per_stage in (3, 4)
    for nmicros in (3, 5)
]


def _assert_phase_schedule_properties(num_stages, layers_per_stage, nmicros):
    """Shared property-check body for both ``_SWEEP_CONFIGS`` and the
    K>1+multistage-focused ``_K_MULTISTAGE_CONFIGS``."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_MultiStageMoEModel(num_stages, layers_per_stage), tempdir)
        graph, phase_nodes_per_stage, sched = _pas_phase_pipeline(graph, num_stages, layers_per_stage, nmicros)
        assert sched.validate()

        seq = _device0_phase_sequence(sched)
        expected = nmicros * layers_per_stage * 4 * 2  # both directions, 4 phases each
        assert len(seq) == expected, (len(seq), expected, seq)
        assert len(_forward_only(seq)) == nmicros * layers_per_stage * 4

        found_dispatch, found_combine = _dispatch_and_combine_windows(seq)
        # A steady-state B(m)/F(m+k) coincidence (k == num_stages, see
        # sched_1f1b's own base formula) only exists at all when
        # nmicros > num_stages (otherwise every forward finishes during
        # warmup, with nothing left to interleave a later backward with --
        # true for plain sched_1f1b/LocalSegmentSched too, not a Step C
        # regression). Only assert the interleave-found property when a
        # steady-state window is actually possible.
        if nmicros > num_stages:
            assert found_dispatch, f"no dispatch-window interleave for {(num_stages, layers_per_stage, nmicros)}: {seq}"
            assert found_combine, f"no combine-window interleave for {(num_stages, layers_per_stage, nmicros)}: {seq}"


@pytest.mark.parametrize('num_stages,layers_per_stage,nmicros', _SWEEP_CONFIGS)
@replace_all_device_with('cpu')
def test_phase_schedule_property_sweep(num_stages, layers_per_stage, nmicros):
    _assert_phase_schedule_properties(num_stages, layers_per_stage, nmicros)


@pytest.mark.parametrize('num_stages,layers_per_stage,nmicros', _K_MULTISTAGE_CONFIGS)
@replace_all_device_with('cpu')
def test_phase_schedule_k_multistage_sweep(num_stages, layers_per_stage, nmicros):
    """K>1 (3-4 MoE layers stacked per physical stage) explicitly combined
    with multistage (2-3 physical stages) -- the same properties as
    :func:`test_phase_schedule_property_sweep`, but this grid was never
    exercised before this remediation (the original sweep capped
    layers_per_stage at 2)."""
    _assert_phase_schedule_properties(num_stages, layers_per_stage, nmicros)


@pytest.mark.parametrize('num_stages,layers_per_stage,nmicros', [
    (1, 1, 4), (2, 1, 4), (1, 2, 4), (2, 2, 6), (3, 1, 6),
])
@replace_all_device_with('cpu')
def test_dispatch_window_is_communication_free(num_stages, layers_per_stage, nmicros):
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_MultiStageMoEModel(num_stages, layers_per_stage), tempdir)
        graph, _, sched = _pas_phase_pipeline(graph, num_stages, layers_per_stage, nmicros)
    seq = _device0_phase_sequence(sched)
    assert _dispatch_window_is_communication_free(seq), seq


@pytest.mark.parametrize('num_stages,layers_per_stage,nmicros', [
    (1, 1, 3), (2, 1, 4), (1, 2, 3), (3, 2, 4), (1, 1, 1),
])
@replace_all_device_with('cpu')
def test_phase_schedule_plan_validates(num_stages, layers_per_stage, nmicros):
    """sched_1f1b_phase_aware already calls sched.finish() (which itself
    asserts validate()) internally, so building the schedule successfully at
    all is itself the check; re-assert explicitly for a directly-readable,
    dedicated test, across configs including edge cases (nmicros=1: no
    steady-state window possible at all)."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_MultiStageMoEModel(num_stages, layers_per_stage), tempdir)
        graph, _, sched = _pas_phase_pipeline(graph, num_stages, layers_per_stage, nmicros)
        assert graph.sched is sched
        assert sched.validate()


@replace_all_device_with('cpu')
def test_phase_schedule_matches_plain_sched_1f1b_fb_pattern():
    """Cross-stage causality sanity check: comparing the *coarse* F/B
    pattern (per device, i.e. which micro-batch's forward/backward runs at
    which position) of a phase-aware schedule against plain
    ``PredefinedSched.sched_1f1b`` on an equivalent plain-Linear stage
    layout -- phases only subdivide *within* one F/B slot, they must never
    change which micro-batch runs on which device at which coarse step."""
    num_stages, nmicros, layers_per_stage = 2, 4, 1

    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_MultiStageMoEModel(num_stages, layers_per_stage), tempdir)
        graph, _, sched = _pas_phase_pipeline(graph, num_stages, layers_per_stage, nmicros)

    class _PlainStack(nn.Module):
        def __init__(self, n):
            super().__init__()
            self.layers = nn.ModuleList(nn.Linear(DIM, DIM, bias=False) for _ in range(n))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x.sum()

    with tempfile.TemporaryDirectory() as tempdir:
        plain_graph = _build_graph(_PlainStack(num_stages), tempdir)
        linears = [n for n in plain_graph.nodes() if isinstance(n, IRFwOperation) and n.name == 'linear']
        plain_graph.staging(linears)
        dataloader = plain_graph.nodes()[0]
        sub_nodes = plain_graph.replicate(dataloader, num_stages)
        for i, sub_node in enumerate(sub_nodes):
            plain_graph.assign(sub_node, i)
        segs = [s for s in plain_graph.select(ntype=IRSegment, flatten=False) if s.isfw()]
        for sid, seg in enumerate(segs):
            for node in seg.nodes():
                plain_graph.assign(node, sid)
        plain_sched = PredefinedSched.sched_1f1b(plain_graph, nmicros, num_stages)

    for dev in range(num_stages):
        phase_blocks = sorted((b for b in sched.all_blocks() if dev in b.device), key=lambda b: sched.start(b))
        plain_blocks = sorted((b for b in plain_sched.all_blocks() if dev in b.device), key=lambda b: plain_sched.start(b))

        def _coarse(blocks):
            # Collapse to one entry per unique (isfw, mid) pair, at its FIRST
            # occurrence position -- NOT merely consecutive-run collapsing,
            # since a phase-aware schedule's steady-state window genuinely
            # interleaves F(m+1)'s phases with B(m)'s (different mid), so one
            # micro-batch's own phases are not necessarily contiguous.
            seen = set()
            out = []
            for b in blocks:
                ev = (b.content.isfw(), b.mid)
                if ev not in seen:
                    seen.add(ev)
                    out.append(ev)
            return out

        phase_coarse = _coarse(phase_blocks)
        plain_coarse = [(b.content.isfw(), b.mid) for b in plain_blocks]
        # (1) same SET of (isfw, mid) coarse events -- phases never invent or
        # drop a micro-batch's forward/backward.
        assert set(phase_coarse) == set(plain_coarse), dev
        # (2) F events still appear in strictly increasing mid order, and so
        # do B events, exactly like the plain schedule -- i.e. phases never
        # reorder *across* micro-batches within one direction, only ever
        # (deliberately, see phase.py's "Scheduling") change the relative
        # B-vs-F order *within* one interleaved steady-state pair.
        phase_f_mids = [mid for isfw, mid in phase_coarse if isfw]
        phase_b_mids = [mid for isfw, mid in phase_coarse if not isfw]
        plain_f_mids = [mid for isfw, mid in plain_coarse if isfw]
        plain_b_mids = [mid for isfw, mid in plain_coarse if not isfw]
        assert phase_f_mids == plain_f_mids, dev
        assert phase_b_mids == plain_b_mids, dev


# ---------------------------------------------------------------------------
# Illegal phase-group tests: missing / duplicate / out-of-order anchors
# ---------------------------------------------------------------------------

class _ScalarWrap(nn.Module):
    """Wraps a layer module (``_MoELayer``/``_IllegalSequenceLayer``) so it
    can be used directly as ``_gen_graph``'s WHOLE model (which requires a
    scalar loss output) -- these layer classes are normally used as one
    layer *inside* ``_MultiStageMoEModel``, which itself does the final
    ``.sum()`` reduction; tests that need a single, standalone layer (e.g.
    to isolate ``lower_layer_to_phases``'s own anchor-sequence validation)
    need this thin wrapper instead."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x).sum()


def _build_illegal_graph(case, tempdir):
    IDGenerator().clear()
    model = _ScalarWrap(_IllegalSequenceLayer(case))
    dummy_input = {'x': torch.randn(2, DIM)}
    model.train()
    graph, _ = _gen_graph(model, dummy_input, tempdir, constant_folding=True, end2end_mode=True)
    return graph


@pytest.mark.parametrize('case', [
    'missing-combine', 'missing-dispatch',
    'duplicate-attention', 'duplicate-dispatch',
    'out-of-order-expert-before-dispatch', 'out-of-order-combine-before-expert',
    'out-of-order-attention-last',
])
@replace_all_device_with('cpu')
def test_illegal_phase_sequence_raises_phase_error(case):
    """``lower_layer_to_phases``'s ``_validate_sequence`` must fail-fast
    with a clear :class:`PhaseError` (not silently mis-schedule, not some
    unrelated/generic exception) for every one of: a phase missing
    entirely, a phase anchored twice (duplicate), and phases anchored out
    of the canonical ATTENTION -> MOE_DISPATCH -> EXPERT_COMPUTE ->
    MOE_COMBINE order."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_illegal_graph(case, tempdir)
        all_ops = [n for n in graph.nodes() if isinstance(n, IRFwOperation)]
        with pytest.raises(PhaseError, match='illegal/incomplete phase sequence'):
            lower_layer_to_phases(graph, all_ops, layer_id=0)


@replace_all_device_with('cpu')
def test_no_phase_anchors_at_all_raises_phase_error():
    """A layer with NO phase anchors at all (e.g. accidentally calling
    ``lower_layer_to_phases`` on a plain, non-phase-anchored node range)
    must raise :class:`PhaseError`, not silently produce zero phases or
    crash with an unrelated error."""
    with tempfile.TemporaryDirectory() as tempdir:
        IDGenerator().clear()

        class _NoAnchorModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(DIM, DIM, bias=False)

            def forward(self, x):
                return self.lin(x).sum()

        dummy_input = {'x': torch.randn(2, DIM)}
        model = _NoAnchorModel()
        model.train()
        graph, _ = _gen_graph(model, dummy_input, tempdir, constant_folding=True, end2end_mode=True)
        all_ops = [n for n in graph.nodes() if isinstance(n, IRFwOperation)]
        with pytest.raises(PhaseError, match='no phase anchors'):
            lower_layer_to_phases(graph, all_ops, layer_id=0)


@replace_all_device_with('cpu')
def test_mismatched_layer_id_raises_phase_error():
    """Passing a layer_id that does not match the anchors' own layer_id
    (e.g. accidentally passing another layer's node range) must raise
    :class:`PhaseError` rather than silently mislabeling the phases."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_ScalarWrap(_MoELayer(layer_id=0)), tempdir)
        all_ops = [n for n in graph.nodes() if isinstance(n, IRFwOperation)]
        with pytest.raises(PhaseError, match='phase anchors for layer id'):
            lower_layer_to_phases(graph, all_ops, layer_id=1)


# ---------------------------------------------------------------------------
# PhaseExecutionIdentity: explicit (microbatch, physical_stage, layer_id,
# phase, direction, seq) uniqueness across the whole schedule table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('num_stages,layers_per_stage,nmicros', [
    (1, 1, 4), (2, 1, 4), (1, 2, 3), (2, 2, 5), (3, 1, 6), (3, 2, 4),
])
@replace_all_device_with('cpu')
def test_phase_execution_identity_is_globally_unique(num_stages, layers_per_stage, nmicros):
    """Every concrete phase execution instance ANYWHERE in the schedule
    table (every stage's device, every microbatch, every phase, both
    directions) must have a UNIQUE :class:`PhaseExecutionIdentity` --
    i.e. the schedule never places two different Blocks that would
    construct the SAME ``(microbatch, physical_stage, layer_id, phase_type,
    direction, seq_in_layer)`` tuple. This is checked from the actual,
    scheduled ``SchedulePlan.all_blocks()`` table directly (not via any
    implicit local variable/tuple bookkeeping in this test itself)."""
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_MultiStageMoEModel(num_stages, layers_per_stage), tempdir)
        graph, phase_nodes_per_stage, sched = _pas_phase_pipeline(graph, num_stages, layers_per_stage, nmicros)

    identities = []
    for blk in sched.all_blocks():
        pei = phase_execution_identity(blk)
        if pei is not None:
            identities.append(pei)

    # sanity: we actually found phase-tagged blocks (not silently empty)
    expected_count = num_stages * nmicros * layers_per_stage * 4 * 2  # both directions
    assert len(identities) == expected_count, (len(identities), expected_count)

    seen = set()
    duplicates = []
    for pei in identities:
        if pei in seen:
            duplicates.append(pei)
        seen.add(pei)
    assert not duplicates, f"duplicate PhaseExecutionIdentity instances found: {duplicates}"
    assert len(seen) == len(identities)

    # Cross-check: two DIFFERENT physical stages running the SAME
    # microbatch/layer/phase/direction/seq (impossible here since each
    # layer_id lives on exactly one stage, but verifies the dataclass
    # genuinely discriminates by physical_stage, not just accidentally
    # unique via layer_id alone) never collide, and identities genuinely
    # differ across microbatches for the SAME (stage, layer, phase, dir, seq).
    by_non_mb = {}
    for pei in identities:
        key = (pei.physical_stage, pei.layer_id, pei.phase_type, pei.direction, pei.seq_in_layer)
        by_non_mb.setdefault(key, set()).add(pei.microbatch)
    for key, mbs in by_non_mb.items():
        assert len(mbs) == nmicros, (key, mbs)


@replace_all_device_with('cpu')
def test_phase_execution_identity_repr_and_equality():
    """Direct, minimal unit test of :class:`PhaseExecutionIdentity` itself
    (independent of a full schedule): equality/hashing by value, distinct
    instances for distinct microbatch/stage/phase combinations, and a
    human-readable repr (useful for the failure messages above)."""
    identity = PhaseIdentity(layer_id=0, phase_type=PhaseType.ATTENTION, direction='forward', seq_in_layer=0)
    a = PhaseExecutionIdentity.from_identity(identity, microbatch=0, physical_stage=(0, 1))
    b = PhaseExecutionIdentity.from_identity(identity, microbatch=0, physical_stage=(0, 1))
    c = PhaseExecutionIdentity.from_identity(identity, microbatch=1, physical_stage=(0, 1))
    d = PhaseExecutionIdentity.from_identity(identity, microbatch=0, physical_stage=(2, 3))
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d
    assert 'mb0' in repr(a) and 'stage(0, 1)' in repr(a)
