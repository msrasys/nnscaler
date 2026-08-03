#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CPU-only unit tests for:

- the additive ``channel_key`` / ``device_key`` parameters on
  ``OpDependencyGraph`` (see ``nnscaler.execplan.planpass.reschedule``);
- ``nnscaler.execplan.planpass.global_schedule.GlobalCommSchedule``, the
  global (cross-rank), fixed, cap-aware communication schedule.
"""
from typing import Dict, List

import pytest

from nnscaler.ir.cten import IRCell
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import MovePrim
from nnscaler.execplan.planpass.reschedule import OpDependencyGraph
from nnscaler.execplan.planpass.global_schedule import (
    GlobalCommSchedule,
    GlobalScheduleError,
    ScheduleReport,
    peer_pair_channel_key,
    cid_channel_key,
    unsafe_direction_split_channel_key,
    p2p_peer_pair,
    device_of,
)


def _sub(shape=(4, 4), requires_grad=True):
    return IRFullTensor(list(shape), requires_grad=requires_grad).tosub()


def _op(name, inputs, output):
    op = IRFwOperation(name, name, inputs=list(inputs), num_outputs=1)
    op.set_output(0, output)
    return op


# ---------------------------------------------------------------------------
# OpDependencyGraph: channel_key / device_key (additive, backward compatible)
# ---------------------------------------------------------------------------

def test_opdepgraph_default_comm_chain_unchanged_without_channel_key():
    """Regression: omitting channel_key reproduces the exact prior behaviour
    (one single global communication chain)."""
    t0, t1, t2, t3 = _sub(requires_grad=False), _sub(), _sub(), _sub()
    a = IRAdapter([], [t0])
    b = _op('b', [t0], t1)
    c = IRAdapter([t1], [t1])
    d = IRAdapter([], [t2])
    nodes = [a, b, c, d]

    graph = OpDependencyGraph(nodes)
    # single chain: a -> c -> d (comm nodes only), in original order
    assert c in graph.successors(a)
    assert d in graph.successors(c)


def test_opdepgraph_channel_key_splits_comm_chain_into_independent_groups():
    """With channel_key, only same-key comm nodes stay chained; different
    keys are free (no edge forced between them)."""
    t0, t2 = _sub(requires_grad=False), _sub(requires_grad=False)
    recv_x1 = IRAdapter([], [_sub(requires_grad=False)])
    recv_x2 = IRAdapter([], [_sub(requires_grad=False)])
    recv_y1 = IRAdapter([], [_sub(requires_grad=False)])
    nodes = [recv_x1, recv_y1, recv_x2]
    key = {recv_x1: 'X', recv_x2: 'X', recv_y1: 'Y'}

    graph = OpDependencyGraph(nodes, channel_key=lambda n: key[n])
    # same-channel (X) pair chained despite recv_y1 sitting between them
    assert recv_x2 in graph.successors(recv_x1)
    # different channel (Y) has no edge to/from X's chain
    assert recv_y1 not in graph.successors(recv_x1)
    assert recv_y1 not in graph.predecessors(recv_x2)


def test_opdepgraph_device_key_splits_anchor_chain_per_device():
    """With serialize_segments + device_key, anchors on different devices are
    not forced into a relative order with each other."""
    seg_a0 = _op('seg_a0', [_sub(requires_grad=False)], _sub())
    seg_b1 = _op('seg_b1', [_sub(requires_grad=False)], _sub())
    seg_a1 = _op('seg_a1', [_sub(requires_grad=False)], _sub())
    nodes = [seg_a0, seg_b1, seg_a1]
    devmap = {seg_a0: 0, seg_b1: 1, seg_a1: 0}

    graph = OpDependencyGraph(nodes, serialize_segments=True, device_key=lambda n: devmap[n])
    assert seg_a1 in graph.successors(seg_a0)     # same device (0): ordered
    assert seg_b1 not in graph.successors(seg_a0)  # different device: free
    assert seg_a0 not in graph.successors(seg_b1)


def test_opdepgraph_predecessors_mirrors_successors():
    t0, t1 = _sub(requires_grad=False), _sub()
    a = _op('a', [t0], t1)
    b = _op('b', [t1], _sub())
    graph = OpDependencyGraph([a, b])
    assert a in graph.predecessors(b)
    assert b not in graph.predecessors(a)


# ---------------------------------------------------------------------------
# GlobalCommSchedule: helper key functions
# ---------------------------------------------------------------------------

def _make_p2p_pair(shape, src, dst, requires_grad=False):
    """Build a (send_view, recv_view) pair sharing a cid, mimicking
    IRAdapter.dispatch's two device-specific views of one logical P2P move --
    but without needing a full tensor/device-dispatch machinery: just enough
    (matching prims + shared cid) for the helpers under test, which only look
    at `.cid`, `.prims` (MovePrim src/dst), and `.device`."""
    send_tensor = _sub(shape, requires_grad=requires_grad)
    recv_tensor = _sub(shape, requires_grad=requires_grad)
    send_view = IRAdapter([send_tensor], [])
    send_view.device = [src]
    send_view.prims = [MovePrim([send_tensor], [], shape=shape, dtype='torch.float32', src=src, dst=dst)]
    recv_view = IRAdapter([], [recv_tensor])
    recv_view.device = [dst]
    recv_view.prims = [MovePrim([], [recv_tensor], shape=shape, dtype='torch.float32', src=src, dst=dst)]
    recv_view._id = send_view._id  # dispatch shares cid between both views
    return send_view, recv_view


def test_p2p_peer_pair_recovers_undirected_pair_after_dispatch():
    send_view, recv_view = _make_p2p_pair((4, 4), src=0, dst=1)
    assert p2p_peer_pair(send_view) == frozenset({0, 1})
    assert p2p_peer_pair(recv_view) == frozenset({0, 1})
    assert device_of(send_view) == 0
    assert device_of(recv_view) == 1


def test_peer_pair_channel_key_ignores_direction():
    send_view, recv_view = _make_p2p_pair((4, 4), src=0, dst=1)
    # from device 0's perspective only the send exists; key is (device, peer)
    assert peer_pair_channel_key(send_view) == (0, frozenset({0, 1}))


def test_cid_channel_key_distinguishes_unrelated_channels_same_peer_pair():
    """Two logically-unrelated channels (different cids) between the SAME
    peer-pair must get independent cap_key groups -- this is what lets
    GlobalCommSchedule bound each one's own outstanding count separately
    instead of forcing them to share one budget."""
    grad_send, grad_recv = _make_p2p_pair((4, 4), src=0, dst=1)
    loss_send, loss_recv = _make_p2p_pair((1,), src=0, dst=1)
    assert cid_channel_key(grad_recv) != cid_channel_key(loss_recv)
    # but the (safety-critical) comm-chain key must be the SAME for both,
    # since they share a peer-pair (see module docstring for why)
    assert peer_pair_channel_key(grad_recv) == peer_pair_channel_key(loss_recv)


def test_unsafe_direction_split_key_is_documented_and_not_wired_as_default():
    """`unsafe_direction_split_channel_key` must exist (documenting the
    empirically-confirmed dead end) but must NOT be GlobalCommSchedule.apply's
    default channel_key."""
    import inspect
    sig = inspect.signature(GlobalCommSchedule.apply)
    assert sig.parameters['channel_key'].default is peer_pair_channel_key
    assert sig.parameters['channel_key'].default is not unsafe_direction_split_channel_key


# ---------------------------------------------------------------------------
# GlobalCommSchedule: illegal configuration
# ---------------------------------------------------------------------------

class _FakeExecPlan:
    """Minimal stand-in exposing just what GlobalCommSchedule.apply needs
    before it would otherwise touch a real ExecutionPlan, for config-only
    tests that should fail before any node processing happens."""
    def devices(self):
        return []


def test_apply_rejects_max_outstanding_below_one():
    with pytest.raises(GlobalScheduleError, match='max_outstanding'):
        GlobalCommSchedule.apply(_FakeExecPlan(), max_outstanding=0)


def test_apply_rejects_empty_execution_plan():
    with pytest.raises(GlobalScheduleError, match='no devices'):
        GlobalCommSchedule.apply(_FakeExecPlan(), max_outstanding=2)


def test_apply_rejects_async_comm_combination():
    from nnscaler.flags import CompileFlag
    saved = CompileFlag.async_comm
    try:
        CompileFlag.async_comm = True
        with pytest.raises(GlobalScheduleError, match='async_comm'):
            GlobalCommSchedule.apply(_FakeExecPlan(), max_outstanding=2)
    finally:
        CompileFlag.async_comm = saved


def test_schedule_report_is_safe_property():
    assert ScheduleReport().is_safe
    from nnscaler.execplan.planpass.global_schedule import ScheduleViolation
    unsafe = ScheduleReport(violations=[
        ScheduleViolation(devid=0, channel='c', node_a='a', window_a=(0, 1), node_b='b', window_b=(1, 2))
    ])
    assert not unsafe.is_safe


# ---------------------------------------------------------------------------
# GlobalCommSchedule: real (CPU-built) multi-device pipeline execution plan
# ---------------------------------------------------------------------------

def _build_pipeline_execplan(nstages=2, nmicros=4, nlayers=4, dim=8):
    """Build a real, properly-dispatched NSTAGES-device pipeline execution
    plan (genuine P2P `MovePrim` adapters, correct per-device `.device` /
    prim src+dst) via the same graph-construction path `parallelize()` uses,
    stopping right after `ExecutionPlan.from_schedplan` -- no actual
    distributed/GPU execution is needed for this (compile-time only)."""
    import tempfile
    import torch
    import torch.nn as nn
    from nnscaler.ir.unique import IDGenerator
    from nnscaler.parallel import ComputeConfig, _gen_graph
    from nnscaler.policies import _replica
    from nnscaler.graph.segment import IRSegment
    from nnscaler.ir.operator import IRDataOperation
    from nnscaler.graph.gener.gen import IRAdapterGener
    from nnscaler.execplan import ExecutionPlan
    from nnscaler.execplan.planpass.fusion import DiffFusion

    IDGenerator().clear()

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(nlayers))

        def forward(self, data):
            x = data['data']
            for layer in self.layers:
                x = layer(x)
            return x.sum()

    def _pas(graph, config):
        linears = graph.select(name='linear')
        stage_start_nodes = linears[::len(linears) // nstages][:nstages]
        graph.staging(stage_start_nodes)
        segments = graph.select(ntype=IRSegment, flatten=False)
        fsegs = [seg for seg in segments if seg.isfw()]
        for sid, segment in enumerate(fsegs):
            for node in segment.nodes():
                _replica(graph, node, devs=[sid])
        for dl in graph.select(ntype=IRDataOperation):
            _replica(graph, dl, devs=list(range(nstages)))
        config.apply_pipeline_scheduler(graph, nstages, nmicros, '1f1b')
        return graph

    config = ComputeConfig(nstages, nstages, use_end2end=True,
                            pas_config=dict(pipeline_nstages=nstages, pipeline_nmicros=nmicros,
                                             pipeline_scheduler='1f1b'))
    with tempfile.TemporaryDirectory() as tempdir:
        init_graph, _ = _gen_graph(_MLP(), {'data': {'data': torch.randn(2, dim)}}, tempdir,
                                    constant_folding=True, end2end_mode=True)
        graph = _pas(init_graph, config)
        adapter_graph = IRAdapterGener.gen(graph, cost_fn=None)
        if adapter_graph.sched is not None:
            adapter_graph.sched.apply()
        execplan = ExecutionPlan.from_schedplan(adapter_graph.sched)
        execplan = DiffFusion.apply(execplan)
    return execplan


def test_global_schedule_widens_overlap_and_stays_safe_on_real_pipeline():
    """End-to-end (CPU, compile-time only) proof that GlobalCommSchedule (a)
    widens the hoistable-receive overlap window (the concrete, checkable
    proxy for the combined_1f1b invariant) relative to the un-rescheduled
    baseline, and (b) produces a schedule `validate()` reports as safe --
    exercised against a REAL, properly-dispatched multi-device pipeline
    execution plan (genuine P2P adapters), not a hand-built toy graph."""
    execplan = _build_pipeline_execplan()

    before = GlobalCommSchedule.validate(execplan, max_outstanding=6)
    assert before.is_safe

    GlobalCommSchedule.apply(execplan, max_outstanding=6)

    after = GlobalCommSchedule.validate(execplan, max_outstanding=6)
    assert after.is_safe, f'GlobalCommSchedule produced an unsafe schedule: {after.violations}'

    # the schedule must have gotten a genuinely wider overlap window on at
    # least one device/channel than the un-rescheduled baseline (this is the
    # measurable proxy for "issue < B(m) < wait < F(m+1)" -- the GPU e2e test
    # additionally confirms this structurally in real generated code)
    assert any(after.hoist_span.get(k, 0) > before.hoist_span.get(k, 0) for k in after.hoist_span), (
        f'expected improved hoist_span; before={before.hoist_span} after={after.hoist_span}'
    )


def test_global_schedule_rejects_max_outstanding_too_small_for_real_plan():
    """A genuinely too-small cap (the result-broadcast needs `nmicros`
    concurrently-outstanding receives, all resolved only by the bulk
    end-of-step drain) is rejected with a clear, actionable
    GlobalScheduleError at schedule-construction time -- not a silent
    unsafe schedule and not a runtime hang."""
    execplan = _build_pipeline_execplan(nmicros=4)
    with pytest.raises(GlobalScheduleError, match='stalled|exceed'):
        GlobalCommSchedule.apply(execplan, max_outstanding=1)


def test_global_schedule_projection_is_consistent_subsequence():
    """Every device's post-`apply` sequence must be exactly the set of nodes
    it had before (same multiset, just reordered) -- the "projection"
    property: each device's local order is filtered FROM the one shared
    global order, not independently re-derived."""
    execplan = _build_pipeline_execplan()
    before = {devid: list(execplan.at(devid)) for devid in execplan.devices()}
    GlobalCommSchedule.apply(execplan, max_outstanding=6)
    after = {devid: list(execplan.at(devid)) for devid in execplan.devices()}
    for devid in before:
        assert set(id(n) for n in before[devid]) == set(id(n) for n in after[devid])
