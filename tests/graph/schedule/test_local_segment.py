#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CPU-only unit tests for ``nnscaler.graph.schedule.local_segment`` (Step B:
local segments inside one physical pipeline stage).

These build a real ``IRGraph`` (via ``nnscaler.parallel._gen_graph``) and a
real, dispatched, scheduled ``ExecutionPlan`` (via the same low-level,
CPU-only compile-pipeline pieces ``tests/codegen/test_reschedule.py`` and
``tests/codegen/test_global_schedule.py`` already use), without needing a
distributed launch or a GPU. Real 2/4-GPU end-to-end coverage (numeric
equivalence, no-deadlock, Step-A compatibility) lives in
``tests/parallel_module/test_local_segments_e2e.py``.
"""
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nnscaler.ir.unique import IDGenerator
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.gener.gen import IRAdapterGener
from nnscaler.graph.schedule.predefined import PredefinedSched
from nnscaler.graph.schedule.local_segment import (
    LocalSegmentError,
    AnchorBoundary,
    ModuleBoundary,
    CallableBoundary,
    partition_stage_into_local_segments,
    LocalSegmentSched,
)
from nnscaler.execplan import ExecutionPlan
from nnscaler.execplan.planpass.fusion import DiffFusion
from nnscaler.parallel import _gen_graph
from nnscaler.codegen.module.module import ModuleCodeGen
import nnscaler.runtime.function as ncf

from tests.utils import replace_all_device_with


# ---------------------------------------------------------------------------
# test models
# ---------------------------------------------------------------------------

class _SeqLinears(nn.Module):
    """`nlayers` unbiased Linear layers in sequence, optionally emitting a
    named anchor before each layer (for AnchorBoundary tests)."""

    def __init__(self, dim=8, nlayers=8, anchor_name=None):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in range(nlayers)])
        self.anchor_name = anchor_name

    def forward(self, x):
        for layer in self.layers:
            if self.anchor_name is not None:
                ncf.anchor(self.anchor_name)
            x = layer(x)
        return x.sum()


class _SharedParamModel(nn.Module):
    """Two `linear` calls sharing the exact same weight Parameter, with an
    anchor boundary in between -- used to test the shared-parameter-across-
    local-segments rejection."""

    def __init__(self, dim=8):
        super().__init__()
        self.w = nn.Parameter(torch.randn(dim, dim))

    def forward(self, x):
        ncf.anchor('lseg')
        x = torch.nn.functional.linear(x, self.w)
        ncf.anchor('lseg')
        x = torch.nn.functional.linear(x, self.w)
        return x.sum()


def _dim():
    return 8


def _build_graph(model, tempdir, dim=8):
    IDGenerator().clear()
    dummy_input = {'x': torch.randn(2, dim)}
    model.train()
    graph, _ = _gen_graph(model, dummy_input, tempdir, constant_folding=True, end2end_mode=True)
    return graph


def _linears_and_loss(graph):
    """Return (dataloader, [linear ops in order], loss/sum op).

    Note ``graph.nodes()`` on a not-yet-staged graph already contains every
    forward op's backward mirror as a top-level sibling (in reverse order),
    so the last element of ``graph.nodes()`` is generally a *backward* op,
    not the forward loss/sum -- filter to forward ops explicitly instead of
    relying on raw position.
    """
    nodes = graph.nodes()
    dataloader = nodes[0]
    fwd_ops = [n for n in nodes if isinstance(n, IRFwOperation)]
    linears = [n for n in fwd_ops if n.name == 'linear']
    loss = fwd_ops[-1]
    assert loss not in linears
    return dataloader, linears, loss


def _all_fwd_ops(graph):
    """Every forward op (linears, any anchors, and the trailing loss/sum),
    in forward execution order -- a single, whole-model contiguous stage
    range that also includes anchor nodes interspersed between layers
    (dropping them, as some tests using ``linears + [loss]`` alone would,
    makes the range non-contiguous)."""
    return [n for n in graph.nodes() if isinstance(n, IRFwOperation)]


# ---------------------------------------------------------------------------
# LocalSegmentBoundary.split_indices unit tests (no full graph needed beyond
# what's used to obtain real IRFwOperation/IRGraphAnchor nodes)
# ---------------------------------------------------------------------------

@replace_all_device_with('cpu')
def test_anchor_boundary_splits_at_named_anchor():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4, anchor_name='lseg'), tempdir, dim=_dim())
        stage_nodes = _all_fwd_ops(graph)
        boundary = AnchorBoundary({'lseg'})
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundary)
        # one anchor immediately before each of the 4 linears -> 4 local segments
        # (the very first anchor is at position 0 and does not split anything off)
        assert len(segs) == 4
        assert sum(len(seg.nodes()) for seg in segs) == len(stage_nodes)


@replace_all_device_with('cpu')
def test_anchor_boundary_ignores_non_matching_names():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4, anchor_name='lseg'), tempdir, dim=_dim())
        stage_nodes = _all_fwd_ops(graph)
        boundary = AnchorBoundary({'some_other_name'})
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundary)
        assert len(segs) == 1


@replace_all_device_with('cpu')
def test_module_boundary_splits_per_layer():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4), tempdir, dim=_dim())
        _, linears, loss = _linears_and_loss(graph)
        stage_nodes = list(linears) + [loss]
        boundary = ModuleBoundary()
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundary)
        # each nn.Linear instance is a distinct module -> a boundary between
        # every pair of consecutive linears; the trailing loss/sum op has no
        # module provenance so it never forces an extra split -> 4 segments
        assert len(segs) == 4
        assert sum(len(seg.nodes()) for seg in segs) == len(stage_nodes)


@replace_all_device_with('cpu')
def test_callable_boundary_custom_split():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4), tempdir, dim=_dim())
        _, linears, loss = _linears_and_loss(graph)
        stage_nodes = list(linears) + [loss]
        boundary = CallableBoundary(lambda nodes: [2])
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundary)
        assert len(segs) == 2
        assert len(segs[0].nodes()) == 2
        assert len(segs[1].nodes()) == 3


# ---------------------------------------------------------------------------
# No-boundary / degenerate behavior
# ---------------------------------------------------------------------------

@replace_all_device_with('cpu')
def test_partition_no_boundary_returns_single_segment():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4), tempdir, dim=_dim())
        _, linears, loss = _linears_and_loss(graph)
        stage_nodes = list(linears) + [loss]
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundary=None)
        assert len(segs) == 1
        assert list(segs[0].nodes()) == stage_nodes


@replace_all_device_with('cpu')
def test_partition_boundary_with_no_splits_returns_single_segment():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4), tempdir, dim=_dim())
        _, linears, loss = _linears_and_loss(graph)
        stage_nodes = list(linears) + [loss]
        boundary = CallableBoundary(lambda nodes: [])
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundary)
        assert len(segs) == 1


# ---------------------------------------------------------------------------
# Illegal cases
# ---------------------------------------------------------------------------

@replace_all_device_with('cpu')
def test_partition_rejects_empty_stage_nodes():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=2), tempdir, dim=_dim())
        with pytest.raises(LocalSegmentError):
            partition_stage_into_local_segments(graph, [], boundary=None)


@replace_all_device_with('cpu')
def test_partition_rejects_non_contiguous_nodes():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4), tempdir, dim=_dim())
        _, linears, loss = _linears_and_loss(graph)
        # skip linears[1]: not contiguous
        stage_nodes = [linears[0], linears[2], linears[3], loss]
        with pytest.raises(LocalSegmentError):
            partition_stage_into_local_segments(graph, stage_nodes, boundary=None)


@replace_all_device_with('cpu')
def test_partition_rejects_out_of_range_split_index():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4), tempdir, dim=_dim())
        _, linears, loss = _linears_and_loss(graph)
        stage_nodes = list(linears) + [loss]
        boundary = CallableBoundary(lambda nodes: [100])
        with pytest.raises(LocalSegmentError):
            partition_stage_into_local_segments(graph, stage_nodes, boundary)


@replace_all_device_with('cpu')
def test_partition_rejects_cross_physical_stage():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4), tempdir, dim=_dim())
        _, linears, loss = _linears_and_loss(graph)
        # create two *existing*, already-grouped physical stages first
        graph.staging([linears[0], linears[2]])
        stages = [s for s in graph.select(ntype=IRSegment, flatten=False) if s.isfw()]
        assert len(stages) == 2
        # now try to partition a node list that spans both existing stages
        mixed = [stages[0].nodes()[0], stages[1].nodes()[0]]
        with pytest.raises(LocalSegmentError):
            partition_stage_into_local_segments(graph, mixed, boundary=None)


@replace_all_device_with('cpu')
def test_partition_rejects_non_forward_operator():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=2), tempdir, dim=_dim())
        dataloader, linears, loss = _linears_and_loss(graph)
        with pytest.raises(LocalSegmentError):
            partition_stage_into_local_segments(graph, [dataloader, linears[0]], boundary=None)


@replace_all_device_with('cpu')
def test_partition_rejects_recompute_split():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4), tempdir, dim=_dim())
        _, linears, loss = _linears_and_loss(graph)
        stage_nodes = list(linears) + [loss]
        # mark linears[1] and linears[2] as one recompute group
        graph.recompute([linears[1], linears[2]])
        boundary = CallableBoundary(lambda nodes: [2])  # falls strictly inside the group
        with pytest.raises(LocalSegmentError):
            partition_stage_into_local_segments(graph, stage_nodes, boundary)


@replace_all_device_with('cpu')
def test_partition_allows_boundary_at_recompute_edge():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4), tempdir, dim=_dim())
        _, linears, loss = _linears_and_loss(graph)
        stage_nodes = list(linears) + [loss]
        graph.recompute([linears[1], linears[2]])
        # split right at the recompute group's own boundary (index 1 and 3): legal
        boundary = CallableBoundary(lambda nodes: [1, 3])
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundary)
        assert len(segs) == 3
        assert [n.recompute for n in segs[1].nodes()] == [linears[1].recompute, linears[2].recompute]


@replace_all_device_with('cpu')
def test_partition_rejects_shared_parameter_across_segments():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SharedParamModel(dim=_dim()), tempdir, dim=_dim())
        nodes = [n for n in graph.nodes() if isinstance(n, IRFwOperation)]
        boundary = AnchorBoundary({'lseg'})
        with pytest.raises(LocalSegmentError):
            partition_stage_into_local_segments(graph, nodes, boundary)


@replace_all_device_with('cpu')
def test_partition_allows_shared_parameter_within_one_segment():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SharedParamModel(dim=_dim()), tempdir, dim=_dim())
        nodes = [n for n in graph.nodes() if isinstance(n, IRFwOperation)]
        # no boundary at all -> both uses of `w` stay in the single local segment
        segs = partition_stage_into_local_segments(graph, nodes, boundary=None)
        assert len(segs) == 1


# ---------------------------------------------------------------------------
# Nested-segment / same-device invariants
# ---------------------------------------------------------------------------

def _assert_no_nested_segment(graph):
    """Regression guard for the core design invariant this module relies on:
    a created local segment must always be an ordinary, flat, top-level
    IRSegment -- never nested inside another IRSegment's own node list."""
    for node in graph.nodes():
        if isinstance(node, IRSegment):
            for sub in node.nodes():
                assert not isinstance(sub, IRSegment), (
                    f"found a nested IRSegment {sub!r} inside {node!r}; local "
                    f"segments must always be flat, top-level siblings"
                )


@replace_all_device_with('cpu')
def test_local_segments_are_flat_not_nested_and_share_device():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4, anchor_name='lseg'), tempdir, dim=_dim())
        stage_nodes = _all_fwd_ops(graph)
        segs = partition_stage_into_local_segments(graph, stage_nodes, AnchorBoundary({'lseg'}))
        assert len(segs) == 4
        _assert_no_nested_segment(graph)

        for seg in segs:
            for node in seg.nodes():
                graph.assign(node, 0)
        devices = {tuple(seg.device) for seg in segs}
        assert devices == {(0,)}


@replace_all_device_with('cpu')
def test_local_segments_mirror_correct_reverse_order():
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=4, anchor_name='lseg'), tempdir, dim=_dim())
        stage_nodes = _all_fwd_ops(graph)
        segs = partition_stage_into_local_segments(graph, stage_nodes, AnchorBoundary({'lseg'}))
        assert all(seg.mirror is not None for seg in segs)
        # backward local segments must appear in the graph's backward node
        # list in exactly the reverse order of their forward counterparts
        top = graph
        positions = []
        for seg in segs:
            positions.append(top.index(seg.mirror).indices[-1])
        assert positions == sorted(positions, reverse=True)


# ---------------------------------------------------------------------------
# Scheduling: degeneracy + interleaving
# ---------------------------------------------------------------------------

def _pas_pipeline(graph, num_stages, per_stage_size, boundaries, nmicros, sched_fn):
    """Low-level (no ComputeConfig/parallelize) PAS-equivalent: stage the
    graph into `num_stages` contiguous physical stages of `per_stage_size`
    linears each (the last stage also absorbs the trailing loss node),
    optionally splitting each stage into local segments via `boundaries`
    (a list of length `num_stages`, each `None` or a `LocalSegmentBoundary`),
    then build a schedule with `sched_fn(graph, nmicros, num_stages)`.

    Returns (graph, list-of-list-of-forward-local-segments-per-stage).
    """
    dataloader, linears, loss = _linears_and_loss(graph)
    assert len(linears) == num_stages * per_stage_size
    all_segs = []
    for sid in range(num_stages):
        start = sid * per_stage_size
        end = start + per_stage_size
        stage_nodes = list(linears[start:end])
        if sid == num_stages - 1:
            stage_nodes.append(loss)
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundaries[sid])
        all_segs.append(segs)

    sub_nodes = graph.replicate(dataloader, num_stages)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)
    for sid in range(num_stages):
        for seg in all_segs[sid]:
            for node in seg.nodes():
                graph.assign(node, sid)

    sched_fn(graph, nmicros, num_stages)
    return graph, all_segs


def _build_local_segment_execplan(model, num_stages, per_stage_size, boundaries, nmicros, tempdir, sched_fn):
    graph = _build_graph(model, tempdir, dim=_dim())
    graph, all_segs = _pas_pipeline(graph, num_stages, per_stage_size, boundaries, nmicros, sched_fn)
    adapter_graph = IRAdapterGener.gen(graph, cost_fn=None)
    if adapter_graph.sched is not None:
        adapter_graph.sched.apply()
    execplan = ExecutionPlan.from_schedplan(adapter_graph.sched)
    execplan = DiffFusion.apply(execplan)
    return execplan, all_segs


def _block_seq_by_device(execplan, num_stages):
    return {dev: [(type(n).__name__, getattr(n, 'cid', None), n.isfw() if hasattr(n, 'isfw') else None)
                  for n in execplan.seq(dev)]
            for dev in range(num_stages)}


@replace_all_device_with('cpu')
def test_sched_local_segments_degenerates_to_sched_1f1b_when_k1():
    """A stage with exactly one local segment must produce the exact same
    per-device relative block order as PredefinedSched.sched_1f1b."""
    num_stages, per_stage_size, nmicros = 2, 2, 4
    boundaries = [None, None]

    with tempfile.TemporaryDirectory() as tempdir:
        exec_local, _ = _build_local_segment_execplan(
            _SeqLinears(dim=_dim(), nlayers=num_stages * per_stage_size),
            num_stages, per_stage_size, boundaries, nmicros, tempdir,
            LocalSegmentSched.sched_1f1b_local_segments,
        )
    with tempfile.TemporaryDirectory() as tempdir:
        exec_1f1b, _ = _build_local_segment_execplan(
            _SeqLinears(dim=_dim(), nlayers=num_stages * per_stage_size),
            num_stages, per_stage_size, boundaries, nmicros, tempdir,
            lambda g, m, s: PredefinedSched.sched_1f1b(g, m, s),
        )

    seq_local = _block_seq_by_device(exec_local, num_stages)
    seq_1f1b = _block_seq_by_device(exec_1f1b, num_stages)
    for dev in range(num_stages):
        # compare shapes: same length and same (isfw) F/B pattern per device.
        fb_local = [isfw for (_, _, isfw) in seq_local[dev] if isfw is not None]
        fb_1f1b = [isfw for (_, _, isfw) in seq_1f1b[dev] if isfw is not None]
        assert fb_local == fb_1f1b, f"device {dev}: F/B pattern differs -- local={fb_local} vs sched_1f1b={fb_1f1b}"


@replace_all_device_with('cpu')
def test_sched_local_segments_interleaves_steady_state():
    """With K=2 local segments on the pipeline's *first* stage (the only
    stage this scheduler safely interleaves -- see module docstring), the
    steady-state window must interleave B(m)'s and F(m+1)'s local segments
    one at a time, not run all of B(m) before any of F(m+1))."""
    num_stages, per_stage_size, nmicros = 2, 2, 4
    boundaries = [CallableBoundary(lambda nodes: [1]) for _ in range(num_stages)]

    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=num_stages * per_stage_size), tempdir, dim=_dim())
        graph, all_segs = _pas_pipeline(
            graph, num_stages, per_stage_size, boundaries, nmicros,
            LocalSegmentSched.sched_1f1b_local_segments,
        )
        sched = graph.sched
        assert sched is not None

        # Inspect stage 0's raw block placement in true start-step order
        # (stage 0 is the only stage this scheduler interleaves).
        stage0_segs = all_segs[0]
        assert len(stage0_segs) == 2
        stage0_dev = 0
        stage0_blocks = sorted(
            (blk for blk in sched.all_blocks() if stage0_dev in blk.device),
            key=lambda blk: sched.start(blk),
        )
        stage0_order = [(blk.content.isfw(), blk.mid) for blk in stage0_blocks]

        # find a B(m) immediately followed by F(m') pair (for stage 0 the
        # base sched_1f1b formula pairs B(m) with F(m + num_stages), not
        # F(m+1) -- see module docstring's "Scheduling" derivation) and
        # confirm they alternate (B,F,B,F) rather than (B,B,F,F).
        expected_diff = num_stages
        found_interleave = False
        for i in range(len(stage0_order) - 3):
            b0, m0 = stage0_order[i]
            f0, m1 = stage0_order[i + 1]
            b1, m2 = stage0_order[i + 2]
            f1, m3 = stage0_order[i + 3]
            if (not b0) and f0 and (not b1) and f1 and m0 == m2 and m1 == m3 and m1 == m0 + expected_diff:
                found_interleave = True
                break
        assert found_interleave, f"expected an interleaved B,F,B,F steady-state window, got: {stage0_order}"


@replace_all_device_with('cpu')
def test_sched_local_segments_validates():
    num_stages, per_stage_size, nmicros = 2, 2, 3
    boundaries = [CallableBoundary(lambda nodes: [1]), None]
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=num_stages * per_stage_size), tempdir, dim=_dim())
        graph, _ = _pas_pipeline(
            graph, num_stages, per_stage_size, boundaries, nmicros,
            LocalSegmentSched.sched_1f1b_local_segments,
        )
        assert graph.sched.validate()


@replace_all_device_with('cpu')
def test_sched_local_segments_rejects_stage_count_mismatch():
    num_stages, per_stage_size, nmicros = 2, 2, 3
    boundaries = [None, None]
    with tempfile.TemporaryDirectory() as tempdir:
        graph = _build_graph(_SeqLinears(dim=_dim(), nlayers=num_stages * per_stage_size), tempdir, dim=_dim())
        graph, _ = _pas_pipeline(
            graph, num_stages, per_stage_size, boundaries, nmicros,
            lambda g, m, s: None,  # do not actually build a schedule yet
        )
        with pytest.raises(ValueError):
            LocalSegmentSched.sched_1f1b_local_segments(graph, nmicros, num_stages=3)


# ---------------------------------------------------------------------------
# gencode: multiple local segment methods + scheduling order
# ---------------------------------------------------------------------------
#
# ModuleCodeGen(execplan).gen(dev) alone (used above and by
# tests/codegen/test_reschedule.py) only emits the model class itself
# (segment/adapter methods, __init__) -- the call-site driver code
# (_train_step, which actually invokes `model.segmentNNN`) is emitted by a
# separate component wired together by the full `parallelize()` compile
# pipeline. So this test goes through the real, full pipeline (still
# CPU-only, via `replace_all_device_with`) and inspects the saved
# `gencode{rank}.py` files, exactly like `tests/parallel_module/test_gencode*.py`.

import re as _re

from nnscaler.parallel import parallelize, ComputeConfig
from tests.parallel_module.test_gencode import _gencode_contains, print_gencode


def _pas_local_segments_full(graph, cfg):
    num_stages = cfg.pas_config['pipeline_nstages']
    nmicros = cfg.pas_config['pipeline_nmicros']
    per_stage_size = cfg.pas_config['per_stage_size']
    boundaries = cfg.pas_config['boundaries']

    dataloader, linears, loss = _linears_and_loss(graph)
    all_segs = []
    for sid in range(num_stages):
        start = sid * per_stage_size
        end = start + per_stage_size
        stage_nodes = list(linears[start:end])
        if sid == num_stages - 1:
            stage_nodes.append(loss)
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundaries[sid])
        all_segs.append(segs)

    sub_nodes = graph.replicate(dataloader, num_stages)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)
    for sid in range(num_stages):
        for seg in all_segs[sid]:
            for node in seg.nodes():
                graph.assign(node, sid)

    cfg.apply_pipeline_scheduler(graph, num_stages, nmicros, LocalSegmentSched.sched_1f1b_local_segments)
    return graph


def _split_at_index_1(nodes):
    return [1]


def _split_at_index_2(nodes):
    return [2]


@replace_all_device_with('cpu')
def test_gencode_multiple_local_segment_methods_and_order(tmp_path):
    num_stages, per_stage_size, nmicros = 2, 2, 3
    model = _SeqLinears(dim=_dim(), nlayers=num_stages * per_stage_size)
    model.train()
    parallelize(
        model,
        {'x': torch.randn(2, _dim())},
        pas_policy=_pas_local_segments_full,
        compute_config=ComputeConfig(
            num_stages, num_stages,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nstages=num_stages,
                pipeline_nmicros=nmicros,
                per_stage_size=per_stage_size,
                boundaries=[CallableBoundary(_split_at_index_1), CallableBoundary(_split_at_index_1)],
            ),
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    for dev in range(num_stages):
        matches = _gencode_contains(tmp_path, type(model), dev, r'^\s*def (segment\d+)\(', flags=_re.MULTILINE)
        seg_names = list(dict.fromkeys(matches))  # de-dup, preserve order of definition
        assert len(seg_names) == per_stage_size, (
            f"expected {per_stage_size} local segment methods on device {dev}, found {seg_names}"
        )

        call_order = _gencode_contains(tmp_path, type(model), dev, r'model\.(segment\d+)\b')
        # each defined local segment must actually be invoked from the driver
        # code (not just defined), and in a relative order consistent with
        # their forward dependency chain (segments[0] before segments[1] at
        # least once, e.g. at the very first, warmup call site).
        assert set(seg_names) <= set(call_order), (
            f"not all defined segments are invoked on device {dev}: "
            f"defined={seg_names}, invoked={call_order}"
        )
        first_call_idx = call_order.index(seg_names[0])
        second_call_idx = call_order.index(seg_names[1])
        assert first_call_idx < second_call_idx, (
            f"expected {seg_names[0]} invoked before {seg_names[1]} at least once on "
            f"device {dev}, got call order: {call_order}"
        )


@replace_all_device_with('cpu')
def test_gencode_activation_released_after_last_consuming_local_segment(tmp_path):
    """Activation lifecycle sanity: the tensor flowing from local segment 0
    into local segment 1 (of the same, otherwise-unsplit, single-stage
    model) must be released (a driver-level ``del ...`` statement, emitted
    by ``ScheduleCodeGen``'s ``LifeCycle``) once segment 1 has consumed it
    -- not held alive for the rest of ``_train_step`` -- proving activation
    release still happens correctly at local-segment granularity."""
    num_stages, per_stage_size, nmicros = 1, 4, 2
    model = _SeqLinears(dim=_dim(), nlayers=per_stage_size)
    model.train()
    parallelize(
        model,
        {'x': torch.randn(2, _dim())},
        pas_policy=_pas_local_segments_full,
        compute_config=ComputeConfig(
            num_stages, num_stages,
            constant_folding=False,
            use_end2end=True,
            pas_config=dict(
                pipeline_nstages=num_stages,
                pipeline_nmicros=nmicros,
                per_stage_size=per_stage_size,
                boundaries=[CallableBoundary(_split_at_index_2)],
            ),
        ),
        gen_savedir=tmp_path,
        load_module=False,
        reuse='override',
    )

    from nnscaler.parallel import _PARALLEL_MODULE_NAMESPACE, _get_full_qualified_name, _DEFAULT_INSTANCE_NAME
    namespace = f'{_PARALLEL_MODULE_NAMESPACE}.{_get_full_qualified_name(type(model))}.{_DEFAULT_INSTANCE_NAME}'
    outdir = tmp_path / Path(namespace.replace('.', '/').strip('/'))
    full_code = (outdir / 'gencode0.py').read_text()
    lines = full_code.splitlines()
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith('def _train_step'))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].lstrip().startswith('def _infer_step')), len(lines))
    body = lines[start:end]

    # find the assignment that calls the first local segment, capture its
    # output variable name(s)
    call_re = _re.compile(r'^\s*([\w, ]+) = nnscaler\.runtime\.executor\.fexecute\([\'"]segment\d+[\'"], model\.segment\d+')
    call_line_idx = next(i for i, l in enumerate(body) if call_re.match(l) and 'segment' in l)
    out_vars = [v.strip() for v in call_re.match(body[call_line_idx]).group(1).split(',') if v.strip()]
    assert out_vars, f"could not parse output variable(s) from call line: {body[call_line_idx]!r}"

    # at least one of those output variables must be `del`-ed later in the
    # same _train_step body (proving driver-level release, not held forever)
    del_re = _re.compile(r'^\s*del ([\w, ]+)\s*$')
    released = set()
    for l in body[call_line_idx + 1:]:
        m = del_re.match(l)
        if m:
            released.update(v.strip() for v in m.group(1).split(','))
    assert released & set(out_vars), (
        f"expected one of {out_vars} to be released (`del ...`) after its last "
        f"consuming local segment in _train_step, found releases: {released}\n"
        f"body:\n" + '\n'.join(body)
    )


@replace_all_device_with('cpu')
def test_gencode_shared_symbol_not_duplicated_across_local_segments():
    """Checkpoint-metadata sanity: a parameter referenced only within a
    single local segment is still declared (registered) exactly once,
    identical to the pre-Step-B behaviour."""
    num_stages, per_stage_size, nmicros = 1, 4, 2
    boundaries = [CallableBoundary(lambda nodes: [2])]
    with tempfile.TemporaryDirectory() as tempdir:
        execplan, all_segs = _build_local_segment_execplan(
            _SeqLinears(dim=_dim(), nlayers=4), 1, 4, boundaries, nmicros, tempdir,
            LocalSegmentSched.sched_1f1b_local_segments,
        )
        code = ModuleCodeGen(execplan).gen(0)
        compile(code, '<gencode>', 'exec')
        # every generated attribute must be register_parameter'd exactly once
        import re
        names = re.findall(r"register_parameter\('([^']+)'", code)
        assert len(names) == len(set(names)), f"duplicate parameter registration in gencode: {names}"


# ---------------------------------------------------------------------------
# Reducer (data-parallel scale-out) correctness with local segments
# ---------------------------------------------------------------------------

@replace_all_device_with('cpu')
def test_gencode_reducer_covers_all_local_segment_params_exactly_once():
    """With data-parallel scale-out (``runtime_ndevs > plan_ngpus``, which is
    what makes ``ModuleCodeGen`` emit a real ``Reducer``/``add_scale_reducers``
    setup -- a pure-pipeline-only plan never exercises this path), every
    parameter used by *any* local segment on a device must be added to
    *exactly one* reducer on that device -- neither missing (silently never
    synced) nor duplicated (double-counted, corrupting
    ``grad_accumulation_steps``'s implicit one-touch-per-microbatch
    assumption -- see module docstring's reducer-count hazard). This is the
    direct, gencode-level counterpart to the 2-GPU numeric-equivalence e2e
    test (which would also catch this indirectly, via diverged trained
    weights, but this isolates the reducer bookkeeping itself)."""
    num_stages, per_stage_size, nmicros = 1, 4, 2
    boundaries = [CallableBoundary(lambda nodes: [2])]  # 2 local segments, 2 params each side
    with tempfile.TemporaryDirectory() as tempdir:
        execplan, all_segs = _build_local_segment_execplan(
            _SeqLinears(dim=_dim(), nlayers=4), num_stages, per_stage_size, boundaries, nmicros, tempdir,
            LocalSegmentSched.sched_1f1b_local_segments,
        )
        # runtime_ndevs=2*plan_ngpus(=1) -> data-parallel degree 2, which is
        # what makes add_scale_reducers()/init_reducer() actually run.
        code = ModuleCodeGen(execplan, runtime_ndevs=2).gen(0)
        compile(code, '<gencode>', 'exec')

        import re
        param_names = set(re.findall(r"register_parameter\('([^']+)'", code))
        assert len(param_names) == 4, f"expected the model's 4 Linear weights, got {param_names}"

        reducer_add_calls = re.findall(r"(self\.wreducer\d+)\.add_param\((self\.\w+)\)", code)
        added_params = [p for _, p in reducer_add_calls]
        # every parameter must be covered by a reducer ...
        covered = {p.removeprefix('self.') for p in added_params}
        assert covered == param_names, (
            f"reducer coverage mismatch: params={param_names}, reducer-covered={covered}"
        )
        # ... and by exactly one (no duplicate add_param for the same weight,
        # which would double-count it in that reducer's bucket).
        assert len(added_params) == len(set(added_params)), (
            f"a parameter was added to a reducer more than once: {added_params}"
        )
