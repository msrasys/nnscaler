#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import os

import torch

from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.adapter import IRAdapter, IRWeightReducer
from nnscaler.graph import IRGraph
from nnscaler.graph.gener.gen import IRAdapterGener
from nnscaler.parallel import _gen_graph
from nnscaler.policies import _tp
from nnscaler.execplan import ExecutionPlan
from nnscaler.execplan.planpass.fusion import DiffFusion
from nnscaler.execplan.planpass.grouping import Grouping
from nnscaler.execplan.planpass.reschedule import (
    OpDependencyGraph,
    Reschedule,
    dump_schedule,
    load_schedule_order,
    config_priority,
    schedule_to_dot,
    visualize_schedule,
)
from nnscaler.graph.segment import IRSegment
from nnscaler.codegen.module.module import ModuleCodeGen
from nnscaler.execplan.execplan import ExeReuseCell

from ..utils import replace_all_device_with


def _sub(shape=(4, 4), requires_grad=True):
    return IRFullTensor(list(shape), requires_grad=requires_grad).tosub()


def _op(name, inputs, output):
    op = IRFwOperation(name, name, inputs=list(inputs), num_outputs=1)
    op.set_output(0, output)
    return op


# ---------------------------------------------------------------------------
# OpDependencyGraph unit tests
# ---------------------------------------------------------------------------

def test_dep_graph_chain():
    """A pure chain has a single legal order."""
    t0 = _sub()
    t1, t2, t3 = _sub(), _sub(), _sub()
    a = _op('a', [t0], t1)
    b = _op('b', [t1], t2)
    c = _op('c', [t2], t3)
    nodes = [a, b, c]

    graph = OpDependencyGraph(nodes)
    assert b in graph.successors(a)
    assert c in graph.successors(b)
    assert c not in graph.successors(a)  # transitive edge not added directly

    assert graph.topological_sort() == [a, b, c]
    assert graph.is_valid_order([a, b, c])
    assert not graph.is_valid_order([b, a, c])
    assert not graph.is_valid_order([a, c, b])


def test_dep_graph_independent_branches_can_reorder():
    """Two independent producers feeding a common consumer can be reordered."""
    t0 = _sub()
    t1, t2, t3 = _sub(), _sub(), _sub()
    a = _op('a', [t0], t1)   # branch 1
    b = _op('b', [t0], t2)   # branch 2, independent of a
    c = _op('c', [t1, t2], t3)  # joins both branches
    nodes = [a, b, c]

    graph = OpDependencyGraph(nodes)
    # both branches must run before the join
    assert c in graph.successors(a)
    assert c in graph.successors(b)
    # the two branches are independent of each other
    assert b not in graph.successors(a)
    assert a not in graph.successors(b)

    # default (stable) order keeps the original order
    assert graph.topological_sort() == [a, b, c]
    # a different priority yields a different but still legal order
    reordered = graph.topological_sort(priority=lambda n: 0 if n is b else 1)
    assert reordered == [b, a, c]
    assert graph.is_valid_order(reordered)
    # the join can never move before a producer
    assert not graph.is_valid_order([c, a, b])


def test_dep_graph_write_after_read_and_write():
    """Reuse of a tensor region creates anti / output dependencies."""
    t0 = _sub()
    shared_parent = IRFullTensor([4, 4], requires_grad=True)
    s_a = shared_parent.tosub()           # a's output
    s_c = shared_parent.tosub()           # c's output (same parent, overlaps s_a)
    out_b = _sub()
    a = _op('a', [t0], s_a)               # writes the shared region
    b = _op('b', [s_a], out_b)            # reads the shared region
    c = _op('c', [t0], s_c)               # writes the shared region again
    nodes = [a, b, c]

    graph = OpDependencyGraph(nodes)
    assert b in graph.successors(a)       # RAW: b reads what a wrote
    assert c in graph.successors(a)       # WAW: c overwrites a's output
    assert c in graph.successors(b)       # WAR: c overwrites what b read
    # the only legal order is the original one
    assert graph.topological_sort() == [a, b, c]
    assert not graph.is_valid_order([a, c, b])


def test_dep_graph_serializes_communication():
    """Communication operators keep their relative order in every topo order."""
    x = _sub()
    y = _sub()
    z = _sub()
    comm1 = IRAdapter([x], [x])
    compute = _op('compute', [z], _sub())   # independent compute op
    comm2 = IRAdapter([y], [y])
    nodes = [comm1, compute, comm2]

    graph = OpDependencyGraph(nodes)
    # the two communication ops are serialized even though they touch
    # independent tensors
    assert comm2 in graph.successors(comm1)

    # even when a priority tries to schedule comm2 first, comm1 stays before comm2
    order = graph.topological_sort(priority=lambda n: 0 if n is comm2 else 1)
    assert order.index(comm1) < order.index(comm2)
    assert graph.is_valid_order(order)


def test_async_recv_can_be_hoisted_early():
    """An input-less async recv (irecv) can be scheduled early to overlap compute.

    This models the async-communication overlap pattern: a receive that only
    *produces* data (it gets its value from a remote rank, so it has no local
    input) does not depend on anything and can therefore be issued as early as
    possible, instead of right before the operator that consumes it.  The compute
    in between then overlaps with the in-flight communication; the wait happens
    implicitly when the consumer reads the received tensor.
    """
    x = _sub()
    y = _sub()                              # value received from a remote rank
    a = _op('compute_a', [x], _sub())
    b = _op('compute_b', [a.output(0)], _sub())
    c = _op('compute_c', [b.output(0)], _sub())
    recv = IRAdapter([], [y])               # irecv: no local input, produces y
    consume = _op('consume', [y, c.output(0)], _sub())

    # original (bad) order: recv sits right before its consumer
    nodes = [a, b, c, recv, consume]
    graph = OpDependencyGraph(nodes)

    # the recv has no dependency: nothing must run before it
    assert all(dst is not recv for _, dst, _ in graph.edges())
    # it is a dependency of its consumer
    assert consume in graph.successors(recv)

    # a communication-early priority hoists the recv to the very front
    order = graph.topological_sort(priority=lambda n: (0 if isinstance(n, IRAdapter) else 1))
    assert order[0] is recv
    assert order.index(recv) < order.index(a)       # before all compute
    assert order.index(consume) > order.index(c)    # wait stays at the consumer
    assert graph.is_valid_order(order)


def test_async_recv_pair_keeps_order():
    """Two async recvs keep their relative order in any schedule (no deadlock)."""
    y1, y2 = _sub(), _sub()
    recv1 = IRAdapter([], [y1])
    recv2 = IRAdapter([], [y2])
    compute = _op('compute', [_sub()], _sub())
    nodes = [recv1, compute, recv2]

    graph = OpDependencyGraph(nodes)
    # even trying to schedule communication last, recv1 stays before recv2
    order = graph.topological_sort(priority=lambda n: (1 if isinstance(n, IRAdapter) else 0))
    assert order.index(recv1) < order.index(recv2)
    assert graph.is_valid_order(order)


def test_dep_graph_unwraps_exereusecell_for_comm():
    """ExeReuseCell-wrapped adapters (pipeline schedules) are treated as comm."""
    x1, y1 = _sub(), _sub()
    x2, y2 = _sub(), _sub()
    adapter1 = IRAdapter([x1], [y1])
    adapter2 = IRAdapter([x2], [y2])
    # wrap the adapters as they would be in a pipeline schedule sequence
    reuse1 = ExeReuseCell(adapter1, [x1], [y1])
    reuse2 = ExeReuseCell(adapter2, [x2], [y2])
    compute = _op('compute', [_sub()], _sub())
    nodes = [reuse1, compute, reuse2]

    graph = OpDependencyGraph(nodes, serialize_segments=True, comm_types=(IRAdapter,))
    # both wrapped adapters are recognized as communication and serialized
    assert graph._is_comm(reuse1) and graph._is_comm(reuse2)
    assert reuse2 in graph.successors(reuse1)         # comm-serialization edge
    # the non-comm anchor keeps its place; the wrapped adapters may still reorder
    order = graph.topological_sort(priority=lambda n: 0 if n is reuse2 else 1)
    assert order.index(reuse1) < order.index(reuse2)  # relative comm order preserved
    assert graph.is_valid_order(order)


def test_dep_graph_topo_sort_always_valid():
    """A topo sort with an arbitrary priority is always a legal order."""
    t0 = _sub()
    t1, t2, t3, t4 = _sub(), _sub(), _sub(), _sub()
    a = _op('a', [t0], t1)
    b = _op('b', [t1], t2)
    c = _op('c', [t0], t3)
    d = _op('d', [t2, t3], t4)
    nodes = [a, b, c, d]

    graph = OpDependencyGraph(nodes)
    for seed in range(5):
        order = graph.topological_sort(priority=lambda n, s=seed: hash((id(n), s)))
        assert graph.is_valid_order(order)
        assert set(order) == set(nodes)


# ---------------------------------------------------------------------------
# Reschedule (segment level) unit test
# ---------------------------------------------------------------------------

def test_reschedule_segment_keeps_validity_and_consistency():
    t0 = _sub((4, 4), requires_grad=False)
    t1, t2, t3 = _sub(), _sub(), _sub()
    a = _op('a', [t0], t1)        # branch 1
    b = _op('b', [t0], t2)        # branch 2 (independent of a)
    c = _op('c', [t1, t2], t3)    # join
    nodes = [a, b, c]
    graph = IRGraph(nodes, [t0], [t3], 'genmodel')
    segment = graph.create_segment([a, b, c])

    original = list(segment.nodes())
    dep = OpDependencyGraph(original)

    # default priority does not change the order (regression-safe)
    changed = Reschedule._reschedule_segment(segment)
    assert not changed
    assert list(segment.nodes()) == original

    # a custom priority reorders the independent branches but stays legal
    changed = Reschedule._reschedule_segment(segment, priority=lambda n: 0 if n is b else 1)
    assert changed
    new_order = list(segment.nodes())
    assert new_order != original
    assert set(new_order) == set(original)
    assert dep.is_valid_order(new_order)

    # producer / consumer bookkeeping stays consistent with the new order
    for ftensor in segment.full_tensors():
        producers = segment.producers(ftensor)
        for producer in producers:
            assert producer in segment.nodes()


# ---------------------------------------------------------------------------
# End-to-end pipeline test (CPU): reschedule must keep the order legal
# ---------------------------------------------------------------------------

class _LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc(x)
        x = x.sum()
        return x


def _pas_tp(graph):
    from nnscaler.policies import _replica
    dataloader = graph.nodes()[0]
    linear = graph.nodes()[1]
    loss = graph.nodes()[2]
    _replica(graph, dataloader, [0, 1])
    _tp(graph, linear, [0, 1], 0, 0)
    _tp(graph, loss, [0, 1], 0, 0)
    return graph


def _build_execplan():
    """Build an execution plan (after the standard plan passes) on CPU."""
    from nnscaler.ir.unique import IDGenerator
    IDGenerator().clear()
    dummy_input = {'x': torch.randn(2, 10)}
    model = _LossModel()
    model.train()
    with tempfile.TemporaryDirectory() as tempdir:
        init_graph, _ = _gen_graph(model, dummy_input, tempdir, constant_folding=True, end2end_mode=True)
        partitioned = _pas_tp(init_graph)
        adapter_graph = IRAdapterGener.gen(partitioned, cost_fn=None)
        execplan = ExecutionPlan.from_graph(adapter_graph)
        execplan = DiffFusion.apply(execplan)
        execplan = Grouping.apply(execplan)
    return execplan


def _fw_segments(execplan):
    return [
        n for n in execplan.seq(0)
        if isinstance(n, IRSegment) and n.isfw()
    ]


@replace_all_device_with('cpu')
def test_reschedule_default_is_noop():
    """With the default priority the order is unchanged (no regression)."""
    execplan = _build_execplan()
    segs = _fw_segments(execplan)
    assert len(segs) >= 1
    original = [list(seg.nodes()) for seg in segs]

    Reschedule.apply(execplan)

    after = [list(seg.nodes()) for seg in segs]
    assert after == original


@replace_all_device_with('cpu')
def test_reschedule_custom_priority_is_legal():
    """A non-default priority keeps the order a legal topological order."""
    execplan = _build_execplan()
    segs = _fw_segments(execplan)
    assert len(segs) >= 1

    # build dependency graphs from the original order (same node objects)
    original = [list(seg.nodes()) for seg in segs]
    dep_graphs = [OpDependencyGraph(nodes) for nodes in original]

    # reschedule the same segments in place with a communication-first priority
    Reschedule.apply(
        execplan,
        priority=lambda n: (0 if isinstance(n, IRAdapter) else 1),
    )

    for dep, seg, orig in zip(dep_graphs, segs, original):
        new_order = list(seg.nodes())
        assert set(new_order) == set(orig)
        assert dep.is_valid_order(new_order)
        # producer / consumer bookkeeping stays consistent
        for ftensor in seg.full_tensors():
            for producer in seg.producers(ftensor):
                assert producer in seg.nodes()


@replace_all_device_with('cpu')
def test_reschedule_codegen_compiles():
    """The module code generated from a rescheduled plan is valid Python."""
    execplan = _build_execplan()
    Reschedule.apply(
        execplan,
        priority=lambda n: (0 if isinstance(n, IRAdapter) else 1),
    )
    code = ModuleCodeGen(execplan).gen(0)
    # must be syntactically valid python
    compile(code, '<gencode>', 'exec')


# ---------------------------------------------------------------------------
# Config-file driven schedule
# ---------------------------------------------------------------------------

def test_config_priority_reorders_by_cid():
    """A config order map reorders independent ops to match the recorded order."""
    t0 = _sub((4, 4), requires_grad=False)
    t1, t2, t3 = _sub(), _sub(), _sub()
    a = _op('a', [t0], t1)        # branch 1
    b = _op('b', [t0], t2)        # branch 2 (independent of a)
    c = _op('c', [t1, t2], t3)    # join
    graph = IRGraph([a, b, c], [t0], [t3], 'genmodel')
    segment = graph.create_segment([a, b, c])

    # config requests b before a; c must stay last (depends on both)
    order_map = {b.cid: 0, a.cid: 1, c.cid: 2}
    changed = Reschedule._reschedule_segment(segment, priority=config_priority(order_map))
    assert changed
    assert list(segment.nodes()) == [b, a, c]


def test_config_priority_illegal_order_is_corrected():
    """An illegal recorded order is corrected to the nearest legal one."""
    t0 = _sub()
    t1, t2, t3 = _sub(), _sub(), _sub()
    a = _op('a', [t0], t1)
    b = _op('b', [t1], t2)
    c = _op('c', [t2], t3)
    dep = OpDependencyGraph([a, b, c])

    # config asks for the reversed (impossible) order
    order_map = {c.cid: 0, b.cid: 1, a.cid: 2}
    order = dep.topological_sort(config_priority(order_map))
    assert order == [a, b, c]            # only legal order
    assert dep.is_valid_order(order)


@replace_all_device_with('cpu')
def test_dump_and_load_schedule_roundtrip():
    """Dumping then loading an unchanged config keeps the schedule unchanged."""
    execplan = _build_execplan()
    segs = _fw_segments(execplan)
    assert len(segs) >= 1

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'schedule.json')
        cfg = dump_schedule(execplan, path)
        assert os.path.exists(path)
        assert cfg['segments']

        order_map = load_schedule_order(path)
        # every operator of every forward segment is recorded
        for seg in segs:
            for node in seg.nodes():
                assert node.cid in order_map

        before = [list(seg.nodes()) for seg in segs]
        Reschedule.apply(execplan, config=path)
        after = [list(seg.nodes()) for seg in segs]
        assert before == after


@replace_all_device_with('cpu')
def test_load_schedule_reversed_is_legal():
    """A reversed recorded order still yields a legal schedule."""
    execplan = _build_execplan()
    segs = _fw_segments(execplan)
    original = [list(seg.nodes()) for seg in segs]
    dep_graphs = [OpDependencyGraph(nodes) for nodes in original]

    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'schedule.json')
        cfg = dump_schedule(execplan, path)

    # reverse the recorded order of every segment
    for seg_cfg in cfg['segments']:
        seg_cfg['order'].reverse()

    Reschedule.apply(execplan, config=cfg)

    for dep, seg, orig in zip(dep_graphs, segs, original):
        new_order = list(seg.nodes())
        assert set(new_order) == set(orig)
        assert dep.is_valid_order(new_order)


@replace_all_device_with('cpu')
def test_dump_schedule_yaml():
    """The schedule can be dumped to and loaded from YAML."""
    import pytest
    pytest.importorskip('yaml')
    execplan = _build_execplan()
    with tempfile.TemporaryDirectory() as tempdir:
        path = os.path.join(tempdir, 'schedule.yaml')
        dump_schedule(execplan, path)
        assert os.path.exists(path)
        order_map = load_schedule_order(path)
        assert len(order_map) > 0


# ---------------------------------------------------------------------------
# Schedule visualization (Graphviz DOT)
# ---------------------------------------------------------------------------

def test_dep_graph_edges_report_kinds():
    """Edges report their dependency kind (raw / war / waw)."""
    t0 = _sub()
    shared_parent = IRFullTensor([4, 4], requires_grad=True)
    s_a = shared_parent.tosub()
    s_c = shared_parent.tosub()
    out_b = _sub()
    a = _op('a', [t0], s_a)
    b = _op('b', [s_a], out_b)
    c = _op('c', [t0], s_c)
    graph = OpDependencyGraph([a, b, c])

    edges = {(s, d): kinds for s, d, kinds in graph.edges()}
    assert 'raw' in edges[(a, b)]
    assert 'waw' in edges[(a, c)]
    assert 'war' in edges[(b, c)]


def test_dep_graph_edges_report_comm_kind():
    """Communication serialization edges are reported with the 'comm' kind."""
    x, y = _sub(), _sub()
    comm1 = IRAdapter([x], [x])
    comm2 = IRAdapter([y], [y])
    compute = _op('compute', [_sub()], _sub())
    graph = OpDependencyGraph([comm1, compute, comm2])

    edges = {(s, d): kinds for s, d, kinds in graph.edges()}
    assert edges[(comm1, comm2)] == ('comm',)


@replace_all_device_with('cpu')
def test_schedule_to_dot_contents():
    """The DOT visualization contains the segment nodes and dependency edges."""
    execplan = _build_execplan()
    dot = schedule_to_dot(execplan)
    assert dot.startswith('digraph')
    assert 'segment cid=' in dot
    assert 'cid=' in dot
    assert '->' in dot          # has edges
    assert 'constraint=false' in dot  # dependency arrows


@replace_all_device_with('cpu')
def test_visualize_schedule_writes_dot():
    """`visualize_schedule` writes a .dot file (and falls back to .dot for images)."""
    execplan = _build_execplan()
    dot = schedule_to_dot(execplan)
    with tempfile.TemporaryDirectory() as tempdir:
        # explicit .dot path
        dot_path = os.path.join(tempdir, 'sched.dot')
        visualize_schedule(execplan, dot_path)
        assert os.path.exists(dot_path)
        assert open(dot_path).read() == dot

        # image path without graphviz still produces a .dot fallback
        png_path = os.path.join(tempdir, 'graph.png')
        visualize_schedule(execplan, png_path)
        assert os.path.exists(os.path.join(tempdir, 'graph.dot'))


# ---------------------------------------------------------------------------
# Cross-segment (execution sequence) rescheduling
# ---------------------------------------------------------------------------

def test_sequence_graph_serializes_anchors():
    """Non-communication nodes keep their relative order; an input-less recv moves."""
    t0 = _sub(requires_grad=False)
    seg_a = _op('seg_a', [t0], _sub())     # stands in for a forward segment producer
    seg_b = _op('seg_b', [seg_a.output(0)], _sub())
    # an input-less recv adapter (irecv) produces a tensor consumed by nobody here
    recv = IRAdapter([], [_sub()])
    nodes = [seg_a, recv, seg_b]

    graph = OpDependencyGraph(nodes, serialize_segments=True, comm_types=(IRAdapter,))
    # recv has no predecessor and can be hoisted to the front
    assert all(dst is not recv for _, dst, _ in graph.edges())
    order = graph.topological_sort(priority=lambda n: (0 if isinstance(n, IRAdapter) else 1))
    assert order[0] is recv
    # the two non-comm anchors keep their relative order
    assert order.index(seg_a) < order.index(seg_b)
    assert graph.is_valid_order(order)


@replace_all_device_with('cpu')
def test_sequence_reschedule_keeps_segments_ordered():
    """The cross-segment reschedule never reorders segments relative to each other."""
    execplan = _build_execplan()
    seq = execplan.seq(0)
    segs_before = [n for n in seq if isinstance(n, IRSegment)]
    assert len(segs_before) >= 2          # forward + backward

    Reschedule.apply(
        execplan,
        scope='sequence',
        priority=lambda n: (0 if isinstance(n, (IRAdapter, IRWeightReducer)) else 1),
    )

    segs_after = [n for n in execplan.seq(0) if isinstance(n, IRSegment)]
    assert segs_after == segs_before      # same segments, same relative order


@replace_all_device_with('cpu')
def test_sequence_hoists_async_recv_before_segments():
    """An async recv in the execution sequence is hoisted ahead of the segments."""
    execplan = _build_execplan()
    seq = list(execplan.seq(0))
    segs = [n for n in seq if isinstance(n, IRSegment)]
    reducers = [n for n in seq if isinstance(n, IRWeightReducer)]
    assert len(segs) >= 2

    # craft an input-less async recv that produces a fresh tensor, placed late
    recv = IRAdapter([], [_sub()])
    nodes = seq + [recv]

    # only adapters move; segments AND reducers are anchors (kept in relative order)
    graph = OpDependencyGraph(
        nodes, serialize_segments=True, comm_types=(IRAdapter,))

    # the non-comm anchors keep their relative order in the dependency graph
    anchors = [n for n in nodes if not isinstance(n, IRAdapter)]
    for i in range(len(anchors) - 1):
        assert anchors[i + 1] in graph.successors(anchors[i])

    # communication-early priority hoists the recv to the very front
    order = graph.topological_sort(
        priority=lambda n: (0 if isinstance(n, IRAdapter) else 1))
    assert order[0] is recv
    assert [n for n in order if isinstance(n, IRSegment)] == segs
    # reducers stay after the segments (their backward dependency is implicit)
    for r in reducers:
        for s in segs:
            assert order.index(s) < order.index(r)
    assert graph.is_valid_order(order)


@replace_all_device_with('cpu')
def test_sequence_scope_codegen_compiles():
    """Code generated after a cross-segment reschedule is valid Python."""
    execplan = _build_execplan()
    Reschedule.apply(
        execplan,
        scope='both',
        priority=lambda n: (0 if isinstance(n, (IRAdapter, IRWeightReducer)) else 1),
    )
    code = ModuleCodeGen(execplan).gen(0)
    compile(code, '<gencode>', 'exec')



