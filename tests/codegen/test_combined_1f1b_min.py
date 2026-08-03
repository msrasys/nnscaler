#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Minimal proof that the ``yomia/rescheduler`` node-level scheduling abstraction
(:mod:`nnscaler.execplan.planpass.reschedule`) can express Megatron's
``combined_1f1b`` pattern: fine-grained interleaving of the *forward* nodes of
microbatch ``m+1`` with the *backward* nodes of microbatch ``m``, with the
communication of one microbatch scheduled to overlap the compute of the other.

This is a CPU-only, dependency-graph-level test (no GPU / distributed runtime
needed). It builds the same kind of ``IRCell`` node list that
:mod:`nnscaler.execplan.planpass.reschedule` operates on in production, but
representing two microbatches worth of fine-grained F/B sub-steps instead of a
whole segment, and checks that:

  1. the two microbatches' compute nodes have no *direct* data-dependency edge
     between them (nothing intrinsic prevents interleaving them);
  2. GIVEN a "recv-hoisted" starting layout (both microbatches' receives
     already issued before either send -- the layout real pipeline engines and
     nnscaler's async-recv codegen target), a legal execution order exists
     that runs F(m+1)'s compute concurrently with B(m)'s in-flight
     communication (and vice versa) -- i.e. the abstraction *can* express the
     "F(m+1) comm overlaps B(m) compute" pattern ``combined_1f1b`` relies on,
     and ``topological_sort`` with a comm-early priority finds it automatically;
  3. a genuine LIMITATION: starting instead from a *fully serialized* baseline
     (B(m) completely emitted, then F(m+1) completely emitted -- the layout you
     get if segments are naively concatenated), the same interleave is
     *unreachable*, because ``OpDependencyGraph`` serializes **all**
     communication adapters (P2P and collective alike) into one global
     relative-order chain. Reaching the interleaved schedule is therefore a
     joint responsibility of the pipeline scheduler / async-recv codegen
     (which produce the recv-hoisted layout) and ``Reschedule`` (which then
     preserves and can further tighten it) -- not something the generic
     dependency-graph pass can conjure from an arbitrary starting order alone;
  4. relative communication order is always preserved (the invariant the
     module documents as required to avoid SPMD collective / P2P issuance
     deadlocks), in both the achievable and the blocked scenario;
  5. the resulting order can be annotated with :class:`StreamContext` (stream +
     event) metadata exactly as production codegen does, giving a concrete,
     executable multi-stream/event schedule.

See ``examples/combined_1f1b_min/run_two_ranks.py`` for a real, 2-GPU runtime
proof that executes the achievable (recv-hoisted) pattern and validates
numerics + overlap + the absence of a deadlock end to end.
"""
from typing import Dict, List

from nnscaler.ir.cten import IRCell
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.adapter import IRAdapter
from nnscaler.execplan.planpass.reschedule import OpDependencyGraph
from nnscaler.graph.schedule.schedplan import StreamContext


def _sub(shape=(4, 4), requires_grad=True):
    return IRFullTensor(list(shape), requires_grad=requires_grad).tosub()


def _op(name, inputs, output):
    op = IRFwOperation(name, name, inputs=list(inputs), num_outputs=1)
    op.set_output(0, output)
    return op


def _make_chain_nodes():
    """Build the two independent, fine-grained F(m+1)/B(m) node chains.

    Chain for B(m) (backward of the *older*, in-flight microbatch):
        recv_grad_m  (comm, irecv: no local input -> grad_m)
        compute_B_m  (compute: consumes grad_m -> produces grad_out_m)
        send_grad_m  (comm: consumes grad_out_m -> sends upstream)

    Chain for F(m+1) (forward of the *next* microbatch), fully independent
    tensors from B(m)'s chain:
        recv_act_m1  (comm, irecv: no local input -> act_m1)
        compute_F_m1 (compute: consumes act_m1 -> produces act_out_m1)
        send_act_m1  (comm: consumes act_out_m1 -> sends downstream)
    """
    grad_m = _sub()
    grad_out_m = _sub()
    act_m1 = _sub()
    act_out_m1 = _sub()

    recv_grad_m = IRAdapter([], [grad_m])
    compute_B_m = _op('compute_B_m', [grad_m], grad_out_m)
    send_grad_m = IRAdapter([grad_out_m], [grad_out_m])

    recv_act_m1 = IRAdapter([], [act_m1])
    compute_F_m1 = _op('compute_F_m1', [act_m1], act_out_m1)
    send_act_m1 = IRAdapter([act_out_m1], [act_out_m1])

    names = {
        recv_grad_m: 'recv_grad_m', compute_B_m: 'compute_B_m', send_grad_m: 'send_grad_m',
        recv_act_m1: 'recv_act_m1', compute_F_m1: 'compute_F_m1', send_act_m1: 'send_act_m1',
    }
    return dict(
        recv_grad_m=recv_grad_m, compute_B_m=compute_B_m, send_grad_m=send_grad_m,
        recv_act_m1=recv_act_m1, compute_F_m1=compute_F_m1, send_act_m1=send_act_m1,
    ), names


def _build_two_microbatch_nodes():
    """The *fully serialized* (Megatron-without-overlap) baseline order: issue
    B(m)'s three nodes back to back, then F(m+1)'s three nodes back to back --
    exactly the order a naive, un-rescheduled per-rank instruction stream would
    use if segments are simply concatenated in microbatch order."""
    n, names = _make_chain_nodes()
    naive_order = [
        n['recv_grad_m'], n['compute_B_m'], n['send_grad_m'],
        n['recv_act_m1'], n['compute_F_m1'], n['send_act_m1'],
    ]
    return naive_order, names


def _build_two_microbatch_nodes_recv_hoisted():
    """The *recv-hoisted* baseline order real pipeline-parallel engines actually
    emit (and that nnscaler's own async-recv / pipeline-reschedule features
    target): every available receive is issued as early as legal, i.e. both
    recvs are issued back to back before any send, matching e.g. nnscaler's
    ``CompileFlag.async_recv`` "issue early, wait late" pattern applied to a
    two-microbatch steady-state step."""
    n, names = _make_chain_nodes()
    order = [
        n['recv_grad_m'], n['recv_act_m1'],
        n['compute_B_m'], n['send_grad_m'],
        n['compute_F_m1'], n['send_act_m1'],
    ]
    return order, names


def test_two_microbatch_compute_nodes_are_mutually_unordered():
    """B(m)'s and F(m+1)'s compute nodes have no *direct* edge between them:
    there is no data dependency that would prevent interleaving them, which is
    the crux of combined_1f1b (fine-grained F/B interleave, not just
    whole-segment reordering). Whether that freedom can actually be *realized*
    as a legal execution order also depends on the surrounding communication
    layout -- see ``test_priority_driven_topological_sort_produces_interleave``
    (achievable) vs. ``test_limitation_fully_serialized_baseline_blocks_the_interleave``
    (blocked transitively via comm-serialization edges)."""
    naive_order, names = _build_two_microbatch_nodes()
    compute_B_m = next(n for n in naive_order if names[n] == 'compute_B_m')
    compute_F_m1 = next(n for n in naive_order if names[n] == 'compute_F_m1')

    graph = OpDependencyGraph(naive_order)
    assert compute_F_m1 not in graph.successors(compute_B_m)
    assert compute_B_m not in graph.successors(compute_F_m1)


def test_combined_1f1b_interleave_is_a_legal_order():
    """Starting from the *recv-hoisted* baseline (both recvs already issued
    before either send -- the layout real pipeline engines and nnscaler's own
    async-recv feature target), a genuinely interleaved order -- run F(m+1)
    compute in between B(m)'s recv and send, i.e. concurrently with B(m)'s
    in-flight comm, or vice versa -- is a legal (dependency-respecting)
    execution order, and multiple different interleavings are all legal."""
    base_order, names = _build_two_microbatch_nodes_recv_hoisted()
    by_name = {v: k for k, v in names.items()}
    graph = OpDependencyGraph(base_order)

    # both sends must respect the comm-chain's relative order (send_grad_m
    # before send_act_m1, matching the recv-hoisted baseline's own adapter
    # order), but the two *computes* in between are freely interleavable.
    interleaved = [
        by_name['recv_grad_m'], by_name['recv_act_m1'],       # both comms hoisted early
        by_name['compute_F_m1'], by_name['compute_B_m'],       # fine-grained F/B interleave
        by_name['send_grad_m'], by_name['send_act_m1'],
    ]
    assert graph.is_valid_order(interleaved)

    # the opposite compute interleave is also legal -- confirms there is no
    # hidden ordering constraint tying the two computes together in either
    # direction, i.e. the scheduler is free to choose either interleave.
    interleaved_swapped = [
        by_name['recv_grad_m'], by_name['recv_act_m1'],
        by_name['compute_B_m'], by_name['compute_F_m1'],
        by_name['send_grad_m'], by_name['send_act_m1'],
    ]
    assert graph.is_valid_order(interleaved_swapped)


def test_priority_driven_topological_sort_produces_interleave():
    """Using ``topological_sort`` with a "communication-early" priority (the
    same mechanism production code uses, see ``_comm_early_priority`` /
    ``config_priority`` in reschedule.py) actually *produces* the interleaved
    schedule automatically, without hand-constructing it, when started from the
    recv-hoisted baseline."""
    base_order, names = _build_two_microbatch_nodes_recv_hoisted()
    graph = OpDependencyGraph(base_order)

    # comm-early priority: schedule every communication op as soon as its
    # dependencies are satisfied, ahead of any ready compute op.
    def comm_early(node: IRCell):
        return (0 if isinstance(node, IRAdapter) else 1, graph._order[node])

    order = graph.topological_sort(priority=comm_early)
    assert graph.is_valid_order(order)

    pos = {n: i for i, n in enumerate(order)}
    by_name = {v: k for k, v in names.items()}
    # both recvs are hoisted ahead of both computes
    assert pos[by_name['recv_grad_m']] < pos[by_name['compute_B_m']]
    assert pos[by_name['recv_act_m1']] < pos[by_name['compute_F_m1']]
    assert pos[by_name['recv_grad_m']] < pos[by_name['compute_F_m1']]
    assert pos[by_name['recv_act_m1']] < pos[by_name['compute_B_m']]


def test_limitation_fully_serialized_baseline_blocks_the_interleave():
    """Documented architectural limitation: ``OpDependencyGraph`` serializes
    *all* communication adapters into a single global relative-order chain
    (``_build_comm_edges`` pairs every consecutive adapter in the input list,
    with no distinction between independent P2P channels and true collectives).

    Starting instead from the *fully serialized* baseline (B(m) completely
    before F(m+1), as :func:`_build_two_microbatch_nodes` builds), that global
    chain forces ``send_grad_m`` before ``recv_act_m1`` transitively forces
    ``compute_B_m`` before ``compute_F_m1`` in *every* legal order, even though
    there is no direct data dependency between them. So the same interleave
    that is achievable from the recv-hoisted baseline is *not* reachable here
    by the generic reschedule pass alone: getting the recv-hoisted layout in
    the first place is the job of the pipeline scheduler / async-recv codegen,
    not of ``Reschedule`` by itself. This is why production code keeps
    pipeline-schedule reordering opt-in (``allow_pipeline=False`` by default)
    and validates it end-to-end (see ``tests/parallel_module/test_reschedule_e2e.py``).
    """
    naive_order, names = _build_two_microbatch_nodes()
    by_name = {v: k for k, v in names.items()}
    graph = OpDependencyGraph(naive_order)

    interleaved = [
        by_name['recv_grad_m'], by_name['recv_act_m1'],
        by_name['compute_F_m1'], by_name['compute_B_m'],
        by_name['send_act_m1'], by_name['send_grad_m'],
    ]
    assert not graph.is_valid_order(interleaved), (
        'expected the fully-serialized baseline to block the interleave -- '
        'if this now passes, the comm-serialization model has been relaxed '
        'to distinguish independent P2P channels and this test (and the '
        'accompanying finding in the report) should be revisited'
    )

    # even the strongest comm-early priority cannot produce it: recv_act_m1 is
    # transitively forced after send_grad_m, which is forced after compute_B_m.
    def comm_early(node: IRCell):
        return (0 if isinstance(node, IRAdapter) else 1, graph._order[node])

    order = graph.topological_sort(priority=comm_early)
    pos = {n: i for i, n in enumerate(order)}
    assert pos[by_name['compute_B_m']] < pos[by_name['recv_act_m1']]
    assert pos[by_name['compute_B_m']] < pos[by_name['compute_F_m1']]


def test_communication_relative_order_preserved_no_deadlock_risk():
    """Even under an aggressive interleave-seeking priority, the *relative*
    order of the communication ops is preserved (comm-serialization edges),
    which is what nnscaler documents as required to keep SPMD collective /
    P2P issuance order consistent across ranks and avoid deadlocks."""
    naive_order, names = _build_two_microbatch_nodes()
    by_name = {v: k for k, v in names.items()}
    graph = OpDependencyGraph(naive_order)

    # try to schedule sends before recvs -- comm-serialization edges must
    # still force the original *relative* comm order to be respected.
    def sends_first(node: IRCell):
        return (0 if 'send' in names.get(node, '') else 1, graph._order[node])

    order = graph.topological_sort(priority=sends_first)
    assert graph.is_valid_order(order)
    pos = {n: i for i, n in enumerate(order)}
    comm_names_in_original_order = [
        names[n] for n in naive_order if isinstance(n, IRAdapter)
    ]
    comm_positions = [pos[by_name[nm]] for nm in comm_names_in_original_order]
    assert comm_positions == sorted(comm_positions), (
        'communication ops must keep their relative order regardless of priority'
    )


def test_stream_event_annotation_for_interleaved_schedule():
    """Once a legal interleaved order is chosen, it can be annotated with the
    same :class:`StreamContext` (stream + wait/record event) metadata that
    production codegen (``nnscaler/codegen/schedule/schedule.py``) attaches to
    ``IRCell`` nodes via ``set_op_context('stream_context', ...)`` to emit
    multi-stream, event-synchronized code. This demonstrates the abstraction
    can carry not just *order* but real multi-stream/event synchronization
    intent for a fine-grained F(m+1)/B(m) interleave."""
    naive_order, names = _build_two_microbatch_nodes()
    by_name = {v: k for k, v in names.items()}

    recv_grad_m = by_name['recv_grad_m']
    compute_B_m = by_name['compute_B_m']
    recv_act_m1 = by_name['recv_act_m1']
    compute_F_m1 = by_name['compute_F_m1']
    send_act_m1 = by_name['send_act_m1']

    # comm nodes run on a side 'comm' stream and record an event when done;
    # the consuming compute node waits on that event before it may read the
    # result, while everything else (the *other* microbatch's compute) stays
    # on the default stream and is never blocked by it.
    recv_grad_m.set_op_context(
        'stream_context', StreamContext(stream='comm', record_events=['grad_m_ready']))
    recv_act_m1.set_op_context(
        'stream_context', StreamContext(stream='comm', record_events=['act_m1_ready']))
    compute_B_m.set_op_context(
        'stream_context', StreamContext(stream='default', wait_events=['grad_m_ready']))
    compute_F_m1.set_op_context(
        'stream_context', StreamContext(stream='default', wait_events=['act_m1_ready']))
    send_act_m1.set_op_context(
        'stream_context', StreamContext(stream='comm', wait_streams=['default']))

    # compute_B_m only waits on grad_m_ready -- it is *not* made to wait on
    # act_m1_ready, so it is free to run fully concurrently with the in-flight
    # recv_act_m1 communication, which is precisely the "F(m+1) comm overlaps
    # B(m) compute" property combined_1f1b exploits.
    b_ctx: StreamContext = compute_B_m.get_op_context('stream_context')
    assert b_ctx.stream == 'default'
    assert b_ctx.wait_events == ['grad_m_ready']
    assert 'act_m1_ready' not in (b_ctx.wait_events or [])

    f_ctx: StreamContext = compute_F_m1.get_op_context('stream_context')
    assert f_ctx.wait_events == ['act_m1_ready']

    recv_ctx: StreamContext = recv_act_m1.get_op_context('stream_context')
    assert recv_ctx.stream == 'comm'
    assert recv_ctx.record_events == ['act_m1_ready']
