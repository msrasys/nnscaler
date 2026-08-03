#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
Global (cross-rank) fixed communication schedule.

``Reschedule.apply(..., scope='sequence')`` (see :mod:`.reschedule`) solves an
*independent* topological sort per device: each device's
:class:`~.reschedule.OpDependencyGraph` is built only from that device's own
node list, so a rank's schedule is (re)discovered on its own, with no shared
structure across ranks beyond the coincidence that they all started from the
same pre-rescheduled plan ("per-rank greedy edge deletion"). That pass also
serializes *all* communication adapters (P2P and collective alike) into a
single relative-order chain per device, which is safe but can be overly
conservative: a hoisted receive can be blocked from moving past unrelated
sends/receives on the *same* device that happen to sit in between it and the
desired hoist target, even though those events have no real data dependency
on it (see ``tests/codegen/test_combined_1f1b_min.py`` and the docstring of
``tests/parallel_module/test_combined_1f1b_pipeline_e2e.py`` for a concrete,
measured case).

:class:`GlobalCommSchedule` targets the same goal
(``issue(F(m+1)) < B(m)`` [a full backward segment] ``< wait(F(m+1)) <
F(m+1)``, i.e. genuinely overlapping a hoisted receive with a full
neighboring compute segment -- Megatron's ``combined_1f1b`` pattern) via:

1. build **one** dependency graph over the nodes of **every** device
   together (nodes are already dispatched to a single device each by
   :class:`~nnscaler.execplan.execplan.ExecutionPlan`, so this is simply the
   concatenation of every device's sequence);
2. run **one** topological sort over that graph with a communication-early
   priority;
3. **project** the single resulting order back onto each device (filter to
   that device's own nodes, keeping their relative order from the shared
   result).

Every device's final sequence is therefore a sub-sequence of the *same*
shared total order -- consistent by construction -- rather than an
independently re-solved order. This is what "same global schedule projected
to all ranks, not per-rank greedy edge deletion" means concretely. (Why a
*global* graph is safe to build in the first place: ``OpDependencyGraph``'s
data-dependency edges are derived purely from each node's own IR input/output
objects, independent of which device it lives on, so combining multiple
devices' nodes into one graph does not, by itself, invent any incorrect
dependency; only the *comm-serialization* edges need a deliberate policy,
which is the ``channel_key`` below.)

Comm-serialization channel key, and the safety story
-----------------------------------------------------
A naive attempt to get more hoisting distance is to split the single
per-device chain into one chain per *directed* peer (e.g. "device 0's sends
to device 1" separate from "device 0's receives from device 1"), or even just
by direction alone (a device's receives from a peer separate from its other
traffic -- sends, collectives -- with that *same* peer). Both were tried in
this and an earlier investigation of this same problem (see the
``implement-fix`` todo history) and **both were empirically confirmed to
deadlock on real GPUs**:

- Concurrent, unbatched, bidirectional point-to-point communication between
  the *same two* ranks (one side's outstanding send racing another side's
  outstanding receive, or vice versa) deadlocks with this environment's
  PyTorch/NCCL point-to-point implementation, with a uniformly-random *and*
  with a fixed/deterministic issue order, warmed up or not. Only issuing such
  a send+recv pair *jointly* via ``torch.distributed.batch_isend_irecv``
  avoided the hang; batching each op *separately* through that same API was
  not sufficient.
- Splitting the per-device chain by *direction only* (§, ``(device,
  peer-pair, direction)``, keeping send/recv independent per peer) was
  believed safe on the reasoning "sends stay synchronous when
  ``CompileFlag.async_comm`` is off, so they are never outstanding". That
  reasoning is **insufficient**: a rank's *synchronous* send to peer P can
  still deadlock while some *other*, unrelated, still-outstanding
  *asynchronous receive from that same peer P* (a different channel/cid --
  e.g. a backward-gradient receive left outstanding while a forward-activation
  send executes) is in flight. This was reproduced concretely: a real 2-stage
  pipeline hung with 100%-utilized (busy-polling), unresponsive GPUs; the
  hang was isolated (via traced executor calls) to exactly this pattern --
  rank 0 blocked forever inside a plain, synchronous ``send`` while two of its
  own asynchronous receives from the same peer were still outstanding.

The hazard is therefore about the **undirected peer-pair as a whole**, not
about any one channel, direction, or synchronous-vs-asynchronous distinction:
if *anything* outstanding remains unresolved between two ranks, no *further*
P2P operation between them (send or receive, sync or async) is safe until it
resolves. Consequently:

- **Construction** groups communication nodes for the serialization chain by
  :func:`peer_pair_channel_key` -- ``(device, peer-pair)``, deliberately not
  split by direction or channel: *all* of a device's P2P traffic with a given
  peer -- every send and every receive, of every channel -- stays in one
  relatively-ordered chain, so nothing on that peer-pair is ever left
  concurrently outstanding with anything else on it. This is more
  conservative than the (deadlocking) direction-split alternative -- see
  :func:`unsafe_direction_split_channel_key` for that dead end, kept
  documented so it is not rediscovered the hard way -- but it is what is
  actually safe to ship without an additional mechanism (e.g. dedicated
  per-channel process groups) that this module does not implement. Distinct
  peer-*pairs* (e.g. ranks (0, 1) vs (1, 2) in a 3-stage pipeline) remain free
  to reorder relative to each other, which is where this pass still
  meaningfully generalizes past a single global per-device chain.
- **Validation** (:func:`GlobalCommSchedule.validate`) is a defensive,
  testable safety net using the same peer-pair grouping: it simulates every
  hoistable receive's issue/first-local-consumer window and raises
  :class:`GlobalScheduleError` if more than the configured
  ``max_outstanding`` are concurrently active for one peer-pair, rather than
  silently risking a hang.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Hashable, List, Optional, Tuple
import logging

from nnscaler.ir.cten import IRCell
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import MovePrim

from nnscaler.execplan import ExecutionPlan
from nnscaler.execplan.execplan import ExeReuseCell
from nnscaler.execplan.planpass.planpass import PlanPass
from nnscaler.execplan.planpass.reschedule import OpDependencyGraph, _comm_early_priority
from nnscaler.flags import CompileFlag


_logger = logging.getLogger(__name__)


class GlobalScheduleError(RuntimeError):
    """Illegal configuration, or an unsafe schedule detected, by
    :class:`GlobalCommSchedule`. See the module docstring for the safety
    invariant being enforced."""


def _unwrap(node: IRCell) -> IRCell:
    return node.cell if isinstance(node, ExeReuseCell) else node


def device_of(node: IRCell) -> Hashable:
    """The single device id of a node already dispatched by ``ExecutionPlan``
    to exactly one device (used as the ``device_key`` for anchor edges, so
    unrelated devices' non-communication nodes are never forced into a
    relative order with each other)."""
    cell = _unwrap(node)
    if len(cell.device) != 1:
        raise GlobalScheduleError(
            f"GlobalCommSchedule requires every node to already be dispatched "
            f"to a single device; got device={list(cell.device)} for {node}"
        )
    return cell.device[0]


def p2p_peer_pair(node: IRCell) -> Optional[FrozenSet[int]]:
    """The undirected (src, dst) peer-pair of a dispatched P2P ``MovePrim``
    -based adapter, or ``None`` if ``node`` is not one (e.g. a collective).

    Dispatch (:meth:`~nnscaler.ir.adapter.adapter.IRAdapter.dispatch`)
    restricts a node's *tensors* to one device, so a dispatched node's own
    ``.device`` is always a singleton (``[this device]``) and cannot be used
    to recover which *other* device it talks to. ``MovePrim`` preserves its
    original ``src``/``dst`` kwargs unchanged through dispatch (both the
    send-side and receive-side dispatched copies keep the *same* values), so
    those are used instead.
    """
    cell = _unwrap(node)
    if not isinstance(cell, IRAdapter):
        return None
    for prim in cell.prims:
        if isinstance(prim, MovePrim):
            src, dst = prim.kwargs.get('src'), prim.kwargs.get('dst')
            if src is not None and dst is not None:
                return frozenset({src, dst})
    return None


def _is_hoistable_recv(node: IRCell) -> bool:
    """Whether `node` is a pure, zero-input receive adapter -- the kind that
    gets an asynchronously-issued launch + a deferred, per-consumer wait (see
    ``FuncEmission.is_async_recv_adapter`` in ``nnscaler.codegen.emit``, which
    this mirrors structurally; duplicated locally to avoid a codegen
    dependency from the execplan/planpass layer)."""
    cell = _unwrap(node)
    if not isinstance(cell, IRAdapter):
        return False
    if cell.differentiable and cell.custom:
        return False
    prims = list(cell.prims)
    return (
        len(cell.inputs()) == 0
        and len(cell.outputs()) == 1
        and len(prims) > 0
        and isinstance(prims[0], MovePrim)
        and len(prims[0].inputs()) == 0
        and len(prims[0].outputs()) == 1
    )


def peer_pair_channel_key(node: IRCell) -> Optional[Hashable]:
    """Conservative ``(device, peer-pair)`` grouping: both directions of P2P
    traffic between the same two ranks are grouped (and hence stay relatively
    ordered) together. Used by :func:`GlobalCommSchedule.validate` regardless
    of the (possibly more permissive) key used for construction. Non-P2P
    (e.g. collective) nodes fall back to grouping by device alone.

    This is also the *default*, and only verified-safe, ``channel_key`` for
    :func:`GlobalCommSchedule.apply`'s comm-serialization chain: a device's
    receive(s) from a peer and its send(s) to that *same* peer must stay in
    one relatively-ordered chain, not be split into independent groups (see
    :func:`unsafe_direction_split_channel_key` for why that alternative,
    despite giving a larger measured hoist distance, was empirically found to
    deadlock).
    """
    peer = p2p_peer_pair(node)
    if peer is None:
        return (device_of(node), 'non-p2p')
    return (device_of(node), peer)


def unsafe_direction_split_channel_key(node: IRCell) -> Optional[Hashable]:
    """A more permissive ``channel_key`` -- ``(device, peer-pair, direction)``
    -- that separates a device's receives from a given peer from its other
    (send) traffic with that *same* peer, letting receives hoist past
    unrelated sends for a measurably larger overlap window.

    NOT SAFE, and NOT the default: confirmed empirically (real 2-GPU run) to
    deadlock. The reasoning that seemed to justify it -- "safe because sends
    stay synchronous when ``CompileFlag.async_comm`` is off" -- is
    insufficient: a *synchronous* send to a peer can still deadlock if some
    *other*, unrelated, still-outstanding asynchronous receive *from that same
    peer* (on a different channel/cid, e.g. a backward-gradient receive
    outstanding while a forward-activation send happens) is in flight --
    concurrent bidirectional P2P to one peer-pair is the hazard (see module
    docstring), regardless of which specific channel/direction each side
    belongs to. Kept here, unused by default, only to document the dead end
    and prevent re-discovering it the hard way; do not wire this in without
    an additional mechanism (e.g. per-channel process groups, not implemented
    by this module) that removes the hazard.
    """
    peer = p2p_peer_pair(node)
    if peer is None:
        return (device_of(node), 'non-p2p', None)
    direction = 'recv' if _is_hoistable_recv(node) else 'other'
    return (device_of(node), peer, direction)


def cid_channel_key(node: IRCell) -> Optional[Hashable]:
    """Per-callsite ``(device, adapter cid)`` grouping: every distinct
    compiled receive (e.g. "the backward-gradient receive" vs "the
    result-broadcast receive" are different cids even when they share a
    peer-pair) gets its own independent outstanding-count budget. This is the
    default ``cap_key`` for :func:`_cap_aware_order` /
    :meth:`GlobalCommSchedule.validate` -- deliberately *finer* than
    :func:`peer_pair_channel_key` (used for the comm-*ordering* chain, where
    only device/peer matter for safety): two unrelated receive
    channels sharing a peer-pair should not compete for the same lifecycle
    budget just because they happen to share a device and direction. A cid is
    stable across an adapter's repeated per-microbatch invocations (the same
    underlying cell is reused via ``ExeReuseCell``), so this naturally caps
    "how many microbatches' worth of *this* channel may be outstanding at
    once" -- matching the ``channel`` identity
    ``CompileFlag.async_recv_channel`` codegen passes to
    ``AsyncCommHandler.issue_recv`` at runtime (see ``nnscaler.codegen.emit``).
    """
    if not _is_hoistable_recv(node):
        return None
    return (device_of(node), _unwrap(node).cid)


@dataclass
class ScheduleViolation:
    """One detected pair of overlapping same-peer-pair receive windows."""
    devid: int
    channel: Hashable
    node_a: str
    window_a: Tuple[int, int]
    node_b: str
    window_b: Tuple[int, int]

    def __str__(self) -> str:
        return (
            f"device {self.devid} channel {self.channel!r}: "
            f"{self.node_a} window={self.window_a} overlaps "
            f"{self.node_b} window={self.window_b}"
        )


@dataclass
class ScheduleReport:
    """Diagnostics returned by :func:`GlobalCommSchedule.validate`.

    ``hoist_span`` maps each ``(devid, channel)`` (``channel`` per
    ``peer_pair_channel_key``) to the distance (in emitted node positions)
    between a hoisted receive's issue and its first local consumer -- i.e.
    how many other nodes now legally sit *between* the issue and the wait,
    the concrete, checkable proxy for "genuine overlap window" used by the
    unit and end-to-end tests.
    """
    hoist_span: Dict[Tuple[int, Hashable], int] = field(default_factory=dict)
    violations: List[ScheduleViolation] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        return not self.violations


def _cap_aware_order(
    graph: OpDependencyGraph,
    nodes: List[IRCell],
    priority: Callable[[IRCell], Hashable],
    max_outstanding: int,
    cap_key: Callable[[IRCell], Optional[Hashable]] = cid_channel_key,
) -> List[IRCell]:
    """Like ``graph.topological_sort(priority)``, but additionally *defers*
    scheduling a hoistable receive once ``max_outstanding`` receives sharing
    its ``cap_key`` group are already scheduled-but-not-yet-resolved.

    A receive counts as "resolved" the moment any node that data-depends on it
    (its earliest such node, in this order -- i.e. its would-be consumer) is
    scheduled, mirroring where the deferred wait is emitted (see
    ``ScheduleReport``/``ScheduleCodeGen``). This makes the construction
    itself respect the same cap ``AsyncCommHandler.issue_recv`` enforces at
    runtime, rather than relying solely on :func:`GlobalCommSchedule.validate`
    to catch a violation after the fact.
    """
    import heapq

    order_index = {n: i for i, n in enumerate(nodes)}
    # indegree/scheduling order uses the FULL graph (data + comm-chain +
    # anchor edges) -- that is what makes the result a legal topological
    # order at all. But "does scheduling `node` resolve some outstanding
    # receive predecessor" must only look at genuine DATA edges (raw/war/waw):
    # a comm-chain or anchor edge to a receive is a scheduling constraint, not
    # a consumption, and must not be mistaken for one (that would release the
    # outstanding budget without the receive's value actually having been
    # read yet).
    indegree = {n: len(graph.predecessors(n)) for n in nodes}
    data_predecessors: Dict[IRCell, List[IRCell]] = {n: [] for n in nodes}
    for src, dst, kinds in graph.edges():
        if any(k in ('raw', 'war', 'waw') for k in kinds):
            data_predecessors[dst].append(src)

    outstanding: Dict[Hashable, int] = {}
    resolved: set = set()

    def _cap_group(node: IRCell) -> Optional[Hashable]:
        if not _is_hoistable_recv(node):
            return None
        return cap_key(node)

    ready: List[Tuple[Hashable, int]] = []
    blocked: List[IRCell] = []

    def _admit(node: IRCell) -> None:
        group = _cap_group(node)
        if group is not None:
            outstanding[group] = outstanding.get(group, 0) + 1
        heapq.heappush(ready, (priority(node), order_index[node]))

    def _try_admit_or_block(node: IRCell) -> None:
        group = _cap_group(node)
        if group is not None and outstanding.get(group, 0) >= max_outstanding:
            blocked.append(node)
        else:
            _admit(node)

    def _release_blocked() -> None:
        nonlocal blocked
        still_blocked = []
        progressed = True
        while progressed:
            progressed = False
            for node in blocked:
                group = _cap_group(node)
                if group is None or outstanding.get(group, 0) < max_outstanding:
                    _admit(node)
                    progressed = True
                else:
                    still_blocked.append(node)
            blocked, still_blocked = still_blocked, []

    for node in nodes:
        if indegree[node] == 0:
            _try_admit_or_block(node)

    result: List[IRCell] = []
    while ready or blocked:
        if not ready:
            stalled = ', '.join(f'{cap_key(n)!r}' for n in blocked[:5])
            raise GlobalScheduleError(
                f"GlobalCommSchedule: cap-aware scheduling stalled with "
                f"{len(blocked)} node(s) waiting for outstanding capacity "
                f"that never frees (max_outstanding={max_outstanding} is too "
                f"small for this plan's genuine concurrency -- e.g. a "
                f"bulk-drained result broadcast with more in-flight receives "
                f"than the configured cap); affected channel(s): {stalled}"
            )
        _, idx = heapq.heappop(ready)
        node = nodes[idx]
        result.append(node)

        for pred in data_predecessors[node]:
            if pred in resolved:
                continue
            group = _cap_group(pred)
            if group is None:
                continue
            # a data edge to a DIFFERENT device is irrelevant here: an
            # outstanding receive is a per-device concept (it occupies a slot
            # in THAT device's own instruction stream until THAT device's own
            # code reads it) -- some other device's node happening to share
            # enough tensor identity to form a cross-device data edge in the
            # combined graph (observed for a result-broadcast: the original,
            # pre-broadcast value on the producing device can appear as a
            # "successor" of the broadcast receive on another device) must
            # not be treated as resolving it, or the cap would be silently
            # bypassed for exactly the receives that need it most (the ones
            # with no genuine *local* consumer).
            if device_of(pred) != device_of(node):
                continue
            resolved.add(pred)
            outstanding[group] = outstanding.get(group, 0) - 1

        _release_blocked()

        for succ in graph.successors(node):
            indegree[succ] -= 1
            if indegree[succ] == 0:
                _try_admit_or_block(succ)

    assert len(result) == len(nodes), (
        'GlobalCommSchedule: internal error, cap-aware scheduling did not '
        'consume every node (dependency graph not acyclic?)'
    )
    return result


class GlobalCommSchedule(PlanPass):
    """Reschedule every device's execution sequence in ``execplan`` from ONE
    shared, global dependency graph. See the module docstring."""

    @staticmethod
    def apply(
        execplan: ExecutionPlan,
        *,
        max_outstanding: int = 2,
        priority: Optional[Callable[[IRCell], Hashable]] = None,
        channel_key: Callable[[IRCell], Optional[Hashable]] = peer_pair_channel_key,
        cap_key: Callable[[IRCell], Optional[Hashable]] = cid_channel_key,
        validate: bool = True,
    ) -> ExecutionPlan:
        """Reschedule ``execplan`` (modified in place) and return it.

        Args:
            execplan: the execution plan to reschedule. Every device's
                sequence must already be dispatched (each node on exactly one
                device), which holds for any ``ExecutionPlan`` returned by
                ``ExecutionPlan.from_graph`` / ``from_schedplan``.
            max_outstanding: the outstanding-op cap enforced BOTH during
                construction (see :func:`_cap_aware_order`) and by the
                corresponding runtime channel tracking (see
                ``CompileFlag.async_recv_max_outstanding``); also validated
                here (``>= 1``) so a misconfiguration is caught at schedule
                time rather than at the first runtime violation.
            priority: tie-breaking key for the topological sort. Defaults to
                :func:`~.reschedule._comm_early_priority` (issue communication
                as early as legally possible), matching the existing
                ``Reschedule`` convention.
            channel_key: groups communication nodes for the comm-serialization
                *ordering* chain (see module docstring for the safety
                reasoning). Defaults to :func:`peer_pair_channel_key` --
                overriding this to something finer-grained (e.g. splitting by
                direction) is a confirmed deadlock risk without an additional
                mechanism (e.g. per-channel process groups) this module does
                not implement; see :func:`unsafe_direction_split_channel_key`.
            cap_key: groups hoistable receives for the outstanding-count
                *cap*, independent of ``channel_key`` -- deliberately a finer
                grouping (see :func:`cid_channel_key`) so two unrelated
                receive channels sharing a peer-pair/device do not compete
                for the same lifecycle budget. This one IS safe to be finer
                than ``channel_key``: it only bounds how many of one specific
                channel's own invocations may be outstanding, it does not
                change the comm-chain's cross-channel relative ordering.
            validate: when True (default), raise :class:`GlobalScheduleError`
                if the resulting schedule still has any channel (per
                ``cap_key``) exceeding ``max_outstanding`` (see
                :func:`validate`) instead of returning a possibly-unsafe
                schedule.

        Raises:
            GlobalScheduleError: for illegal configuration (``max_outstanding
                < 1``, an empty execution plan, or ``CompileFlag.async_comm``
                enabled at the same time -- asynchronous sends add a further,
                separately-unverified risk on top of the peer-pair chain, see
                module docstring), if cap-aware scheduling stalls
                (``max_outstanding`` too small for the plan's genuine
                concurrency), or, when ``validate`` is set, an unsafe
                resulting schedule.
        """
        if max_outstanding < 1:
            raise GlobalScheduleError(
                f"max_outstanding must be >= 1, got {max_outstanding}"
            )
        if CompileFlag.async_comm:
            raise GlobalScheduleError(
                "GlobalCommSchedule does not support being combined with "
                "CompileFlag.async_comm (asynchronous sends): even with the "
                "safe (undirected peer-pair) channel_key, an asynchronous "
                "send left outstanding is an additional, separately "
                "unverified risk on top of the confirmed peer-pair hazard. "
                "See module docstring for the full reasoning."
            )
        devices = execplan.devices()
        if not devices:
            raise GlobalScheduleError(
                "execution plan has no devices to schedule"
            )

        if priority is None:
            priority = _comm_early_priority

        # ONE shared node list spanning every device (already single-device
        # per node); order here is only the tie-break seed for determinism,
        # NOT a device-relative-order assumption -- devices are otherwise
        # unordered relative to each other except via real dependencies.
        all_nodes: List[IRCell] = []
        for devid in devices:
            all_nodes.extend(execplan.at(devid))

        graph = OpDependencyGraph(
            all_nodes,
            serialize_segments=True,
            comm_types=(IRAdapter,),
            channel_key=channel_key,
            device_key=device_of,
        )
        # a cap-aware order, not the plain `graph.topological_sort(priority)`:
        # the latter has no notion of "outstanding" and will happily hoist
        # every eligible receive on a channel to the front with nothing in
        # between, which can silently exceed `max_outstanding` (verified to
        # occur for e.g. a multi-microbatch result-broadcast, all resolved
        # only by the bulk end-of-step drain). `_cap_aware_order` builds the
        # cap into the construction itself (keyed by `cap_key`, deliberately
        # finer than `channel_key` -- see module docstring); `validate` below
        # remains as a defensive re-check with the same `cap_key`.
        global_order = _cap_aware_order(
            graph, all_nodes, priority, max_outstanding, cap_key=cap_key)
        assert graph.is_valid_order(global_order), \
            'GlobalCommSchedule: internal error, cap-aware order is not a valid topological order'

        # project the single global order back onto each device (filter,
        # preserve relative order) -- this is the "projection", not a
        # per-device re-solve.
        for devid in devices:
            seq = execplan.at(devid)          # the real list, mutated in place
            local_ids = {id(n) for n in seq}
            seq[:] = [n for n in global_order if id(n) in local_ids]

        if validate:
            report = GlobalCommSchedule.validate(
                execplan, max_outstanding=max_outstanding, channel_key=cap_key)
            if not report.is_safe:
                violation_lines = '\n  '.join(str(v) for v in report.violations)
                raise GlobalScheduleError(
                    f"GlobalCommSchedule produced an unsafe schedule: "
                    f"{len(report.violations)} channel(s) exceed "
                    f"max_outstanding={max_outstanding} concurrently-active "
                    f"receives (would exceed the runtime outstanding cap, or "
                    f"risk the buffer/handle lifecycle growing unbounded):\n"
                    f"  {violation_lines}"
                )
        return execplan

    @staticmethod
    def validate(
        execplan: ExecutionPlan,
        *,
        max_outstanding: int = 2,
        channel_key: Callable[[IRCell], Optional[Hashable]] = cid_channel_key,
    ) -> ScheduleReport:
        """Simulate each device's hoistable-receive issue/first-local-consumer
        windows and report any channel where more than ``max_outstanding``
        receives are simultaneously outstanding.

        A receive's "window" is ``[issue_position, wait_position)`` where
        ``wait_position`` is the position of its earliest same-device data
        successor (mirroring where ``ScheduleCodeGen`` emits the deferred
        wait -- right before the first consumer), or the end of the sequence
        if it has none, e.g. a step-output receive drained only in bulk at the
        very end (conservatively "still outstanding at the end"; see
        ``AsyncCommHandler.drain``).

        Note this counts *concurrently outstanding* receives, not merely
        *overlapping* ones: several receives from the same peer legitimately
        being in flight at once (ordinary receive pipelining, resolved FIFO by
        NCCL/``AsyncCommHandler``) is safe and expected -- only exceeding
        ``max_outstanding`` at any point is flagged, mirroring exactly the cap
        ``AsyncCommHandler.issue_recv`` enforces at runtime, so a
        misconfiguration is caught here at schedule time instead of at the
        first runtime violation.
        """
        hoist_span: Dict[Tuple[int, Hashable], int] = {}
        violations: List[ScheduleViolation] = []

        for devid in execplan.devices():
            seq = execplan.at(devid)
            if not seq:
                continue
            position = {n: i for i, n in enumerate(seq)}
            # a plain (single-device) dependency graph, only to look up each
            # receive's local DATA successors -- `comm_types=()` disables the
            # comm-serialization chain entirely for this internal graph, so
            # `successors()` only reflects genuine raw/war/waw data edges, not
            # the (scheduling-only) comm-chain edge to the "next" adapter,
            # which would otherwise dominate the min() below and make every
            # receive look like it is "resolved" by whichever adapter happens
            # to be scheduled right after it.
            dep = OpDependencyGraph(seq, comm_types=())

            windows: List[Tuple[Hashable, int, int, IRCell]] = []
            for node in seq:
                if not _is_hoistable_recv(node):
                    continue
                key = channel_key(node)
                if key is None:
                    continue
                issue_pos = position[node]
                succ_positions = [position[s] for s in dep.successors(node) if s in position]
                wait_pos = min(succ_positions) if succ_positions else len(seq)
                windows.append((key, issue_pos, wait_pos, node))
                span = wait_pos - issue_pos
                hoist_span[(devid, key)] = max(hoist_span.get((devid, key), 0), span)

            by_channel: Dict[Hashable, List[Tuple[int, int, IRCell]]] = {}
            for key, issue_pos, wait_pos, node in windows:
                by_channel.setdefault(key, []).append((issue_pos, wait_pos, node))

            for key, wins in by_channel.items():
                # sweep-line over issue/wait events: track how many windows on
                # this channel are concurrently active, flagging any point
                # where that count exceeds max_outstanding. Ties are broken so
                # a resolution (-1) at position p is processed before a new
                # issue (+1) at the same position p -- i.e. "wait, then the
                # next issue" is not counted as a moment of (count+1) overlap.
                events: List[Tuple[int, int, IRCell]] = []
                for issue_pos, wait_pos, node in wins:
                    events.append((issue_pos, 1, node))
                    events.append((wait_pos, -1, node))
                events.sort(key=lambda e: (e[0], e[1]))

                active: List[IRCell] = []
                active_window: Dict[IRCell, Tuple[int, int]] = {
                    node: (issue_pos, wait_pos) for issue_pos, wait_pos, node in wins
                }
                for _, delta, node in events:
                    if delta == 1:
                        active.append(node)
                        if len(active) > max_outstanding:
                            newest = active[-1]
                            prior = active[-2]
                            violations.append(ScheduleViolation(
                                devid=devid, channel=key,
                                node_a=repr(prior), window_a=active_window[prior],
                                node_b=repr(newest), window_b=active_window[newest],
                            ))
                    else:
                        if node in active:
                            active.remove(node)

        return ScheduleReport(hoist_span=hoist_span, violations=violations)
