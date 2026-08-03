#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Step B of the combined-1F1B work: *local segments* inside one physical
pipeline stage.

Goal
----
Step A (``nnscaler.execplan.planpass.global_schedule``) lets a receive be
issued early and waited late so it can overlap a *whole* neighboring
segment's compute. That only helps if there is something coarser than a
single monolithic per-stage segment to overlap with in the first place: as
long as one physical stage's forward (and its mirrored backward) is exactly
*one* opaque :class:`~nnscaler.graph.segment.IRSegment`, "overlap" can only
ever mean "run some communication concurrently with one giant compute
block" -- there is no way to interleave *part* of one micro-batch's compute
with *part* of another's.

This module adds that missing piece: a physical stage's forward node range
can be split into multiple, ordered, independently codegen'd and
independently schedulable **local segments**, *without* adding a physical
pipeline stage, *without* any cross-device adapter between them (they always
stay on the exact same device(s) as their enclosing stage), and *without*
introducing a new IR hierarchy level. A new schedule builder,
:meth:`LocalSegmentSched.sched_1f1b_local_segments`, can then place
``(microbatch, physical_stage, local_segment, F/B)`` blocks individually --
in particular, the pipeline's *first* stage can have its own micro-batch
``m``'s local backward segments genuinely interleave with micro-batch
``m+1``'s local forward segments in the steady state (see "Scheduling" and
"Known limitations" below for exactly which stages this applies to, and why),
matching, at the ``SchedulePlan`` level (nnScaler's compile-time scheduling
IR, not a post-hoc gencode edit), the same "split a stage's compute into
smaller F/B units and interleave them across micro-batches" idea behind
Megatron's ``combined_1f1b``.

Both new capabilities are purely additive: nothing here is invoked unless a
PAS policy explicitly calls :func:`partition_stage_into_local_segments` with
a boundary, and/or explicitly uses :class:`LocalSegmentSched` instead of
:class:`~nnscaler.graph.schedule.predefined.PredefinedSched`. No existing
module is modified by this file. A stage that is not partitioned (the
default -- see "No-boundary behavior" below) is scheduled byte-for-byte
identically to today, and a partitioned stage still round-trips through the
*exact same*, unmodified ``SchedulePlan`` / ``ScheduleDependency`` /
``ExecutionPlan`` / codegen / recompute / ``LifeCycle`` / reducer /
checkpoint-metadata machinery A already builds on -- see "Why no core-file
changes are needed" below.

Why "local segment", not nested ``IRSegment``
-----------------------------------------------
The task that motivated this module suggested two possible designs: nested
``IRSegment``s, or a distinct "local segment" abstraction. This module uses
the latter, for a concrete, load-bearing reason found while reading the
existing implementation (not a stylistic preference):
``nnscaler.graph.segment.IRSegment.select`` documents an explicit, existing
invariant of the IR --

    "Current IRGraph can have at most a 2-level hierarchy
    (IRGraph[IRSegment]). We don't allow IRSegment inside IRSegment."

True nesting (creating an ``IRSegment`` whose own ``._nodes`` contains
*another* ``IRSegment``) would violate that invariant and put this feature
at odds with any code that assumes it (``select(flatten=False)``, dispatch,
adapter generation, etc. -- code this module does not want to have to
re-audit). Instead, a "local segment" here is an **ordinary, flat,
top-level** ``IRSegment``, created by calling the existing
``IRGraph.group()`` primitive *multiple times* over consecutive sub-ranges
of what would otherwise become one physical stage's node range, *before*
partitioning/adapter-generation -- exactly the same precondition under which
``IRGraph.blocking()``/``IRGraph.staging()`` already call ``group()`` once
per *physical* stage. The result is several adjacent, sibling top-level
``IRSegment``s sharing one physical stage's device assignment, in the same
shape ``PredefinedSched.sched_1f1b_interleaved`` already uses for *virtual*
pipeline stages (its ``devs2segs``: multiple forward segments sharing one
device tuple) -- except local segments are always *contiguous* in graph
order and always share their *physical* stage's device set, so (unlike
interleaved/virtual stages) no adapter is ever generated between them: two
adjacent nodes on the same device never trigger cross-device adapter
insertion in the first place, so nothing needs to be suppressed.

Why no core-file changes are needed
------------------------------------
Because a local segment is an ordinary top-level ``IRSegment`` (not a
floating/orphan object kept outside the graph), every downstream consumer
that already understands stage segments understands local segments too,
with no modification:

- ``ScheduleDependency.build()`` finds them via the same
  ``graph.select(ntype=IRSegment, flatten=False)`` it already uses, so
  adapter sender/receiver detection (``_place_adapters``) and
  ``SchedulePlan.validate()``'s direct-dependency checks work unchanged --
  a local segment's ``.inputs()``/``.outputs()`` are computed by the same
  ``create_segment`` producer/consumer analysis used for physical stages,
  so the segment that actually feeds (or is fed by) a cross-stage adapter is
  correctly identified even though it is only *part of* a physical stage.
- Codegen (``nnscaler.codegen.module.module.ModuleCodeGen``) emits one
  method per ``IRSegment`` found in ``execplan.seq(device)`` (built from the
  schedule's own topological order), so each local segment becomes its own
  generated method -- exactly the "multiple local segment methods" gencode
  structure this feature is meant to enable.
- Recompute grouping (``emit_segment``) groups *consecutive nodes with the
  same non-``None`` ``.recompute`` id within one segment's own node list*.
  Splitting a stage into local segments changes which nodes share a
  segment, so :func:`partition_stage_into_local_segments` explicitly
  forbids placing a boundary in the middle of a recompute group (see
  "Validation" below) -- otherwise one checkpoint region would silently
  become two, changing recompute/memory semantics without any error.
- ``LifeCycle`` (``nnscaler.codegen.lifecycle``) computes tensor
  release points from whatever nodes/segments it is given; local segments
  just give it more numerous, smaller units, at worst enabling *earlier*
  (never *later*, never incorrect) release of a tensor that is not needed
  past its local segment's own boundary.
- The parameter/buffer "first declared here" bookkeeping
  (``ModuleCodeGen._init_attributes`` + its module-wide ``SymbolTable``) is
  keyed by attribute identity, not by segment, so a parameter used inside
  exactly one local segment is declared exactly once, at the same place it
  would be today; this is also how model-checkpoint metadata
  (``add_full_map``) stays unaffected by local-segment splitting.
- The gradient reducer's count-based "fire after N backward touches" trigger
  (``grad_accumulation_steps``) is set to the number of micro-batches,
  implicitly assuming each parameter is touched by exactly one ``.backward()``
  call per micro-batch. Splitting a stage's backward into several
  independently-called local backward segments preserves this *for any
  parameter used inside exactly one local segment* (still exactly one
  ``.backward()`` touch per micro-batch) but would silently under-count (and
  hence fire the all-reduce with an incomplete gradient) for a parameter
  referenced from *multiple* local segments of the same stage. This module
  does not attempt to fix the count for that case; instead
  :func:`partition_stage_into_local_segments` detects and rejects it (see
  "Validation" below) rather than ship a subtle correctness hazard.

Boundary sources
-----------------
:func:`partition_stage_into_local_segments` accepts an optional
:class:`LocalSegmentBoundary`, with three concrete sources (a caller may
also implement their own subclass):

- :class:`AnchorBoundary` -- splits at ``nnscaler.runtime.function.anchor(name)``
  markers (:class:`~nnscaler.graph.function.anchor.IRGraphAnchor` nodes),
  matched by name (or any anchor, if no names given). This reuses the exact
  same anchor mechanism ``IRGraphAnchor`` already documents as "user hints
  of staging boundary inside the graph"; a policy can use *different* anchor
  names (or the same anchor, filtered by a name set) for physical-stage
  boundaries versus local-segment boundaries.
- :class:`ModuleBoundary` -- splits wherever the originating ``nn.Module``
  changes, using the existing ``IRCell.module_stack`` provenance (optionally
  at a specific nesting depth). Nodes with no module provenance (e.g. a
  freestanding anchor) never force a split.
- :class:`CallableBoundary` -- an arbitrary user function
  ``List[IRFwOperation] -> List[int]`` returning split indices directly.

No-boundary behavior
---------------------
``partition_stage_into_local_segments(graph, stage_nodes)`` with
``boundary=None`` (the default) returns ``[graph.group(stage_nodes)]``: a
single local segment spanning the whole stage, which is exactly what
``IRGraph.blocking()``/``.staging()`` already produce today. A stage that
never opts in to local-segment splitting is therefore completely unaffected
-- same segment count, same schedule, same generated code.

Validation
----------
:func:`partition_stage_into_local_segments` raises :class:`LocalSegmentError`
(rather than silently producing something subtly wrong) when:

- ``stage_nodes`` is empty, contains a non-forward-operation node, or the
  nodes are not all part of the same not-yet-grouped region (i.e. they
  would cross an existing, different segment -- "illegal cross physical
  stage").
- a computed split point is out of range, or the resulting groups would not
  be contiguous / would not cover ``stage_nodes`` exactly once each
  ("non-contiguous nodes"); this also surfaces (with a clearer message) the
  equivalent, pre-existing contiguity assertions in ``IRGraph.group()``.
- a split point falls strictly inside a maximal run of consecutive nodes
  sharing one non-``None`` ``.recompute`` group id ("recompute must not
  cross a local segment boundary").
- a parameter/buffer (attribute) tensor would end up read or written from
  more than one of the resulting local segments ("shared parameter across
  local segments" -- see the reducer-count hazard above).

Scheduling
----------
:class:`LocalSegmentSched.sched_1f1b_local_segments` groups all top-level
forward ``IRSegment``s by device tuple (mirroring
``PredefinedSched.sched_1f1b_interleaved``'s ``devs2segs``) to recover, per
physical stage, its ordered list of local segments (a stage that was never
split simply has a list of length 1). It reuses ``PredefinedSched.sched_1f1b``'s
exact per-stage coarse step/micro-batch formula unchanged (so cross-stage
causality is inherited from already-validated code, not re-derived), and
only changes *what* is scheduled at each coarse (stage, step) slot:

- A stage with exactly one local segment behaves exactly as
  ``sched_1f1b`` (its coarse step numbers are uniformly scaled by a
  constant factor, which does not change the relative order of any block on
  any device, hence not the generated code) -- this is checked directly by
  a regression test asserting per-device block order equality with
  ``sched_1f1b`` when every stage has one local segment.
- **Only the pipeline's first stage** (the one with no incoming
  cross-stage forward dependency -- i.e. ``wait_steps[sid] == 0`` in
  ``sched_1f1b``'s own formula) additionally *interleaves*, in the steady
  state where the base schedule places ``B(m)`` immediately followed by
  ``F(m+1)``, its local segments one at a time: ``B(segK-1,m),
  F(seg0,m+1), B(segK-2,m), F(seg1,m+1), ..., B(seg0,m), F(segK-1,m+1)``.
  Every other stage places its local segments sequentially per slot
  (``F(seg0..segK-1)`` then, at the next slot, ``B(segK-1..seg0)``) --
  see "Known limitations" for why this is not merely a simplification but
  a correctness requirement, found and fixed via a real CPU
  ``SchedulePlan.validate()`` failure during development.

Compatibility with Step A
--------------------------
Local-segment interleaving (on the pipeline's first stage) only ever
reorders *what happens between* that stage's own recv-issue and recv-wait
for its *own* incoming async communication (if any is configured on it by a
different stage/direction); it never moves work from *after* a wait to
*before* it, because the first local segment of any stage is exactly the one
whose input is that stage's cross-stage adapter's output (see "Why no
core-file changes are needed" above), so a real data dependency -- checked
by the same, unmodified ``ScheduleDependency``/``validate()`` machinery A
already relies on -- keeps it (and, transitively, every later local segment)
after the wait. In other words Step A's
``issue(F(m+1)) < B(m) < wait(F(m+1)) < F(m+1)`` invariant generalizes to
``issue(F(m+1)) < B(m)'s local segments (any order) < wait(F(m+1)) <
F(m+1)'s local segments (in dependency order, seg0 first)``. This is
exercised directly by a real 2-GPU test combining local segments with Step
A's ``async_recv``/``enable_global_p2p_reschedule``.

Known limitations (honestly scoped out of Step B)
----------------------------------------------------
- **Interleaving is only performed for the pipeline's first stage.** This
  was originally attempted for *every* stage and caught, by a real CPU
  ``SchedulePlan.validate()`` failure (not a theoretical concern), as
  unsound in general: for any stage other than the first, ``F(m+1)`` has a
  genuine cross-device data dependency on the *previous* stage's
  ``F(m+1))``, which ``sched_1f1b``'s base formula already places at the
  minimum safe distance (exactly one coarse step later); pulling it earlier
  to interleave with this stage's own ``B(m)`` schedules it before its
  producer finishes. The symmetric idea -- push a non-last stage's ``B(m)``
  *later* to interleave from the other side -- fares no better: it would
  delay the gradient this stage's ``B(m)`` feeds to the *previous* stage's
  ``B(m)``, which the base formula also already places at the minimum safe
  distance. (The *last* stage has no such outgoing-gradient constraint, and
  the *first* stage has no such incoming-activation constraint, but with
  the equal per-stage local-segment counts this module supports generally,
  the same minimum-safe-distance argument leaves no slack for the last
  stage either -- only the first stage's case remains unconditionally safe
  in general.) Extending genuine interleaving to more stages would need a
  fundamentally different (not just locally-adjusted) pipeline schedule --
  arguably a zero-bubble-style redesign, not a local, incremental change,
  and explicitly out of scope for Step B.
- Combining local segments with *virtual* pipeline stages
  (``sched_1f1b_interleaved``) is not supported by
  :class:`LocalSegmentSched` in this module -- device-tuple grouping cannot
  distinguish "several local segments of one physical stage" from "several
  virtual stages round-robined onto one device" by device tuple alone, and
  virtual stages additionally *do* need adapters between them. Doing both
  at once is future work, not attempted here.
- Only a 1F1B-shaped schedule is provided
  (:meth:`LocalSegmentSched.sched_1f1b_local_segments`); other predefined
  schedules (GPipe, 1F1B-plus, Chimera-direct, infer-pipe) are not given a
  local-segment-aware counterpart, since they were not asked for and adding
  them without a concrete use case would be speculative.
- The scheduler intentionally does not try to be memory- or
  performance-optimal (e.g. it does not attempt to minimize peak activation
  memory by choosing a particular interleave order beyond the one described
  above); Step B's scope is a correct, general mechanism, not a tuned policy.
"""

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.schedule.schedplan import SchedulePlan
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRSubTensor


class LocalSegmentError(ValueError):
    """Raised when a local-segment boundary would be unsafe.

    Kept as a dedicated (but still ``ValueError``-compatible, so existing
    ``except ValueError`` callers keep working) exception type so tests can
    assert on it precisely, mirroring
    ``nnscaler.runtime.executor.AsyncCommError`` from Step A.
    """


# --------------------------------------------------------------------------
# Boundary sources
# --------------------------------------------------------------------------

class LocalSegmentBoundary:
    """Base class for a local-segment boundary policy.

    A boundary policy only ever *proposes* split points; final legality
    (contiguity, recompute, shared-attribute) is always enforced by
    :func:`partition_stage_into_local_segments` itself, regardless of the
    boundary source used.
    """

    def split_indices(self, nodes: Sequence[IRFwOperation]) -> List[int]:
        """Return sorted, de-duplicated indices ``i`` (``0 < i < len(nodes)``)
        such that a new local segment should start at ``nodes[i]``, i.e.
        ``nodes[:i]`` and ``nodes[i:]`` should end up in different local
        segments.

        Args:
            nodes: the candidate stage's forward node list (not yet grouped).

        Returns:
            List[int]: candidate split indices (need not be adjusted for
            recompute/multiref/anchor -- the caller does that).
        """
        raise NotImplementedError


class AnchorBoundary(LocalSegmentBoundary):
    """Split immediately before each matching
    :class:`~nnscaler.graph.function.anchor.IRGraphAnchor` node.

    This is the same anchor mechanism used for *physical* stage boundaries
    (``nnscaler.runtime.function.anchor(name)`` in user model code); pass a
    disjoint ``names`` set from whatever names are used for physical staging
    to keep the two concerns independent, e.g.::

        nnscaler.runtime.function.anchor('stage')       # physical stage
        nnscaler.runtime.function.anchor('local_seg')   # local segment

        AnchorBoundary({'local_seg'})
    """

    def __init__(self, names: Optional[Set[str]] = None):
        """
        Args:
            names: anchor names that mark a local-segment boundary. If
                ``None``, *every* :class:`IRGraphAnchor` found is treated as
                a boundary.
        """
        self.names = set(names) if names is not None else None

    def split_indices(self, nodes: Sequence[IRFwOperation]) -> List[int]:
        indices = []
        for i, node in enumerate(nodes):
            if not isinstance(node, IRGraphAnchor):
                continue
            if self.names is not None and node.kwargs.get('name') not in self.names:
                continue
            if i > 0:  # an anchor at position 0 does not split anything off
                indices.append(i)
        return indices


class ModuleBoundary(LocalSegmentBoundary):
    """Split wherever the originating ``nn.Module`` changes.

    Uses ``IRCell.module_stack`` (already recorded by the tracer for every
    node with a ``forward``-having originating module). Nodes with no module
    provenance (``module_stack`` is ``None``/empty -- e.g. a freestanding
    anchor or a system-inserted node) never force a split: they are treated
    as a continuation of whatever module last had a boundary decision.
    """

    def __init__(self, depth: Optional[int] = None):
        """
        Args:
            depth: if given, compare the module stack's ``depth``-th entry
                (0 = the outermost recorded module) instead of the full
                (deepest) stack; useful to split per top-level block while
                ignoring differences among a block's own children.
        """
        self.depth = depth

    def _key(self, node: IRFwOperation):
        stack = node.module_stack
        if not stack:
            return None
        keys = list(stack.keys())
        if self.depth is None:
            return keys[-1]
        if self.depth < len(keys):
            return keys[self.depth]
        return keys[-1]

    def split_indices(self, nodes: Sequence[IRFwOperation]) -> List[int]:
        indices = []
        last_key = None
        last_key_set = False
        for i, node in enumerate(nodes):
            key = self._key(node)
            if key is None:
                continue  # no provenance: never forces a boundary
            if last_key_set and key != last_key:
                indices.append(i)
            last_key = key
            last_key_set = True
        return indices


class CallableBoundary(LocalSegmentBoundary):
    """Wrap an arbitrary ``List[IRFwOperation] -> List[int]`` policy function."""

    def __init__(self, fn: Callable[[Sequence[IRFwOperation]], List[int]]):
        self.fn = fn

    def split_indices(self, nodes: Sequence[IRFwOperation]) -> List[int]:
        return list(self.fn(nodes))


# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------

def _adjust_for_pull_in(nodes: Sequence[IRFwOperation], indices: List[int]) -> List[int]:
    """Pull a split index earlier past any immediately-preceding
    ``multiref``/:class:`IRGraphAnchor` node, mirroring the exact convention
    ``IRGraph.blocking()``/``.staging()`` already use for *physical* stage
    boundaries (such nodes are pulled into the *following* group)."""
    adjusted = []
    for idx in indices:
        while idx > 0 and (nodes[idx - 1].name == 'multiref' or isinstance(nodes[idx - 1], IRGraphAnchor)):
            idx -= 1
        adjusted.append(idx)
    return adjusted


def _validate_contiguous_indices(nodes: Sequence[IRFwOperation], indices: List[int]) -> List[int]:
    n = len(nodes)
    indices = sorted(set(indices))
    for idx in indices:
        if not (0 < idx < n):
            raise LocalSegmentError(
                f"Local segment boundary index {idx} is out of range for a stage "
                f"with {n} nodes; a boundary must strictly separate two non-empty "
                f"groups (0 < index < {n})."
            )
    return indices


def _groups_from_indices(nodes: Sequence[IRFwOperation], indices: List[int]) -> List[List[IRFwOperation]]:
    bounds = [0] + indices + [len(nodes)]
    groups = []
    for start, end in zip(bounds[:-1], bounds[1:]):
        if start >= end:
            # only possible if the same index sneaks in twice after adjustment;
            # _validate_contiguous_indices + set() dedup already rules this out,
            # but keep the check for defense in depth.
            raise LocalSegmentError(
                f"Local segment boundary computation produced an empty group "
                f"(range [{start}, {end})); this indicates two boundaries "
                f"collapsed onto the same position after multiref/anchor "
                f"pull-in adjustment."
            )
        groups.append(list(nodes[start:end]))
    return groups


def _validate_no_recompute_split(nodes: Sequence[IRFwOperation], indices: List[int]) -> None:
    for idx in indices:
        prev_rc = getattr(nodes[idx - 1], 'recompute', None)
        curr_rc = getattr(nodes[idx], 'recompute', None)
        if prev_rc is not None and prev_rc == curr_rc:
            raise LocalSegmentError(
                f"Local segment boundary at index {idx} would split recompute "
                f"group {prev_rc} (node {nodes[idx - 1]!r} and node "
                f"{nodes[idx]!r} are both in recompute group {prev_rc}, but "
                f"would end up in different local segments). A local segment "
                f"boundary must not fall inside a recompute group; move the "
                f"boundary outside the group instead."
            )


def _attr_key(tensor: IRSubTensor):
    # Two IRSubTensor views of the same underlying attribute (parameter or
    # buffer) share the same full tensor; that identity is what "the same
    # attribute" means for reducer-count purposes, regardless of how it is
    # sliced in any one particular view.
    return tensor.parent


def _validate_no_shared_attribute_split(groups: List[List[IRFwOperation]]) -> None:
    attr_to_groups: Dict[object, Set[int]] = {}
    attr_example: Dict[object, IRSubTensor] = {}
    for gi, group in enumerate(groups):
        for node in group:
            for t in IRSegment.get_objects_from_complex(node.inputs()):
                if isinstance(t, IRSubTensor) and t.is_attr():
                    key = _attr_key(t)
                    attr_to_groups.setdefault(key, set()).add(gi)
                    attr_example.setdefault(key, t)
    for key, gset in attr_to_groups.items():
        if len(gset) > 1:
            raise LocalSegmentError(
                f"Parameter/buffer {attr_example[key]!r} is referenced from "
                f"{len(gset)} different local segments ({sorted(gset)}) of "
                f"the same physical stage. This is rejected rather than "
                f"silently supported: the runtime's gradient reducer expects "
                f"exactly one backward touch per micro-batch per parameter "
                f"(`grad_accumulation_steps` is set to the micro-batch "
                f"count), and splitting this parameter's uses across "
                f"multiple local segments -- each an independent "
                f"`.backward()` call -- would touch it multiple times per "
                f"micro-batch, triggering the all-reduce with an incomplete "
                f"gradient. Keep all uses of a shared parameter/buffer "
                f"within one local segment (adjust the boundary policy so "
                f"no split point falls between them)."
            )


def _validate_same_ungrouped_region(graph: IRGraph, stage_nodes: Sequence[IRFwOperation]) -> None:
    if len(stage_nodes) == 0:
        raise LocalSegmentError("Cannot partition an empty stage (stage_nodes is empty).")
    for node in stage_nodes:
        if not isinstance(node, IRFwOperation):
            raise LocalSegmentError(
                f"Local segment partitioning only accepts forward operators, "
                f"got {type(node)}: {node!r}."
            )
    # `graph.segment(node)` returns the lowest *existing* segment containing
    # `node`. If the given nodes are not all directly inside the same
    # not-yet-grouped region (e.g. some are already part of a *different*,
    # previously-created physical-stage segment, or of the top graph while
    # others are inside a segment), this is an illegal cross-physical-stage
    # request: `graph.group()` would raise its own
    # "cross-segment grouping is not allowed yet" assertion once it hit the
    # mismatch, but we check first, with a clearer, Step-B-specific message,
    # since this is an explicitly required diagnosable case (not merely an
    # incidental one).
    enclosing = {graph.segment(node) for node in stage_nodes}
    if len(enclosing) != 1:
        raise LocalSegmentError(
            f"stage_nodes are not all part of the same (not yet grouped) "
            f"region -- found {len(enclosing)} distinct enclosing scopes. "
            f"This usually means the given node list spans more than one "
            f"physical stage (e.g. nodes from two different, already-staged "
            f"IRSegments were mixed together), which is illegal: local "
            f"segments may only subdivide a *single* physical stage."
        )
    # `stage_nodes` itself (regardless of whether a boundary later
    # subdivides it) must be a contiguous run in its enclosing scope's node
    # order -- e.g. `[nodes[0], nodes[2], nodes[3]]` (skipping nodes[1]) is
    # illegal. `graph.group()` would eventually also catch this via its own
    # "nodes should be in consecutive order" assertion, but only once it is
    # actually called (e.g. never, for a custom boundary whose groups happen
    # to each individually be contiguous even though the *whole* range is
    # not); check it here, unconditionally and with a clearer message.
    (fgraph,) = enclosing
    indices = [fgraph.index(node)[0] for node in stage_nodes]
    if max(indices) - min(indices) + 1 != len(stage_nodes):
        raise LocalSegmentError(
            f"stage_nodes are not contiguous in the graph's node order "
            f"(indices {indices} span a wider range than the {len(stage_nodes)} "
            f"given nodes cover) -- a physical stage's node range (and any "
            f"local segment sub-range of it) must be a contiguous run, with "
            f"no node skipped and no node belonging to a different stage "
            f"interleaved in between."
        )


def partition_stage_into_local_segments(
    graph: IRGraph,
    stage_nodes: Sequence[IRFwOperation],
    boundary: Optional[LocalSegmentBoundary] = None,
) -> List[IRSegment]:
    """Partition one physical stage's forward node range into one or more
    local segments.

    This must be called at the same point in a PAS policy where the stage
    would otherwise be grouped directly, i.e. *before* any operator
    partition (TP sharding) or adapter generation -- the same precondition
    ``IRGraph.group()``/``.blocking()``/``.staging()`` already require. It
    replaces a single ``graph.group(stage_nodes)`` call: call this instead,
    then continue assigning devices/partitioning ops on *each* returned
    local segment's own ``.nodes()`` (all local segments of one stage should
    end up assigned to the same device(s), since they are meant to stay
    parts of one physical stage).

    Args:
        graph: the (root) graph these nodes belong to.
        stage_nodes: the candidate physical stage's forward node list, in
            forward execution order. Must be contiguous and not yet part of
            any existing segment (see :class:`LocalSegmentError` cases
            below).
        boundary: the boundary policy to use. If ``None`` (default), no
            split is performed -- the whole stage becomes a single local
            segment, byte-for-byte identical to calling
            ``graph.group(stage_nodes)`` directly. If ``boundary`` is given
            but proposes no valid split points, the result is likewise a
            single local segment.

    Returns:
        List[IRSegment]: the created *forward* local segments, in forward
        execution order (``segments[i].mirror`` is the corresponding
        backward local segment). Length 1 when there is no boundary (or the
        boundary proposes no splits); length > 1 otherwise.

    Raises:
        LocalSegmentError: if ``stage_nodes`` is empty, contains a
            non-forward-operator node, spans more than one existing
            (already-grouped) region, if a computed boundary is
            out-of-range, if a boundary would split a recompute group, or
            if a boundary would split usage of a shared parameter/buffer
            across local segments.
    """
    stage_nodes = list(stage_nodes)
    _validate_same_ungrouped_region(graph, stage_nodes)

    if boundary is None:
        return [graph.group(stage_nodes)]

    raw_indices = list(boundary.split_indices(stage_nodes))
    raw_indices = _validate_contiguous_indices(stage_nodes, raw_indices)
    indices = _adjust_for_pull_in(stage_nodes, raw_indices)
    # pulling in past multiref/anchor nodes can duplicate or invalidate an
    # index (e.g. pull two neighboring indices onto the same position, or
    # pull the smallest index down to 0); re-validate after adjustment.
    indices = sorted(set(indices))
    indices = [idx for idx in indices if idx > 0]
    indices = _validate_contiguous_indices(stage_nodes, indices)

    if not indices:
        return [graph.group(stage_nodes)]

    _validate_no_recompute_split(stage_nodes, indices)
    groups = _groups_from_indices(stage_nodes, indices)
    _validate_no_shared_attribute_split(groups)

    segments = [graph.group(group) for group in groups]
    return segments


# --------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------

def _stage_local_segments(graph: IRGraph, num_stages: int) -> List[List[IRSegment]]:
    """Recover, per physical stage (grouped by device tuple, mirroring
    ``PredefinedSched.sched_1f1b_interleaved``'s ``devs2segs``), the ordered
    list of its local segments (length 1 for a stage that was never split).

    Order across stages follows first-appearance order in
    ``graph.select(ntype=IRSegment, flatten=False)``, which walks the
    graph's node list in (forward) execution order; since local segments of
    one physical stage are always contiguous (see module docstring), this
    is exactly the pipeline stage order.
    """
    segments = graph.select(ntype=IRSegment, flatten=False)
    fsegs = [seg for seg in segments if seg.isfw()]
    devs2segs: Dict[Tuple[int, ...], List[IRSegment]] = {}
    for seg in fsegs:
        devs2segs.setdefault(tuple(seg.device), []).append(seg)
    if len(devs2segs) != num_stages:
        raise ValueError(
            f"Mismatch of physical stage number ({len(devs2segs)}, inferred "
            f"from distinct device tuples among forward IRSegments) with "
            f"num_stages ({num_stages})."
        )
    return list(devs2segs.values())


class LocalSegmentSched:
    """Local-segment-aware predefined schedules.

    Kept separate from ``nnscaler.graph.schedule.predefined.PredefinedSched``
    so that module is not modified by Step B; use directly as a callable
    ``pipeline_scheduler`` (``ComputeConfig.apply_pipeline_scheduler`` accepts
    an arbitrary callable, not just a name registered on ``PredefinedSched``).
    """

    @staticmethod
    def sched_1f1b_local_segments(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """1F1B scheduling with local-segment interleaving on the first stage.

        Identical cross-stage causality/step formula to
        ``PredefinedSched.sched_1f1b`` (reused verbatim below, per-stage);
        the only difference is that each stage's ``F(m)``/``B(m)`` slot is
        expanded into its constituent local segments, and -- only for the
        pipeline's first stage -- a steady state ``B(m)`` immediately
        followed by ``F(m+1)`` is interleaved segment-by-segment when that
        stage has more than one local segment. See the module docstring's
        "Scheduling" and "Known limitations" sections for the exact
        interleave order, why it is restricted to the first stage, and the
        degeneracy property (a stage with exactly one local segment behaves
        exactly as ``sched_1f1b``).
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches} ")
        stage_local_segs = _stage_local_segments(graph, num_stages)

        sched = SchedulePlan(graph, num_microbatches)

        wait_steps = [sid for sid in range(num_stages)]
        bw_ofst = [num_stages - 1 - sid for sid in range(num_stages)]
        total_steps = num_microbatches * 2 + (num_stages - 1) * 2
        # Enough fine-grained sub-steps per coarse step to (a) place every
        # local segment of the largest-K stage individually and (b) merge a
        # steady-state B(m)+F(m+1) pair (2x the local segments) without
        # collision. Uniformly scaling every stage's coarse step by the same
        # constant factor preserves each device's own relative block order
        # exactly (no two blocks on one device ever share a step either way),
        # so this never changes cross-stage causality, only "spreads out" the
        # step numbering to make room for interleaving.
        max_k = max((len(segs) for segs in stage_local_segs), default=1)
        scale = max(1, 2 * max_k)

        def event_at(sid: int, step: int) -> Optional[Tuple[str, int]]:
            ofst = wait_steps[sid]
            if step < ofst:
                return None
            fw_idx = (step - ofst) // 2
            is_fwd = (step - ofst) % 2 == 0
            mb_idx = fw_idx if is_fwd else fw_idx - bw_ofst[sid]
            if mb_idx < 0 or mb_idx >= num_microbatches:
                return None
            return ('F', mb_idx) if is_fwd else ('B', mb_idx)

        for sid in range(num_stages):
            local_segs = stage_local_segs[sid]
            k = len(local_segs)
            # Only the pipeline's first stage (wait_steps[sid] == 0) can
            # safely interleave its own B(m) with F(m+1): F(m+1) for any
            # OTHER stage has a real cross-device data dependency on the
            # *previous* stage's F(m+1) (checked directly by
            # ScheduleDependency/validate()), which the base sched_1f1b
            # formula already places at the minimum safe distance (one
            # coarse step later) -- pulling it earlier to interleave with
            # this stage's own B(m) would schedule it before its producer
            # finishes. Symmetrically, pushing a non-last stage's B(m) later
            # to interleave from the other side would delay the gradient it
            # feeds to the *previous* stage's B(m), which the base formula
            # also already places at the minimum safe distance. So for any
            # stage other than the first, local segments are still placed
            # individually (this stage's F/B compute is still split into
            # independently codegen'd/scheduled units -- see module
            # docstring), just not cross-microbatch-interleaved. See the
            # module docstring's "Known limitations" for the full rationale
            # (this was found, root-caused, and fixed via a real CPU
            # validate() failure during development -- not a design
            # preference).
            can_interleave = (wait_steps[sid] == 0)
            step = 0
            while step < total_steps:
                ev = event_at(sid, step)
                if ev is None:
                    step += 1
                    continue
                kind, mb = ev
                nxt = event_at(sid, step + 1) if step + 1 < total_steps else None
                if can_interleave and kind == 'B' and k > 1 and nxt is not None and nxt[0] == 'F':
                    # Steady-state pair: interleave this B(mb) with the
                    # immediately-following F(nxt_mb) one local segment at a
                    # time, backward segments in reverse (mirror) order.
                    fw_mb = nxt[1]
                    base = step * scale
                    sub = 0
                    for i in range(k):
                        bseg = local_segs[k - 1 - i].mirror
                        sched.add_segment(bseg, mb, base + sub)
                        sub += 1
                        fseg = local_segs[i]
                        sched.add_segment(fseg, fw_mb, base + sub)
                        sub += 1
                    step += 2  # this coarse step and the next are both consumed
                    continue
                base = step * scale
                if kind == 'F':
                    for i in range(k):
                        sched.add_segment(local_segs[i], mb, base + i)
                else:
                    for i in range(k):
                        bseg = local_segs[k - 1 - i].mirror
                        sched.add_segment(bseg, mb, base + i)
                step += 1

        sched.finish()
        return sched
