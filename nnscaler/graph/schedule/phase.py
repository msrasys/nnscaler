#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Step C of the combined-1F1B work: an explicit **phase IR** for splitting one
transformer layer's forward (and its mirrored backward) into named,
first-class, independently schedulable *phases* -- ``ATTENTION``,
``MOE_DISPATCH``, ``EXPERT_COMPUTE``, ``MOE_COMBINE`` -- and a phase-*kind*-aware
schedule builder that places ``F(m+1)``'s communication phases (the MoE
dispatch/combine all-to-all) so they genuinely interleave with ``B(m)``'s
independent, unrelated attention/expert *compute* phases: the classic
Megatron ``combined_1f1b`` pattern, expressed at nnScaler's compile-time
scheduling IR level (not a hand-edited/post-hoc gencode change, and not merely
Step B's local segments renamed -- see "Relationship to Step B" below for the
concrete, load-bearing difference).

Why this needs to be a new IR layer, not just Step B renamed
--------------------------------------------------------------
Step B (:mod:`nnscaler.graph.schedule.local_segment`) already lets a physical
stage's forward node range be split into several, ordered, flat, independently
codegen'd/schedulable ``IRSegment``s ("local segments"), and already has a
schedule builder (``LocalSegmentSched.sched_1f1b_local_segments``) that
interleaves one stage's ``B(m)`` and ``F(m+1)`` local segments on the
pipeline's first stage. This module reuses that machinery directly (a phase
*is* a local segment -- see below) but adds two things Step B's generic,
*phase-blind* pairwise alternation cannot provide:

1. **Explicit phase identity and semantics** (:class:`PhaseType`,
   :class:`PhaseIdentity`, :class:`PhaseNode`): a local segment on its own
   carries no information about *what kind of work* it does. Step B's
   ``sched_1f1b_local_segments`` therefore alternates ``B``/``F`` local
   segments purely by *position* (``B[k-1], F[0], B[k-2], F[1], ...``),
   with no notion of "this segment issues an outstanding async communication
   that should be overlapped with independent compute from the other
   micro-batch". A concrete, worked-out counter-example (see the module-level
   derivation this file's author performed before writing
   :func:`PhaseAwareSched.sched_1f1b_phase_aware`, and the property-sweep
   tests in ``tests/graph/schedule/test_phase_schedule_sweep.py``): applying
   Step B's *unmodified* alternation formula to the canonical
   ``[ATTENTION, MOE_DISPATCH, EXPERT_COMPUTE, MOE_COMBINE]`` phase sequence
   places exactly **one** of ``B(m)``'s phases between ``F(m+1)``'s dispatch
   issue and its wait -- and depending on parity that one phase can just as
   easily be ``B(m)``'s *own* ``MOE_DISPATCH`` backward (which itself issues a
   real, synchronous all-to-all) as it can be a genuinely independent compute
   phase. Reusing Step B's placement logic unmodified therefore cannot
   *reliably* deliver "communication hides behind independent compute" --
   it is luck-of-parity, not a designed guarantee. :class:`PhaseIdentity`'s
   ``issues_async``/``has_communication`` properties give
   :func:`PhaseAwareSched.sched_1f1b_phase_aware` the information it needs to
   deliberately place independent-compute phases inside a comm phase's
   issue-to-wait window instead (see "Scheduling" below).
2. **Fail-fast structural validation specific to the MoE phase pattern**:
   legal phase order/completeness (:func:`lower_layer_to_phases` requires
   *exactly* the canonical ``[ATTENTION, MOE_DISPATCH, EXPERT_COMPUTE,
   MOE_COMBINE]`` or ``[ATTENTION]``-only sequence -- not merely "some
   anchors, in some order"), same-physical-stage/layer-locality
   (:func:`validate_phase_layout`), plus reuse of Step B's own recompute- and
   shared-parameter-boundary checks. None of this exists for a plain,
   untagged local segment.

A phase *is*, structurally, still an ordinary flat top-level ``IRSegment``
(created via :func:`~nnscaler.graph.schedule.local_segment.partition_stage_into_local_segments`,
the exact same call Step B itself uses), tagged with a :class:`PhaseNode`
recorded in ``op_context['phase']`` (the same generic per-node metadata
channel :class:`~nnscaler.graph.schedule.schedplan.StreamContext` already
uses -- see ``nnscaler/graph/schedule/schedplan.py``'s module docstring, and
confirmed to survive ``IRSegment.dispatch()``/mirror projection via
``IRSegment._copy_meta`` and ``ExecutionPlan``'s ``ExeReuseCell``, i.e. it is
visible, unmodified, all the way to codegen). Consequently every downstream
consumer that already understands Step B's local segments (``ScheduleDependency``,
``SchedulePlan.validate()``, codegen, recompute grouping, ``LifeCycle``, the
reducer, checkpoint metadata) understands phases too, with **no core-file
changes** -- exactly Step B's own "why no core-file changes are needed"
argument, inherited unmodified.

Phase pattern
-------------
One MoE-FFN transformer layer lowers to exactly 4 forward phases, in this
order (see :data:`MOE_PHASE_SEQUENCE`); a dense (non-MoE) layer lowers to
exactly 1 (:data:`DENSE_PHASE_SEQUENCE`):

- ``ATTENTION``: self-attention block. Pure compute, no communication either
  direction.
- ``MOE_DISPATCH``: gating/router + local capacity-buffer scatter, ending
  with an **async issue** of the dispatch all-to-all
  (:mod:`nnscaler.runtime.adapter.moe`). Forward only issues; backward runs
  the adjoint all-to-all (also real, kept synchronous -- see "Scope" below).
- ``EXPERT_COMPUTE``: starts with the **deferred wait** for dispatch's
  result, runs the local expert FFN, and ends with the async **issue** of the
  combine all-to-all. Both a comm consumer and a comm producer.
- ``MOE_COMBINE``: starts with the **deferred wait** for combine's result,
  then unpermutes/weights/adds the residual. Pure compute after its wait;
  critically, the *backward* of a wait is an identity (no-op), so
  ``MOE_COMBINE``'s backward phase has no communication of its own (see
  :data:`_BWD_HAS_COMM`).

Backward phases run in the mirrored (reverse) order:
``MOE_COMBINE(bwd) -> EXPERT_COMPUTE(bwd) -> MOE_DISPATCH(bwd) -> ATTENTION(bwd)``.

Lowering API
------------
A PAS policy calls :func:`phase_anchor` inside model ``forward()`` (a plain,
value-returning-``None`` call exactly like
:func:`nnscaler.runtime.function.anchor`, of which it is a thin, structured
wrapper -- zero numerical effect, so a phase-anchored model and its
un-anchored twin are byte-for-byte the same math, letting e2e tests compare a
"phased" and a "serial baseline" compile of the *same* model class) once per
phase, before the sub-module(s) implementing that phase. At compile time, the
policy calls :func:`lower_layer_to_phases` once per (stage, layer) with that
layer's contiguous, not-yet-grouped forward node range; it replaces (mirrors
Step B's own ``partition_stage_into_local_segments`` call convention) what
would otherwise be a single ``graph.group(...)`` call. The returned
:class:`PhaseNode` list is then assigned to devices exactly like Step B's
local segments (all phases of one layer to the same device tuple -- see
"Same physical stage" below). Finally
:meth:`PhaseAwareSched.sched_1f1b_phase_aware` replaces
``LocalSegmentSched.sched_1f1b_local_segments`` as the ``pipeline_scheduler``.

Fail-fast validation (:func:`lower_layer_to_phases`)
-----------------------------------------------------
- **Legal phase order / completeness**: the anchors found in ``layer_nodes``
  (in graph order) must spell out *exactly* :data:`MOE_PHASE_SEQUENCE` or
  :data:`DENSE_PHASE_SEQUENCE` -- any other order, a missing phase, a
  duplicate, or an unrecognized phase raises :class:`PhaseError`.
- **Continuity / same physical stage**: inherited unmodified from
  :func:`~nnscaler.graph.schedule.local_segment.partition_stage_into_local_segments`
  (``_validate_same_ungrouped_region``/contiguity checks) -- a layer's phase
  anchors must be a contiguous run of not-yet-grouped nodes, so they cannot
  spill across an already-existing (different) physical stage segment.
  :func:`validate_phase_layout` additionally checks, *after* device
  assignment, that every phase of one ``layer_id`` ends up on the exact same
  device tuple (see below) -- this is a redundant, phase-specific,
  clearer-error-message safety net on top of what
  ``LocalSegmentSched._stage_local_segments``'s device-tuple-contiguity check
  already structurally enforces, in the same spirit as Step B's own
  post-commit-audit-added redundant checks.
- **Recompute boundary / shared parameter**: inherited unmodified from the
  same function (``_validate_no_recompute_split`` / ``_validate_no_shared_attribute_split``).
- **Mirror preserved**: :func:`lower_layer_to_phases` raises
  :class:`PhaseError` if a produced forward phase segment has no ``.mirror``,
  so every phase is guaranteed independently ``executor.backward()``-able
  (exercised directly by
  ``tests/graph/schedule/test_phase.py::test_phase_backward_independent``).

Same physical stage
--------------------
Exactly like Step B's local segments, a phase never introduces a *new*
physical pipeline stage or a *new* cross-device adapter: all 4 phases of one
MoE layer are assigned to the same device tuple (the physical stage's device
set) by the PAS policy, matching how a real PP x EP mesh works in practice --
a stage's device set plays *both* roles (e.g. a TP/replica group for
attention, an EP group for the experts), so the all-to-all inside
``MOE_DISPATCH``/``MOE_COMBINE`` is an **op-level** collective inside one
segment, never a new top-level ``IRSegment``/``IRAdapter`` boundary. This is
also exactly why Step A's ``GlobalCommSchedule`` (which only ever reorders
*segment*/*adapter*-level blocks, see
``nnscaler.execplan.planpass.global_schedule``) is structurally unaware of --
and therefore never conflicts with -- the MoE communication living *inside* a
phase segment; compatibility with Step A is architectural, not incidental
(exercised directly by
``tests/parallel_module/test_phase_moe_e2e.py::test_phase_moe_compatible_with_step_a_global_schedule``).

Scheduling
----------
:meth:`PhaseAwareSched.sched_1f1b_phase_aware` reuses
``LocalSegmentSched``'s exact per-stage coarse step/microbatch formula
(copied intentionally -- see its docstring) and, for each stage's steady-state
``B(m)``-then-``F(m+1)`` coincidence (the pipeline's first stage only, for
the exact same soundness reason documented in Step B's module docstring --
this module does not revisit or relax that restriction), replaces Step B's
position-based alternation with a **phase-kind-aware interleave**:

    Walk ``F(m+1)``'s phases in order. After placing a phase whose forward
    code ends with an async issue (:attr:`PhaseIdentity.issues_async` --
    ``MOE_DISPATCH`` and ``EXPERT_COMPUTE``), drain ``B(m)``'s phases (in
    their own, mirrored order) into the schedule *until* one without its own
    communication (:attr:`PhaseIdentity.has_communication`) has been placed,
    or ``B(m)`` is exhausted. After ``F(m+1)`` is exhausted, drain any
    remaining ``B(m)`` phases.

This guarantees, for every comm-issuing phase, **at least one** genuinely
independent, communication-free ``B(m)`` compute phase is scheduled strictly
between its issue and its (next-phase) wait -- the concrete, checkable
``issue < compute < wait`` property
``tests/graph/schedule/test_phase_schedule_sweep.py`` asserts across a sweep
of microbatch/stage/layer counts. Untagged local segments (no phase metadata
anywhere in a stage) fall back to *exactly* (byte-for-byte;
regression-tested) Step B's own alternation -- Step C is a strict,
opt-in extension of Step B, never a behavior change for existing callers.

Honest, analyzed limitation of the two-window case
----------------------------------------------------
For the canonical 4-phase pattern, ``EXPERT_COMPUTE`` is *simultaneously* the
close of ``MOE_DISPATCH``'s window and the open of ``MOE_COMBINE``'s window,
so both windows share one contiguous slice of ``B(m)``'s 4 phases. ``B(m)``'s
own communication-free phases are ``MOE_COMBINE(bwd)`` (first) and
``ATTENTION(bwd)`` (last) -- i.e. at the two *extremes* of its own sequence --
while its two communication-bearing phases (``EXPERT_COMPUTE(bwd)``,
``MOE_DISPATCH(bwd)``) sit contiguously in the *middle*. Consequently the
"drain until communication-free" rule above always gives the dispatch window
a clean, communication-free filler (``MOE_COMBINE(bwd)`` alone) but the
combine window ends up with ``B(m)``'s remaining 3 phases, i.e. its own two
communication-bearing phases *plus* a trailing communication-free one. This
is a genuine structural property of this specific, adjacent-window phase
pattern (verified by hand-derivation and by
``test_phase_schedule_sweep.py::test_dispatch_window_is_communication_free``),
not a bug: the literal, checked property this module guarantees is "at least
one independent compute phase between issue and wait" (true for *both*
windows), not "the window contains *only* compute" (only true for the
dispatch window). Whether ``B(m)``'s own communication-bearing phases sharing
the combine window with ``F(m+1)``'s outstanding combine is *safe* (not just
"scheduled", i.e. does not deadlock/corrupt data) is a claim about the
runtime/NCCL, not the schedule table -- justified here because every rank in
one EP process group runs the *same*, compiled-once, deterministic schedule
(so any two collectives on that group are necessarily enqueued in the same
relative order by every rank, satisfying NCCL's same-communicator ordering
requirement regardless of which CUDA stream issues them), and empirically
confirmed (not just argued) by the real multi-GPU, repeated, deadlock-guarded
e2e tests in ``tests/parallel_module/test_phase_moe_e2e.py`` /
``test_phase_moe_multistage_e2e.py``.

Scope (what this step deliberately does not do)
--------------------------------------------------
- Backward-direction MoE communication (the adjoint all-to-alls inside
  ``MOE_DISPATCH(bwd)``/``EXPERT_COMPUTE(bwd)``) is issued synchronously
  (issue-and-immediately-wait); only *forward* ``F(m+1)``'s dispatch/combine
  get the deferred-wait treatment. This matches the literal ask ("F(m+1)'s
  MoE communication phase interleaves with B(m)'s independent compute
  phase") and keeps the backward path simple; a symmetric
  backward-overlaps-forward-compute scheme is not attempted here.
- dgrad/wgrad splitting is explicitly out of scope (not requested for this
  step); ``EXPERT_COMPUTE``'s backward computes both together, same as every
  other phase.
- Only the pipeline's first stage gets genuine cross-microbatch interleaving,
  for the exact same reason Step B's own local segments are restricted to it
  (a non-first stage's ``F(m+1)`` has a real cross-device data dependency on
  the *previous* stage's ``F(m+1)``, already placed at the minimum safe
  distance by the base ``sched_1f1b`` formula) -- this module does not
  revisit that restriction, it inherits it.
"""

import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from nnscaler.graph.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.graph.schedule.schedplan import SchedulePlan
from nnscaler.graph.schedule.local_segment import (
    LocalSegmentError,
    AnchorBoundary,
    partition_stage_into_local_segments,
    _stage_local_segments,
)
from nnscaler.ir.operator import IRFwOperation

import nnscaler.runtime.function as ncf


class PhaseError(LocalSegmentError):
    """Raised when a phase graph/layout/schedule would be illegal.

    Subclasses :class:`~nnscaler.graph.schedule.local_segment.LocalSegmentError`
    (itself a ``ValueError``), mirroring that class's own relationship to
    plain ``ValueError`` -- existing ``except ValueError`` (or
    ``except LocalSegmentError``) callers keep working, while tests can
    assert on :class:`PhaseError` precisely for phase-specific failures.
    """


class PhaseType(enum.Enum):
    ATTENTION = 'attention'
    MOE_DISPATCH = 'moe_dispatch'
    EXPERT_COMPUTE = 'expert_compute'
    MOE_COMBINE = 'moe_combine'


class PhaseKind(enum.Enum):
    """Coarse compute/communication classification of a phase, derived from
    (:class:`PhaseType`, direction) -- see :attr:`PhaseIdentity.kind`."""
    COMPUTE = 'compute'
    COMM = 'comm'


#: Canonical forward-order phase sequence for one MoE-FFN transformer layer.
MOE_PHASE_SEQUENCE: Tuple[PhaseType, ...] = (
    PhaseType.ATTENTION,
    PhaseType.MOE_DISPATCH,
    PhaseType.EXPERT_COMPUTE,
    PhaseType.MOE_COMBINE,
)

#: A dense (non-MoE) layer may legally use just the ``ATTENTION`` phase.
DENSE_PHASE_SEQUENCE: Tuple[PhaseType, ...] = (PhaseType.ATTENTION,)

_LEGAL_PHASE_SEQUENCES: Tuple[Tuple[PhaseType, ...], ...] = (MOE_PHASE_SEQUENCE, DENSE_PHASE_SEQUENCE)

#: Forward phases whose OWN code ends with an async communication issue that a
#: *later* phase (always the immediately next one -- see module docstring)
#: must wait on: ``MOE_DISPATCH`` (issues dispatch) and ``EXPERT_COMPUTE``
#: (issues combine, after waiting for dispatch).
_ISSUES_ASYNC = {PhaseType.MOE_DISPATCH, PhaseType.EXPERT_COMPUTE}

#: Per-(phase type, forward) whether that phase's own code performs any real
#: communication (an issue or a wait) -- see module docstring "Phase pattern".
_FWD_HAS_COMM: Dict[PhaseType, bool] = {
    PhaseType.ATTENTION: False,
    PhaseType.MOE_DISPATCH: True,
    PhaseType.EXPERT_COMPUTE: True,
    PhaseType.MOE_COMBINE: True,
}

#: Per-(phase type, backward) whether that phase's mirror code performs any
#: real communication. ``MOE_COMBINE``'s backward has none: the backward of a
#: *wait* is an identity (no-op) -- the adjoint all-to-all for ``MOE_COMBINE``'s
#: own forward issue is attributed to ``EXPERT_COMPUTE(bwd)`` instead, since
#: that issue call is textually inside ``EXPERT_COMPUTE``'s forward code (and
#: therefore inside its mirror's backward). Symmetrically, the adjoint for
#: ``MOE_DISPATCH``'s issue lives in ``MOE_DISPATCH(bwd)`` itself (the issue
#: is the last op in its own forward code).
_BWD_HAS_COMM: Dict[PhaseType, bool] = {
    PhaseType.ATTENTION: False,
    PhaseType.MOE_DISPATCH: True,
    PhaseType.EXPERT_COMPUTE: True,
    PhaseType.MOE_COMBINE: False,
}

ANCHOR_PREFIX = '__phase__'
OP_CONTEXT_KEY = 'phase'


def _anchor_name(layer_id: int, phase_type: PhaseType) -> str:
    return f'{ANCHOR_PREFIX}{layer_id}:{phase_type.value}'


def phase_anchor(layer_id: int, phase_type: PhaseType) -> None:
    """Emit a phase-boundary anchor in model ``forward()`` code.

    A thin, structured wrapper around
    :func:`nnscaler.runtime.function.anchor` (itself a no-op returning
    ``None`` -- zero numerical effect, real or traced). Call once per phase,
    immediately before the sub-module(s)/ops implementing that phase, e.g.::

        def forward(self, x):
            phase_anchor(self.layer_id, PhaseType.ATTENTION)
            x = self.attn(x)
            phase_anchor(self.layer_id, PhaseType.MOE_DISPATCH)
            dispatched = self.moe.dispatch(x)
            phase_anchor(self.layer_id, PhaseType.EXPERT_COMPUTE)
            combined_in = self.moe.expert_compute(dispatched)
            phase_anchor(self.layer_id, PhaseType.MOE_COMBINE)
            x = self.moe.combine(combined_in)
            return x
    """
    return ncf.anchor(_anchor_name(layer_id, phase_type))


def _parse_anchor_name(name: str) -> Optional[Tuple[int, PhaseType]]:
    if not isinstance(name, str) or not name.startswith(ANCHOR_PREFIX):
        return None
    body = name[len(ANCHOR_PREFIX):]
    layer_str, sep, type_str = body.partition(':')
    if not sep:
        raise PhaseError(f"Malformed phase anchor name {name!r} (missing ':' separator)")
    try:
        layer_id = int(layer_str)
    except ValueError:
        raise PhaseError(f"Malformed phase anchor name {name!r} (non-integer layer id {layer_str!r})")
    try:
        phase_type = PhaseType(type_str)
    except ValueError:
        raise PhaseError(f"Malformed phase anchor name {name!r} (unknown phase type {type_str!r})")
    return layer_id, phase_type


@dataclass(frozen=True)
class PhaseIdentity:
    """Static identity of one phase node: which layer, which phase type,
    which direction, and its canonical (forward) position within its layer's
    phase sequence -- independent of *which* microbatch instance a schedule
    later places (that is :class:`~nnscaler.graph.schedule.schedplan.Block`'s
    ``mid``, a runtime/schedule-time concept, not part of this static
    identity).
    """
    layer_id: int
    phase_type: PhaseType
    direction: str  # 'forward' or 'backward'
    seq_in_layer: int  # 0-based canonical forward-order index within its layer's phase sequence

    def __post_init__(self):
        if self.direction not in ('forward', 'backward'):
            raise PhaseError(f"direction must be 'forward' or 'backward', got {self.direction!r}")

    @property
    def has_communication(self) -> bool:
        """Whether this phase's own (mirrored, if backward) code performs any
        real communication (an issue and/or a wait)."""
        table = _FWD_HAS_COMM if self.direction == 'forward' else _BWD_HAS_COMM
        return table[self.phase_type]

    @property
    def issues_async(self) -> bool:
        """Whether this phase's own code ends with an async communication
        issue that the immediately-next phase (in its own direction's order)
        must wait on. Only ever true for forward ``MOE_DISPATCH``/``EXPERT_COMPUTE``
        -- see module docstring "Scope"."""
        return self.direction == 'forward' and self.phase_type in _ISSUES_ASYNC

    @property
    def kind(self) -> PhaseKind:
        return PhaseKind.COMM if self.has_communication else PhaseKind.COMPUTE

    def __repr__(self):
        arrow = 'fwd' if self.direction == 'forward' else 'bwd'
        return f"Phase(L{self.layer_id}.{self.phase_type.value}.{arrow}#{self.seq_in_layer})"


@dataclass
class PhaseNode:
    """A phase's static identity plus the (already direction-matched)
    ``IRSegment`` it is attached to -- ``identity.direction == 'forward'``
    implies ``segment.isfw()``, and vice versa."""
    identity: PhaseIdentity
    segment: IRSegment

    def __repr__(self):
        return f"PhaseNode({self.identity!r}, seg={self.segment.cid})"


def get_phase(node) -> Optional[PhaseNode]:
    """Read back a node's :class:`PhaseNode` metadata (set by
    :func:`lower_layer_to_phases`), if any. Returns ``None`` for a plain,
    untagged node (including an untagged Step B local segment, or any
    ``IRCell`` without an ``op_context``)."""
    ctx_get = getattr(node, 'get_op_context', None)
    if ctx_get is None:
        return None
    return ctx_get(OP_CONTEXT_KEY)


@dataclass(frozen=True)
class PhaseExecutionIdentity:
    """The full identity of one *concrete, scheduled execution instance* of
    a phase -- i.e. :class:`PhaseIdentity` (static: layer/phase-type
    /direction/seq, the same for every microbatch and every run of the
    schedule) PLUS the two runtime/schedule-time coordinates that make one
    execution instance concretely distinct from another: which microbatch,
    and which physical stage (device group) it executes on.

    ``PhaseIdentity`` itself deliberately stays static/microbatch
    -independent (see its own docstring) -- microbatch and physical-stage
    identity already exist on :class:`~nnscaler.graph.schedule.schedplan.Block`
    (``.mid``, ``.device``) but, before this dataclass, were never bundled
    together with a phase's own static identity into one explicit, directly
    -constructible, directly-comparable object; call sites had to
    separately track a block's ``.mid``/``.device`` alongside its
    ``PhaseNode`` via implicit local variables/tuples. This makes that
    bundling explicit and gives it a name.

    Every concrete phase execution instance placed anywhere in a schedule's
    table must have a UNIQUE ``PhaseExecutionIdentity`` -- see
    :func:`phase_execution_identity` (the standard constructor from a
    ``(Block, PhaseNode)`` pair) and
    ``tests/graph/schedule/test_phase_schedule_sweep.py``'s
    ``test_phase_execution_identity_is_globally_unique`` for the property
    test proving this holds across an entire real, multi-stage,
    multi-microbatch schedule table.
    """
    microbatch: int
    physical_stage: Tuple[int, ...]
    layer_id: int
    phase_type: PhaseType
    direction: str
    seq_in_layer: int

    @classmethod
    def from_identity(cls, identity: PhaseIdentity, microbatch: int,
                       physical_stage: Tuple[int, ...]) -> 'PhaseExecutionIdentity':
        return cls(microbatch, tuple(physical_stage), identity.layer_id,
                    identity.phase_type, identity.direction, identity.seq_in_layer)

    def __repr__(self):
        arrow = 'fwd' if self.direction == 'forward' else 'bwd'
        return (f"PhaseExec(mb{self.microbatch}@stage{self.physical_stage}:"
                f"L{self.layer_id}.{self.phase_type.value}.{arrow}#{self.seq_in_layer})")


def phase_execution_identity(block, phase_node: Optional[PhaseNode] = None) -> Optional[PhaseExecutionIdentity]:
    """Construct a :class:`PhaseExecutionIdentity` for one scheduled
    ``Block``, reading its phase's static identity from ``phase_node``
    (defaults to ``get_phase(block.content)``, i.e. the block's own tagged
    metadata) and its microbatch/physical-stage coordinates directly off
    the block itself (``block.mid``/``block.device``). Returns ``None`` if
    the block carries no :class:`PhaseNode` (a plain, non-phase node, e.g.
    an untagged dataloader/reducer block)."""
    if phase_node is None:
        phase_node = get_phase(block.content)
    if phase_node is None:
        return None
    return PhaseExecutionIdentity.from_identity(phase_node.identity, block.mid, tuple(block.device))


def _tag(segment: IRSegment, identity: PhaseIdentity) -> None:
    segment.set_op_context(OP_CONTEXT_KEY, PhaseNode(identity, segment))


def _validate_sequence(layer_id: int, found: Sequence[PhaseType]) -> None:
    if tuple(found) not in _LEGAL_PHASE_SEQUENCES:
        legal = [[t.value for t in seq] for seq in _LEGAL_PHASE_SEQUENCES]
        raise PhaseError(
            f"layer {layer_id}: illegal/incomplete phase sequence "
            f"{[t.value for t in found]} (found via phase anchors, in graph "
            f"order); must be exactly one of {legal}. This usually means a "
            f"phase anchor is missing, duplicated, or out of the canonical "
            f"order (ATTENTION -> MOE_DISPATCH -> EXPERT_COMPUTE -> MOE_COMBINE)."
        )


def lower_layer_to_phases(
    graph: IRGraph,
    layer_nodes: Sequence[IRFwOperation],
    layer_id: int,
) -> List[PhaseNode]:
    """Lower one layer's contiguous, not-yet-grouped forward node range into
    its constituent phase segments.

    Must be called at the same point in a PAS policy where the layer would
    otherwise become part of a single physical-stage ``graph.group(...)``
    call -- i.e. *before* any operator partition (TP/EP sharding) or adapter
    generation, the same precondition
    :func:`~nnscaler.graph.schedule.local_segment.partition_stage_into_local_segments`
    (which this function calls directly) already requires. ``layer_nodes``
    must contain :func:`phase_anchor` markers spelling out *exactly*
    :data:`MOE_PHASE_SEQUENCE` or :data:`DENSE_PHASE_SEQUENCE`, all for the
    same ``layer_id`` (see :class:`PhaseError` cases below).

    Args:
        graph: the (root) graph these nodes belong to.
        layer_nodes: the candidate layer's forward node list, in forward
            execution order, containing this layer's :func:`phase_anchor`
            markers.
        layer_id: the layer id these anchors are expected to carry (a
            mismatch, e.g. accidentally passing another layer's nodes,
            raises :class:`PhaseError` rather than silently mislabeling).

    Returns:
        List[PhaseNode]: one :class:`PhaseNode` per forward phase, in
        canonical (forward) order. ``result[i].segment.mirror`` is the
        corresponding backward phase segment, itself tagged with a
        :class:`PhaseNode` of ``direction='backward'``.

    Raises:
        PhaseError: if no phase anchors are found, if anchors for more than
            one layer id are mixed in, if the anchored sequence is not
            exactly :data:`MOE_PHASE_SEQUENCE` or :data:`DENSE_PHASE_SEQUENCE`,
            or if a produced forward phase segment has no ``.mirror``.
        LocalSegmentError: propagated unmodified from
            ``partition_stage_into_local_segments`` for non-contiguity,
            recompute-boundary, or shared-attribute violations.
    """
    layer_nodes = list(layer_nodes)

    found_types: List[PhaseType] = []
    anchor_layer_ids = set()
    for node in layer_nodes:
        if not isinstance(node, IRGraphAnchor):
            continue
        parsed = _parse_anchor_name(node.kwargs.get('name'))
        if parsed is None:
            continue  # an unrelated anchor (e.g. a physical-stage boundary); not ours
        found_layer_id, phase_type = parsed
        anchor_layer_ids.add(found_layer_id)
        found_types.append(phase_type)

    if not found_types:
        raise PhaseError(
            f"layer {layer_id}: no phase anchors "
            f"(nnscaler.graph.schedule.phase.phase_anchor(...)) found in the "
            f"given layer_nodes; lower_layer_to_phases requires at least the "
            f"ATTENTION phase to be anchored explicitly."
        )
    if anchor_layer_ids != {layer_id}:
        raise PhaseError(
            f"layer {layer_id}: phase anchors for layer id(s) "
            f"{sorted(anchor_layer_ids)} found instead of only {layer_id}; "
            f"lower_layer_to_phases must be called with the node range of "
            f"exactly one layer's phase anchors (one call per layer)."
        )
    _validate_sequence(layer_id, found_types)

    boundary = AnchorBoundary({_anchor_name(layer_id, t) for t in found_types})
    segments = partition_stage_into_local_segments(graph, layer_nodes, boundary)

    if len(segments) != len(found_types):
        # Defensive: unreachable given partition_stage_into_local_segments's
        # own contiguity/dedup guarantees (one boundary index per anchor,
        # producing exactly len(found_types) groups), but guard explicitly
        # rather than silently mis-zip segments <-> phase types below.
        raise PhaseError(
            f"layer {layer_id}: {len(found_types)} phase anchors produced "
            f"{len(segments)} local segments; expected these counts to match "
            f"(internal inconsistency -- please file a bug)."
        )

    phase_nodes = []
    for seq, (seg, ptype) in enumerate(zip(segments, found_types)):
        if seg.mirror is None:
            raise PhaseError(
                f"layer {layer_id} phase {ptype.value}: forward segment "
                f"{seg.cid} has no mirror (backward) segment; every phase "
                f"must have an independently executor.backward()-able "
                f"counterpart."
            )
        fwd_identity = PhaseIdentity(layer_id, ptype, 'forward', seq)
        bwd_identity = PhaseIdentity(layer_id, ptype, 'backward', seq)
        _tag(seg, fwd_identity)
        _tag(seg.mirror, bwd_identity)
        phase_nodes.append(PhaseNode(fwd_identity, seg))
    return phase_nodes


def validate_phase_layout(graph: IRGraph, num_stages: int) -> None:
    """Post-device-assignment fail-fast check: every phase belonging to one
    ``layer_id`` must resolve to the exact same physical stage (device
    tuple).

    This is a redundant, phase-specific safety net -- deliberately checked
    directly here, with a clear, phase-specific error message, rather than
    relying only on
    ``LocalSegmentSched._stage_local_segments``'s more generic device-tuple
    contiguity check (which :func:`PhaseAwareSched.sched_1f1b_phase_aware`
    also runs, and which would eventually catch the same misconfiguration
    less legibly) -- the same "add an explicit, direct check instead of an
    incidental downstream failure" pattern Step B's own post-commit audit
    established for :func:`~nnscaler.graph.schedule.local_segment._stage_local_segments`.

    Raises:
        PhaseError: if any two phase segments sharing a ``layer_id`` resolve
            to different device tuples.
    """
    segments = graph.select(ntype=IRSegment, flatten=False)
    layer_devices: Dict[int, Tuple[Tuple[int, ...], int]] = {}
    for seg in segments:
        if not seg.isfw():
            continue
        phase = get_phase(seg)
        if phase is None:
            continue
        devs = tuple(seg.device)
        layer_id = phase.identity.layer_id
        if layer_id in layer_devices and layer_devices[layer_id][0] != devs:
            other_devs, other_seq = layer_devices[layer_id]
            raise PhaseError(
                f"layer {layer_id}: phase {phase.identity.phase_type.value} "
                f"(seq {phase.identity.seq_in_layer}) is assigned to device "
                f"tuple {devs}, but another phase of the same layer (seq "
                f"{other_seq}) is assigned to {other_devs}; all phases of one "
                f"layer must share the same physical stage's device tuple "
                f"(a phase never introduces a new physical pipeline stage)."
            )
        layer_devices.setdefault(layer_id, (devs, phase.identity.seq_in_layer))


# --------------------------------------------------------------------------
# Scheduling
# --------------------------------------------------------------------------

def _seg_stream_context(seg: IRSegment):
    """Read back an optional :class:`~nnscaler.graph.schedule.schedplan.StreamContext`
    a PAS policy attached to a phase segment via
    ``seg.set_op_context('stream_context', StreamContext(...))`` (the same
    generic ``op_context`` channel :func:`get_phase`/:class:`PhaseNode` use),
    so :meth:`PhaseAwareSched.sched_1f1b_phase_aware` can pass it through to
    ``SchedulePlan.add_segment``'s own ``stream_context`` parameter --
    required for it to actually reach codegen: ``ExecutionPlan.from_schedplan``
    keys its real/CUDA-stream-block-emission decision off each *Block's own*
    ``stream_context`` attribute (set only via ``add_segment(...,
    stream_context=...)``), not off whatever ``op_context`` happens to
    already be set on the underlying segment node. This lets a PAS policy
    give a specific phase (e.g. ``MOE_DISPATCH``) a dedicated communication
    stream using the exact same, pre-existing, first-class extension point
    Step A's own ``test_combined_1f1b_pipeline_e2e.py`` (``_pas_multi_stream``)
    already demonstrates for inter-segment P2P adapters -- applied here to a
    phase segment instead of an adapter, with no core-file changes beyond
    this one, generic (not MoE-specific) lookup helper.
    """
    return seg.get_op_context('stream_context')


@dataclass
class _PlacedBlock:
    segment: IRSegment
    mid: int


def _interleave_window(local_segs: Sequence[IRSegment], b_mid: int, f_mid: int) -> List[_PlacedBlock]:
    """Compute the placement order for one steady-state ``B(m)`` + ``F(m+1)``
    interleave window over one stage's local segments.

    If none of ``local_segs`` carry phase metadata, reproduces
    ``LocalSegmentSched.sched_1f1b_local_segments``'s exact interleave order
    byte-for-byte (regression-tested directly in
    ``tests/graph/schedule/test_phase_schedule_sweep.py``). If any of them
    do, all of them must (mixed tagging within one stage is rejected), and
    the phase-kind-aware algorithm from the module docstring's "Scheduling"
    section is used instead.
    """
    k = len(local_segs)
    tags = [get_phase(seg) is not None for seg in local_segs]
    if any(tags) and not all(tags):
        raise PhaseError(
            "mixing phase-tagged and untagged local segments within one "
            "physical stage's interleave window is not supported; tag all "
            "of a stage's local segments via lower_layer_to_phases (one "
            "call per layer) or leave all of them untagged."
        )

    if not any(tags):
        # Byte-for-byte LocalSegmentSched.sched_1f1b_local_segments behavior.
        blocks = []
        for i in range(k):
            blocks.append(_PlacedBlock(local_segs[k - 1 - i].mirror, b_mid))
            blocks.append(_PlacedBlock(local_segs[i], f_mid))
        return blocks

    fwd = list(local_segs)
    bwd = [local_segs[k - 1 - i].mirror for i in range(k)]  # B(m)'s own execution order

    merged: List[_PlacedBlock] = []
    j = 0
    for seg in fwd:
        merged.append(_PlacedBlock(seg, f_mid))
        if get_phase(seg).identity.issues_async:
            while j < len(bwd):
                bseg = bwd[j]
                merged.append(_PlacedBlock(bseg, b_mid))
                j += 1
                if not get_phase(bseg).identity.has_communication:
                    break
    while j < len(bwd):
        merged.append(_PlacedBlock(bwd[j], b_mid))
        j += 1
    return merged


class PhaseAwareSched:
    """Phase-kind-aware predefined schedule.

    Kept separate from
    :class:`~nnscaler.graph.schedule.local_segment.LocalSegmentSched` (itself
    kept separate from ``PredefinedSched``) so neither of those modules is
    modified by Step C; use directly as a callable ``pipeline_scheduler``.
    """

    @staticmethod
    def sched_1f1b_phase_aware(graph: IRGraph, num_microbatches: int, num_stages: int) -> SchedulePlan:
        """1F1B scheduling with phase-kind-aware interleaving on the first
        stage. See module docstring "Scheduling" for the exact interleave
        rule, and "Known limitations" (inherited from Step B) for why only
        the first stage is interleaved.
        """
        if num_microbatches <= 0:
            raise ValueError(f"expected num_microbatches > 0, but got {num_microbatches}")

        validate_phase_layout(graph, num_stages)
        stage_local_segs = _stage_local_segments(graph, num_stages)

        sched = SchedulePlan(graph, num_microbatches)

        wait_steps = [sid for sid in range(num_stages)]
        bw_ofst = [num_stages - 1 - sid for sid in range(num_stages)]
        total_steps = num_microbatches * 2 + (num_stages - 1) * 2
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
                    fw_mb = nxt[1]
                    base = step * scale
                    for sub, block in enumerate(_interleave_window(local_segs, mb, fw_mb)):
                        sched.add_segment(block.segment, block.mid, base + sub,
                                           stream_context=_seg_stream_context(block.segment))
                    step += 2
                    continue
                base = step * scale
                if kind == 'F':
                    for i in range(k):
                        sched.add_segment(local_segs[i], mb, base + i,
                                           stream_context=_seg_stream_context(local_segs[i]))
                else:
                    for i in range(k):
                        bseg = local_segs[k - 1 - i].mirror
                        sched.add_segment(bseg, mb, base + i, stream_context=_seg_stream_context(bseg))
                step += 1

        sched.finish()
        return sched
