#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Step C of the combined-1F1B work: a minimal, real MoE expert-parallel
all-to-all communication primitive (``moe_dispatch``/``moe_combine``, plus
their deferred-wait counterparts), registered as ordinary nnScaler custom ops
(:func:`nnscaler.graph.parser.register.register_op`) so a model's
``forward()`` can call them directly and have nnScaler trace, partition, and
codegen them like any other operator -- exactly the same mechanism
``nnscaler.customized_ops.ring_attention`` already uses for a different real
collective (ring attention's P2P rotation).

Why a new primitive, not the existing ``all_to_all``/``all_to_all_single``
-------------------------------------------------------------------------
``nnscaler.runtime.adapter.collectives.all_to_all``/``all_to_all_single``
already wrap a real ``torch.distributed.all_to_all``/``all_to_all_single``,
so the underlying *collective* is reused, not reinvented (confirmed via a
repo-wide search before writing this module -- there is no existing
MoE/expert/dispatch/combine-specific communication primitive anywhere in
``nnscaler/``, only this general tensor-resharding one, and the alternative
"replicate + shard-expert-weights + reduce" strategy the ``deepseek_coder_v2_lite``
example uses, which is a fundamentally different EP strategy than the
all-to-all-based dispatch/combine explicitly asked for here). Two concrete
gaps, found while evaluating reuse, motivate the small amount of genuinely
new code below instead of calling those functions directly:

1. Their contract is "reshard a distributed tensor from partition dim
   ``idim`` to ``odim``" (a fixed, compile-time-known transform, used
   internally by nnScaler's own automatic adapter-generation for tensor-
   parallel resharding). MoE dispatch/combine instead moves each rank's own,
   *distinctly-valued* local capacity buffer to its destination EP rank(s) --
   structurally an all-to-all over one leading (already-per-rank) dimension,
   not a reshard of one logical global tensor, and it must be driven by an
   explicit, model-chosen ``ep_ranks`` group (the same "explicit device-group
   argument, defaulting to a harmless identity path so tracing never performs
   real communication" pattern
   ``nnscaler.customized_ops.ring_attention.ring_attn.wrap_ring_attn_func``
   already uses for ``process_group`` -- see :func:`moe_dispatch`), not
   something nnScaler's own partition-annotation system should discover.
2. ``all_to_all_single``'s existing ``async_op=True`` path was found, while
   building this module, to be dormant/untested and subtly inconsistent: it
   tracks the in-flight ``Work`` keyed by the *input* tensor
   (``AsyncCommHandler().submit(tensor, [work], ...)``) while *returning* a
   different, freshly-allocated output buffer (``otensor``) -- unlike
   ``all_gather``'s analogous path in the same file, which correctly sets
   ``otensor = tensor`` so the returned placeholder *is* the tracked key. Its
   only real caller (``nnscaler.runtime.adapter.nn.alltoall_alltoall``) never
   passes ``async_op=True``, so this is latent, not currently reachable --
   fixing it is out of scope for this step (touching a shared, differently-
   tested function to fix an unrelated dormant bug is a bigger, separate
   change than "add MoE phase IR"), so this module implements its own,
   correctly-tracked async issue/wait pair from scratch instead of building
   on top of that path. It reuses the exact same, real, already-proven
   infrastructure underneath: ``nnscaler.runtime.device.DeviceGroup`` for the
   process group and ``nnscaler.runtime.executor.AsyncCommHandler`` (the
   very same channel/sequence-tracked issue/wait API Step A added for P2P
   async-recv, see ``nnscaler.runtime.executor``'s module docstring) for the
   deferred-wait bookkeeping.

Issue/wait split, and why the backward is just another all-to-all
-------------------------------------------------------------------
:func:`moe_dispatch`/:func:`moe_combine` ("issue") and
:func:`moe_dispatch_wait`/:func:`moe_combine_wait` ("wait") are deliberately
four *separate* registered ops (hence four separate, independently
schedulable graph nodes) rather than one op that issues-and-immediately-waits:
this is what lets :mod:`nnscaler.graph.schedule.phase`'s phase lowering put
the issue at the end of one phase segment and the wait at the start of the
*next* phase segment, so the schedule can genuinely place independent work
from another micro-batch in between (see that module's docstring). An issue
returns the (not-yet-populated) output buffer itself as the "pending" value --
the same "the returned placeholder tensor object is the tracked key" idiom
``all_gather``'s existing async path already uses -- and the matching wait
resolves it via ``AsyncCommHandler().wait(...)``.

Both the issue and the wait are real ``torch.autograd.Function``s, so
gradient flow is standard autograd, not special-cased: the issue's backward
performs the *adjoint* all-to-all on the incoming gradient (synchronous --
see "Scope" below), and the wait's backward is the identity (waiting performs
no computation). For an equal-chunk all-to-all over one group, the adjoint of
"redistribute chunks across ranks" is the exact same redistribution applied
to the gradient (a chunk-transpose is its own inverse), matching the existing
``nnscaler.runtime.adapter.nn.AllToAllAllToAllSingle``'s backward, which
likewise just calls the forward collective again with swapped roles.

Scope
-----
Only the *forward* issue is asynchronous with a deferred wait (matching
:mod:`nnscaler.graph.schedule.phase`'s documented scope: only ``F(m+1)``'s
communication needs to hide behind ``B(m)``'s independent compute). The
backward adjoint all-to-all (invoked from :meth:`_MoEDispatchIssue.backward`/
:meth:`_MoECombineIssue.backward`) is issued synchronously
(``async_op=False``) -- no separate "backward issue"/"backward wait" op pair
is introduced.

Static shape / capacity
------------------------
The transported buffer's leading dimension must equal ``len(ep_ranks)``
(one equal-sized chunk per EP rank) and its shape is otherwise arbitrary but
must be identical, and known at compile time, on every participating rank --
i.e. callers are expected to use a fixed-*capacity* token buffer (the
standard GShard/Switch-Transformer-style expert-capacity buffer: gate, then
scatter into a ``[num_experts, capacity, hidden]`` buffer with overflow
dropped and underflow zero-padded), never a data-dependent shape. This is
exactly nnScaler's static-codegen philosophy applied to MoE: the *routing
decision* (which expert a token goes to) is real, data-dependent runtime
compute, but the *communication shape* it feeds is fixed at compile time.
"""

from typing import Optional, Sequence, Tuple

import torch

from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.executor import AsyncCommHandler
from nnscaler.profiler.timer import CudaTimer
from nnscaler.graph.parser.register import register_op


class MoECommError(ValueError):
    """Raised for illegal MoE all-to-all configuration (bad ``ep_ranks``,
    shape/capacity mismatch, or illegal channel/outstanding-cap config) --
    mirrors ``nnscaler.graph.schedule.local_segment.LocalSegmentError`` and
    ``nnscaler.runtime.executor.AsyncCommError``'s own "dedicated,
    ValueError-compatible exception type" convention.
    """


def _check_ep_ranks(buffer: torch.Tensor, ep_ranks: Sequence[int]) -> Tuple[int, ...]:
    ep_ranks = tuple(ep_ranks)
    if len(ep_ranks) == 0:
        raise MoECommError("ep_ranks must be non-empty")
    if len(set(ep_ranks)) != len(ep_ranks):
        raise MoECommError(f"ep_ranks must not contain duplicate ranks, got {ep_ranks}")
    if buffer.dim() < 1:
        raise MoECommError(f"buffer must have at least 1 dimension (got shape {tuple(buffer.shape)})")
    if buffer.shape[0] != len(ep_ranks):
        raise MoECommError(
            f"buffer's leading dimension ({buffer.shape[0]}) must equal "
            f"len(ep_ranks) ({len(ep_ranks)}) -- exactly one equal-sized "
            f"chunk per EP rank is required (a static-capacity buffer); got "
            f"buffer shape {tuple(buffer.shape)} for ep_ranks {ep_ranks}."
        )
    return ep_ranks


def _all_to_all_ep(buffer: torch.Tensor, ep_ranks: Tuple[int, ...], async_op: bool):
    buffer = buffer.contiguous() if not buffer.is_contiguous() else buffer
    group = DeviceGroup().get_group(ep_ranks)
    otensor = torch.empty_like(buffer)
    work = torch.distributed.all_to_all_single(otensor, buffer, group=group, async_op=async_op)
    return otensor, work


def _sync_all_to_all_ep(buffer: torch.Tensor, ep_ranks: Tuple[int, ...]) -> torch.Tensor:
    CudaTimer().start(field_name='comm', predefined=True)
    otensor, _ = _all_to_all_ep(buffer, ep_ranks, async_op=False)
    CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def _issue_all_to_all_ep(
    buffer: torch.Tensor,
    ep_ranks: Tuple[int, ...],
    channel,
    max_outstanding: Optional[int],
) -> torch.Tensor:
    if channel is not None:
        if max_outstanding is None:
            raise MoECommError("max_outstanding is required (must be >= 1) when channel is given")
    otensor, work = _all_to_all_ep(buffer, ep_ranks, async_op=True)
    if channel is not None:
        AsyncCommHandler().issue_recv(channel, otensor, [work], max_outstanding)
    else:
        AsyncCommHandler().submit(otensor, [work])
    return otensor


class _MoEDispatchIssue(torch.autograd.Function):
    """Real forward dispatch all-to-all (async issue); backward is the
    adjoint (synchronous) all-to-all -- see module docstring."""

    @staticmethod
    def forward(ctx, buffer: torch.Tensor, ep_ranks: Tuple[int, ...], channel, max_outstanding):
        ctx.ep_ranks = ep_ranks
        return _issue_all_to_all_ep(buffer, ep_ranks, channel, max_outstanding)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _sync_all_to_all_ep(grad_output, ctx.ep_ranks)
        return grad_input, None, None, None


class _MoEDispatchWait(torch.autograd.Function):
    """Deferred wait for a pending dispatch buffer; backward is the identity
    (a wait performs no computation of its own)."""

    @staticmethod
    def forward(ctx, pending: torch.Tensor):
        return AsyncCommHandler().wait(pending)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class _MoECombineIssue(torch.autograd.Function):
    """Real forward combine all-to-all (async issue); kept as a distinct
    class from :class:`_MoEDispatchIssue` (even though the collective itself
    is structurally identical) so registered signatures -- and hence
    generated code -- name dispatch and combine distinctly (see
    ``tests/codegen/test_phase_gencode.py``)."""

    @staticmethod
    def forward(ctx, buffer: torch.Tensor, ep_ranks: Tuple[int, ...], channel, max_outstanding):
        ctx.ep_ranks = ep_ranks
        return _issue_all_to_all_ep(buffer, ep_ranks, channel, max_outstanding)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = _sync_all_to_all_ep(grad_output, ctx.ep_ranks)
        return grad_input, None, None, None


class _MoECombineWait(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pending: torch.Tensor):
        return AsyncCommHandler().wait(pending)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def moe_dispatch(
    buffer: torch.Tensor,
    ep_ranks: Tuple[int, ...],
    channel=None,
    max_outstanding: Optional[int] = None,
) -> torch.Tensor:
    """Issue the (real, async) MoE dispatch all-to-all across ``ep_ranks``.

    ``buffer`` must be shaped ``[len(ep_ranks), ...]`` -- one equal-sized,
    already rank-ordered chunk per destination EP rank (build it with a
    fixed-capacity scatter first; see module docstring "Static shape").
    Returns a *pending* tensor of the same shape/dtype; call
    :func:`moe_dispatch_wait` on it (later -- possibly after other, unrelated
    work, to let the two genuinely overlap) before reading its values.

    ``ep_ranks=(r,)`` (a single rank) degenerates to a true no-op identity
    (correct single-EP-rank/no-expert-parallelism reference semantics, and
    also what tracing uses when a model calls this without having been
    assigned a real multi-rank EP group).

    Args:
        channel: optional stable, hashable identity for this dispatch
            call-site (e.g. a layer id), mirroring
            ``nnscaler.runtime.adapter.collectives.move``'s ``channel``
            parameter: tracks the issue/wait pair via
            ``nnscaler.runtime.executor.AsyncCommHandler``'s FIFO
            channel bookkeeping. Default ``None`` skips channel tracking
            (only the plain tensor-keyed wait applies).
        max_outstanding: required (>= 1) when ``channel`` is given.

    Raises:
        MoECommError: if ``ep_ranks`` is empty/has duplicates, if
            ``buffer``'s leading dimension does not equal ``len(ep_ranks)``,
            or if ``channel`` is given without ``max_outstanding``.
    """
    ep_ranks = _check_ep_ranks(buffer, ep_ranks)
    if len(ep_ranks) == 1:
        return buffer
    return _MoEDispatchIssue.apply(buffer, ep_ranks, channel, max_outstanding)


def moe_dispatch_wait(pending: torch.Tensor) -> torch.Tensor:
    """Deferred wait for a pending :func:`moe_dispatch` buffer."""
    return _MoEDispatchWait.apply(pending)


def moe_combine(
    buffer: torch.Tensor,
    ep_ranks: Tuple[int, ...],
    channel=None,
    max_outstanding: Optional[int] = None,
) -> torch.Tensor:
    """Issue the (real, async) MoE combine all-to-all across ``ep_ranks``
    (the structural inverse leg of :func:`moe_dispatch`: expert outputs
    travel back to the rank that originally owned each token). See
    :func:`moe_dispatch` for the buffer-shape contract, degenerate
    single-rank case, and ``channel``/``max_outstanding`` semantics.
    """
    ep_ranks = _check_ep_ranks(buffer, ep_ranks)
    if len(ep_ranks) == 1:
        return buffer
    return _MoECombineIssue.apply(buffer, ep_ranks, channel, max_outstanding)


def moe_combine_wait(pending: torch.Tensor) -> torch.Tensor:
    """Deferred wait for a pending :func:`moe_combine` buffer."""
    return _MoECombineWait.apply(pending)


# --------------------------------------------------------------------------
# nnScaler custom-op registration
# --------------------------------------------------------------------------
# All dims are frozen ('^'): the transported buffer's own shape is fixed at
# compile time (see module docstring "Static shape"), and its distribution
# across `ep_ranks` is driven entirely by the explicit `ep_ranks` argument,
# not discovered by nnScaler's own partition-annotation system -- the same
# "explicit device-group argument, opaque to auto-partitioning" contract
# `nnscaler.customized_ops.ring_attention.ring_attn.wrap_ring_attn_func` uses
# for its own `process_group` argument.
_MOE_A2A_ANNO = 'e^ c^ h^ -> e^ c^ h^'


def _fake_identity(buffer: torch.Tensor, ep_ranks, channel=None, max_outstanding=None) -> torch.Tensor:
    """Lightweight, communication-free tracing substitute for
    :func:`moe_dispatch`/:func:`moe_combine`: an all-to-all's *shape*
    contract is identity (same shape/dtype in and out), so returning
    ``buffer`` unchanged is sufficient for nnScaler to infer this op's
    shape/dtype/``requires_grad`` during tracing -- real distributed
    communication only ever happens through the registered ``runtime_fn``
    itself, invoked separately by the actually-compiled/generated code
    (never during tracing)."""
    return buffer


def _fake_identity_wait(pending: torch.Tensor) -> torch.Tensor:
    return pending


register_op(_MOE_A2A_ANNO, fake_fn=_fake_identity)(moe_dispatch)
register_op('e^ c^ h^ -> e^ c^ h^', fake_fn=_fake_identity_wait)(moe_dispatch_wait)
register_op(_MOE_A2A_ANNO, fake_fn=_fake_identity)(moe_combine)
register_op('e^ c^ h^ -> e^ c^ h^', fake_fn=_fake_identity_wait)(moe_combine_wait)
