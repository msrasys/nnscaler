#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

r"""
Executor for runtime
"""
import atexit

from collections import deque
from dataclasses import dataclass
from typing import Tuple, Any, Callable, List, Dict, Deque, Hashable, Iterable, Optional, Union, Iterator
import torch
import logging
from torch.distributed import Work

from ._patch_torch import stage_backward_input, stage_backward_weight


_logger = logging.getLogger(__name__)

_ALLOW_GRAD_DTYPES = (torch.double, torch.float32, torch.float16, torch.bfloat16)


def debug_id(tensors, msg: str, rank: int):
    if torch.distributed.get_rank() == rank:
        if torch.is_tensor(tensors):
            print(f'[{torch.distributed.get_rank()}] {msg}: [{id(tensors)}]')
        else:
            print(f'[{torch.distributed.get_rank()}] {msg}: {[id(t) for t in tensors]}')


class AsyncCommError(RuntimeError):
    """Illegal channel/sequence/lifecycle usage of async P2P communication.

    Raised by :meth:`_AsyncCommHandler.issue_recv`, and by the channel-aware
    check inside :meth:`_AsyncCommHandler.wait`, instead of letting an
    outstanding-op count grow unbounded or a mismatched issue/wait pairing
    resolve silently (which could otherwise hang, or resolve the wrong
    buffer). See ``CompileFlag.async_recv_channel`` / ``async_recv_max_outstanding``.
    """


@dataclass
class _ChannelEntry:
    """One outstanding (issued, not yet waited) op tracked on a channel."""
    seq: int
    tensor: torch.Tensor
    works: List[Work]


class _AsyncCommHandler:
    def __init__(self) -> None:
        self._works: Dict[torch.Tensor, Union[torch.Tensor, List[Work]]] = {}
        self._callbacks: Dict[torch.Tensor, Callable] = {}
        self._send_holds: List[Tuple[torch.Tensor, Work]] = []

        # ---- channel/sequence/lifecycle tracking (opt-in, additive) ----
        # A "channel" is any hashable, stable identity chosen by the caller for
        # a repeatedly-issued P2P callsite (e.g. an adapter's IR cell id). Each
        # `issue_recv` on a channel is assigned a monotonically increasing
        # sequence number; the matching resolution (via `wait`, or via a bulk
        # drain) must consume outstanding entries FIFO. This layer only adds
        # bookkeeping/validation on top of the plain tensor-keyed `submit` --
        # it does not change what is transported. (There is no `issue_send`:
        # an earlier attempt to also channel-track async sends turned out to
        # be structurally unreachable from codegen -- the only path that
        # would emit one, `CompileFlag.async_comm`, is unconditionally
        # rejected by `GlobalCommSchedule` -- so it was removed rather than
        # kept as untested, misleadingly-named dead code; see
        # `nnscaler.runtime.adapter.collectives.move`.)
        # channel -> FIFO queue of outstanding entries
        self._channel_pending: Dict[Hashable, 'deque[_ChannelEntry]'] = {}
        # channel -> next sequence number to hand out on issue
        self._channel_next_seq: Dict[Hashable, int] = {}
        # tensor identity -> (channel, seq), so the plain `wait(tensor)` entry
        # point (used unmodified by existing codegen) transparently becomes
        # channel-checked when the tensor was issued through a channel
        self._tensor_channel: Dict[torch.Tensor, Tuple[Hashable, int]] = {}

    def wait(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Wait until the finish of the communication

        @param tensor torch.Tensor
        @return tensor torch.Tensor
        """
        if tensor not in self._works:
            return tensor

        if tensor in self._tensor_channel:
            self._resolve_channel_strict(tensor)

        tensor_or_works = self._works.pop(tensor)
        if isinstance(tensor_or_works, torch.Tensor):
            return tensor_or_works

        for work in tensor_or_works:
            work.wait()
        callback = self._callbacks.pop(tensor)
        if callback is not None:
            tensor = callback(tensor)
        return tensor

    def submit(self, tensor: torch.Tensor, works: List[Work], callback: Optional[Callable] = None):
        """
        Submit an async communication
        """
        self._works[tensor] = works
        self._callbacks[tensor] = callback

    def issue_recv(
        self,
        channel: Hashable,
        tensor: torch.Tensor,
        works: List[Work],
        max_outstanding: int,
        callback: Optional[Callable] = None,
    ) -> int:
        """Register an asynchronously-issued receive on ``channel``, in
        addition to the plain tensor-keyed :meth:`submit`.

        Assigns and returns the channel's next FIFO sequence number, and
        enforces that no more than ``max_outstanding`` ops issued on this
        channel may be outstanding (issued but not yet waited) at once -- this
        bounds the buffer/handle lifecycle instead of letting it grow without
        limit. The later :meth:`wait` for ``tensor`` transparently validates
        and clears this bookkeeping (no separate "wait_recv" call needed).

        Raises:
            AsyncCommError: if ``max_outstanding < 1`` (illegal configuration),
                if ``tensor`` already has an outstanding issue, or if issuing
                would exceed the outstanding cap for ``channel``.
        """
        seq = self._channel_issue(channel, tensor, works, max_outstanding)
        self.submit(tensor, works, callback)
        return seq

    def _channel_issue(
        self,
        channel: Hashable,
        tensor: torch.Tensor,
        works: List[Work],
        max_outstanding: int,
    ) -> int:
        if max_outstanding < 1:
            raise AsyncCommError(
                f"channel {channel!r}: max_outstanding must be >= 1, got {max_outstanding}"
            )
        if tensor in self._tensor_channel:
            prior_channel, _ = self._tensor_channel[tensor]
            raise AsyncCommError(
                f"channel {channel!r}: tensor already has an outstanding issue "
                f"on channel {prior_channel!r}; a buffer must be waited before "
                f"it is reissued"
            )
        pending = self._channel_pending.setdefault(channel, deque())
        if len(pending) >= max_outstanding:
            raise AsyncCommError(
                f"channel {channel!r}: {len(pending)} outstanding op(s) already "
                f"in flight, exceeds max_outstanding={max_outstanding}; a prior "
                f"issue on this channel must be waited/drained before issuing "
                f"another (increase max_outstanding if more pipelining depth "
                f"is genuinely intended)"
            )
        seq = self._channel_next_seq.get(channel, 0)
        self._channel_next_seq[channel] = seq + 1
        pending.append(_ChannelEntry(seq, tensor, works))
        self._tensor_channel[tensor] = (channel, seq)
        return seq

    def _resolve_channel_strict(self, tensor: torch.Tensor) -> None:
        """Validate and clear the channel bookkeeping for ``tensor``'s wait.

        Enforces that waits are consumed in the exact FIFO order they were
        issued on their channel, and that the waited tensor is exactly the
        buffer that was issued -- catching mismatched issue/wait pairing (a
        scheduling/codegen bug) early with a clear error instead of a silent
        hang or a wrongly-resolved buffer. Used by the deliberate, program-order
        -driven :meth:`wait`; the opportunistic bulk-drain paths use the more
        lenient :meth:`_release_channel_any` instead (see its docstring).

        Atomic on failure: every check runs against ``self._tensor_channel``
        / ``self._channel_pending`` *before* either is mutated, so a raised
        ``AsyncCommError`` leaves both exactly as they were (no partially
        popped/cleared state for a subsequent call to trip over).
        """
        channel, seq = self._tensor_channel[tensor]
        pending = self._channel_pending.get(channel)
        if not pending:
            raise AsyncCommError(
                f"channel {channel!r}: attempted to resolve seq={seq} but no "
                f"outstanding issue is recorded on this channel (already "
                f"resolved, or never issued)"
            )
        entry = pending[0]
        if entry.seq != seq or entry.tensor is not tensor:
            raise AsyncCommError(
                f"channel {channel!r}: sequence mismatch on wait -- the oldest "
                f"outstanding issue is seq={entry.seq}, but wait() resolved "
                f"seq={seq}; issue/wait pairs on a channel must be resolved "
                f"strictly FIFO"
            )
        # all checks passed: only now mutate state, so a raise above never
        # leaves `_tensor_channel` and `_channel_pending` inconsistent with
        # each other.
        del self._tensor_channel[tensor]
        pending.popleft()

    def _release_channel_any(self, tensor: torch.Tensor) -> None:
        """Release ``tensor``'s channel-tracking entry from wherever it sits
        in its channel's outstanding queue, if it has one; a no-op otherwise.

        Used by the bulk drain paths (:meth:`drain`, :meth:`drain_all_completed`)
        over ``_works`` (receives, or collectives with a callback), which
        opportunistically resolve whichever transport op has completed first
        -- not necessarily FIFO issue order -- so, unlike :meth:`wait`, this
        does not enforce (or require) the entry to be at the front. There is
        no channel-tracked send (see module docstring), so ``_send_holds``
        entries (drained via :meth:`drain_sends` / :meth:`drain_sends_completed`)
        never actually have anything for this to release; it is only called
        here at all for genuinely channel-tracked (receive) tensors.
        """
        entry = self._tensor_channel.pop(tensor, None)
        if entry is None:
            return
        channel, seq = entry
        pending = self._channel_pending.get(channel)
        if not pending:
            return
        for i, e in enumerate(pending):
            if e.seq == seq:
                del pending[i]
                break

    def hold_send(self, tensor: torch.Tensor, work: Work):
        self._send_holds.append((tensor, work))

    def drain_sends_completed(self):
        running: list[tuple[torch.Tensor, Work]] = []
        for tensor, work in self._send_holds:
            if work.is_completed():
                work.wait()
            else:
                running.append((tensor, work))
        self._send_holds[:] = running

    def drain_sends(self):
        for tensor, work in self._send_holds:
            work.wait()
        self._send_holds.clear()

    def has_pending(self) -> bool:
        """Fast path for executor boundaries with no async work to poll."""
        return bool(self._works or self._send_holds)

    def drain_all_completed(self):
        self.drain_sends_completed()

        for tensor, tensor_or_works in list(self._works.items()):
            if isinstance(tensor_or_works, torch.Tensor):
                continue

            if not all(work.is_completed() for work in tensor_or_works):
                continue

            for work in tensor_or_works:
                work.wait()

            callback = self._callbacks.pop(tensor)
            self._works[tensor] = tensor if callback is None else callback(tensor)
            self._release_channel_any(tensor)

    def drain(self):
        """
        Blocking-wait every still-pending communication and clear the handler.

        Used before a step returns when async-recv is enabled: an async-recv
        whose output is a step output (e.g. the gathered loss) is never consumed
        by a later node, so it is never explicitly waited. This drains the held
        sends and any such callback-less pending receives so the handler is fully
        cleared (see :meth:`check_clear`). Entries that carry a callback are left
        untouched, as their result must be rebound at an explicit ``wait``.
        """
        self.drain_sends()
        for tensor, tensor_or_works in list(self._works.items()):
            if isinstance(tensor_or_works, torch.Tensor):
                # an already-resolved value that was never consumed; drop it
                self._works.pop(tensor, None)
                self._callbacks.pop(tensor, None)
                self._release_channel_any(tensor)
                continue
            if self._callbacks.get(tensor) is not None:
                continue
            for work in tensor_or_works:
                work.wait()
            self._works.pop(tensor, None)
            self._callbacks.pop(tensor, None)
            self._release_channel_any(tensor)

    def check_clear(self):
        assert len(self._works) == 0 and len(self._callbacks) == 0 and len(self._send_holds) == 0, \
            f"AsyncCommHandler not cleared: works={len(self._works)}, callbacks={len(self._callbacks)}, send_holds={len(self._send_holds)}"
        leaked_channels = {c: len(p) for c, p in self._channel_pending.items() if len(p) > 0}
        assert len(self._tensor_channel) == 0 and not leaked_channels, \
            f"AsyncCommHandler channel state not cleared: tracked_tensors={len(self._tensor_channel)}, " \
            f"channels_with_outstanding={leaked_channels}"

    def force_clear_after_exception(self) -> None:
        """Forcibly discard ALL pending bookkeeping -- tensor-keyed
        works/callbacks/send-holds AND channel/sequence state -- without
        waiting on any underlying ``Work``.

        For use when the step that issued this pending communication raised
        an exception partway through (e.g. between an ``issue_recv`` and its
        matching ``wait``) and is being abandoned: without this, the
        now-stale bookkeeping would persist into a *subsequent* step (this
        handler is a process-wide singleton, not reset between steps other
        than by a normal, successful ``drain()``/explicit ``wait()``), and
        that subsequent step's legitimate ``issue_recv`` calls
        on the SAME channel could then spuriously hit the outstanding-count
        cap or a FIFO mismatch -- a confusing, misattributed error that masks
        (and postdates) the real root cause. Deliberately does NOT call
        ``work.wait()`` on anything: the underlying communicator may itself be
        in a broken state after an unrelated crash, and waiting on it could
        hang or raise a new, unrelated error in place of the original one.
        Never raises (best-effort; any internal failure is only logged), so
        it is always safe to call from an ``except`` block that must
        re-raise the original exception unmodified afterward.
        """
        try:
            n_works, n_channels = len(self._works), len(self._tensor_channel)
            self._works.clear()
            self._callbacks.clear()
            self._send_holds.clear()
            self._channel_pending.clear()
            self._channel_next_seq.clear()
            self._tensor_channel.clear()
            if n_works or n_channels:
                _logger.warning(
                    f"AsyncCommHandler: force-cleared {n_works} pending tensor-keyed "
                    f"op(s) and {n_channels} channel-tracked op(s) left outstanding "
                    f"by a step that raised an exception (not waited on -- see "
                    f"force_clear_after_exception docstring)."
                )
        except Exception:
            _logger.exception("AsyncCommHandler.force_clear_after_exception itself failed")


_instance: Optional[_AsyncCommHandler] = None

def AsyncCommHandler() -> _AsyncCommHandler:
    global _instance
    if _instance is None:
        _instance = _AsyncCommHandler()
    return _instance


TensorPairs = List[Tuple[int, torch.Tensor]]


@dataclass
class _WeightBackwardState:
    param_groups: Optional[List[Dict[str, Any]]] = None
    output_tensors: Optional[Tuple[torch.Tensor, ...]] = None
    output_tensor_grads: Optional[Tuple[Optional[torch.Tensor], ...]] = None


@dataclass(frozen=True)
class PhaseSpec:
    """Cached input schema for one scheduled phase execution slot.

    The generated schedule gives every `(microbatch, stage, layer, phase)` a
    stable integer slot.  This schema avoids repeated generic id-dictionary
    construction while retaining alias and dynamic-input fallbacks.
    """
    input_arity: int
    tensor_mask: Tuple[bool, ...]
    grad_mask: Tuple[bool, ...]
    grad_positions: Tuple[int, ...]
    alias_groups: Tuple[Tuple[int, ...], ...]

    @classmethod
    def from_inputs(cls, inputs: Tuple[Any, ...]) -> 'PhaseSpec':
        tensor_mask = tuple(torch.is_tensor(value) for value in inputs)
        grad_mask = tuple(
            bool(value.requires_grad) if is_tensor else False
            for value, is_tensor in zip(inputs, tensor_mask)
        )
        grad_positions = tuple(index for index, requires_grad in enumerate(grad_mask) if requires_grad)
        groups: List[List[int]] = []
        positions_by_id: Dict[int, int] = {}
        for index in grad_positions:
            tensor_id = id(inputs[index])
            group_index = positions_by_id.get(tensor_id)
            if group_index is None:
                positions_by_id[tensor_id] = len(groups)
                groups.append([index])
            else:
                groups[group_index].append(index)
        return cls(
            input_arity=len(inputs),
            tensor_mask=tensor_mask,
            grad_mask=grad_mask,
            grad_positions=grad_positions,
            alias_groups=tuple(tuple(group) for group in groups),
        )

    def matches(self, inputs: Tuple[Any, ...]) -> bool:
        if len(inputs) != self.input_arity:
            return False
        for index, value in enumerate(inputs):
            if bool(torch.is_tensor(value)) != self.tensor_mask[index]:
                return False
            if self.tensor_mask[index] and bool(value.requires_grad) != self.grad_mask[index]:
                return False
        group_heads = []
        for group in self.alias_groups:
            head = id(inputs[group[0]])
            if any(id(inputs[index]) != head for index in group[1:]):
                return False
            group_heads.append(head)
        return len(group_heads) == len(set(group_heads))


@dataclass
class _PhaseState:
    detached_inputs: Tuple[torch.Tensor, ...]


class PhaseExecutor:
    """Model-owned slot executor for independently schedulable phase islands.

    It intentionally preserves one autograd graph and one backward invocation
    per phase.  The fast path only removes generic string/FIFO bookkeeping and
    rebuilds cached detach/alias metadata when an input schema changes.
    """

    def __init__(self, slot_count: int):
        if slot_count <= 0:
            raise ValueError(f'phase executor requires slot_count > 0, got {slot_count}')
        self._states: List[Optional[_PhaseState]] = [None] * slot_count
        self._specs: List[Optional[PhaseSpec]] = [None] * slot_count

    @property
    def slot_count(self) -> int:
        return len(self._states)

    def _check_slot(self, slot: int) -> None:
        if not isinstance(slot, int) or slot < 0 or slot >= self.slot_count:
            raise RuntimeError(f'invalid phase execution slot {slot!r}; expected [0, {self.slot_count})')

    @staticmethod
    def _sync_inputs(inputs: Tuple[Any, ...]) -> Tuple[Any, ...]:
        # Match Executor.sync_tensors exactly when work exists.  Most phase
        # boundaries have no outstanding async work, so avoid its generic list
        # construction and handler scan in that common case.
        handler = AsyncCommHandler()
        if not handler.has_pending():
            return inputs
        handler.drain_all_completed()
        return tuple(handler.wait(value) if torch.is_tensor(value) else value for value in inputs)

    def forward(self, slot: int, subgraph: Callable, *inputs: Any, requires_grad: bool = True,
                sync_inputs: bool = True):
        self._check_slot(slot)
        # Codegen disables this only for phase inputs proven not to be a
        # pending async tensor. Consumers of dispatch/combine pending buffers
        # keep the exact wait-before-detach behavior.
        if sync_inputs:
            inputs = self._sync_inputs(inputs)
        if not requires_grad:
            with torch.no_grad():
                return subgraph(*inputs)
        if self._states[slot] is not None:
            raise RuntimeError(
                f'phase execution slot {slot} already holds a forward state; '
                'the prior phase instance must run backward or be cleared first'
            )

        spec = self._specs[slot]
        if spec is None or not spec.matches(inputs):
            spec = PhaseSpec.from_inputs(inputs)
            self._specs[slot] = spec

        detached = list(inputs)
        for alias_group in spec.alias_groups:
            source = inputs[alias_group[0]]
            dtensor = source.detach().requires_grad_()
            for index in alias_group:
                detached[index] = dtensor
        outputs = subgraph(*detached)
        self._states[slot] = _PhaseState(tuple(detached[index] for index in spec.grad_positions))
        return outputs

    def backward(self, slot: int, output_tensors: List[torch.Tensor],
                 output_tensor_grads: List[Optional[torch.Tensor]]):
        self._check_slot(slot)
        output_tensor_grads = list(self._sync_inputs(tuple(output_tensor_grads)))
        state = self._states[slot]
        if state is None:
            raise RuntimeError(f'no pending forward state for phase execution slot {slot}')
        # Clear before autograd to match Executor's popleft-before-backward
        # lifecycle on an exception as well as on normal completion.
        self._states[slot] = None
        if len(output_tensors) == 0:
            return None

        dtensors = state.detached_inputs
        input_tensors = []
        for tensor in dtensors:
            if torch.is_tensor(tensor) and tensor.requires_grad:
                tensor.retain_grad()
                input_tensors.append(tensor)

        visited = set()
        dedup_outputs = []
        dedup_grads = []
        for tensor, grad in zip(output_tensors, output_tensor_grads):
            pair = (id(tensor), id(grad))
            if pair not in visited:
                visited.add(pair)
                dedup_outputs.append(tensor)
                dedup_grads.append(grad)

        if Executor._backward_pre_hook is not None:
            input_tensors, dedup_outputs, dedup_grads = Executor._backward_pre_hook(
                input_tensors, dedup_outputs, dedup_grads
            )
        torch.autograd.backward(dedup_outputs, grad_tensors=dedup_grads)
        grads = tuple(tensor.grad for tensor in input_tensors)
        assert all(grad is not None for grad in grads), 'RuntimeError: got gradient None'
        if len(grads) == 0:
            return None
        if len(grads) == 1:
            return grads[0]
        return grads

    def clear(self) -> None:
        self._states[:] = [None] * self.slot_count

    def check_clear(self) -> None:
        occupied = [index for index, state in enumerate(self._states) if state is not None]
        if occupied:
            raise AssertionError(f'phase executor has pending forward state in slots {occupied}')


def phase_fexecute(phase_executor: PhaseExecutor, slot: int, subgraph: Callable,
                   *inputs: Any, requires_grad: bool = True, sync_inputs: bool = True):
    return phase_executor.forward(
        slot, subgraph, *inputs, requires_grad=requires_grad, sync_inputs=sync_inputs
    )


def phase_backward(phase_executor: PhaseExecutor, slot: int,
                   output_tensors: List[torch.Tensor],
                   output_tensor_grads: List[Optional[torch.Tensor]]):
    return phase_executor.backward(slot, output_tensors, output_tensor_grads)


class Executor:

    # We consider each segment as an isolated graph. By
    # executing the forward of graph, the input tensors will be detached
    # from previous graph and saved for backward.
    # Each graph has its name, and multiple call for the graph will append
    # (instant id -> detached) input tensor pairs for FIFO backward reference.
    _detach: Dict[str, Deque[TensorPairs]] = dict()
    # Weight-backward states follow the same per-segment FIFO order as `_detach`.
    _weight_backward_states: Dict[str, Deque[_WeightBackwardState]] = dict()
    _backward_pre_hook: Optional[Callable] = None

    @staticmethod
    def fexecute(name: str, subgraph: Callable, *input_tensors: Tuple[Any], requires_grad=True):
        """
        forward the sub-graph.
        """
        input_tensors = Executor.sync_tensors(input_tensors)

        if not requires_grad:
            with torch.no_grad():
                outputs = subgraph(*input_tensors)
            return outputs

        # everytime forward a segment, detach the tensor from previous graph
        mapping: Dict[int, torch.Tensor] = dict()
        for itensor in input_tensors:
            if torch.is_tensor(itensor) and itensor.requires_grad:
                mapping[id(itensor)] = itensor.detach().requires_grad_()
        input_dtensors = tuple(mapping[id(t)] if id(t) in mapping else t for t in input_tensors)

        saved_pairs = [(id(itensor), dtensor) for itensor, dtensor in zip(input_tensors, input_dtensors)]
        Executor._detach.setdefault(name, deque()).append(saved_pairs)

        outputs = subgraph(*input_dtensors)
        return outputs

    @staticmethod
    def aexecute(subgraph: Callable, *input_tensors: Tuple[Any], requires_grad=True):
        """
        execute adapter
        """
        if not requires_grad:
            with torch.no_grad():
                outputs = subgraph(*input_tensors)
        else:
            outputs = subgraph(*input_tensors)
            if isinstance(outputs, tuple):
                outputs = (t.requires_grad_() if torch.is_tensor(t) and t.dtype in _ALLOW_GRAD_DTYPES else t for t in outputs)
            elif torch.is_tensor(outputs) and outputs.dtype in _ALLOW_GRAD_DTYPES:
                outputs = outputs.requires_grad_()
        return outputs

    @staticmethod
    def backward(name: str,
                 input_tensors: List[torch.Tensor],
                 output_tensors: List[torch.Tensor],
                 output_tensor_grads: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        """
        Backward Procedure.

        @param input_tensors List[torch.Tensor]
            tensors that their gradient need to be computed, including parameters.
            Correspoinding forward input tensors.

        @param output_tensors List[torch.Tensor]
            tensors that start for gradient backward computation.
            Corresponding to forward output tensors.

        @param output_tensor_grads List[torch.Tensor]:
            gradient tensors corresponding to output_tensors.

        @return gradients List[torch.Tensor]:
            gradient tensors corresponding to input_tensors.
        """
        output_tensor_grads = Executor.sync_tensors(output_tensor_grads)

        saved_pairs = Executor._detach[name].popleft()
        tensor_ids: List[int] = [pair[0] for pair in saved_pairs]
        dtensors: List[torch.Tensor] = [pair[1] for pair in saved_pairs]
        for t in input_tensors:
            if id(t) not in tensor_ids:
                import traceback
                _logger.warning(
                    f"rank {torch.distributed.get_rank()}: input {name} doesn't match. "
                    f"Make sure in scheduling, earlier forward perform earlier backward. "
                    f"Remain {len(Executor._detach[name])} segments.\n"
                    f"{''.join(traceback.format_stack())}"
                )

        if len(output_tensors) == 0: return None

        input_tensors = []
        for t in dtensors:
            if torch.is_tensor(t) and t.requires_grad:
                t.retain_grad()
                input_tensors.append(t)

        visited = set()
        dedup_output_tensors = []
        dedup_output_tensor_grads = []
        for t, g in zip(output_tensors, output_tensor_grads):
            # filter out duplicated output tensor and its grad.
            pair = (id(t), id(g))
            if pair not in visited:
                visited.add(pair)
                dedup_output_tensors.append(t)
                dedup_output_tensor_grads.append(g)

        # apply hook before backward
        if Executor._backward_pre_hook is not None:
            input_tensors, dedup_output_tensors, dedup_output_tensor_grads = \
                Executor._backward_pre_hook(
                    input_tensors,
                    dedup_output_tensors,
                    dedup_output_tensor_grads
                )

        torch.autograd.backward(
            dedup_output_tensors,
            grad_tensors=dedup_output_tensor_grads,
        )
        grads = tuple(t.grad for t in input_tensors)
        assert all(grad is not None for grad in grads), "RuntimeError: got gradient None"

        if    len(grads) == 0: return None
        elif  len(grads) == 1: return grads[0]
        else: return grads

    @staticmethod
    def backward_input(
        name: str,
        input_tensors: List[torch.Tensor],
        output_tensors: List[torch.Tensor],
        output_tensor_grads: List[Optional[torch.Tensor]],
        weights: Iterable[torch.nn.Parameter],
    ) -> Any:
        """Compute input gradients and defer weight gradients.

        ``backward_weight`` must later be called for the same segment name and
        in the same order as ``backward_input``.
        """
        output_tensor_grads = Executor.sync_tensors(output_tensor_grads)
        weights = tuple(weights)

        saved_pairs = Executor._detach[name].popleft()
        tensor_ids: List[int] = [pair[0] for pair in saved_pairs]
        dtensors: List[torch.Tensor] = [pair[1] for pair in saved_pairs]
        for tensor in input_tensors:
            if id(tensor) not in tensor_ids:
                import traceback
                _logger.warning(
                    f"rank {torch.distributed.get_rank()}: input {name} doesn't match. "
                    f"Make sure in scheduling, earlier forward perform earlier backward. "
                    f"Remain {len(Executor._detach[name])} segments.\n"
                    f"{''.join(traceback.format_stack())}"
                )

        if len(output_tensors) == 0:
            Executor._weight_backward_states.setdefault(name, deque()).append(
                _WeightBackwardState(param_groups=[])
            )
            return None

        input_tensors = [
            tensor for tensor in dtensors
            if torch.is_tensor(tensor) and tensor.requires_grad
        ]

        visited = set()
        dedup_output_tensors = []
        dedup_output_tensor_grads = []
        for tensor, grad in zip(output_tensors, output_tensor_grads):
            pair = (id(tensor), id(grad))
            if pair not in visited:
                visited.add(pair)
                dedup_output_tensors.append(tensor)
                dedup_output_tensor_grads.append(grad)

        if Executor._backward_pre_hook is not None:
            input_tensors, dedup_output_tensors, dedup_output_tensor_grads = \
                Executor._backward_pre_hook(
                    input_tensors,
                    dedup_output_tensors,
                    dedup_output_tensor_grads,
                )

        if not input_tensors:
            Executor._weight_backward_states.setdefault(name, deque()).append(
                _WeightBackwardState(
                    output_tensors=tuple(dedup_output_tensors),
                    output_tensor_grads=tuple(dedup_output_tensor_grads),
                )
            )
            return None

        # PyTorch's helper detaches stage outputs in-place. A view cannot be
        # detached in-place, so use a differentiable clone for that case.
        stage_outputs = [
            tensor.clone() if tensor._is_view() else tensor
            for tensor in dedup_output_tensors
        ]
        grads, param_groups = stage_backward_input(
            stage_outputs,
            dedup_output_tensor_grads,
            input_tensors,
            iter(weights),
        )
        Executor._weight_backward_states.setdefault(name, deque()).append(
            _WeightBackwardState(param_groups=param_groups)
        )

        assert all(grad is not None for grad in grads), "RuntimeError: got gradient None"
        if len(grads) == 0: return None
        elif len(grads) == 1: return grads[0]
        else: return grads

    @staticmethod
    def backward_weight(
        name: str,
        weights: Iterable[torch.nn.Parameter],
    ) -> None:
        """Compute weight gradients deferred by ``backward_input``."""
        weights = tuple(weights)
        states = Executor._weight_backward_states.get(name)
        if not states:
            raise RuntimeError(f"No pending weight backward for segment {name}.")
        state = states.popleft()

        if state.param_groups is not None:
            if weights and state.param_groups:
                stage_backward_weight(weights, state.param_groups)
            return

        if weights:
            # This branch is only taken when the segment has no grad-requiring
            # inputs, so a plain backward reaches only the weight leaves and does
            # not recompute input gradients. Running through each weight's
            # AccumulateGrad node makes reducer hooks fire (they move `param.grad`
            # into the reducer's buffer). `inputs=weights` would fire the hooks
            # too, but it is unnecessary here and only `torch.autograd.grad` /
            # manual `weight.grad = dw` would bypass AccumulateGrad.
            torch.autograd.backward(
                state.output_tensors,
                grad_tensors=state.output_tensor_grads,
            )

    @staticmethod
    def sync_tensors(tensors: List[Any]) -> List[Any]:
        """
        Wait until the finish of synchronized tensors.

        Most small phase boundaries have no outstanding async work.  Avoid a
        singleton lookup plus a full work-dictionary completion scan in that
        common case; when work exists this preserves the exact prior drain and
        per-tensor FIFO wait behavior.
        """
        handler = AsyncCommHandler()
        if handler.has_pending():
            handler.drain_all_completed()
        return [handler.wait(t) if torch.is_tensor(t) else t for t in tensors]


    @staticmethod
    def register_backward_pre_hook(hook: Optional[Callable]):
        """Register a backward hook for the right before the backward executor.

        The backward hook will be called with the following arguments:
            hook(input_tensors, output_tensors, output_tensor_grads) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]

        The backward hook mainly serves for the scenarios like loss scaling.

        Notes:
            Users can only register one backward pre_hook. If there was a hook
            registered before, it will be overwritten.

        Args:
            hook (Callable or None): the backward hook to be registered. The hook takes
                input_tensors (List[torch.Tensor]),
                output_tensors (List[torch.Tensor]),
                output_tensor_grads (List[torch.Tensor]) as inputs and returns the
                same format of updated tensors.
        """
        Executor._backward_pre_hook = hook

    @staticmethod
    def clear():
        Executor._detach = dict()
        Executor._weight_backward_states = dict()
        Executor._backward_pre_hook = None

    @staticmethod
    def check_clear():
        for name, npairs in Executor._detach.items():
            assert len(npairs) == 0, \
                f"Fine remaining segment needs backward: {name}, remaining times: {len(npairs)}"
        for name, states in Executor._weight_backward_states.items():
            assert len(states) == 0, \
                f"Fine remaining segment needs weight backward: {name}, remaining times: {len(states)}"


fexecute = Executor.fexecute
aexecute = Executor.aexecute
backward = Executor.backward
backward_input = Executor.backward_input
backward_weight = Executor.backward_weight
sync_tensors = Executor.sync_tensors


# register checking for normal exit
atexit.register(Executor.check_clear)
atexit.register(AsyncCommHandler().check_clear)
