#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

r"""
Executor for runtime
"""
import atexit
import weakref
from copy import copy
import sys

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Tuple, Any, Callable, List, Dict, Iterable, Optional, Union, Generator, Set, TypedDict
import torch
import logging
from torch.autograd.graph import Node
from torch.distributed import Work

from ._patch_torch import stage_backward_input, stage_backward_weight


_logger = logging.getLogger(__name__)

_ALLOW_GRAD_DTYPES = (torch.double, torch.float32, torch.float16, torch.bfloat16)
# Number of holders of a tensor's TensorImpl, exposed since torch 2.4.0. On the
# older versions this package still supports (>= 2.0) it is None and
# `release_storage` is a no-op, i.e. no storage is freed but nothing breaks.
_use_count = torch.Tensor._use_count if torch.__version__ >= (2, 4, 0) else None


def _sole_owner_refcounts() -> Tuple[int, int]:
    """
    Python references `release_storage` sees for a stage output held only by its
    `_train_step` local, and for that output's view base, which outlives its last
    Python name because its TensorImpl preserves the Python object. Measured on a
    probe shaped like the generated step rather than hardcoded, since the exact
    numbers are CPython details (e.g. 3.14 stops counting borrowed stack references).
    """
    def _fexecute(data):
        # the output is a view over an internal result, the shape ZeRO-3's backward
        # hook produces; the base loses its only Python name when the segment returns
        return (data * 2.0).view(2)

    def _send(tensor):
        # a sync send keeps no reference; an async one holds a `detach()` alias,
        # which is another Python object and does not count either
        pass

    def _release_storage(*tensors):
        # the loop of `Executor.release_storage`, measuring instead of releasing
        for t in tensors:
            base = t._base
            return sys.getrefcount(t), sys.getrefcount(base)

    # the generated step; under inference mode (if this module is first imported
    # inside one) the probe would be an inference tensor whose view has no `_base`
    with torch.inference_mode(False):
        x = _fexecute(torch.ones(2))
        _send(x)
        return _release_storage(x)


_SOLE_OWNER_REFS, _SOLE_OWNER_BASE_REFS = _sole_owner_refcounts()
# Set by `_check_release_storage` at import; False disables `release_storage`.
_release_storage_enabled = _use_count is not None
# Attribute `release_storage` sets on a tensor it released.
# A backward pre-hook must hand such a tensor back as is,
# see `Executor._apply_backward_pre_hook`.
_RELEASED_TAG = '_nnscaler_storage_released'


def debug_id(tensors, msg: str, rank: int):
    if torch.distributed.get_rank() == rank:
        if torch.is_tensor(tensors):
            print(f'[{torch.distributed.get_rank()}] {msg}: [{id(tensors)}]')
        else:
            print(f'[{torch.distributed.get_rank()}] {msg}: {[id(t) for t in tensors]}')


class _AsyncCommHandler:
    def __init__(self) -> None:
        self._works: Dict[torch.Tensor, List[Work]] = {}
        self._callbacks: Dict[torch.Tensor, Callable] = {}
        self._send_holds: List[Tuple[torch.Tensor, Work]] = []

    def wait(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Wait until the finish of the communication

        @param tensor torch.Tensor
        @return tensor torch.Tensor
        """
        works = self._works.pop(tensor, None)
        if works is not None:
            self._finish(tensor, works)
        return tensor

    def _finish(self, tensor: torch.Tensor, works: List[Work]):
        """Wait for `works` and run the callback that completes `tensor` in place."""
        for work in works:
            work.wait()
        callback = self._callbacks.pop(tensor)
        if callback is not None:
            assembled = callback(tensor)
            assert assembled is tensor, "async callbacks must assemble the output into the registered tensor"

    def submit(self, tensor: torch.Tensor, works: List[Work], callback: Optional[Callable] = None):
        """
        Submit an async communication.

        `tensor` is both the handle consumers wait on and the final output. A
        `callback(tensor)` may assemble the result into it after the works
        finish, but must not replace it: consumers keep referring to `tensor`,
        and a second wait on an already resolved tensor is a no-op that must
        still see the complete data.
        """
        self._works[tensor] = works
        self._callbacks[tensor] = callback

    def hold_send(self, tensor: torch.Tensor, work: Work):
        # Hold an alias instead of the tensor itself: `Executor.release_storage`
        # may swap the sender's output to a placeholder right after the send is
        # issued, and the alias keeps the real buffer alive until the work completes.
        self._send_holds.append((tensor.detach(), work))

    def drain_sends_completed(self):
        running: list[tuple[torch.Tensor, Work]] = []
        for tensor, work in self._send_holds:
            if work.is_completed():
                work.wait()
            else:
                running.append((tensor, work))
        self._send_holds[:] = running

    def drain_sends(self):
        for _, work in self._send_holds:
            work.wait()
        self._send_holds.clear()

    def check_clear(self):
        assert len(self._works) == 0 and len(self._callbacks) == 0 and len(self._send_holds) == 0, \
            f"AsyncCommHandler not cleared: works={len(self._works)}, callbacks={len(self._callbacks)}, send_holds={len(self._send_holds)}"


_instance: Optional[_AsyncCommHandler] = None

def AsyncCommHandler() -> _AsyncCommHandler:
    global _instance
    if _instance is None:
        _instance = _AsyncCommHandler()
    return _instance


TensorPairs = List[Tuple[int, torch.Tensor]]


class _FBWParamGroupRequired(TypedDict):
    params: Set[Node]
    intermediates: Union[Set[Node], List[Node]]


class FBWParamGroup(_FBWParamGroupRequired, total=False):
    """Mutable graph-boundary state shared by B and W.

    Graph analysis creates the required ``params`` and ``intermediates`` keys,
    with intermediates initially stored as a set. B normalizes that set to a
    list, then its Node pre-hooks lazily add ``grads``. Each list slot matches
    one intermediate; each tuple slot matches that Node's output index. Slots
    remain ``None`` until the corresponding pre-hook runs.
    """

    grads: List[Optional[Tuple[Optional[torch.Tensor], ...]]]


@dataclass
class _WeightBackwardState:
    param_groups: Optional[List[FBWParamGroup]] = None
    output_tensors: Optional[Tuple[torch.Tensor, ...]] = None
    output_tensor_grads: Optional[Tuple[Optional[torch.Tensor], ...]] = None


@dataclass
class _InputGradCallback:
    # Holding the tensor keeps its id from being reused before the callback runs.
    tensor: torch.Tensor
    callback: Callable[[Optional[torch.Tensor]], None]
    # Sum of the gradients from consumer segments that already ran backward;
    # stays None while no consumer has contributed a gradient.
    grad: Optional[torch.Tensor] = None


class Executor:

    # We consider each segment as an isolated graph. By
    # executing the forward of graph, the input tensors will be detached
    # from previous graph and saved for backward.
    # Each graph has its name, and multiple call for the graph will append
    # (instant id -> detached) input tensor pairs for backward reference.
    _detach: Dict[str, List[TensorPairs]] = dict()
    # Weight-backward states follow the same per-segment FIFO order as `_detach`.
    _weight_backward_states: Dict[str, List[_WeightBackwardState]] = dict()
    _backward_pre_hook: Optional[Callable] = None

    _input_grad_callbacks: Dict[int, _InputGradCallback] = dict()
    # Tensors whose callback already fired, so that a later consumer cannot
    # silently lose its gradient. Weak references neither keep the outer graph
    # alive nor match a new tensor that reuses the id of a released one.
    # A mark lives as long as its tensor, independent of train_step:
    # it is dropped when the tensor is released, when the tensor is registered
    # again for a new round, or by clear().
    _delivered_input_grads: Dict[int, weakref.ref] = dict()
    # ids of the saved pairs of every segment whose boundary grads Executor
    # handled (retained grads written back, callbacks dispatched) while a
    # custom_fbw replacement runs, so the wrapper can tell whether the
    # replacement already went through Executor for its own segment.
    # A set, scoped to one replacement call, stays correct when the replacement
    # or a callback runs further segments; it is None outside such a call.
    # ids avoid keeping the detached inputs alive.
    _handled_pairs_ids: Optional[Set[int]] = None

    @staticmethod
    def register_input_grad_callback(
        tensor: torch.Tensor,
        callback: Callable[[Optional[torch.Tensor]], None],
    ) -> None:
        """Call ``callback`` once with the gradient of ``tensor`` from the segments it feeds.

        ``fexecute`` detaches segment inputs, so autograd never reaches ``tensor``.
        The callback lets an outer eager module resume its own backward as soon as the gradient is ready,
        for example once per micro-batch inside a pipeline ``train_step``.

        The gradients of all consumer segments are summed.
        The callback runs right after the backward (or ``backward_input``) of the last consumer
        whose forward has already run.
        So every consumer must run its forward before the first of them runs backward.
        A segment that consumes ``tensor`` after the callback fired raises,
        e.g. when one tensor object is fed to several micro-batches;
        You should pass each micro-batch its own tensor, or register again to start a new round.

        Unlike an ordinary hook, a registration is one-shot:

        - It fires at most once, in whichever backward completes its last consumer,
          including a backward that another callback runs through ``Executor``;
          any later backward skips it.
        - A pending registration cannot be replaced: registering ``tensor`` again raises.
          Registering it again after the callback fired starts a new round,
          which collects only the consumers whose forward runs afterwards.

        Register the callback before the ``train_step`` of the module that consumes ``tensor``,
        with no other end-to-end module's ``train_step`` in between, or while that ``train_step`` runs.
        ``ParallelModule.train_step`` raises if any callback is still pending when it returns,
        since it cannot tell which module a callback was meant for.

        A consumer without gradient roots (its backward gets no output tensors) contributes no gradient.
        If no consumer contributes, the callback receives ``None``, PyTorch's convention for "no gradient".
        A zero tensor would not be equivalent:
        optimizers such as Adam still update parameters with a zero gradient through momentum and weight decay,
        while they skip parameters whose ``grad`` is ``None``.
        Whether a segment has gradient roots is fixed in the generated schedule,
        so the callback receives ``None`` for every micro-batch of the step or for none of them.

        On ``None`` the caller should skip the outer backward.
        The reducers of the outer module then see no gradient in this step:

        - sync reducer: the parameters get zero gradients;
        - async reducer: ``sync_grads`` raises because the reduction was never triggered;
        - either reducer with ``ComputeConfig.reducer_none_grad=True``:
          the parameters keep ``grad=None`` and the optimizer skips them.

        The callback owns the outer propagation, including retained grads:
        ``Executor`` does not also write the retained grad of ``tensor``.
        ``tensor.grad`` after its consumers ran backward:

        =================  ===========  ==============================================================
        ``retain_grad()``  callback     ``tensor.grad``
        =================  ===========  ==============================================================
        no                 no           ``None``: segments detach their inputs.
        no                 registered   Passed to the callback, which decides.
                                        ``torch.autograd.backward(tensor, grad)`` in the callback
                                        sets it only if ``tensor`` is a leaf.
        yes                no           Written by ``Executor``, summed over consumer segments.
        yes                registered   Passed to the callback, which decides.
                                        ``torch.autograd.backward(tensor, grad)`` in the callback
                                        sets it (to the sum over consumer segments).
        =================  ===========  ==============================================================

        The callback runs synchronously inside the generated schedule,
        on the current stream and under the runtime flags the schedule has set
        (e.g. ``RuntimeFlag.skip_reducer``).
        Only the ranks of the pipeline stage that consumes ``tensor`` reach this point together,
        so collectives in the callback must stay within that stage.
        A blocking collective that also needs ranks of other stages waits for ranks
        that are busy with their own part of the schedule, and can deadlock.
        """
        if not torch.is_tensor(tensor) or not tensor.requires_grad:
            raise ValueError('input gradient callbacks require a grad-requiring tensor')
        tensor_id = id(tensor)
        if tensor_id in Executor._input_grad_callbacks:
            raise RuntimeError('an input gradient callback is already registered')
        Executor._delivered_input_grads.pop(tensor_id, None)
        Executor._input_grad_callbacks[tensor_id] = _InputGradCallback(tensor, callback)

    @staticmethod
    def _mark_input_grad_delivered(tensor: torch.Tensor) -> None:
        tensor_id = id(tensor)
        # Bind the dict marked now: forget may run after clear() replaced it,
        # or at interpreter shutdown when module globals are already torn down.
        delivered = Executor._delivered_input_grads

        def forget(ref):
            # Defensive: only drop the entry this reference still owns.
            if delivered.get(tensor_id) is ref:
                del delivered[tensor_id]

        delivered[tensor_id] = weakref.ref(tensor, forget)

    @staticmethod
    def _check_input_grad_not_delivered(name: str, tensor: torch.Tensor) -> None:
        ref = Executor._delivered_input_grads.get(id(tensor))
        if ref is not None and ref() is tensor:
            raise RuntimeError(
                f'segment {name} consumes a tensor whose input gradient callback already fired; '
                f'its gradient would be lost. Every consumer must run forward before the first '
                f'consumer runs backward: pass each micro-batch its own tensor, '
                f'or register the callback again.'
            )

    @staticmethod
    def check_input_grad_callbacks_fired() -> None:
        """Raise if a registered input gradient callback has not fired.

        A pending callback never fires once its consumers have run,
        and would keep its tensor and the outer autograd graph alive across steps.
        The callbacks are dropped before raising so that the graph is released.
        """
        pending = len(Executor._input_grad_callbacks)
        if pending:
            Executor._input_grad_callbacks = dict()
            raise RuntimeError(
                f'{pending} input gradient callback(s) did not fire: '
                f'their tensors were not consumed by a segment requiring gradients. '
                f'Register a callback only on the ranks whose pipeline stage consumes the tensor, '
                f'and pass the tensor itself (not inside a container) as a module input.'
            )

    @staticmethod
    def _run_input_grad_callbacks(
        saved_pairs: TensorPairs,
        grad_by_dtensor_id: Optional[Dict[int, torch.Tensor]],
    ) -> None:
        """Accumulate registered input grads of one segment and fire completed callbacks.

        ``grad_by_dtensor_id`` is None when the segment has no gradient roots,
        so it contributes no gradient.
        """
        # Executor writes retained grads right before dispatching callbacks,
        # so this marks the boundary grads of the segment as handled.
        if Executor._handled_pairs_ids is not None:
            Executor._handled_pairs_ids.add(id(saved_pairs))
        # Accumulate every registered input of the segment before firing any
        # callback: a callback that runs another segment's backward through
        # Executor may otherwise fire a callback this segment still owes a share to.
        # A repeated input shares one detached tensor whose grad already sums
        # all its uses; dict() collapses the repeats so it is counted once.
        entries = []
        try:
            for tensor_id, dtensor in dict(saved_pairs).items():
                entry = Executor._input_grad_callbacks.get(tensor_id)
                if entry is None:
                    continue
                entries.append((tensor_id, entry))
                if grad_by_dtensor_id is not None:
                    grad = grad_by_dtensor_id.get(id(dtensor))
                    if grad is None:
                        raise RuntimeError('segment did not produce a registered input gradient')
                    grad = grad.detach()
                    entry.grad = grad if entry.grad is None else entry.grad + grad
            for tensor_id, entry in entries:
                # A registration fires at most once: an earlier callback may have run
                # another segment's backward that already fired it, after which the
                # tensor may even be registered again for a new round.
                if Executor._input_grad_callbacks.get(tensor_id) is not entry:
                    continue
                # Another segment that consumed the tensor still owes its share.
                if any(
                    tid == tensor_id
                    for queue in Executor._detach.values()
                    for pairs in queue
                    for tid, _ in pairs
                ):
                    continue
                del Executor._input_grad_callbacks[tensor_id]
                Executor._mark_input_grad_delivered(entry.tensor)
                entry.callback(entry.grad)
        except BaseException:
            # The segment has left _detach, so its remaining registrations would
            # never fire, yet keep their tensors and outer graphs alive if the
            # caller recovers. Drop only those this call collected and that are
            # still current; a callback may have registered a new round meanwhile.
            for tensor_id, entry in entries:
                if Executor._input_grad_callbacks.get(tensor_id) is entry:
                    del Executor._input_grad_callbacks[tensor_id]
            raise

    @staticmethod
    def _retained_inputs(input_tensors: Iterable[Any]) -> Dict[int, torch.Tensor]:
        """Caller tensors that asked to keep their gradient, keyed by id.

        ``fexecute`` detaches segment inputs, so autograd never reaches the
        caller's tensors.  Honor ``retain_grad()`` on them for callers that
        continue the backward in an outer eager graph (a nested end2end module).
        """
        return {
            id(t): t for t in input_tensors
            if torch.is_tensor(t) and t.retains_grad
        }

    @staticmethod
    def _restore_retained_grads(
        retained_inputs: Dict[int, torch.Tensor],
        saved_pairs: TensorPairs,
        grad_by_dtensor: Dict[int, torch.Tensor],
    ) -> None:
        # An input passed several times maps to one detached tensor whose grad
        # is already the sum over all uses; dict() collapses the repeats so the
        # gradient is copied once.  Accumulate across segments like retain_grad.
        # A registered input-gradient callback owns the outer propagation,
        # including retained grads, so writing them here would count twice.
        for tid, dtensor in dict(saved_pairs).items():
            tensor = retained_inputs.get(tid)
            if tensor is not None and tid not in Executor._input_grad_callbacks:
                grad = grad_by_dtensor[id(dtensor)]
                tensor.grad = grad if tensor.grad is None else tensor.grad + grad

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
                # Sharing a tensor across segments is fine while their forwards all run
                # before its callback fires. A segment consuming it afterwards, e.g. the same
                # tensor fed to a later micro-batch, would silently lose its gradient.
                Executor._check_input_grad_not_delivered(name, itensor)
                mapping[id(itensor)] = itensor.detach().requires_grad_()
        input_dtensors = tuple(mapping[id(t)] if id(t) in mapping else t for t in input_tensors)

        saved_pairs = [(id(itensor), dtensor) for itensor, dtensor in zip(input_tensors, input_dtensors)]
        Executor._detach.setdefault(name, []).append(saved_pairs)

        outputs = subgraph(*input_dtensors)
        return outputs

    @staticmethod
    def aexecute(subgraph: Callable, *input_tensors: Tuple[Any], requires_grad=True):
        """
        execute adapter
        """
        input_tensors = Executor.sync_tensors(input_tensors)
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

        saved_pairs = Executor._detach[name].pop(0)
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

        if len(output_tensors) == 0:
            Executor._run_input_grad_callbacks(saved_pairs, None)
            return None

        retained_inputs = Executor._retained_inputs(input_tensors)
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
        input_tensors, dedup_output_tensors, dedup_output_tensor_grads = \
            Executor._apply_backward_pre_hook(input_tensors, dedup_output_tensors, dedup_output_tensor_grads)

        torch.autograd.backward(
            dedup_output_tensors,
            grad_tensors=dedup_output_tensor_grads,
        )
        grads = tuple(t.grad for t in input_tensors)
        assert all(grad is not None for grad in grads), "RuntimeError: got gradient None"

        grad_by_dtensor = {id(t): t.grad for t in input_tensors}
        Executor._restore_retained_grads(retained_inputs, saved_pairs, grad_by_dtensor)
        Executor._run_input_grad_callbacks(saved_pairs, grad_by_dtensor)

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

        saved_pairs = Executor._detach[name].pop(0)
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
            # Even without output tensors (an uncommon case), record a paired
            # no-op state for the matching backward_weight call to consume.
            Executor._weight_backward_states.setdefault(name, []).append(
                _WeightBackwardState(param_groups=[])
            )
            Executor._run_input_grad_callbacks(saved_pairs, None)
            return None

        retained_inputs = Executor._retained_inputs(input_tensors)
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

        input_tensors, dedup_output_tensors, dedup_output_tensor_grads = \
            Executor._apply_backward_pre_hook(input_tensors, dedup_output_tensors, dedup_output_tensor_grads)

        if not input_tensors:
            Executor._weight_backward_states.setdefault(name, []).append(
                _WeightBackwardState(
                    output_tensors=tuple(dedup_output_tensors),
                    output_tensor_grads=tuple(dedup_output_tensor_grads),
                )
            )
            return None

        # stage_backward_input miscomputes gradients when an input is listed more
        # than once; deduplicate and return one gradient per occurrence instead.
        unique_inputs = list({id(tensor): tensor for tensor in input_tensors}.values())
        unique_grads, param_groups = stage_backward_input(
            dedup_output_tensors,
            dedup_output_tensor_grads,
            unique_inputs,
            iter(weights),
        )
        Executor._weight_backward_states.setdefault(name, []).append(
            _WeightBackwardState(param_groups=param_groups)
        )

        assert all(grad is not None for grad in unique_grads), "RuntimeError: got gradient None"
        grad_by_dtensor = dict(zip(map(id, unique_inputs), unique_grads))
        grads = tuple(grad_by_dtensor[id(tensor)] for tensor in input_tensors)
        Executor._restore_retained_grads(retained_inputs, saved_pairs, grad_by_dtensor)
        Executor._run_input_grad_callbacks(saved_pairs, grad_by_dtensor)
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
        state = states.pop(0)

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
    def release_storage(*tensors: torch.Tensor):
        """
        Free the storage of forward outputs that are only kept as roots for a
        later `Executor.backward`, e.g. a pipeline stage output after it was sent.
        The data is swapped for a stride-0 view of one NaN element: the tensor
        keeps its shape and grad_fn, so backward (and its grad shape check) works
        unchanged, while the real buffer is released. Real floating outputs get a
        float8 placeholder on purpose: autograd validates gradients against the
        metadata recorded when the graph was built, so backward through the root is
        unaffected, while hardly any kernel accepts float8 and it never promotes,
        so anything that still reads the tensor (a forward consumer, or an alias
        saved for backward) fails at that op instead of computing with garbage.

        Swapping `.data` rewrites the TensorImpl in place, so it is only safe when
        nothing else can observe that impl. Holders show up in one of two counts:

        * `_use_count`: C++ handles to the impl. Autograd saves a node's *input*
          (and the base of a view) as the original impl, but a node's *output*
          (e.g. tanh's result) as a detached shallow copy that merely shares the
          storage, because holding the original would form the cycle
          Node -> SavedVariable -> Variable.grad_fn -> Node that refcounting never
          frees. Such copies are not affected by the swap (and keep the buffer
          alive, so nothing is gained for them).
        * `sys.getrefcount`: Python references to the tensor object. Another name
          for the same object in the generated step (an op that returned its input,
          a sibling output), a `ctx.out = out` attribute, and any tensor saved under
          `saved_tensors_hooks` (ZeRO-3's `save_params_hooks`, CPU offloading) are
          held this way: a pack hook receives the tensor's own Python object and
          returning it stores a Python reference, not an impl handle, so the first
          count stays 1 for them.

        A tensor is released only when both counts show no holder but the caller's
        local; a view is released together with its root base when that base is
        held by nothing but its own (impl-preserved) Python object and this view.
        This covers outputs aliasing an internal tensor (ZeRO-3's
        `insert_backward_hook` views) without knowing the op. The expected counts
        are verified on the running interpreter at import, see
        `_check_release_storage`.
        """
        if not _release_storage_enabled:
            return
        for t in tensors:
            if not (torch.is_tensor(t) and t.grad_fn is not None and t.layout == torch.strided):
                continue
            if _use_count(t) != 1 or sys.getrefcount(t) != _SOLE_OWNER_REFS:
                continue
            base = t._base
            if base is None:
                targets = (t,)
            elif _use_count(base) == 2 and sys.getrefcount(base) == _SOLE_OWNER_BASE_REFS:
                # the impl is held by the base's own Python object and this view only
                targets = (t, base)
            else:
                continue
            for x in targets:
                # float8 so that a stray reader fails loudly (see the docstring);
                # complex roots must keep their dtype for the backward grad check,
                # and NaN keeps the misuse visible where the dtype trick cannot apply
                # (complex, or an output that is float8 already)
                dtype = torch.float8_e4m3fn if x.dtype.is_floating_point else x.dtype
                x.data = torch.full((), float('nan'), dtype=dtype, device=x.device).expand(x.shape)
            setattr(t, _RELEASED_TAG, True)

    @staticmethod
    def _apply_backward_pre_hook(input_tensors, output_tensors, output_tensor_grads):
        """
        Run the registered backward pre-hook and check what it returns.

        The hook must keep the number of inputs, outputs and grads:
        outputs and grads pair up as backward roots, and the generated code unpacks one
        gradient per input, so a changed count would fail later with a confusing error
        or drop a gradient silently.
        It must also return every released output as the same object.
        Both are checked against a snapshot taken before the call,
        because a hook may write into the lists it receives (the built-in loss-scaling
        hook does), and then the lists passed in no longer show what was there.

        A tensor the hook builds on top of a released output (even a view) records
        the float8 placeholder dtype in its new autograd node,
        so backward from it would cast the incoming gradient to float8 and truncate it silently.
        Reading values through such a tensor sees NaN.
        A pipeline-internal output is already sent, so changing it can only change
        the gradient flowing back; the hook should change `output_tensor_grads` instead.
        """
        if Executor._backward_pre_hook is None:
            return input_tensors, output_tensors, output_tensor_grads
        counts_before = [len(input_tensors), len(output_tensors), len(output_tensor_grads)]
        # (position, tensor) of every released output, as the hook will see them
        released = [(idx, t) for idx, t in enumerate(output_tensors) if getattr(t, _RELEASED_TAG, False)]
        new_inputs, new_outputs, new_grads = Executor._backward_pre_hook(
            input_tensors, output_tensors, output_tensor_grads)
        counts_after = [len(new_inputs), len(new_outputs), len(new_grads)]
        if counts_before != counts_after:
            raise RuntimeError(
                f'The backward pre-hook must keep the number of inputs, outputs and grads, '
                f'got {counts_before} -> {counts_after}.'
            )
        for idx, old in released:
            if new_outputs[idx] is not old:
                raise RuntimeError(
                    f'The backward pre-hook replaced output {idx}, whose storage was released after it '
                    f'was sent to the next pipeline stage. A hook must return released outputs unchanged '
                    f'and adjust output_tensor_grads instead, or disable '
                    f'ComputeConfig.pipeline_deallocate_outputs.'
                )
        return new_inputs, new_outputs, new_grads

    @staticmethod
    def sync_tensors(tensors: List[Any]) -> List[Any]:
        """
        Wait until the finish of synchornized tensors
        """
        # Only wait on this consumer's tensors; completed send buffers have no
        # stream dependency and can be released eagerly.
        AsyncCommHandler().drain_sends_completed()
        return [AsyncCommHandler().wait(t) if torch.is_tensor(t) else t for t in tensors]


    @staticmethod
    def register_backward_pre_hook(hook: Optional[Callable]):
        """Register a backward hook for the right before the backward executor.

        The backward hook will be called with the following arguments:
            hook(input_tensors, output_tensors, output_tensor_grads) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]

        The backward hook mainly serves for the scenarios like loss scaling.

        Notes:
            Users can only register one backward pre_hook. If there was a hook
            registered before, it will be overwritten.

            The hook must return as many inputs, outputs and grads as it received,
            otherwise backward raises a RuntimeError.

            With pipeline output deallocation (`ComputeConfig.pipeline_deallocate_outputs`),
            an output tensor other than the loss may already have had its storage
            released by `Executor.release_storage`.
            It then only keeps its shape and grad_fn, and its dtype is float8.
            The hook must return such a tensor unchanged (the same object),
            otherwise backward raises a RuntimeError;
            to change what flows back, change its entry in output_tensor_grads.
            The loss is a graph output and is never released,
            which is all the built-in loss-scaling hook touches.

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
        Executor._input_grad_callbacks = dict()
        Executor._delivered_input_grads = dict()
        Executor._handled_pairs_ids = None

    @staticmethod
    def check_clear():
        for name, npairs in Executor._detach.items():
            assert len(npairs) == 0, \
                f"Fine remaining segment needs backward: {name}, remaining times: {len(npairs)}"
        for name, states in Executor._weight_backward_states.items():
            assert len(states) == 0, \
                f"Fine remaining segment needs weight backward: {name}, remaining times: {len(states)}"
        assert len(Executor._input_grad_callbacks) == 0, \
            f"Input gradient callbacks remain: {len(Executor._input_grad_callbacks)}"


fexecute = Executor.fexecute
aexecute = Executor.aexecute
backward = Executor.backward
register_input_grad_callback = Executor.register_input_grad_callback
backward_input = Executor.backward_input
backward_weight = Executor.backward_weight
sync_tensors = Executor.sync_tensors
release_storage = Executor.release_storage


def _check_release_storage() -> bool:
    """
    Verify `_SOLE_OWNER_REFS`/`_SOLE_OWNER_BASE_REFS` end to end on this
    interpreter: a view over an internal base must be released together with the
    base, and a tensor with a second Python name must be left alone. On a
    mismatch, or if the probes raise (e.g. a torch build rejecting `.data` on a
    view), the feature is disabled instead of failing the import.
    """
    if _use_count is None:
        return False

    def segment(x):
        return (x * 2.0).view(2)

    try:
        # the probes need grad_fn even if this module is first imported under
        # no_grad or inference_mode; enable_grad alone does not leave inference mode
        with torch.inference_mode(False), torch.enable_grad():
            leaf = torch.ones(2, requires_grad=True)
            view = segment(leaf)
            release_storage(view)
            released = (view.untyped_storage().nbytes() == view.element_size()
                        and view._base.untyped_storage().nbytes() == view.element_size())
            shared = leaf * 2.0
            other_name = shared  # another Python name for the same tensor
            # can't release
            release_storage(shared)
            kept = shared.untyped_storage().nbytes() == 2 * shared.element_size()
    except Exception:
        _logger.warning('release_storage self-check raised', exc_info=True)
        return False
    return released and kept


_release_storage_enabled = _check_release_storage()
if _use_count is not None and not _release_storage_enabled:
    _logger.warning('release_storage self-check failed on this interpreter; stage outputs are not deallocated.')


@contextmanager
def _record_fbw_grads(indexed_weights, received):
    """Record which indexed parameters receive an autograd contribution."""
    handles = []

    def make_hook(index):
        def hook(grad):
            # Tensor hooks run when autograd reaches the parameter's
            # AccumulateGrad edge. They observe a contribution without relying
            # on `.grad`, which reducers may move into a separate buffer.
            if grad is not None:
                received.add(index)
            return grad
        return hook

    try:
        # Keep the original index from the generated `weights` tuple so error
        # messages still identify the right parameter after reachability
        # filtering.
        for index, weight in indexed_weights:
            handles.append(weight.register_hook(make_hook(index)))
        yield
    finally:
        # B and W install temporary hooks independently. Never leave diagnostic
        # hooks attached after either implementation returns or raises.
        for handle in handles:
            handle.remove()


def _checked_fbw(input_fn, weight_fn):
    """Wrap a custom FBW pair with best-effort parameter coverage checks."""
    # A segment can have several microbatches between B and W. Each B publishes
    # one state and the matching W consumes the oldest state for that segment.
    pending = {}

    def checked_input(name, input_tensors, output_tensors, output_tensor_grads, weights):
        weights = tuple(weights)
        # Walk the output-side autograd graph before B can release any of it.
        # AccumulateGrad nodes expose their leaf tensor through `variable`, so
        # collecting those object ids identifies structurally reachable weights.
        nodes = [tensor.grad_fn for tensor in output_tensors if tensor.requires_grad]
        reachable = {id(tensor) for tensor in output_tensors if tensor.requires_grad}
        visited = set()
        while nodes:
            node = nodes.pop()
            if node is None or node in visited:
                continue
            visited.add(node)
            variable = getattr(node, 'variable', None)
            if variable is not None:
                reachable.add(id(variable))
            nodes.extend(parent for parent, _ in node.next_functions)
        expected = tuple(
            (index, weight)
            for index, weight in enumerate(weights)
            if weight.requires_grad and id(weight) in reachable
        )
        # Contributions may arrive in either B or W. Share one set across the
        # paired calls and require every reachable parameter to appear at least
        # once; this intentionally cannot prove that all contributions arrived.
        received = set()
        with _record_fbw_grads(expected, received):
            result = input_fn(name, input_tensors, output_tensors, output_tensor_grads, weights)
        pending.setdefault(name, []).append((expected, received))
        return result

    def checked_weight(name, weights):
        queue = pending.get(name)
        if not queue:
            raise RuntimeError(f'FBW check: no pending B for segment {name!r}')
        # Preserve generated schedule order when several microbatches of the
        # same segment are pending.
        expected, received = queue.pop(0)
        with _record_fbw_grads(expected, received):
            result = weight_fn(name, weights)
        missing = [index for index, _ in expected if index not in received]
        if missing:
            raise RuntimeError(f'FBW check: segment {name!r}, reachable parameter indices {missing} received no gradient in B/W')
        return result

    return checked_input, checked_weight, pending


def _with_boundary_grads(input_fn):
    """Write retained grads and dispatch callbacks for a custom ``backward_input``.

    ``Executor.backward_input`` does both for the segment inputs;
    a replacement that bypasses it would otherwise silently drop them.
    """
    @wraps(input_fn)
    def backward_input_with_boundary_grads(name, input_tensors, output_tensors, output_tensor_grads, weights):
        # Custom implementations can consume _detach without calling Executor.
        # Save the boundary association before they pop the FIFO entry.
        queue = Executor._detach.get(name, ())
        saved_pairs = queue[0] if queue else ()
        retained_inputs = Executor._retained_inputs(input_tensors)
        outer_handled = Executor._handled_pairs_ids
        Executor._handled_pairs_ids = handled = set()
        try:
            result = input_fn(name, input_tensors, output_tensors, output_tensor_grads, weights)
        finally:
            Executor._handled_pairs_ids = outer_handled
            if outer_handled is not None:
                outer_handled |= handled
        # A custom backward_input that called Executor.backward or
        # Executor.backward_input for this segment has already handled it,
        # even if it ran further segments afterwards; doing it again would count twice.
        # ``saved_pairs`` stays alive here, so no other object can reuse its id.
        if id(saved_pairs) in handled:
            return result
        boundary_ids = {
            tid for tid, _ in saved_pairs
            if tid in retained_inputs or tid in Executor._input_grad_callbacks
        }
        if not boundary_ids:
            return result
        # Like Executor.backward_input, a segment without output tensors has no
        # gradient roots: retained grads stay as they are, and callbacks get no contribution.
        if len(output_tensors) == 0:
            Executor._run_input_grad_callbacks(saved_pairs, None)
            return result
        if result is None:
            raise RuntimeError('custom backward_input returned no gradient for a retained or callback input')
        inputs = [t for _, t in saved_pairs if torch.is_tensor(t) and t.requires_grad]
        grads = (result,) if torch.is_tensor(result) else result
        if len(inputs) != len(grads):
            raise RuntimeError('custom backward_input returned an unexpected number of input gradients')
        grad_by_dtensor_id = {id(t): grad for t, grad in zip(inputs, grads) if grad is not None}
        if any(id(dtensor) not in grad_by_dtensor_id for tid, dtensor in saved_pairs if tid in boundary_ids):
            raise RuntimeError('custom backward_input returned no gradient for a retained or callback input')
        # Same order as Executor.backward_input.
        Executor._restore_retained_grads(retained_inputs, saved_pairs, grad_by_dtensor_id)
        Executor._run_input_grad_callbacks(saved_pairs, grad_by_dtensor_id)
        return result

    return backward_input_with_boundary_grads


@contextmanager
def custom_fbw(
    backward_input_fn: Callable,
    backward_weight_fn: Callable,
    *,
    check_grad_coverage: bool = False,
) -> Generator[None, None, None]:
    """Temporarily replace the FBW functions used by generated schedules.

    Schedule code resolves the module-level ``backward_input`` and
    ``backward_weight`` attributes at runtime. Replacements must keep the same
    call signatures as the corresponding :class:`Executor` methods.
    When the replacement calls neither :meth:`Executor.backward_input` nor :meth:`Executor.backward` for the segment,
    the input gradients it returns are written to retained grads
    and passed to input-gradient callbacks (see :meth:`Executor.register_input_grad_callback`),
    as the default implementation does.
    A replacement must therefore return the input gradients of a segment with output tensors;
    returning None for an input that retains its grad or has a callback raises.

    ``check_grad_coverage`` enables strict diagnostics for graph-reachable
    parameters receiving no gradient across a paired B/W call. Intentional
    None gradients can trigger this check. It does not detect partial missing
    contributions. Stop training on failure; gradient accumulation may already
    have occurred. Direct calls to Executor methods bypass these wrappers.

    Args:
        backward_input_fn (Callable): Replacement function for backward input computation.
        backward_weight_fn (Callable): Replacement function for backward weight computation.
        check_grad_coverage (bool, optional):
            Enable strict gradient coverage checking. Defaults to False.
            Only turn this on for debugging purposes or when you need to ensure full gradient coverage.

    Returns:
        Generator[None, None, None]: Context manager for custom FBW functions.

    """
    if not callable(backward_input_fn) or not callable(backward_weight_fn):
        raise TypeError("backward_input_fn and backward_weight_fn must be callable")

    if check_grad_coverage:
        backward_input_fn, backward_weight_fn, pending = _checked_fbw(
            backward_input_fn, backward_weight_fn,
        )
    else:
        pending = {}

    global backward_input, backward_weight
    previous_backward_input = backward_input
    previous_backward_weight = backward_weight
    backward_input = _with_boundary_grads(backward_input_fn)
    backward_weight = backward_weight_fn
    try:
        yield
        if any(pending.values()):
            raise RuntimeError('FBW check: unmatched B calls at context exit')
    finally:
        backward_input = previous_backward_input
        backward_weight = previous_backward_weight
        pending.clear()


# register checking for normal exit
atexit.register(Executor.check_clear)
atexit.register(AsyncCommHandler().check_clear)
