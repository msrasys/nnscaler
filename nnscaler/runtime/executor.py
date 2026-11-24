#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

r"""
Executor for runtime
"""
import atexit
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

from dataclasses import dataclass
from typing import Tuple, Any, Callable, List, Dict, Iterable, Optional, Union, Iterator
import torch
import logging
from torch.distributed import Work

from ._patch_torch import stage_backward_input, stage_backward_weight
from .stream import input_stream_lifetime, drain_stream_holds
from .pipeline_input import (
    InputReference, OutputReference, InputGradSlot, make_releasable_input,
    release_input_tensor as _make_input_reference,
    input_id, saved_input_id, is_grad_target, supports_input_slots, supports_output_references,
)


_logger = logging.getLogger(__name__)

_ALLOW_GRAD_DTYPES = (torch.double, torch.float32, torch.float16, torch.bfloat16)
_forward_output_observers = ContextVar('nnscaler_forward_output_observers', default=())


@contextmanager
def observe_forward_outputs(callback):
    """Observe segment boundaries without changing their outputs or schedule.

    ``callback(name, outputs)`` must treat outputs as read-only. Keep metadata or
    weak references when measuring lifetime. Registration is context-local;
    nested observers compose and are removed on exit, including exceptions.
    Callback failures propagate to the caller.
    """
    if not callable(callback):
        raise TypeError('A forward output observer must be callable')
    token = _forward_output_observers.set((*_forward_output_observers.get(), callback))
    try:
        yield
    finally:
        _forward_output_observers.reset(token)


def _observe_forward_outputs(name, outputs):
    for callback in _forward_output_observers.get():
        callback(name, outputs)

try:
    from torch.autograd.graph import get_gradient_edge
except ImportError:
    get_gradient_edge = None


class AsyncCommHandler:

    class __AsyncCommHandler:
        def __init__(self):
            self._works: Dict[int, List] = {}
            self._callbacks: Dict[int, Callable] = {}
            self._send_holds: List = []
            self._send_bundle_queues: Dict[Any, List[List]] = {}
            self._active_send_bundle: Optional[Tuple[Any, List]] = None

    instance = None

    def __init__(self) -> None:
        if not AsyncCommHandler.instance:
            AsyncCommHandler.instance = AsyncCommHandler.__AsyncCommHandler()

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def wait(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Wait until the finish of the communication

        @param tensor torch.Tensor
        @return tensor torch.Tensor
        """
        if id(tensor) not in self._works:
            return tensor

        tensor_or_works = self._works.pop(id(tensor))
        if isinstance(tensor_or_works, torch.Tensor):
            return tensor_or_works

        for work in tensor_or_works:
            work.wait()
        callback = self._callbacks.pop(id(tensor))
        if callback is not None:
            # A callback may enqueue a copy/cat that still reads the placeholder
            # after returning its replacement. Protect both sides of resolution.
            with input_stream_lifetime((tensor,), True):
                tensor = callback(tensor)
        return tensor

    def submit(self, tensor: torch.Tensor, works: List[Work], callback: Optional[Callable] = None):
        """
        Submit an async communication
        """
        self._works[id(tensor)] = works
        self._callbacks[id(tensor)] = callback

    def hold_send(self, tensor: torch.Tensor, work, callback: Optional[Callable] = None):
        hold = (tensor, work, callback)
        state = self.instance
        if state._active_send_bundle is None:
            state._send_holds.append(hold)
        else:
            state._active_send_bundle[1].append(hold)

    @staticmethod
    def _complete_send_holds(holds):
        for _, work, callback in holds:
            work.wait()
            if callback is not None:
                callback()

    def reserve_send_bundle(self, key, max_pending: int = 2):
        """Wait for capacity before materializing another pipeline output."""
        if max_pending < 1:
            raise ValueError(f'max_pending must be positive, got {max_pending}')

        queue = self._send_bundle_queues.setdefault(key, [])
        while len(queue) >= max_pending:
            self._complete_send_holds(queue.pop(0))

    def begin_send_bundle(self, key):
        """Start collecting sends for one pipeline boundary."""
        state = self.instance
        if state._active_send_bundle is not None:
            raise RuntimeError('A send bundle is already active')
        state._active_send_bundle = (key, [])

    def end_send_bundle(self):
        """Commit the active pipeline-boundary send bundle."""
        state = self.instance
        if state._active_send_bundle is None:
            raise RuntimeError('No send bundle is active')
        key, holds = state._active_send_bundle
        state._active_send_bundle = None
        if holds:
            state._send_bundle_queues.setdefault(key, []).append(holds)

    def drain_sends(self, wait: bool = True):
        if self._active_send_bundle is not None:
            raise RuntimeError('Cannot drain sends while a send bundle is active')

        if wait:
            self._complete_send_holds(self._send_holds)
            self._send_holds.clear()
            for queue in self._send_bundle_queues.values():
                for bundle in queue:
                    self._complete_send_holds(bundle)
            self._send_bundle_queues.clear()
            return

        pending = []
        for tensor, work, callback in self._send_holds:
            is_completed = getattr(work, 'is_completed', None)
            if is_completed is not None and is_completed():
                work.wait()
                if callback is not None:
                    callback()
            else:
                pending.append((tensor, work, callback))
        self._send_holds[:] = pending

    def drain(self):
        self.drain_sends()

        for tid, works in list(self._works.items()):
            callback = self._callbacks.get(tid)
            if callback is not None:
                continue
            for work in works:
                work.wait()
            self._works.pop(tid, None)
            self._callbacks.pop(tid, None)
        drain_stream_holds(wait=True)

    def clear(self):
        try:
            drain_stream_holds(wait=True)
        finally:
            AsyncCommHandler.instance = AsyncCommHandler.__AsyncCommHandler()

    def check_clear(self):
        assert (
            len(self._works) == 0
            and len(self._callbacks) == 0
            and len(self._send_holds) == 0
            and len(self._send_bundle_queues) == 0
            and self._active_send_bundle is None
        ), (
            f"AsyncCommHandler is not clear: works={len(self._works)}, "
            f"callbacks={len(self._callbacks)}, send_holds={len(self._send_holds)}, "
            f"send_bundle_queues={len(self._send_bundle_queues)}, "
            f"active_send_bundle={self._active_send_bundle is not None}"
        )


TensorPairs = List[Tuple[Any, Any]]


@dataclass
class _WeightBackwardState:
    param_groups: Optional[List[Dict[str, Any]]] = None
    output_tensors: Optional[Tuple[torch.Tensor, ...]] = None
    output_tensor_grads: Optional[Tuple[Optional[torch.Tensor], ...]] = None
    opaque_inputs: Tuple[Any, ...] = ()


class Executor:

    # We consider each segment as an isolated graph. By
    # executing the forward of graph, the input tensors will be detached
    # from previous graph and saved for backward.
    # Each graph has its name, and multiple call for the graph will append
    # (instant id -> detached) input tensor pairs for backward reference.
    _detach: Dict[str, List[TensorPairs]] = dict()
    # Opaque owners may hide CUDA tensors, so retain them conservatively.
    # Ordinary tensors use record_stream; single-stream forwards append ().
    _backward_owners: Dict[str, List[Tuple[Any, ...]]] = dict()
    _pseudo_free_grad_edges: Dict[int, Any] = dict()
    _pseudo_free_pending_sends: Dict[int, int] = dict()
    _pseudo_free_unavailable_warned = False
    # Weight-backward states follow the same per-segment FIFO order as `_detach`.
    _weight_backward_states: Dict[str, List[_WeightBackwardState]] = dict()
    _backward_pre_hook: Optional[Callable] = None
    _input_grad_callbacks: Dict[
        int, Tuple[torch.Tensor, Callable[[torch.Tensor], None]]
    ] = dict()
    # Weak callback registrations are keyed by an identity object, not an
    # integer tensor id that may be reused after forward data is released.
    _released_input_grad_callbacks: Dict[InputReference, Tuple[Callable, Optional[Callable]]] = dict()

    @staticmethod
    def register_input_grad_callback(
        tensor: torch.Tensor,
        callback: Callable[[torch.Tensor], None],
        *,
        release_input: bool = False,
        on_data_release: Optional[Callable[[], None]] = None,
    ) -> None:
        """Run ``callback`` once a detached segment produces ``tensor``'s grad.

        Weak registrations must precede forward. Their callback must retain
        any required outer autograd edge independently of the input tensor.
        ``on_data_release`` runs only at the schedule's final data-read marker.
        """
        if not torch.is_tensor(tensor) or not tensor.requires_grad:
            raise ValueError('input gradient callbacks require a grad-requiring tensor')
        tensor_id = id(tensor)
        if (tensor_id in Executor._input_grad_callbacks
                or any(key.original() is tensor for key in Executor._released_input_grad_callbacks)):
            raise RuntimeError('an input gradient callback is already registered')
        if not isinstance(release_input, bool) or (on_data_release is not None and not release_input):
            raise ValueError('on_data_release requires a weak release_input callback')
        if release_input:
            for pending in Executor._detach.values():
                for pairs in pending:
                    if any((key.original() is tensor if isinstance(key, InputReference)
                            else key == tensor_id) for key, _ in pairs):
                        raise RuntimeError('A weak input gradient callback must be registered before forward')
            Executor._released_input_grad_callbacks[InputReference(tensor)] = (callback, on_data_release)
            return
        # Keep the tensor alive so its id cannot be reused before the callback.
        Executor._input_grad_callbacks[tensor_id] = (tensor, callback)

    @staticmethod
    def release_input_tensor(tensor):
        reference = _make_input_reference(tensor)
        if reference is tensor:
            return reference
        for key in tuple(Executor._released_input_grad_callbacks):
            if key.original() is tensor:
                callback, release = Executor._released_input_grad_callbacks[key]
                if release is not None:
                    Executor._released_input_grad_callbacks[key] = (callback, None)
                    release()
        return reference

    @staticmethod
    def release_output_tensor(tensor):
        """Keep an immutable pipeline output's backward edge after its last read.

        Leaves, scalar losses, unsupported layouts/APIs and unknown backward
        hooks retain the tensor. No data pointer, shape or storage is modified.
        Existing owners and autograd saves can still keep its storage alive.
        """
        if isinstance(tensor, OutputReference):
            return tensor
        if (get_gradient_edge is None or not supports_output_references(Executor._backward_pre_hook)
                or not torch.is_tensor(tensor) or tensor.is_leaf or not tensor.requires_grad
                or tensor.layout != torch.strided or tensor.numel() <= 1):
            return tensor
        # Transfer ownership from the legacy P2P placeholder mechanism. Pending
        # sends retain their own tensor until completion, so the new reference
        # needs neither a second edge registry entry nor a later data mutation.
        edge = Executor._pseudo_free_grad_edges.pop(id(tensor), None)
        Executor._pseudo_free_pending_sends.pop(id(tensor), None)
        if edge is None:
            edge = get_gradient_edge(tensor)
        if tensor.device.type == 'cuda':
            tensor.record_stream(torch.cuda.current_stream(tensor.device))
        # A tensor may also have an external producer with a final-read callback.
        Executor.release_input_tensor(tensor)
        return OutputReference(tensor, edge)

    @staticmethod
    def register_input_backward(
        tensor: torch.Tensor,
        *,
        context_factory: Optional[Callable] = None,
        on_complete: Optional[Callable[[], None]] = None,
        on_data_release: Optional[Callable[[], None]] = None,
    ) -> None:
        """Connect an isolated segment's input gradient to an outer eager graph.

        A non-leaf can retain its autograd edge without owning the output data.
        When early data release is requested, the caller only supplies cleanup
        of its external references. Leaves (and older PyTorch without gradient
        edges) keep the tensor alive. Saved values required by autograd always
        remain owned by that graph. Register before forward for early release.

        ``context_factory`` is entered only while running the outer backward,
        for example to apply the caller's gradient synchronization policy.
        """
        if not torch.is_tensor(tensor) or not tensor.requires_grad:
            raise ValueError('input backward connections require a grad-requiring tensor')
        release_input = on_data_release is not None and not tensor.is_leaf and get_gradient_edge is not None
        root = get_gradient_edge(tensor) if release_input else tensor

        def resume_backward(grad):
            nonlocal root
            with (context_factory() if context_factory is not None else nullcontext()):
                torch.autograd.backward(root, grad)
            root = None
            if on_complete is not None:
                on_complete()

        Executor.register_input_grad_callback(
            tensor, resume_backward, release_input=release_input,
            on_data_release=on_data_release if release_input else None,
        )

    @staticmethod
    def _run_input_grad_callbacks(
        saved_pairs: TensorPairs,
        grad_by_dtensor_id: Dict[int, Optional[torch.Tensor]],
    ) -> None:
        for key, dtensor in saved_pairs:
            if dtensor is None:
                continue
            if isinstance(key, InputReference) and key in Executor._released_input_grad_callbacks:
                callback, _ = Executor._released_input_grad_callbacks.pop(key)
                grad = grad_by_dtensor_id.get(id(dtensor))
                if grad is None:
                    raise RuntimeError('segment did not produce a registered input gradient')
                callback(grad.detach())
                continue
            tensor_id = saved_input_id(key)
            entry = Executor._input_grad_callbacks.get(tensor_id)
            if entry is None:
                continue
            if isinstance(key, InputReference) and key.original() is not entry[0]:
                # The original input can now die before backward. Never run
                # a callback registered on a later tensor that reused its id.
                continue
            Executor._input_grad_callbacks.pop(tensor_id)
            grad = grad_by_dtensor_id.get(id(dtensor))
            if grad is None:
                raise RuntimeError('segment did not produce a registered input gradient')
            _, callback = entry
            callback(grad.detach())

    @staticmethod
    def _pop_forward_inputs(name, requested):
        saved_pairs = Executor._detach[name].pop(0)
        opaque_inputs = Executor._backward_owners[name].pop(0)
        pruned_ids = {saved_input_id(tid) for tid, tensor in saved_pairs if tensor is None}
        for tensor in requested:
            if torch.is_tensor(tensor) and tensor.requires_grad and id(tensor) in pruned_ids:
                raise RuntimeError('Segment input requires_grad was enabled after forward')
        return saved_pairs, opaque_inputs

    @staticmethod
    def _apply_backward_pre_hook(inputs, outputs, grads, track_stream):
        if Executor._backward_pre_hook is None:
            return inputs, outputs, grads
        # The hook may read an old tensor then replace it with a new one.
        # Protect the old storage before the hook submits CUDA work.
        with input_stream_lifetime((inputs, outputs, grads), track_stream):
            return Executor._backward_pre_hook(inputs, outputs, grads)

    @staticmethod
    def fexecute(name: str, subgraph: Callable, *input_tensors: Tuple[Any],
                 requires_grad=True, track_stream=True, release_inputs: Tuple[int, ...] = ()):
        """
        Forward the sub-graph. Tracking defaults on for direct callers;
        generated single-stream plans explicitly pass ``track_stream=False``.
        """
        original_inputs = input_tensors
        input_tensors = Executor.sync_tensors(input_tensors)

        if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(input_tensors)
               for i in release_inputs):
            raise ValueError('release_inputs must contain valid input argument indices')
        if not requires_grad:
            with torch.no_grad(), input_stream_lifetime(input_tensors, track_stream):
                outputs = subgraph(*input_tensors)
                _observe_forward_outputs(name, outputs)
            return outputs

        # everytime forward a segment, detach the tensor from previous graph
        mapping: Dict[int, torch.Tensor] = dict()
        selected_ids = {id(original_inputs[i]) for i in release_inputs}
        slots, keys = {}, {}
        owner = getattr(subgraph, '__self__', None)
        find_owner = getattr(owner, 'find_buffer_or_param_owner', None)
        for original, itensor in zip(original_inputs, input_tensors):
            if not torch.is_tensor(itensor) or not (itensor.requires_grad or original.requires_grad):
                continue
            tid = id(original)
            if id(itensor) in mapping:
                continue
            for key in Executor._released_input_grad_callbacks:
                if key.tensor_id == tid and key.original() is original:
                    keys[tid] = key
                    break
            if tid in selected_ids:
                keys.setdefault(tid, InputReference(original))
                if (supports_input_slots(Executor._backward_pre_hook)
                        and itensor.layout == torch.strided
                        and itensor.dtype in _ALLOW_GRAD_DTYPES
                        and not isinstance(itensor, torch.nn.Parameter)
                        and not (find_owner and find_owner(itensor) is not None)
                        and not getattr(itensor.untyped_storage(), '_nnscaler_activation_resident', False)):
                    mapping[id(itensor)], slots[tid] = make_releasable_input(itensor)
                    continue
            mapping[id(itensor)] = itensor.detach().requires_grad_()
        input_dtensors = tuple(mapping[id(t)] if id(t) in mapping else t for t in input_tensors)
        saved_pairs = [(keys.get(id(original), id(original)), slots.get(id(original), dtensor))
                       for original, dtensor in zip(original_inputs, input_dtensors)]
        Executor._detach.setdefault(name, []).append(saved_pairs)

        from nnscaler.runtime.cpu_offloading import mark_retained_scope_tensors
        mark_retained_scope_tensors(tuple(dtensor for original, dtensor in zip(original_inputs, input_dtensors)
                                         if id(original) not in slots))
        owners = ()
        try:
            with input_stream_lifetime(input_dtensors, track_stream) as opaque:
                owners = tuple(opaque)
                Executor._backward_owners.setdefault(name, []).append(owners)
                outputs = subgraph(*input_dtensors)
                _observe_forward_outputs(name, outputs)
        except BaseException:
            pending = Executor._detach[name]
            pending[:] = [pairs for pairs in pending if pairs is not saved_pairs]
            owner_queue = Executor._backward_owners.get(name, [])
            if owner_queue and owner_queue[-1] is owners:
                owner_queue.pop()
            raise
        saved_pairs[:] = [
            (key, tensor if is_grad_target(tensor) or (
                torch.is_tensor(tensor) and saved_input_id(key) in Executor._input_grad_callbacks
            ) else None)
            for key, tensor in saved_pairs
        ]
        return outputs

    @staticmethod
    def aexecute(subgraph: Callable, *input_tensors: Tuple[Any],
                 requires_grad=True, track_stream=True):
        """
        execute adapter
        """
        input_tensors = Executor.sync_tensors(input_tensors)
        if not requires_grad:
            with torch.no_grad(), input_stream_lifetime(input_tensors, track_stream):
                outputs = subgraph(*input_tensors)
        else:
            with input_stream_lifetime(input_tensors, track_stream):
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
                 output_tensor_grads: List[torch.Tensor], *, track_stream=True) -> Tuple[torch.Tensor]:
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

        saved_pairs, opaque_inputs = Executor._pop_forward_inputs(name, input_tensors)
        tensor_ids: List[int] = [saved_input_id(pair[0]) for pair in saved_pairs]
        dtensors: List[torch.Tensor] = [pair[1] for pair in saved_pairs]
        requested_input_tensors = input_tensors
        requested_tensor_ids = [
            input_id(t) for t in requested_input_tensors if isinstance(t, (torch.Tensor, InputReference))
        ]
        dtensor_by_input_id = {
            saved_input_id(tid): dtensor
            for tid, dtensor in saved_pairs
            if isinstance(dtensor, (torch.Tensor, InputGradSlot))
        }

        for t in requested_input_tensors:
            if isinstance(t, (torch.Tensor, InputReference)) and input_id(t) not in tensor_ids:
                import traceback
                _logger.warning(
                    f"rank {torch.distributed.get_rank()}: input {name} doesn't match. "
                    f"Make sure in scheduling, earlier forward perform earlier backward. "
                    f"Remain {len(Executor._detach[name])} segments.\n"
                    f"{''.join(traceback.format_stack())}"
                )

        if len(output_tensors) == 0: return None

        input_tensors = []
        for tid in requested_tensor_ids:
            t = dtensor_by_input_id.get(tid)
            if is_grad_target(t):
                t.retain_grad()
                input_tensors.append(t)

        visited = set()
        dedup_output_tensors = []
        dedup_output_tensor_grads = []
        for t, g in zip(output_tensors, output_tensor_grads):
            # filter out duplicated output tensor and its grad.
            pair = (t.backward_identity if isinstance(t, OutputReference) else id(t), id(g))
            if pair not in visited:
                visited.add(pair)
                dedup_output_tensors.append(t)
                dedup_output_tensor_grads.append(g)

        # apply hook before backward
        if Executor._backward_pre_hook is not None:
            if any(isinstance(t, InputGradSlot) for t in input_tensors) and not supports_input_slots(Executor._backward_pre_hook):
                raise RuntimeError('The backward hook changed after pipeline inputs were released; it must support input gradient slots')
            if (any(isinstance(t, OutputReference) for t in dedup_output_tensors)
                    and not supports_output_references(Executor._backward_pre_hook)):
                raise RuntimeError('The backward hook changed after pipeline outputs were released; it must support output references')
            input_tensors, dedup_output_tensors, dedup_output_tensor_grads = \
                Executor._apply_backward_pre_hook(
                    input_tensors, dedup_output_tensors, dedup_output_tensor_grads,
                    track_stream or bool(opaque_inputs),
                )

        pseudo_free_output_ids = []
        backward_roots = []
        for tensor in dedup_output_tensors:
            if isinstance(tensor, OutputReference):
                if tensor.edge is None:
                    raise RuntimeError('A released pipeline output was already consumed by backward')
                backward_roots.append(tensor.edge)
                continue
            edge = Executor._pseudo_free_grad_edges.get(id(tensor))
            if edge is None:
                backward_roots.append(tensor)
            else:
                pseudo_free_output_ids.append(id(tensor))
                backward_roots.append(edge)

        with input_stream_lifetime(
            (opaque_inputs, input_tensors, dedup_output_tensors, dedup_output_tensor_grads),
            track_stream or bool(opaque_inputs),
        ):
            try:
                torch.autograd.backward(
                    backward_roots,
                    grad_tensors=dedup_output_tensor_grads,
                )
            finally:
                for tensor in (*output_tensors, *dedup_output_tensors):
                    if isinstance(tensor, OutputReference):
                        tensor.edge = None
                for tensor_id in pseudo_free_output_ids:
                    Executor._pseudo_free_grad_edges.pop(tensor_id, None)
                    Executor._pseudo_free_pending_sends.pop(tensor_id, None)
            grads = tuple(t.grad for t in input_tensors)
            assert all(grad is not None for grad in grads), "RuntimeError: got gradient None"

            # ``fexecute`` deliberately detaches segment inputs.  Preserve the
            # normal PyTorch ``retain_grad`` contract for callers that need a
            # gradient to cross from an end-to-end generated subgraph back into an
            # outer eager graph.
            grad_by_input_id = {
                saved_input_id(tid): dtensor.grad
                for tid, dtensor in saved_pairs
                if is_grad_target(dtensor)
            }
            Executor._accumulate_retained_input_grads(
                [tensor.original() if isinstance(tensor, InputReference) else tensor
                 for tensor in requested_input_tensors],
                grad_by_input_id,
            )

            Executor._run_input_grad_callbacks(
                saved_pairs,
                {
                    id(dtensor): dtensor.grad
                    for dtensor in dtensors
                    if isinstance(dtensor, (torch.Tensor, InputGradSlot))
                },
            )
            for target in dtensors:
                if isinstance(target, InputGradSlot):
                    # The returned gradients/callbacks own any still-needed data.
                    # An output retained by a caller must not keep dInput alive.
                    target.grad = None

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
        *, track_stream=True,
    ) -> Any:
        """Compute input gradients and defer weight gradients.

        ``backward_weight`` must later be called for the same segment name and
        in the same order as ``backward_input``.
        """
        if any(isinstance(t, OutputReference) for t in output_tensors):
            raise RuntimeError('Released pipeline outputs do not support split backward')
        if any(isinstance(key, InputReference) for key, _ in Executor._detach[name][0]):
            raise RuntimeError('Released pipeline inputs do not support split backward')
        output_tensor_grads = Executor.sync_tensors(output_tensor_grads)
        weights = tuple(weights)
        requested_input_tensors = input_tensors

        saved_pairs, opaque_inputs = Executor._pop_forward_inputs(name, input_tensors)
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
            Executor._weight_backward_states.setdefault(name, []).append(
                _WeightBackwardState(param_groups=[], opaque_inputs=opaque_inputs)
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

        input_tensors, dedup_output_tensors, dedup_output_tensor_grads = \
            Executor._apply_backward_pre_hook(
                input_tensors, dedup_output_tensors, dedup_output_tensor_grads,
                track_stream or bool(opaque_inputs),
            )

        if not input_tensors:
            Executor._weight_backward_states.setdefault(name, []).append(
                _WeightBackwardState(
                    output_tensors=tuple(dedup_output_tensors),
                    output_tensor_grads=tuple(dedup_output_tensor_grads),
                    opaque_inputs=opaque_inputs,
                )
            )
            return None

        # PyTorch's helper detaches stage outputs in-place. A view cannot be
        # detached in-place, so use a differentiable clone for that case.
        stage_outputs = [
            tensor.clone() if tensor._is_view() else tensor
            for tensor in dedup_output_tensors
        ]
        # The split helper installs one gradient hook per input. Repeated
        # arguments share a leaf and must not install that hook repeatedly.
        unique_inputs = list({id(tensor): tensor for tensor in input_tensors}.values())
        with input_stream_lifetime(
            (opaque_inputs, unique_inputs, dedup_output_tensors, stage_outputs, dedup_output_tensor_grads),
            track_stream or bool(opaque_inputs),
        ):
            unique_grads, param_groups = stage_backward_input(
                stage_outputs,
                dedup_output_tensor_grads,
                unique_inputs,
                iter(weights),
            )
            grad_by_id = {id(tensor): grad for tensor, grad in zip(unique_inputs, unique_grads)}
            grads = tuple(grad_by_id[id(tensor)] for tensor in input_tensors)
            Executor._weight_backward_states.setdefault(name, []).append(
                _WeightBackwardState(param_groups=param_groups, opaque_inputs=opaque_inputs)
            )

            assert all(grad is not None for grad in grads), "RuntimeError: got gradient None"
            Executor._accumulate_retained_input_grads(
                requested_input_tensors,
                {tid: grad_by_id.get(id(tensor)) for tid, tensor in saved_pairs},
            )
            Executor._run_input_grad_callbacks(
                saved_pairs,
                {id(tensor): grad for tensor, grad in zip(input_tensors, grads)},
            )
            if len(grads) == 0: return None
            elif len(grads) == 1: return grads[0]
            else: return grads

    @staticmethod
    def backward_weight(
        name: str,
        weights: Iterable[torch.nn.Parameter],
        *, track_stream=True,
    ) -> None:
        """Compute weight gradients deferred by ``backward_input``."""
        weights = tuple(weights)
        states = Executor._weight_backward_states.get(name)
        if not states:
            raise RuntimeError(f"No pending weight backward for segment {name}.")
        state = states.pop(0)

        with input_stream_lifetime(
            (state.opaque_inputs, weights, state.output_tensors, state.output_tensor_grads),
            track_stream or bool(state.opaque_inputs),
        ):
            if state.param_groups is not None:
                if weights and state.param_groups:
                    stage_backward_weight(weights, state.param_groups)
                return

            if weights:
                # With no grad-requiring inputs, backward reaches only weight
                # leaves. AccumulateGrad still fires the reducer hooks.
                torch.autograd.backward(
                    state.output_tensors,
                    grad_tensors=state.output_tensor_grads,
                )

    @staticmethod
    def _accumulate_retained_input_grads(input_tensors, grad_by_input_id):
        retained_ids = set()
        for tensor in input_tensors:
            if (
                torch.is_tensor(tensor)
                and tensor.retains_grad
                and id(tensor) not in retained_ids
                and id(tensor) not in Executor._input_grad_callbacks
                and (grad := grad_by_input_id.get(id(tensor))) is not None
            ):
                retained_ids.add(id(tensor))
                # A callback that resumes outer autograd owns this update.
                # Otherwise preserve retain_grad once per boundary identity.
                tensor.grad = grad.detach() if tensor.grad is None else tensor.grad + grad.detach()

    @staticmethod
    def sync_tensors(tensors: List[Any]) -> List[Any]:
        """
        Wait until the finish of synchornized tensors
        """
        handler = AsyncCommHandler()
        # The scheduler may consume adapter inputs while constructing a send
        # bundle. Wait for those inputs, but leave bundle ownership and source
        # release to its explicit end/reserve operations.
        if handler._active_send_bundle is None:
            handler.drain_sends(wait=False)
        replacements = {}
        result = []
        for tensor in tensors:
            if torch.is_tensor(tensor):
                # wait consumes the pending entry. Every occurrence of a
                # placeholder must resolve to the same completed tensor.
                if id(tensor) not in replacements:
                    replacements[id(tensor)] = handler.wait(tensor)
                tensor = replacements[id(tensor)]
            result.append(tensor)
        return result


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
    def _can_pseudo_free_tensor(tensor: torch.Tensor) -> bool:
        if get_gradient_edge is None:
            if not Executor._pseudo_free_unavailable_warned:
                _logger.warning(
                    'Pipeline output pseudo-free requires '
                    'torch.autograd.graph.get_gradient_edge; leaving outputs allocated.'
                )
                Executor._pseudo_free_unavailable_warned = True
            return False
        if not torch.is_tensor(tensor):
            return False
        if not tensor.requires_grad or tensor.grad_fn is None:
            return False
        if tensor.layout != torch.strided or tensor.numel() <= 1:
            return False
        if getattr(tensor, '_base', None) is not None:
            return False
        return True

    @staticmethod
    def _record_pseudo_free_edge(tensor: torch.Tensor) -> bool:
        if not Executor._can_pseudo_free_tensor(tensor):
            return False
        tensor_id = id(tensor)
        if tensor_id not in Executor._pseudo_free_grad_edges:
            Executor._pseudo_free_grad_edges[tensor_id] = get_gradient_edge(tensor)
        return True

    @staticmethod
    def defer_pseudo_free_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if Executor._record_pseudo_free_edge(tensor):
            tensor_id = id(tensor)
            Executor._pseudo_free_pending_sends[tensor_id] = \
                Executor._pseudo_free_pending_sends.get(tensor_id, 0) + 1
        return tensor

    @staticmethod
    def complete_deferred_pseudo_free_tensor(tensor: torch.Tensor) -> torch.Tensor:
        tensor_id = id(tensor)
        pending = Executor._pseudo_free_pending_sends.get(tensor_id)
        if pending is None:
            return tensor
        if pending > 1:
            Executor._pseudo_free_pending_sends[tensor_id] = pending - 1
            return tensor

        Executor._pseudo_free_pending_sends.pop(tensor_id, None)
        if tensor_id not in Executor._pseudo_free_grad_edges:
            return tensor
        return Executor.pseudo_free_tensor(tensor)

    @staticmethod
    def pseudo_free_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """
        Replace a non-leaf output tensor's payload with a 1-element placeholder
        while keeping its autograd edge for a later Executor.backward call.
        """
        if not Executor._record_pseudo_free_edge(tensor):
            return tensor
        tensor.data = torch.empty((1,), dtype=tensor.dtype, device=tensor.device)
        return tensor

    @staticmethod
    def clear():
        try:
            drain_stream_holds(wait=True)
        finally:
            Executor._detach = dict()
            Executor._backward_owners = dict()
            Executor._pseudo_free_grad_edges = dict()
            Executor._pseudo_free_pending_sends = dict()
            Executor._weight_backward_states = dict()
            Executor._backward_pre_hook = None
            Executor._input_grad_callbacks = dict()
            Executor._released_input_grad_callbacks = dict()

    @staticmethod
    def check_clear():
        for name, owners in Executor._backward_owners.items():
            assert not owners, f'Opaque inputs still need backward: {name}, remaining: {len(owners)}'
        for name, npairs in Executor._detach.items():
            assert len(npairs) == 0, \
                f"Fine remaining segment needs backward: {name}, remaining times: {len(npairs)}"
        assert (
            len(Executor._pseudo_free_grad_edges) == 0
            and len(Executor._pseudo_free_pending_sends) == 0
        ), (
            f"Pseudo-free output tensors remain: "
            f"edges={len(Executor._pseudo_free_grad_edges)}, "
            f"pending_sends={len(Executor._pseudo_free_pending_sends)}"
        )
        for name, states in Executor._weight_backward_states.items():
            assert len(states) == 0, \
                f"Fine remaining segment needs weight backward: {name}, remaining times: {len(states)}"
        assert len(Executor._input_grad_callbacks) == 0, \
            f"Input gradient callbacks remain: {len(Executor._input_grad_callbacks)}"
        assert not Executor._released_input_grad_callbacks, \
            f"Released input gradient callbacks remain: {len(Executor._released_input_grad_callbacks)}"


fexecute = Executor.fexecute
aexecute = Executor.aexecute
backward = Executor.backward
pseudo_free_tensor = Executor.pseudo_free_tensor
defer_pseudo_free_tensor = Executor.defer_pseudo_free_tensor
complete_deferred_pseudo_free_tensor = Executor.complete_deferred_pseudo_free_tensor
backward_input = Executor.backward_input
backward_weight = Executor.backward_weight
register_input_grad_callback = Executor.register_input_grad_callback
register_input_backward = Executor.register_input_backward
release_input_tensor = Executor.release_input_tensor
release_output_tensor = Executor.release_output_tensor
sync_tensors = Executor.sync_tensors


# register checking for normal exit
atexit.register(Executor.check_clear)
atexit.register(AsyncCommHandler().check_clear)
