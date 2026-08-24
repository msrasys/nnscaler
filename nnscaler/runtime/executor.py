#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

r"""
Executor for runtime
"""
import atexit

from dataclasses import dataclass
from typing import Tuple, Any, Callable, List, Dict, Iterable, Optional, Union, Iterator
import torch
import logging
from torch.distributed import Work

from ._patch_torch import stage_backward_input, stage_backward_weight


_logger = logging.getLogger(__name__)

_ALLOW_GRAD_DTYPES = (torch.double, torch.float32, torch.float16, torch.bfloat16)

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

    def clear(self):
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


TensorPairs = List[Tuple[int, torch.Tensor]]


@dataclass
class _WeightBackwardState:
    param_groups: Optional[List[Dict[str, Any]]] = None
    output_tensors: Optional[Tuple[torch.Tensor, ...]] = None
    output_tensor_grads: Optional[Tuple[Optional[torch.Tensor], ...]] = None


class Executor:

    # We consider each segment as an isolated graph. By
    # executing the forward of graph, the input tensors will be detached
    # from previous graph and saved for backward.
    # Each graph has its name, and multiple call for the graph will append
    # (instant id -> detached) input tensor pairs for backward reference.
    _detach: Dict[str, List[TensorPairs]] = dict()
    _pseudo_free_grad_edges: Dict[int, Any] = dict()
    _pseudo_free_pending_sends: Dict[int, int] = dict()
    _pseudo_free_unavailable_warned = False
    # Weight-backward states follow the same per-segment FIFO order as `_detach`.
    _weight_backward_states: Dict[str, List[_WeightBackwardState]] = dict()
    _backward_pre_hook: Optional[Callable] = None
    _input_grad_callbacks: Dict[
        int, Tuple[torch.Tensor, Callable[[torch.Tensor], None]]
    ] = dict()

    @staticmethod
    def register_input_grad_callback(
        tensor: torch.Tensor,
        callback: Callable[[torch.Tensor], None],
    ) -> None:
        """Run ``callback`` once a detached segment produces ``tensor``'s grad."""
        if not torch.is_tensor(tensor) or not tensor.requires_grad:
            raise ValueError('input gradient callbacks require a grad-requiring tensor')
        tensor_id = id(tensor)
        if tensor_id in Executor._input_grad_callbacks:
            raise RuntimeError('an input gradient callback is already registered')
        # Keep the tensor alive so its id cannot be reused before the callback.
        Executor._input_grad_callbacks[tensor_id] = (tensor, callback)

    @staticmethod
    def _run_input_grad_callbacks(
        saved_pairs: TensorPairs,
        grad_by_dtensor_id: Dict[int, Optional[torch.Tensor]],
    ) -> None:
        for tensor_id, dtensor in saved_pairs:
            entry = Executor._input_grad_callbacks.pop(tensor_id, None)
            if entry is None:
                continue
            grad = grad_by_dtensor_id.get(id(dtensor))
            if grad is None:
                raise RuntimeError('segment did not produce a registered input gradient')
            _, callback = entry
            callback(grad.detach())

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
        Executor._detach.setdefault(name, []).append(saved_pairs)

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

        saved_pairs = Executor._detach[name].pop(0)
        tensor_ids: List[int] = [pair[0] for pair in saved_pairs]
        dtensors: List[torch.Tensor] = [pair[1] for pair in saved_pairs]
        requested_input_tensors = input_tensors
        requested_tensor_ids = [
            id(t) for t in requested_input_tensors if torch.is_tensor(t)
        ]
        dtensor_by_input_id = {
            tid: dtensor
            for tid, dtensor in saved_pairs
            if torch.is_tensor(dtensor)
        }

        for t in requested_input_tensors:
            if torch.is_tensor(t) and id(t) not in tensor_ids:
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

        pseudo_free_output_ids = []
        backward_roots = []
        for tensor in dedup_output_tensors:
            edge = Executor._pseudo_free_grad_edges.get(id(tensor))
            if edge is None:
                backward_roots.append(tensor)
            else:
                pseudo_free_output_ids.append(id(tensor))
                backward_roots.append(edge)

        try:
            torch.autograd.backward(
                backward_roots,
                grad_tensors=dedup_output_tensor_grads,
            )
        finally:
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
            tid: dtensor.grad
            for tid, dtensor in saved_pairs
            if torch.is_tensor(dtensor) and dtensor.requires_grad
        }
        for tensor in requested_input_tensors:
            if (
                torch.is_tensor(tensor)
                and tensor.retains_grad
                and (grad := grad_by_input_id.get(id(tensor))) is not None
            ):
                tensor.grad = (
                    grad.detach()
                    if tensor.grad is None
                    else tensor.grad + grad.detach()
                )

        Executor._run_input_grad_callbacks(
            saved_pairs,
            {
                id(dtensor): dtensor.grad
                for dtensor in dtensors
                if torch.is_tensor(dtensor)
            },
        )

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
            Executor._weight_backward_states.setdefault(name, []).append(
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
            Executor._weight_backward_states.setdefault(name, []).append(
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
        Executor._weight_backward_states.setdefault(name, []).append(
            _WeightBackwardState(param_groups=param_groups)
        )

        assert all(grad is not None for grad in grads), "RuntimeError: got gradient None"
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
    def sync_tensors(tensors: List[Any]) -> List[Any]:
        """
        Wait until the finish of synchornized tensors
        """
        AsyncCommHandler().drain_sends(wait=False)
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
        Executor._detach = dict()
        Executor._pseudo_free_grad_edges = dict()
        Executor._pseudo_free_pending_sends = dict()
        Executor._weight_backward_states = dict()
        Executor._backward_pre_hook = None
        Executor._input_grad_callbacks = dict()

    @staticmethod
    def check_clear():
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


fexecute = Executor.fexecute
aexecute = Executor.aexecute
backward = Executor.backward
pseudo_free_tensor = Executor.pseudo_free_tensor
defer_pseudo_free_tensor = Executor.defer_pseudo_free_tensor
complete_deferred_pseudo_free_tensor = Executor.complete_deferred_pseudo_free_tensor
backward_input = Executor.backward_input
backward_weight = Executor.backward_weight
register_input_grad_callback = Executor.register_input_grad_callback
sync_tensors = Executor.sync_tensors


# register checking for normal exit
atexit.register(Executor.check_clear)
atexit.register(AsyncCommHandler().check_clear)
