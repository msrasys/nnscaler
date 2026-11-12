#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

r"""
Executor for runtime
"""
import atexit
import sys

from contextlib import contextmanager
from dataclasses import dataclass
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


def debug_id(tensors, msg: str, rank: int):
    if torch.distributed.get_rank() == rank:
        if torch.is_tensor(tensors):
            print(f'[{torch.distributed.get_rank()}] {msg}: [{id(tensors)}]')
        else:
            print(f'[{torch.distributed.get_rank()}] {msg}: {[id(t) for t in tensors]}')


class _AsyncCommHandler:
    def __init__(self) -> None:
        self._works: Dict[torch.Tensor, Union[torch.Tensor, List[Work]]] = {}
        self._callbacks: Dict[torch.Tensor, Callable] = {}
        self._send_holds: List[Tuple[torch.Tensor, Work]] = []

    def wait(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Wait until the finish of the communication

        @param tensor torch.Tensor
        @return tensor torch.Tensor
        """
        if tensor not in self._works:
            return tensor

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

        grads, param_groups = stage_backward_input(
            dedup_output_tensors,
            dedup_output_tensor_grads,
            input_tensors,
            iter(weights),
        )
        Executor._weight_backward_states.setdefault(name, []).append(
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

    @staticmethod
    def sync_tensors(tensors: List[Any]) -> List[Any]:
        """
        Wait until the finish of synchornized tensors
        """
        AsyncCommHandler().drain_all_completed()
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

            With pipeline output deallocation (`ComputeConfig.pipeline_deallocate_outputs`)
            an output tensor other than the loss may already have had its storage
            released by `Executor.release_storage` and only keeps its shape and
            grad_fn (its dtype is then float8). The hook may replace or scale such
            entries but must not read their values. The loss is a graph output and is never released, which is all the
            built-in loss-scaling hook touches.

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
    backward_input = backward_input_fn
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
