#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

r"""
Executor for runtime
"""
import atexit
import os

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Tuple, Any, Callable, List, Dict, Iterable, Optional
import torch
import logging
from torch.distributed import Work

from nnscaler.flags import RuntimeFlag

from ._patch_torch import (
    stage_backward_input,
    stage_backward_input_selective,
    stage_backward_weight,
)
from ._patch_torch_checkpoint import ReusableGraphExecGroup

_logger = logging.getLogger(__name__)

_FBW_NATIVE_MODULE_MAX_GROUP_WEIGHT_ELEMENTS = int(os.getenv(
    'FBW_NATIVE_MODULE_MAX_GROUP_WEIGHT_ELEMENTS', str(1 << 23)
))
_FBW_NATIVE_MODULE_MAX_PENDING_BYTES = int(os.getenv(
    'FBW_NATIVE_MODULE_MAX_PENDING_BYTES', str(256 << 20)
))
_FBW_NATIVE_MODULE_MAX_PENDING_GROUPS = int(os.getenv(
    'FBW_NATIVE_MODULE_MAX_PENDING_GROUPS', '4'
))
# Reject module W work larger than the measured PP P2P window by default.
# This keeps the 64.4B-FMA shared-FFN FC1 in I while still permitting an
# explicitly selected 12.9B-FMA attention K/V projection.
_FBW_NATIVE_MODULE_MAX_GROUP_FMA = int(os.getenv(
    'FBW_NATIVE_MODULE_MAX_GROUP_FMA', '20000000000'
))
_FBW_DIRECT_MODULE_WEIGHT_TASKS = os.getenv(
    'FBW_DIRECT_MODULE_WEIGHT_TASKS', '1'
).lower() in ('1', 'true', 'yes', 'on')
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

    @staticmethod
    def _has_incomplete_work(works) -> bool:
        """Return whether a submitted communication still has GPU work left.

        A receive/send handle can already be complete by the time generated
        code reaches its wait adapter.  Running a deferred wgrad before an
        already-complete wait does not fill a bubble; it only moves the GEMM
        onto the critical path.  ProcessGroup work handles expose a
        nonblocking completion query, so use it to distinguish a real
        communication window from an empty one.  Conservatively regard an
        unknown handle as incomplete.
        """
        for work in works:
            is_completed = getattr(work, 'is_completed', None)
            if is_completed is None:
                return True
            try:
                if not is_completed():
                    return True
            except Exception:
                return True
        return False

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

        # The receive has already been submitted on its communication stream.
        # Give one retained module-local W group this pipeline wait window
        # before blocking the compute stream. If no receive is ever consumed,
        # backward/reducer cleanup remains the correctness fallback.
        if self._has_incomplete_work(tensor_or_works):
            Executor.finish_native_module_weight_tasks()
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
            # A full queue means the CPU is about to block for an older P2P
            # send. Consume one retained module W only when the send is still
            # in flight; a completed handle has no bubble left to fill.
            holds = queue.pop(0)
            if self._has_incomplete_work(
                work for _, work, _ in holds
            ):
                Executor.finish_native_module_weight_tasks()
            self._complete_send_holds(holds)

    def begin_send_bundle(self, key):
        """Start collecting sends for one pipeline boundary."""
        state = self.instance
        if state._active_send_bundle is not None:
            raise RuntimeError('A send bundle is already active')
        state._active_send_bundle = (key, [])

    def end_send_bundle(
        self,
        run_native_weight_tasks: Optional[bool] = None,
    ):
        """Commit one pipeline send bundle and optionally launch its W work.

        A backward bundle is complete only after all of its ``isend`` calls
        have been submitted.  Launching the retained module-local dWeight at
        that point mirrors Megatron's ``post_backward(); backward_dw()``
        ordering and lets the compute-stream GEMM overlap the P2P transfer.
        Forward bundles must not consume backward W work, so new codegen opts
        in only for backward sends. ``None`` preserves compatibility with
        already-generated schedules: a retained W can only have been created
        by the immediately preceding backward action in those schedules.
        """
        state = self.instance
        if state._active_send_bundle is None:
            raise RuntimeError('No send bundle is active')
        key, holds = state._active_send_bundle
        state._active_send_bundle = None
        if holds:
            state._send_bundle_queues.setdefault(key, []).append(holds)
        if (
            run_native_weight_tasks is True
            or (
                run_native_weight_tasks is None
                and (
                    Executor._native_module_weight_tasks
                    or Executor._native_module_weight_groups
                )
            )
        ):
            Executor.finish_native_module_weight_tasks()

    def drain_sends(self, wait: bool = True):
        # Correctness fallback for a rank/boundary without a consumed P2P
        # receive. A nonblocking poll must leave W available for the following
        # ``AsyncCommHandler.wait``; otherwise ``sync_tensors`` drains it just
        # before reaching the actual receive window.
        if wait:
            Executor.finish_native_module_weight_tasks(force=True)
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
    # Non-reentrant checkpoint keeps recomputed activations per graph-execution
    # group. Reusing the group in W lets it consume the disjoint activations
    # materialized by I instead of replaying the checkpointed forward again.
    graph_exec_group: Optional[Any] = None


@dataclass
class _NativeModuleWeightGroup:
    """One module-local dWeight unit retained across pipeline actions."""
    segment: Optional[str]
    tasks: Tuple[Any, ...]
    retained_bytes: int


def _partition_fine_grained_weight_tasks(tasks):
    """Keep one fitting, earliest-forward module for pipeline scheduling.

    Megatron completes expert/dense wgrad inside the layer backward and moves
    only the first layer's pre-dispatch wgrad across the following P2P call.
    Custom kernels opt in by attaching ``_nnscaler_fbw_schedule_group``. The
    final group observed by autograd is the earliest module in forward order.
    A whole group must fit the configured weight-work budget; this prevents a
    module containing several individually reasonable projections from
    creating a new critical-path W tail. Legacy tasks without this metadata
    retain the original all-deferred behavior.
    """
    grouped = [
        task for task in tasks
        if hasattr(task, '_nnscaler_fbw_schedule_group')
        and task._nnscaler_fbw_schedule_group is not None
    ]
    if not grouped:
        if any(
            hasattr(task, '_nnscaler_fbw_schedule_group') for task in tasks
        ):
            return list(tasks), [], None
        return [], list(tasks), None

    group_elements = {}
    group_cost_fma = {}
    group_retained_bytes = {}
    group_target_ids = {}
    for task in grouped:
        schedule_group = task._nnscaler_fbw_schedule_group
        total = group_elements.setdefault(schedule_group, 0)
        target_ids = group_target_ids.setdefault(schedule_group, set())
        for target in getattr(task, '_nnscaler_fbw_targets', ()):
            target_id = id(target)
            if target_id in target_ids:
                continue
            target_ids.add(target_id)
            numel = getattr(target, 'numel', None)
            if numel is not None:
                total += int(numel())
        group_elements[schedule_group] = total
        group_cost_fma[schedule_group] = (
            group_cost_fma.get(schedule_group, 0)
            + int(getattr(task, '_nnscaler_fbw_cost_fma', 0))
        )
        group_retained_bytes[schedule_group] = (
            group_retained_bytes.get(schedule_group, 0)
            + int(getattr(task, '_nnscaler_fbw_retained_bytes', 0))
        )

    fitting_groups = {
        schedule_group
        for schedule_group, total in group_elements.items()
        if total <= _FBW_NATIVE_MODULE_MAX_GROUP_WEIGHT_ELEMENTS
        and group_retained_bytes.get(schedule_group, 0)
        <= _FBW_NATIVE_MODULE_MAX_PENDING_BYTES
        and (
            _FBW_NATIVE_MODULE_MAX_GROUP_FMA <= 0
            or group_cost_fma.get(schedule_group, 0)
            <= _FBW_NATIVE_MODULE_MAX_GROUP_FMA
        )
    }
    delayed_group = next((
        task._nnscaler_fbw_schedule_group
        for task in reversed(grouped)
        if task._nnscaler_fbw_schedule_group in fitting_groups
    ), None)
    eager = []
    delayed = []
    for task in tasks:
        if not hasattr(task, '_nnscaler_fbw_schedule_group'):
            # Preserve compatibility for custom Functions that have not opted
            # in to fine-grained scheduling.
            delayed.append(task)
        elif (
            delayed_group is not None
            and task._nnscaler_fbw_schedule_group == delayed_group
        ):
            delayed.append(task)
        else:
            eager.append(task)
    return eager, delayed, delayed_group


def _select_fine_grained_weight_group(param_groups):
    """Select one movable module-level W group for a whole segment.

    Selective backward groups callbacks by their reducer/parameter ownership,
    which is unrelated to pipeline scheduling.  Choosing the last group in
    each parameter group therefore leaves several W actions after one I.  Use
    the callback registration order recorded during autograd instead.  The
    last registered explicit group is the earliest module in forward order,
    matching the small first-layer W that Megatron moves across P2P.

    ``None`` means every explicitly annotated task should finish in I.  A
    segment containing only legacy, unannotated callbacks returns no
    selection and keeps the legacy all-deferred behavior.
    """
    ordered_groups = []
    group_elements = {}
    group_cost_fma = {}
    group_retained_bytes = {}
    group_target_ids = {}
    fallback_order = 0
    has_explicit_metadata = False
    for param_group in param_groups:
        for task in param_group.get('deferred_tasks', ()):
            if not hasattr(task, '_nnscaler_fbw_schedule_group'):
                fallback_order += 1
                continue
            has_explicit_metadata = True
            schedule_group = task._nnscaler_fbw_schedule_group
            if schedule_group is not None:
                total = group_elements.setdefault(schedule_group, 0)
                target_ids = group_target_ids.setdefault(
                    schedule_group, set()
                )
                for target in getattr(task, '_nnscaler_fbw_targets', ()):
                    target_id = id(target)
                    if target_id in target_ids:
                        continue
                    target_ids.add(target_id)
                    numel = getattr(target, 'numel', None)
                    if numel is not None:
                        total += int(numel())
                group_elements[schedule_group] = total
                group_cost_fma[schedule_group] = (
                    group_cost_fma.get(schedule_group, 0)
                    + int(getattr(task, '_nnscaler_fbw_cost_fma', 0))
                )
                group_retained_bytes[schedule_group] = (
                    group_retained_bytes.get(schedule_group, 0)
                    + int(getattr(task, '_nnscaler_fbw_retained_bytes', 0))
                )
                ordered_groups.append((
                    getattr(
                        task,
                        '_nnscaler_fbw_registration_order',
                        fallback_order,
                    ),
                    fallback_order,
                    schedule_group,
                ))
            fallback_order += 1
    if not ordered_groups:
        return None, has_explicit_metadata
    fitting_groups = {
        schedule_group
        for schedule_group, total in group_elements.items()
        if total <= _FBW_NATIVE_MODULE_MAX_GROUP_WEIGHT_ELEMENTS
        and group_retained_bytes.get(schedule_group, 0)
        <= _FBW_NATIVE_MODULE_MAX_PENDING_BYTES
        and (
            _FBW_NATIVE_MODULE_MAX_GROUP_FMA <= 0
            or group_cost_fma.get(schedule_group, 0)
            <= _FBW_NATIVE_MODULE_MAX_GROUP_FMA
        )
    }
    fitting_ordered_groups = [
        item for item in ordered_groups if item[2] in fitting_groups
    ]
    if not fitting_ordered_groups:
        return None, True
    return max(fitting_ordered_groups, key=lambda item: item[:2])[2], True


def _partition_fine_grained_param_groups(param_groups):
    """Partition all callback groups with one segment-wide W selection."""
    delayed_group, has_explicit_metadata = (
        _select_fine_grained_weight_group(param_groups)
    )
    if not has_explicit_metadata:
        return [([], list(group.get('deferred_tasks', ())))
                for group in param_groups], None

    partitions = []
    for param_group in param_groups:
        eager = []
        delayed = []
        for task in param_group.get('deferred_tasks', ()):
            if not hasattr(task, '_nnscaler_fbw_schedule_group'):
                delayed.append(task)
            elif (
                delayed_group is not None
                and task._nnscaler_fbw_schedule_group == delayed_group
            ):
                delayed.append(task)
            else:
                eager.append(task)
        partitions.append((eager, delayed))
    return partitions, delayed_group


def _run_self_finalizing_weight_tasks(tasks) -> bool:
    """Run module-level dWeight callables without the generic GraphTask path.

    Phase-aware Linear kernels can accumulate directly into their final
    reducer buffers and complete the corresponding reducer lifecycle inside
    the callback.  Megatron's ``backward_dw()`` follows this same direct path.
    Avoiding ``stage_backward_weight`` here removes parameter metadata setup
    and generic contribution mapping from every pipeline W action.
    """
    if not _FBW_DIRECT_MODULE_WEIGHT_TASKS:
        return False
    tasks = tuple(tasks)
    if not tasks or not all(
        getattr(task, '_nnscaler_fbw_self_finalizing', False)
        for task in tasks
    ):
        return False

    previous_phase = RuntimeFlag.fbw_phase
    previous_tasks = RuntimeFlag.fbw_deferred_tasks
    RuntimeFlag.fbw_phase = 'weight'
    RuntimeFlag.fbw_deferred_tasks = None
    try:
        for task in tasks:
            contributions = task()
            if contributions:
                raise RuntimeError(
                    'A self-finalizing dWeight task returned generic '
                    'gradient contributions'
                )
    finally:
        RuntimeFlag.fbw_phase = previous_phase
        RuntimeFlag.fbw_deferred_tasks = previous_tasks
    return True


def _run_self_finalizing_param_groups(param_groups) -> bool:
    """Fast-path a W state that contains only direct module callbacks."""
    tasks = []
    for param_group in param_groups:
        if (
            param_group.get('params')
            or param_group.get('intermediates')
            or any(grad is not None for grad in param_group.get('grads', ()))
        ):
            return False
        tasks.extend(param_group.get('deferred_tasks', ()))
    if not _run_self_finalizing_weight_tasks(tasks):
        return False
    for param_group in param_groups:
        param_group.pop('deferred_tasks', None)
    return True


class Executor:

    # We consider each segment as an isolated graph. By
    # executing the forward of graph, the input tensors will be detached
    # from previous graph and saved for backward.
    # Each graph has its name, and multiple call for the graph will append
    # (instant id -> detached) input tensor pairs for backward reference.
    _detach: Dict[str, List[TensorPairs]] = dict()
    # Deferred weight-backward states use the same per-segment FIFO order.
    _weight_backward_states: Dict[str, List[_WeightBackwardState]] = dict()
    # Stable per-segment module groups selected during the first dInput. This
    # lets later microbatches avoid creating and immediately draining callbacks
    # for Attention modules that the scheduler will never move.
    _fine_grained_schedule_groups: Dict[str, frozenset[Any]] = dict()
    # Legacy single-group fields remain for generated code/tests that may set
    # them directly. New backward calls enqueue module groups below so a small
    # W can survive a short P2P window and be consumed by a later real wait.
    _native_module_weight_tasks: List[Any] = []
    _native_module_weight_segment: Optional[str] = None
    _native_module_weight_groups: List[_NativeModuleWeightGroup] = []
    _native_module_weight_pending_bytes: int = 0
    _pseudo_free_grad_edges: Dict[int, Any] = dict()
    _pseudo_free_pending_sends: Dict[int, int] = dict()
    _pseudo_free_unavailable_warned = False
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
        # A later segment may safely retain a different module's W. Before
        # re-entering the same segment, drain the FIFO through its pending
        # group so reducer state for that parameter cannot overlap itself.
        Executor.finish_native_module_weight_tasks(segment=name)

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
        input_grad_dtypes = []
        requested_dtype_by_id = {
            id(tensor): tensor.dtype
            for tensor in requested_input_tensors
            if torch.is_tensor(tensor)
        }
        for tid in requested_tensor_ids:
            t = dtensor_by_input_id.get(tid)
            if torch.is_tensor(t) and t.requires_grad:
                t.retain_grad()
                input_tensors.append(t)
                input_grad_dtypes.append(requested_dtype_by_id[tid])

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

        native_module_tasks = []
        native_module_overlap = RuntimeFlag.fbw_native_module_overlap
        previous_native_phase = RuntimeFlag.fbw_native_module_phase
        previous_tasks = RuntimeFlag.fbw_deferred_tasks
        previous_accumulate = RuntimeFlag.fbw_accumulate_undeferred_grads
        previous_schedule_groups = RuntimeFlag.fbw_schedule_groups
        if native_module_overlap:
            RuntimeFlag.fbw_native_module_phase = True
            RuntimeFlag.fbw_deferred_tasks = native_module_tasks
            RuntimeFlag.fbw_accumulate_undeferred_grads = True
            RuntimeFlag.fbw_schedule_groups = (
                Executor._fine_grained_schedule_groups.get(name)
            )
        try:
            torch.autograd.backward(
                backward_roots,
                grad_tensors=dedup_output_tensor_grads,
            )
        finally:
            RuntimeFlag.fbw_native_module_phase = previous_native_phase
            RuntimeFlag.fbw_deferred_tasks = previous_tasks
            RuntimeFlag.fbw_accumulate_undeferred_grads = previous_accumulate
            RuntimeFlag.fbw_schedule_groups = previous_schedule_groups
            for tensor_id in pseudo_free_output_ids:
                Executor._pseudo_free_grad_edges.pop(tensor_id, None)
                Executor._pseudo_free_pending_sends.pop(tensor_id, None)
        if native_module_overlap:
            eager_tasks, delayed_tasks, delayed_group = (
                _partition_fine_grained_weight_tasks(native_module_tasks)
            )
            if eager_tasks and not _run_self_finalizing_weight_tasks(eager_tasks):
                raise RuntimeError(
                    'Native module-local FBW requires self-finalizing eager '
                    f'dWeight callbacks for segment {name}'
                )
            if name not in Executor._fine_grained_schedule_groups:
                Executor._fine_grained_schedule_groups[name] = frozenset(
                    () if delayed_group is None else (delayed_group,)
                )
            if delayed_tasks:
                if not all(
                    getattr(task, '_nnscaler_fbw_self_finalizing', False)
                    for task in delayed_tasks
                ):
                    raise RuntimeError(
                        'Native module-local FBW requires self-finalizing '
                        f'delayed callbacks for segment {name}'
                    )
                Executor._enqueue_native_module_weight_tasks(
                    name, delayed_tasks
                )
        grads = tuple(
            tensor.grad.to(dtype)
            if tensor.grad.dtype != dtype else tensor.grad
            for tensor, dtype in zip(input_tensors, input_grad_dtypes)
        )
        assert all(grad is not None for grad in grads), "RuntimeError: got gradient None"
        if    len(grads) == 0: return None
        elif  len(grads) == 1: return grads[0]
        else: return grads

    @staticmethod
    def _promote_legacy_native_module_weight_tasks() -> None:
        tasks = tuple(Executor._native_module_weight_tasks)
        if not tasks:
            return
        retained_bytes = sum(int(getattr(
            task, '_nnscaler_fbw_retained_bytes', 0
        )) for task in tasks)
        Executor._native_module_weight_groups.append(
            _NativeModuleWeightGroup(
                Executor._native_module_weight_segment,
                tasks,
                retained_bytes,
            )
        )
        Executor._native_module_weight_pending_bytes += retained_bytes
        Executor._native_module_weight_tasks = []
        Executor._native_module_weight_segment = None

    @staticmethod
    def _enqueue_native_module_weight_tasks(
        segment: str,
        tasks: Iterable[Any],
    ) -> None:
        """Retain a bounded FIFO of independently schedulable module W work."""
        Executor._promote_legacy_native_module_weight_tasks()
        tasks = tuple(tasks)
        retained_bytes = sum(int(getattr(
            task, '_nnscaler_fbw_retained_bytes', 0
        )) for task in tasks)
        while Executor._native_module_weight_groups and (
            len(Executor._native_module_weight_groups)
            >= _FBW_NATIVE_MODULE_MAX_PENDING_GROUPS
            or Executor._native_module_weight_pending_bytes + retained_bytes
            > _FBW_NATIVE_MODULE_MAX_PENDING_BYTES
        ):
            Executor.finish_native_module_weight_tasks()
        Executor._native_module_weight_groups.append(
            _NativeModuleWeightGroup(segment, tasks, retained_bytes)
        )
        Executor._native_module_weight_pending_bytes += retained_bytes

    @staticmethod
    def finish_native_module_weight_tasks(
        force: bool = False,
        segment: Optional[str] = None,
    ) -> None:
        """Consume queued module W work in FIFO order.

        A normal P2P boundary consumes one group. ``segment`` drains through
        the last pending occurrence of that segment before its next GraphTask,
        while ``force`` is reserved for reducer/iteration finalization.
        """
        Executor._promote_legacy_native_module_weight_tasks()
        groups = Executor._native_module_weight_groups
        if force:
            count = len(groups)
        elif segment is not None:
            matching = [
                index for index, group in enumerate(groups)
                if group.segment == segment
            ]
            count = matching[-1] + 1 if matching else 0
        else:
            count = min(1, len(groups))

        for _ in range(count):
            group = groups.pop(0)
            Executor._native_module_weight_pending_bytes -= (
                group.retained_bytes
            )
            if (group.tasks and not
                    _run_self_finalizing_weight_tasks(group.tasks)):
                raise RuntimeError(
                    'Native module-local FBW retained a non-self-finalizing '
                    f'dWeight callback for segment {group.segment}'
                )

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
        requested_tensor_ids = [
            id(tensor) for tensor in input_tensors if torch.is_tensor(tensor)
        ]
        dtensor_by_input_id = {
            tensor_id: dtensor
            for tensor_id, dtensor in saved_pairs
            if torch.is_tensor(dtensor)
        }
        tensor_ids: List[int] = [pair[0] for pair in saved_pairs]
        for tensor in input_tensors:
            if torch.is_tensor(tensor) and id(tensor) not in tensor_ids:
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

        detached_inputs = []
        input_grad_dtypes = []
        requested_dtype_by_id = {
            id(tensor): tensor.dtype
            for tensor in input_tensors
            if torch.is_tensor(tensor)
        }
        for tensor_id in requested_tensor_ids:
            tensor = dtensor_by_input_id.get(tensor_id)
            if torch.is_tensor(tensor) and tensor.requires_grad:
                detached_inputs.append(tensor)
                input_grad_dtypes.append(requested_dtype_by_id[tensor_id])

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
            detached_inputs, dedup_output_tensors, dedup_output_tensor_grads = \
                Executor._backward_pre_hook(
                    detached_inputs,
                    dedup_output_tensors,
                    dedup_output_tensor_grads,
                )

        if (
            not detached_inputs
            and not RuntimeFlag.fbw_accumulate_undeferred_grads
        ):
            Executor._weight_backward_states.setdefault(name, []).append(
                _WeightBackwardState(
                    output_tensors=tuple(dedup_output_tensors),
                    output_tensor_grads=tuple(dedup_output_tensor_grads),
                )
            )
            return None

        # Selective FBW completes the ordinary autograd graph in I and its
        # module-owned callbacks retain their concrete GEMM operands.  It
        # never re-enters a checkpoint GraphTask in W, so constructing a
        # reusable graph-execution group for every segment/microbatch is pure
        # overhead.  The generic split path still needs the group when W
        # consumes checkpoint recomputation state produced by I.
        graph_exec_group = (
            ReusableGraphExecGroup()
            if (
                ReusableGraphExecGroup is not None
                and not RuntimeFlag.fbw_accumulate_undeferred_grads
            )
            else None
        )
        completed = False
        previous_schedule_groups = RuntimeFlag.fbw_schedule_groups
        RuntimeFlag.fbw_schedule_groups = (
            Executor._fine_grained_schedule_groups.get(name)
        )
        try:
            with graph_exec_group if graph_exec_group is not None else nullcontext():
                if RuntimeFlag.fbw_accumulate_undeferred_grads:
                    grads, param_groups = stage_backward_input_selective(
                        dedup_output_tensors,
                        dedup_output_tensor_grads,
                        detached_inputs,
                    )
                    # Fine-grained phase-aware kernels mark only the module
                    # whose wgrad should cross the P2P boundary. Complete all
                    # other module/expert tasks now, matching Megatron's
                    # backward_dw placement and keeping the pending state
                    # small. The scheduled W action consumes the retained
                    # group after the generated async send.
                    partitions, delayed_group = (
                        _partition_fine_grained_param_groups(param_groups)
                    )
                    for param_group, (eager_tasks, delayed_tasks) in zip(
                        param_groups, partitions, strict=True
                    ):
                        tasks = param_group.get('deferred_tasks')
                        if not tasks:
                            continue
                        param_group['deferred_tasks'] = delayed_tasks
                        if eager_tasks:
                            if not _run_self_finalizing_weight_tasks(
                                eager_tasks
                            ):
                                stage_backward_weight(iter(weights), [{
                                    'params': set(),
                                    'intermediates': [],
                                    'grads': [],
                                    'deferred_tasks': eager_tasks,
                                }])
                    if name not in Executor._fine_grained_schedule_groups:
                        Executor._fine_grained_schedule_groups[name] = (
                            frozenset(
                                () if delayed_group is None
                                else (delayed_group,)
                            )
                        )
                    # Keep a FIFO state for the generated W action, but do not
                    # send an empty callback-only group through the generic
                    # stage_backward_weight machinery.  This is the common
                    # case for segments whose small/native dWeights all stay
                    # in I and whose expert wgrad was finalized at dispatch.
                    param_groups = [
                        param_group for param_group in param_groups
                        if (
                            param_group.get('params')
                            or param_group.get('intermediates')
                            or any(
                                grad is not None
                                for grad in param_group.get('grads', ())
                            )
                            or param_group.get('deferred_tasks')
                        )
                    ]
                else:
                    grads, param_groups = stage_backward_input(
                        dedup_output_tensors,
                        dedup_output_tensor_grads,
                        detached_inputs,
                        iter(weights),
                    )
            completed = True
        finally:
            RuntimeFlag.fbw_schedule_groups = previous_schedule_groups
            if graph_exec_group is not None and not completed:
                graph_exec_group.release()
        grads = tuple(
            grad.to(dtype) if grad is not None and grad.dtype != dtype else grad
            for grad, dtype in zip(grads, input_grad_dtypes)
        )
        needs_graph_exec_group = any(
            param_group.get("params")
            for param_group in param_groups
        )
        if graph_exec_group is not None and not needs_graph_exec_group:
            graph_exec_group.release()
            graph_exec_group = None
        state = _WeightBackwardState(
            param_groups=param_groups,
            graph_exec_group=graph_exec_group,
        )
        Executor._weight_backward_states.setdefault(name, []).append(state)

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
        try:
            if state.param_groups is not None:
                if weights and state.param_groups:
                    if not _run_self_finalizing_param_groups(
                        state.param_groups
                    ):
                        with (
                            state.graph_exec_group
                            if state.graph_exec_group is not None
                            else nullcontext()
                        ):
                            stage_backward_weight(
                                iter(weights), state.param_groups
                            )
                return

            if weights:
                # No stage input required gradients. A regular backward reaches only
                # weight leaves and, importantly, fires their AccumulateGrad hooks.
                torch.autograd.backward(
                    state.output_tensors,
                    grad_tensors=state.output_tensor_grads,
                )
        finally:
            if state.graph_exec_group is not None:
                state.graph_exec_group.release()

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
        Executor.finish_native_module_weight_tasks(force=True)
        Executor._detach = dict()
        Executor._weight_backward_states = dict()
        Executor._fine_grained_schedule_groups = dict()
        Executor._native_module_weight_tasks = []
        Executor._native_module_weight_segment = None
        Executor._native_module_weight_groups = []
        Executor._native_module_weight_pending_bytes = 0
        Executor._pseudo_free_grad_edges = dict()
        Executor._pseudo_free_pending_sends = dict()
        Executor._backward_pre_hook = None

    @staticmethod
    def check_clear():
        for name, npairs in Executor._detach.items():
            assert len(npairs) == 0, \
                f"Fine remaining segment needs backward: {name}, remaining times: {len(npairs)}"
        assert (
            len(Executor._pseudo_free_grad_edges) == 0
            and len(Executor._pseudo_free_pending_sends) == 0
            and len(Executor._native_module_weight_tasks) == 0
            and len(Executor._native_module_weight_groups) == 0
            and Executor._native_module_weight_pending_bytes == 0
        ), (
            f"Pseudo-free output tensors remain: "
            f"edges={len(Executor._pseudo_free_grad_edges)}, "
            f"pending_sends={len(Executor._pseudo_free_pending_sends)}, "
            f"native_weight_groups="
            f"{len(Executor._native_module_weight_groups)}, "
            f"native_weight_bytes="
            f"{Executor._native_module_weight_pending_bytes}"
        )
        for name, states in Executor._weight_backward_states.items():
            assert len(states) == 0, \
                f"Fine remaining segment needs weight backward: {name}, remaining times: {len(states)}"


fexecute = Executor.fexecute
aexecute = Executor.aexecute
backward = Executor.backward
backward_input = Executor.backward_input
backward_weight = Executor.backward_weight
pseudo_free_tensor = Executor.pseudo_free_tensor
defer_pseudo_free_tensor = Executor.defer_pseudo_free_tensor
complete_deferred_pseudo_free_tensor = Executor.complete_deferred_pseudo_free_tensor
sync_tensors = Executor.sync_tensors


# register checking for normal exit
atexit.register(Executor.check_clear)
atexit.register(AsyncCommHandler().check_clear)
