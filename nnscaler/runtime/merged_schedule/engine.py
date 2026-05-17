"""
MoE FWD-BWD overlap scheduling engine.

Provides ScheduleNode (CUDA stream/event scheduling primitive) and
MergedScheduler (merged forward-backward with communication/computation overlap).

4-Phase Overlap (MoE layers):
  Phase 1: f_attn_router(COMP) || b_combine(COMM)
  Phase 2: b_expert (COMP) || f_dispatch   (COMM)
  Phase 3: f_expert (COMP) || b_dispatch(COMM)
           delayed-wgrad mode: b_expert_wgrad(COMP) || b_dispatch(COMM),
           then f_expert(COMP)
  Phase 4: b_attn   (COMP) || f_combine    (COMM)

2-Phase Overlap (dense layers):
  Phase 1: body_bwd(prev)[COMP] || attn_fwd(next)[COMP]
  Phase 2: attn_bwd(prev)[COMP] || body_fwd(next)[COMP]

MoE layers use 4 ScheduleNodes (attn_router, dispatch, expert, combine)
alternating COMP/COMM streams for true communication/computation overlap.
Dense layers use 2 ScheduleNodes (attn, ffn) both on COMP stream.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.autograd import Variable

import nnscaler
from nnscaler.runtime import device as runtime_device


_logger = logging.getLogger(__name__)

_COMP_STREAM = None   # computation stream (non-default)
_COMM_STREAM = None   # communication stream (non-default)


def set_streams():
    """Initialize global COMP/COMM streams.
    BOTH streams are non-default so that CUDA's legacy
    default-stream implicit sync does not serialise them.
    """
    global _COMP_STREAM, _COMM_STREAM
    _COMP_STREAM = torch.cuda.Stream()
    _COMM_STREAM = torch.cuda.Stream()
    # Merged scheduling intentionally accumulates gradients on non-default
    # streams. Suppress this warning only when the merged scheduler streams
    # are initialized, rather than at module import time.
    if hasattr(torch.autograd.graph, 'set_warn_on_accumulate_grad_stream_mismatch'):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)


def get_comp_stream():
    assert _COMP_STREAM is not None, "call set_streams() first"
    return _COMP_STREAM


def get_comm_stream():
    assert _COMM_STREAM is not None, "call set_streams() first"
    return _COMM_STREAM


def manual_sync_grads(parallel_module):
    """Manually trigger synchronous allreduce after all backward calls.

    The merged scheduler uses skip_reducer=True, so hooks copy grads to the
    contiguous buffer but skip counting/triggering allreduce. This function
    performs the allreduce and sets param.grad from the buffer.
    """
    pm = parallel_module
    if hasattr(pm, 'backbone'):
        pm = pm.backbone
    if not hasattr(pm, '_reducers'):
        _logger.warning("No _reducers found on parallel module, skipping manual sync")
        return

    for reducer in pm._reducers:
        for bucket in reducer._buckets:
            old_async = bucket._async
            bucket._async = False
            try:
                bucket.sync_grads()
            finally:
                bucket._async = old_async
            bucket.reset()


def _unwrap_parallel_module(parallel_module):
    pm = parallel_module
    if hasattr(pm, 'backbone'):
        pm = pm.backbone
    return pm


@torch.no_grad()
def manual_accumulate_param_grad(parallel_module, param, grad):
    """Accumulate a delayed parameter gradient into nnscaler reducer buffers.

    The merged MoE scheduler can intentionally return no autograd wgrad from a
    fused expert op, then compute that wgrad later on the COMP stream. This
    helper mirrors the reducer's post-AccumulateGrad hook so the later wgrad
    lands in the same contiguous gradient buffer used by manual_sync_grads().

    Returns False when the parameter is not owned by a reducer, allowing callers
    outside ParallelModule/reducer contexts to fall back to param.grad.
    """
    if grad is None:
        return True

    pm = _unwrap_parallel_module(parallel_module)
    if not hasattr(pm, '_reducers'):
        return False

    for reducer in pm._reducers:
        for bucket in reducer._buckets:
            if param not in bucket._pofset:
                continue
            bucket.accumulate_param_grad(param, grad)
            return True

    return False


def _make_viewless(t):
    """Ensure tensor has its own storage view (avoids grad-engine pitfalls)."""
    if isinstance(t, torch.Tensor) and t._base is not None:
        return t.clone()
    return t


def _detach_for_layer_state(t):
    if t is None:
        return None
    t = _make_viewless(t)
    detached = t.detach()
    detached.requires_grad = t.requires_grad
    return detached


_TENSOR_STREAM_ATTR = '_nnscaler_merged_schedule_stream'


def _iter_tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def _same_stream(lhs, rhs):
    return lhs is rhs or getattr(lhs, 'cuda_stream', None) == getattr(rhs, 'cuda_stream', None)


def _set_tensor_stream(tensor, stream):
    try:
        setattr(tensor, _TENSOR_STREAM_ATTR, stream)
    except Exception:
        pass


def _get_tensor_stream(tensor):
    stream = getattr(tensor, _TENSOR_STREAM_ATTR, None)
    if stream is not None:
        return stream
    base = getattr(tensor, '_base', None)
    if isinstance(base, torch.Tensor):
        return getattr(base, _TENSOR_STREAM_ATTR, None)
    return None


def _copy_tensor_stream(src, dst):
    stream = _get_tensor_stream(src)
    if stream is not None:
        _set_tensor_stream(dst, stream)


def _mark_tensors_stream(value, stream):
    for tensor in _iter_tensors(value):
        if _get_tensor_stream(tensor) is None:
            _set_tensor_stream(tensor, stream)


def _order_cross_stream_tensors_until_done(value, stream):
    producer_streams = []
    seen = set()
    for tensor in _iter_tensors(value):
        producer_stream = _get_tensor_stream(tensor)
        if producer_stream is None or _same_stream(producer_stream, stream):
            continue
        stream_key = getattr(producer_stream, 'cuda_stream', id(producer_stream))
        if stream_key not in seen:
            producer_streams.append(producer_stream)
            seen.add(stream_key)
    for producer_stream in producer_streams:
        runtime_device.wait_stream_for_release(
            producer_stream=producer_stream,
            consumer_stream=stream,
        )


class ScheduleNode:
    """A node that executes *forward_func* on *stream*, synchronized via *event*.

    The event protocol:
        event.wait(stream)      # wait for previous node on this mb
        <execute on stream>
        event.record(stream)    # signal completion to next node on this mb

    When checkpoint=True (activation checkpointing):
      - Forward runs with torch.no_grad(), saving only inputs (not intermediates).
      - Backward recomputes forward from saved inputs to rebuild the autograd
        graph, then runs backward through it.
      - Do NOT use checkpoint for nodes containing collective communication
        (dispatch/combine) as recomputation would deadlock.
    """

    def __init__(
        self,
        forward_func,
        stream,
        event,
        backward_func=None,
        free_input=False,
        name="schedule_node",
        checkpoint=False,
    ):
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func if backward_func else self._default_backward
        self.stream = stream
        self.event = event
        self.free_input = free_input
        self.checkpoint = checkpoint
        self.inputs = None
        self.output = None
        self._output_is_tuple = False
        self._output_arity = 1
        self._skip_event = False  # When True, skip event wait/record in _stream_ctx

    def forward(self, inputs=()):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(*inputs)

    def _forward(self, *inputs):
        with self._stream_ctx(f"{self.name} fwd"):
            self.inputs = []
            for inp in inputs:
                if inp is None:
                    self.inputs.append(None)
                elif isinstance(inp, torch.Tensor):
                    d = inp.detach()
                    _copy_tensor_stream(inp, d)
                    d.requires_grad = inp.requires_grad
                    self.inputs.append(d)
                else:
                    self.inputs.append(inp)

            if self.checkpoint:
                with torch.no_grad():
                    data = self.forward_func(*self.inputs)
                self._set_output_metadata(data)
                if not isinstance(data, tuple):
                    data = _make_viewless(data).detach().requires_grad_(True)
                else:
                    data = tuple(
                        _make_viewless(e).detach().requires_grad_(True)
                        if isinstance(e, torch.Tensor) else e
                        for e in data
                    )
            else:
                data = self.forward_func(*self.inputs)
                self._set_output_metadata(data)

            _mark_tensors_stream(data, self.stream)
            self.output = data

        self._free_forward_inputs(inputs)

        return self.output

    def get_output(self):
        return self.output

    def backward(self, output_grad, retain_graph=False):
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(*output_grad, retain_graph=retain_graph)

    def _backward(self, *output_grad, retain_graph=False):
        with self._stream_ctx(f"{self.name} bwd"):
            if self.checkpoint:
                recomputed = self.forward_func(*self.inputs)
                if not isinstance(recomputed, tuple):
                    recomputed = (recomputed,)

                tensor_outputs = []
                tensor_grads = []
                for i, out in enumerate(recomputed):
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        g = self._output_grad_or_none(output_grad, i)
                        if g is None:
                            continue
                        tensor_outputs.append(out)
                        tensor_grads.append(g)
                if tensor_outputs:
                    self.backward_func(tuple(tensor_outputs), tuple(tensor_grads),
                                       retain_graph=retain_graph)
            else:
                outputs = self.output if isinstance(self.output, tuple) else (self.output,)
                tensor_outputs = []
                tensor_grads = []
                for i, out in enumerate(outputs):
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        g = self._output_grad_or_none(output_grad, i)
                        if g is None:
                            continue
                        tensor_outputs.append(out)
                        tensor_grads.append(g)
                if tensor_outputs:
                    self.backward_func(tuple(tensor_outputs), tuple(tensor_grads),
                                       retain_graph=retain_graph)

        _order_cross_stream_tensors_until_done(output_grad, self.stream)
        grads = self.get_grad()
        _mark_tensors_stream(grads, self.stream)
        self._release()
        return grads

    def get_grad(self):
        grad = tuple(e.grad if e is not None else None for e in self.inputs)
        return grad[0] if len(grad) == 1 else grad

    def _set_output_metadata(self, output):
        self._output_is_tuple = isinstance(output, tuple)
        self._output_arity = len(output) if self._output_is_tuple else 1

    def output_arity(self):
        return self._output_arity

    def output_is_tuple(self):
        return self._output_is_tuple

    @staticmethod
    def _output_grad_or_none(output_grad, index):
        return output_grad[index] if index < len(output_grad) else None

    @staticmethod
    def _default_backward(outputs, output_grad, retain_graph=False):
        Variable._execution_engine.run_backward(
            tensors=outputs,
            grad_tensors=output_grad,
            keep_graph=retain_graph,
            create_graph=False,
            inputs=tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )

    def _free_forward_inputs(self, inputs):
        """Release selected forward input storages after stream-safe use.

        record_stream() only makes cross-stream allocator reuse safe. The
        memory-pressure reduction comes from explicitly resizing safe input
        storages to zero. free_input may be False, True, or a collection of
        integer input indexes.
        """
        if not self.free_input:
            return

        indexes = range(len(inputs)) if self.free_input is True else self.free_input
        for idx in indexes:
            if idx >= len(inputs):
                continue
            inp = inputs[idx]
            if isinstance(inp, torch.Tensor):
                inp.record_stream(self.stream)
                inp.untyped_storage().resize_(0)

    @contextmanager
    def _stream_ctx(self, name=None):
        if not self._skip_event:
            self.event.wait(self.stream)
        if name:
            torch.cuda.nvtx.range_push(name)
        try:
            with torch.cuda.stream(self.stream):
                runtime_device.prune_deferred_releases()
                yield
        finally:
            if name:
                torch.cuda.nvtx.range_pop()
            if not self._skip_event:
                self.event.record(self.stream)

    def _release(self):
        _order_cross_stream_tensors_until_done((self.inputs, self.output), self.stream)
        self.inputs = None
        self.output = None

        if not getattr(self, '_defer_step_data_release', False):
            self._release_step_data()

        for attr in (
            'loss_aux_tensors',
            'loss_aux_step_data',
            'forward_func',
            'backward_func',
        ):
            self._release_attr(attr)

    def _release_step_data(self):
        step_data = getattr(self, 'step_data', None)
        if isinstance(step_data, dict):
            step_data.clear()
        self._release_attr('step_data')

    def _release_attr(self, attr):
        if hasattr(self, attr):
            setattr(self, attr, None)
            delattr(self, attr)


@dataclass
class LayerCallables:
    """User-provided callable description for each layer step."""
    # MoE 4-node mode:
    attn_fn: Callable = None          # (h) -> (h, h_ln) for MoE, (h) -> h for dense
    dispatch_fn: Callable = None      # (h_ln) -> (sorted_tokens, sorted_probs)
    expert_fn: Callable = None        # (sorted_tokens, sorted_probs) -> expert_outs
    expert_wgrad_fn: Callable = None  # delayed routed expert wgrad drain
    combine_fn: Callable = None       # (expert_outs) -> h_out; residual/shared state in step_data

    # Dense-only:
    body_fn: Callable = None          # (h) -> h_out (wraps FFN)

    is_moe: bool = False
    step_data: dict = field(default_factory=dict)
    loss_aux_data: dict = field(default_factory=dict)

    # Special steps (yoco_proj):
    is_special: bool = False
    special_forward: Callable = None  # (h) -> (h, special_data)
    special_backward: Callable = None  # (grad_h) -> grad_h


class MergedScheduler:
    """Communication/computation overlap merged forward-backward scheduler.

    MoE layers use 4-node scheduling with COMP/COMM stream alternation:
      - attn_router: attention + residual + gate + routing (COMP)
      - dispatch:    all2all dispatch (COMM)
      - expert:      fused FFN (COMP)
      - combine:     all2all combine + shared expert + residual (COMM)

    Dense layers use 2-node scheduling:
      - attn_node: attention + residual (COMP)
      - body_node: FFN (COMP)
    """

    def __init__(self, parallel_module, num_layers, *,
                 use_checkpoint=False,
                 early_attn_memory_release=False,
                 delay_wgrad_compute=False):
        self.parallel_module = parallel_module
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self._use_4node = True
        self.early_attn_memory_release = early_attn_memory_release
        self.delay_wgrad_compute = delay_wgrad_compute

        # Async overlap: launch COMM and COMP ops from separate CPU threads
        # so their kernel launches can interleave, enabling true GPU overlap.
        # Controlled by ASYNC_4PHASE env var (default=1, set 0 to disable).
        _async_on = os.environ.get('ASYNC_4PHASE', '1') not in ('0', '')
        self._async_pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix='moe-overlap')
            if _async_on else None
        )

        # Pre-allocate CUDA events for per-dependency stream synchronization.
        # 4-phase MoE scheduling has three cross-stream dependency points:
        #   0: f_attn -> f_dispatch,      b_combine -> b_expert
        #   1: f_dispatch -> f_expert,    b_expert -> b_dispatch
        #   2: f_expert -> f_combine,     b_dispatch -> b_attn
        # Each slot has one COMP-produced event and one COMM-produced event.
        self._dep_comp_evts = [torch.cuda.Event() for _ in range(3)]
        self._dep_comm_evts = [torch.cuda.Event() for _ in range(3)]
        # 2 uni-directional sync events (comp->comm and comm->comp)
        self._sync_c2m_evt = torch.cuda.Event()
        self._sync_m2c_evt = torch.cuda.Event()

    def shutdown(self):
        """Release scheduler-owned CPU worker resources."""
        runtime_device.prune_deferred_releases()
        pool = self._async_pool
        if pool is not None:
            self._async_pool = None
            pool.shutdown(wait=True)

    close = shutdown

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def run(self, samples, layer_callables_fn, embed_fn, loss_fn):
        """Execute merged FWD-BWD schedule.

        Args:
            samples: micro-batch list (>= 2)
            layer_callables_fn: (step_idx, sample) -> LayerCallables
            embed_fn: (sample) -> h_float32
            loss_fn: (h, sample, routing_maps, gate_scores_list) -> (loss_node, output_info)

        Returns: [output_info] per micro-batch
        """
        num_mbs = len(samples)
        assert num_mbs >= 2, "merged FWD-BWD requires at least 2 micro-batches"
        runtime_device.prune_deferred_releases()

        # Ensure COMP/COMM streams see all prior default-stream work
        # (optimizer.step, zero_grad from the previous training step).
        # Non-blocking streams do NOT auto-synchronize with the default stream,
        # so without this explicit sync, the forward pass might read stale
        # parameters or see non-zeroed gradients.
        default_done = torch.cuda.Event()
        default_done.record(torch.cuda.default_stream())
        get_comp_stream().wait_event(default_done)
        get_comm_stream().wait_event(default_done)

        num_steps = self.num_layers
        events = [torch.cuda.Event() for _ in range(num_mbs)]
        results = [None] * num_mbs

        _logger.debug("Warmup: forward mb0")
        with torch.cuda.stream(get_comp_stream()):
            h0 = embed_fn(samples[0])
        _embed_h_list = [h0]  # Save embedding outputs for backward

        lc_list_0 = []
        for si in range(num_steps):
            lc_list_0.append(layer_callables_fn(si, samples[0]))

        h0, all_nodes_0, rmaps_0, eprobs_0 = self._forward_all_layers(
            h0, lc_list_0, events[0])
        del lc_list_0

        # Sync COMM→COMP: last forward node may be on COMM (MoE combine),
        # but loss_node runs on COMP with a fresh event (no wait).
        self._sync_comm_to_comp()

        loss_node_0, output_info_0 = loss_fn(h0, samples[0], rmaps_0, eprobs_0)
        loss_0 = loss_node_0.forward((h0,))
        results[0] = output_info_0['output_tuple']

        with torch.cuda.stream(get_comp_stream()):
            loss_grad = torch.ones_like(loss_0)
        del h0, rmaps_0, eprobs_0, output_info_0

        prev_all_nodes = all_nodes_0
        prev_loss_node = loss_node_0

        for mb_i in range(num_mbs - 1):
            fwd_sample = samples[mb_i + 1]
            fwd_event = events[mb_i + 1]

            _logger.debug(f"[MERGED] bwd(mb{mb_i}) + fwd(mb{mb_i+1})")

            # Launch embed on COMM — overlaps with loss_bwd on COMP.
            # embed_fn only reads sample data + embedding weight, independent of COMP.
            with torch.cuda.stream(get_comm_stream()):
                fwd_h = embed_fn(fwd_sample)
            _embed_h_list.append(fwd_h)

            # Loss backward on COMP — overlaps with embed on COMM
            with nnscaler.sync_grad_when(False):
                grad_h = prev_loss_node.backward(loss_grad)

            prev_loss_node._release()

            # Create layer callables (CPU work, overlaps with GPU)
            fwd_lc_list = []
            for si in range(num_steps):
                fwd_lc_list.append(layer_callables_fn(si, fwd_sample))

            fwd_routing_maps = []
            fwd_expert_probs = []
            fwd_all_nodes = [None] * num_steps

            bwd_idx = num_steps - 1
            fwd_idx = 0

            # (4) Sync COMM→COMP: embed result available for forward layers
            self._sync_comm_to_comp()

            while fwd_idx < num_steps and bwd_idx >= 0:
                fwd_lc = fwd_lc_list[fwd_idx]

                if fwd_lc.is_special:
                    # Sync COMM→COMP: previous MoE layer's combine ran on COMM.
                    self._sync_comm_to_comp()
                    with torch.cuda.stream(get_comp_stream()):
                        fwd_h, special_data = fwd_lc.special_forward(fwd_h)
                    fwd_all_nodes[fwd_idx] = ('special', fwd_lc)
                    fwd_lc_list[fwd_idx] = None
                    fwd_lc = None
                    fwd_idx += 1
                    continue

                bwd_entry = prev_all_nodes[bwd_idx]
                if isinstance(bwd_entry, tuple) and len(bwd_entry) == 2 and bwd_entry[0] == 'special':
                    with nnscaler.sync_grad_when(False):
                        grad_h = self._special_backward_and_release(bwd_entry, grad_h)
                    prev_all_nodes[bwd_idx] = None
                    bwd_idx -= 1
                    continue
                if bwd_entry is None:
                    bwd_idx -= 1
                    continue

                with nnscaler.sync_grad_when(False):
                    fwd_h, grad_h, fwd_entry = self._merged_step_general(
                        bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h)

                prev_all_nodes[bwd_idx] = None

                fwd_all_nodes[fwd_idx] = fwd_entry

                if fwd_lc.is_moe:
                    routing_map = self._get_loss_aux_value(fwd_lc, 'routing_map')
                    gate_scores = self._get_loss_aux_value(fwd_lc, 'gate_scores')
                    if routing_map is not None or gate_scores is not None:
                        fwd_routing_maps.append(routing_map)
                        fwd_expert_probs.append(gate_scores)

                fwd_lc_list[fwd_idx] = None
                fwd_lc = None
                fwd_idx += 1
                bwd_idx -= 1

            while fwd_idx < num_steps:
                fwd_lc = fwd_lc_list[fwd_idx]
                if fwd_lc.is_special:
                    # Sync COMM→COMP: previous MoE layer's combine ran on COMM.
                    self._sync_comm_to_comp()
                    with torch.cuda.stream(get_comp_stream()):
                        fwd_h, special_data = fwd_lc.special_forward(fwd_h)
                    fwd_all_nodes[fwd_idx] = ('special', fwd_lc)
                    fwd_lc_list[fwd_idx] = None
                    fwd_lc = None
                    fwd_idx += 1
                    continue

                if fwd_lc.is_moe and self._use_4node:
                    fwd_h, fwd_entry = self._forward_single_layer_4node(
                        fwd_h, fwd_lc, fwd_event)
                else:
                    fwd_h, fwd_entry = self._forward_single_layer(
                        fwd_h, fwd_lc, fwd_event)
                fwd_all_nodes[fwd_idx] = fwd_entry
                if fwd_lc.is_moe:
                    routing_map = self._get_loss_aux_value(fwd_lc, 'routing_map')
                    gate_scores = self._get_loss_aux_value(fwd_lc, 'gate_scores')
                    if routing_map is not None or gate_scores is not None:
                        fwd_routing_maps.append(routing_map)
                        fwd_expert_probs.append(gate_scores)
                fwd_lc_list[fwd_idx] = None
                fwd_lc = None
                fwd_idx += 1

            while bwd_idx >= 0:
                bwd_entry = prev_all_nodes[bwd_idx]
                if isinstance(bwd_entry, tuple) and len(bwd_entry) == 2 and bwd_entry[0] == 'special':
                    with nnscaler.sync_grad_when(False):
                        grad_h = self._special_backward_and_release(bwd_entry, grad_h)
                    prev_all_nodes[bwd_idx] = None
                    bwd_idx -= 1
                    continue
                if bwd_entry is None:
                    bwd_idx -= 1
                    continue
                with nnscaler.sync_grad_when(False):
                    grad_h = self._backward_entry_and_release(bwd_entry, grad_h)
                prev_all_nodes[bwd_idx] = None
                bwd_idx -= 1

            del prev_all_nodes
            # Propagate gradient through embedding graph to tok_embed weight
            with nnscaler.sync_grad_when(False):
                with torch.cuda.stream(get_comp_stream()):
                    _embed_h_list[mb_i].backward(grad_h)
                    _embed_h_list[mb_i] = None
                    del grad_h

            # Sync COMM→COMP: last fwd node may be on COMM (MoE combine),
            # but loss_node runs on COMP with a fresh event.
            self._sync_comm_to_comp()

            fwd_loss_node, fwd_output_info = loss_fn(
                fwd_h, fwd_sample, fwd_routing_maps, fwd_expert_probs)
            fwd_loss = fwd_loss_node.forward((fwd_h,))
            results[mb_i + 1] = fwd_output_info['output_tuple']

            prev_all_nodes = fwd_all_nodes
            prev_loss_node = fwd_loss_node

            if mb_i == 0:
                del all_nodes_0, loss_node_0, loss_0
            del fwd_h, fwd_loss, fwd_routing_maps, fwd_expert_probs
            del fwd_lc_list

        _logger.debug(f"Cooldown: backward mb{num_mbs-1}")
        with nnscaler.sync_grad_when(False):
            grad_h = prev_loss_node.backward(loss_grad)
            for i in reversed(range(num_steps)):
                entry = prev_all_nodes[i]
                if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == 'special':
                    grad_h = self._special_backward_and_release(entry, grad_h)
                    prev_all_nodes[i] = None
                    continue
                if entry is None:
                    continue
                grad_h = self._backward_entry_and_release(entry, grad_h)
                prev_all_nodes[i] = None
            # Propagate gradient through embedding graph to tok_embed weight
            with torch.cuda.stream(get_comp_stream()):
                _embed_h_list[-1].backward(grad_h)
                _embed_h_list[-1] = None
                del grad_h

        # Make default stream wait for COMP/COMM to finish without blocking host.
        comp_done = torch.cuda.Event()
        comm_done = torch.cuda.Event()
        comp_done.record(get_comp_stream())
        comm_done.record(get_comm_stream())
        torch.cuda.default_stream().wait_event(comp_done)
        torch.cuda.default_stream().wait_event(comm_done)

        for i in range(len(results)):
            if results[i] is not None:
                results[i] = tuple(
                    t.detach() if isinstance(t, torch.Tensor) else t
                    for t in results[i]
                )

        _embed_h_list.clear()
        runtime_device.prune_deferred_releases()

        del prev_all_nodes, prev_loss_node

        return results

    def _create_nodes(self, lc, event):
        """Create 2 ScheduleNodes for a layer (dense or MoE fallback)."""
        comp_stream = get_comp_stream()

        attn_node = ScheduleNode(
            lc.attn_fn, comp_stream, event,
            name="attn", checkpoint=self.use_checkpoint)

        if lc.is_moe:
            raise NotImplementedError(
                "2-node MoE fallback is unsupported. MoE layers must use the "
                "4-node dispatch/expert/combine schedule.")

        body_node = ScheduleNode(
            lc.body_fn, comp_stream, event,
            name="ffn", checkpoint=self.use_checkpoint)

        return (attn_node, body_node)

    def _create_nodes_4(self, lc, event):
        """Create 4 ScheduleNodes for MoE layer, alternating COMP/COMM streams."""
        comp_stream = get_comp_stream()
        comm_stream = get_comm_stream()

        attn_node = ScheduleNode(
            lc.attn_fn, comp_stream, event,
            name="attn_router", checkpoint=self.use_checkpoint)

        dispatch_node = ScheduleNode(
            lc.dispatch_fn, comm_stream, event,
            name="dispatch", checkpoint=False)

        expert_node = ScheduleNode(
            lc.expert_fn, comp_stream, event,
            name="expert", checkpoint=self.use_checkpoint)
        expert_node.wgrad_func = lc.expert_wgrad_fn

        combine_node = ScheduleNode(
            lc.combine_fn, comm_stream, event,
            free_input=True, name="combine", checkpoint=False)
        combine_node.step_data = lc.step_data
        # combine.backward() returns before residual/shared grads are collected.
        combine_node._defer_step_data_release = True

        return (attn_node, dispatch_node, expert_node, combine_node)

    def _sync_comm_to_comp(self):
        self._sync_m2c_evt.record(get_comm_stream())
        get_comp_stream().wait_event(self._sync_m2c_evt)

    def _sync_comp_to_comm(self):
        self._sync_c2m_evt.record(get_comp_stream())
        get_comm_stream().wait_event(self._sync_c2m_evt)

    def _record_comp_dependency(self, slot):
        self._dep_comp_evts[slot].record(get_comp_stream())

    def _record_comm_dependency(self, slot):
        self._dep_comm_evts[slot].record(get_comm_stream())

    def _comm_waits_for_comp_dependency(self, slot):
        get_comm_stream().wait_event(self._dep_comp_evts[slot])

    def _comp_waits_for_comm_dependency(self, slot):
        get_comp_stream().wait_event(self._dep_comm_evts[slot])

    def _order_cross_stream_for_comp(self, tensors):
        """Order producer streams after queued COMP work that uses tensors."""
        _order_cross_stream_tensors_until_done(tensors, get_comp_stream())

    @staticmethod
    def _prepare_combine_state(lc, h_residual=None, shared_expert_out=None):
        if h_residual is not None:
            lc.step_data['_combine_residual'] = _detach_for_layer_state(h_residual)
        if shared_expert_out is not None:
            lc.step_data['_combine_shared_expert_out'] = _detach_for_layer_state(
                shared_expert_out)

    def _collect_combine_grads(self, combine_node, combine_grads):
        """Collect grads from combine's detached layer-state tensors."""
        try:
            if isinstance(combine_grads, tuple) and len(combine_grads) == 3:
                return combine_grads

            if isinstance(combine_grads, tuple):
                grad_expert_out = combine_grads[0]
            else:
                grad_expert_out = combine_grads
            step_data = getattr(combine_node, 'step_data', None) or {}
            backward_tensors = step_data.pop('_combine_backward_tensors', None)
            if backward_tensors is not None:
                residual, shared_expert_out = backward_tensors
            else:
                residual = step_data.pop('_combine_residual', None)
                shared_expert_out = step_data.pop('_combine_shared_expert_out', None)

            grad_residual = residual.grad if isinstance(residual, torch.Tensor) else None
            grad_shared = (
                shared_expert_out.grad if isinstance(shared_expert_out, torch.Tensor) else None
            )

            _mark_tensors_stream((grad_residual, grad_shared), combine_node.stream)
            _order_cross_stream_tensors_until_done(
                (residual, shared_expert_out), combine_node.stream)

            return grad_expert_out, grad_residual, grad_shared
        finally:
            combine_node._release_step_data()

    def _prepare_loss_aux_tensors(self, attn_node, lc, attn_out):
        aux_data = self._select_loss_aux_data(lc)
        if aux_data is None:
            return
        aux_tensors = aux_data.get('_loss_aux_tensors')
        if aux_tensors is None:
            return
        if not isinstance(aux_tensors, (tuple, list)):
            aux_tensors = (aux_tensors,)
        if not aux_tensors:
            attn_node.loss_aux_tensors = ()
            return

        outputs = attn_out if isinstance(attn_out, tuple) else (attn_out,)
        if len(outputs) < 3 + len(aux_tensors):
            raise ValueError(
                "MoE aux loss tensors must be returned by attn_fn after "
                "(h_residual, h_ln, routing_probs).")

        replacements = {}
        grad_receivers = []
        for idx, aux_tensor in enumerate(aux_tensors):
            output = outputs[3 + idx]
            if aux_tensor is None:
                grad_receivers.append(None)
                continue
            if (not isinstance(aux_tensor, torch.Tensor)
                    or not isinstance(output, torch.Tensor)):
                raise TypeError("MoE aux loss tensors must correspond to tensor attn_fn outputs.")

            if attn_node.checkpoint:
                receiver = output
            else:
                receiver = _make_viewless(output).detach()
                receiver.requires_grad = output.requires_grad

            if receiver.requires_grad and not receiver.is_leaf:
                receiver.retain_grad()
            replacements[id(aux_tensor)] = receiver
            replacements[id(output)] = receiver
            grad_receivers.append(receiver)

        prepared_aux_data = {}
        for key, value in list(aux_data.items()):
            replaced, changed = self._replace_tensors_by_identity(value, replacements)
            if changed:
                aux_data[key] = replaced
                prepared_aux_data[key] = replaced
        aux_data['_loss_aux_tensors'] = tuple(grad_receivers)
        prepared_aux_data['_loss_aux_tensors'] = tuple(grad_receivers)
        attn_node.loss_aux_tensors = tuple(grad_receivers)
        attn_node.loss_aux_step_data = prepared_aux_data
        attn_node.loss_aux_data_ref = aux_data

    @staticmethod
    def _replace_tensors_by_identity(value, replacements):
        replacement = replacements.get(id(value))
        if replacement is not None:
            return replacement, True
        if isinstance(value, tuple):
            replaced = []
            changed = False
            for v in value:
                item, item_changed = MergedScheduler._replace_tensors_by_identity(v, replacements)
                replaced.append(item)
                changed = changed or item_changed
            return (tuple(replaced) if changed else value), changed
        if isinstance(value, list):
            replaced = []
            changed = False
            for v in value:
                item, item_changed = MergedScheduler._replace_tensors_by_identity(v, replacements)
                replaced.append(item)
                changed = changed or item_changed
            return (replaced if changed else value), changed
        if isinstance(value, dict):
            replaced = {}
            changed = False
            for k, v in value.items():
                item, item_changed = MergedScheduler._replace_tensors_by_identity(v, replacements)
                replaced[k] = item
                changed = changed or item_changed
            return (replaced if changed else value), changed
        return value, False

    @staticmethod
    def _restore_loss_aux_step_data(attn_node, lc):
        prepared_step_data = getattr(attn_node, 'loss_aux_step_data', None)
        if prepared_step_data:
            aux_data = getattr(attn_node, 'loss_aux_data_ref', None)
            if aux_data is None:
                aux_data = MergedScheduler._select_loss_aux_data(lc, create=True)
            aux_data.update(prepared_step_data)
        attn_node._release_attr('loss_aux_step_data')
        attn_node._release_attr('loss_aux_data_ref')

    @staticmethod
    def _select_loss_aux_data(lc, create=False):
        aux_data = getattr(lc, 'loss_aux_data', None)
        if isinstance(aux_data, dict) and (aux_data or create):
            return aux_data
        step_data = getattr(lc, 'step_data', None)
        if isinstance(step_data, dict) and '_loss_aux_tensors' in step_data:
            return step_data
        if create:
            if aux_data is None:
                lc.loss_aux_data = {}
            return lc.loss_aux_data
        return None

    @staticmethod
    def _get_loss_aux_value(lc, key):
        aux_data = MergedScheduler._select_loss_aux_data(lc)
        if isinstance(aux_data, dict) and key in aux_data:
            return aux_data.get(key)
        step_data = getattr(lc, 'step_data', None)
        if isinstance(step_data, dict):
            return step_data.get(key)
        return None

    @staticmethod
    def _release_node_runtime_state(node):
        if node is None:
            return
        for attr in ('inputs', 'output'):
            if hasattr(node, attr):
                setattr(node, attr, None)
        if hasattr(node, '_release_step_data'):
            node._release_step_data()
        if hasattr(node, '_release_attr'):
            for attr in (
                'loss_aux_tensors',
                'loss_aux_step_data',
                'loss_aux_data_ref',
                'forward_func',
                'backward_func',
                'wgrad_func',
            ):
                node._release_attr(attr)

    @staticmethod
    def _release_layer_callable_state(lc):
        if lc is None:
            return
        step_data = getattr(lc, 'step_data', None)
        if isinstance(step_data, dict):
            step_data.clear()
        loss_aux_data = getattr(lc, 'loss_aux_data', None)
        if isinstance(loss_aux_data, dict):
            loss_aux_data.clear()
        for attr in (
            'attn_fn',
            'dispatch_fn',
            'expert_fn',
            'expert_wgrad_fn',
            'combine_fn',
            'body_fn',
            'special_forward',
            'special_backward',
            'step_data',
            'loss_aux_data',
        ):
            if hasattr(lc, attr):
                setattr(lc, attr, None)

    def _release_entry_state(self, entry):
        """Drop Python references for an entry after its backward is complete."""
        if not isinstance(entry, tuple) or len(entry) != 2:
            return
        tag, payload = entry
        if tag == 'special':
            self._release_layer_callable_state(payload)
            return
        if tag not in ('layer2', 'layer4'):
            return
        nodes = payload if isinstance(payload, (tuple, list)) else (payload,)
        for node in nodes:
            self._release_node_runtime_state(node)

    def _special_backward_and_release(self, entry, grad_h):
        """Run a special layer backward and immediately drop its callable state."""
        try:
            _, lc = entry
            with torch.cuda.stream(get_comp_stream()):
                return lc.special_backward(grad_h)
        finally:
            self._release_entry_state(entry)

    def _backward_entry_and_release(self, entry, grad_h):
        """Run a layer backward and immediately drop its saved runtime state."""
        try:
            return self._backward_entry(entry, grad_h)
        finally:
            self._release_entry_state(entry)

    @staticmethod
    def _attn_backward_grads(attn_node, grad_h, grad_h_ln, grad_routing):
        if not attn_node.output_is_tuple():
            return grad_h

        grads = [grad_h, grad_h_ln, grad_routing]
        aux_tensors = getattr(attn_node, 'loss_aux_tensors', ())
        for tensor in aux_tensors:
            if isinstance(tensor, torch.Tensor):
                grads.append(tensor.grad)
            else:
                grads.append(None)
        while len(grads) < attn_node.output_arity():
            grads.append(None)
        return tuple(grads[:attn_node.output_arity()])

    def _has_delayed_expert_wgrad(self, expert_node):
        return (
            self.delay_wgrad_compute
            and callable(getattr(expert_node, 'wgrad_func', None))
        )

    def _run_expert_wgrad(self, expert_node):
        wgrad_func = getattr(expert_node, 'wgrad_func', None)
        if not self.delay_wgrad_compute or wgrad_func is None:
            expert_node._release_attr('wgrad_func')
            return
        with torch.cuda.stream(get_comp_stream()):
            torch.cuda.nvtx.range_push("expert wgrad")
            try:
                wgrad_func()
            finally:
                torch.cuda.nvtx.range_pop()
        expert_node._release_attr('wgrad_func')

    def _dispatch_backward_with_optional_wgrad(self, dispatch_node, expert_node, expert_grads, pool=None):
        dispatch_grad_inputs = (expert_grads[0], expert_grads[1])
        if not self._has_delayed_expert_wgrad(expert_node):
            expert_node._release_attr('wgrad_func')
            return dispatch_node.backward(dispatch_grad_inputs)

        self._run_expert_wgrad(expert_node)
        if pool is not None:
            fut_dispatch_bwd = pool.submit(dispatch_node.backward, dispatch_grad_inputs)
            return fut_dispatch_bwd.result()

        return dispatch_node.backward(dispatch_grad_inputs)

    def _forward_all_layers(self, h, lc_list, event):
        """Warmup: forward all layers, collect nodes and routing data."""
        all_nodes = []
        routing_maps = []
        expert_probs = []

        for si, lc in enumerate(lc_list):
            if lc.is_special:
                # Sync COMM→COMP: previous MoE layer's combine ran on COMM,
                # special_forward runs on COMP and needs the COMM-produced h.
                self._sync_comm_to_comp()
                with torch.cuda.stream(get_comp_stream()):
                    h, special_data = lc.special_forward(h)
                all_nodes.append(('special', lc))
                lc_list[si] = None
                lc = None
                continue

            if lc.is_moe and self._use_4node:
                h, entry = self._forward_single_layer_4node(h, lc, event)
            else:
                nodes = self._create_nodes(lc, event)
                attn_n, body_n = nodes

                attn_out = attn_n.forward((h,))
                h = body_n.forward(attn_out)

                if attn_n.checkpoint:
                    attn_n.output = None
                if body_n.checkpoint:
                    body_n.output = None

                entry = ('layer2', nodes)

            if lc.is_moe:
                routing_map = self._get_loss_aux_value(lc, 'routing_map')
                gate_scores = self._get_loss_aux_value(lc, 'gate_scores')
                if routing_map is not None or gate_scores is not None:
                    routing_maps.append(routing_map)
                    expert_probs.append(gate_scores)

            all_nodes.append(entry)
            lc_list[si] = None
            lc = None

        return h, all_nodes, routing_maps, expert_probs

    def _forward_single_layer(self, h, lc, event):
        """Forward a single dense layer."""
        nodes = self._create_nodes(lc, event)
        attn_n, body_n = nodes

        attn_out = attn_n.forward((h,))
        h = body_n.forward(attn_out)

        if attn_n.checkpoint:
            attn_n.output = None
        if body_n.checkpoint:
            body_n.output = None

        return h, ('layer2', nodes)

    def _forward_single_layer_4node(self, h, lc, event):
        """Forward a single MoE layer through 4 nodes.

        expert_fn now includes shared expert: takes (sorted_tokens, sorted_probs, h_ln),
        returns (expert_outs, shared_expert_out). combine_fn reads detached
        residual/shared state from step_data, keeping combine inputs safe to free.
        """
        nodes = self._create_nodes_4(lc, event)
        attn_n, dispatch_n, expert_n, combine_n = nodes

        attn_out = attn_n.forward((h,))
        h_residual, h_ln, routing_probs = attn_out[:3]
        self._prepare_combine_state(lc, h_residual=h_residual)
        self._prepare_loss_aux_tensors(attn_n, lc, attn_out)
        dispatch_out = dispatch_n.forward((h_ln, routing_probs))
        expert_result = expert_n.forward((*dispatch_out, h_ln))
        expert_out, shared_expert_out = expert_result
        self._prepare_combine_state(lc, shared_expert_out=shared_expert_out)
        h_out = combine_n.forward((expert_out,))
        self._restore_loss_aux_step_data(attn_n, lc)

        for n in nodes:
            if n.checkpoint:
                n.output = None

        return h_out, ('layer4', nodes)

    def _backward_layer(self, nodes, grad_h):
        """Backward through a 2-node layer: body -> attn."""
        attn_n, body_n = nodes
        # Ensure all intermediate ops run on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        with torch.cuda.stream(get_comp_stream()):
            body_grads = body_n.backward(grad_h)
            grad_x = attn_n.backward(body_grads)
        return grad_x

    def _backward_layer_4node(self, nodes, grad_h):
        """Backward through a 4-node MoE layer: combine→expert→dispatch→attn.

        expert_fn includes shared expert, so expert backward produces grad_h_ln
        as its 3rd output gradient (for the h_ln input).
        """
        attn_n, dispatch_n, expert_n, combine_n = nodes

        # Ensure all intermediate ops run on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        with torch.cuda.stream(get_comp_stream()):
            self._sync_comp_to_comm()

            # combine_grads: (grad_expert_outs, grad_h_residual, grad_shared_expert_out)
            combine_grads = combine_n.backward(grad_h)
            combine_grads = self._collect_combine_grads(combine_n, combine_grads)
            # expert backward needs grads for both outputs: expert_outs and shared_expert_out
            expert_grads = expert_n.backward((combine_grads[0], combine_grads[2]))
            # expert_grads: (grad_sorted_tokens, grad_sorted_probs, grad_h_ln)
            dispatch_grads = self._dispatch_backward_with_optional_wgrad(
                dispatch_n, expert_n, expert_grads, self._async_pool)

            self._sync_comm_to_comp()

            grad_h_ln_total = dispatch_grads[0] + expert_grads[2]

            attn_grads = self._attn_backward_grads(
                attn_n, combine_grads[1], grad_h_ln_total, dispatch_grads[1])
            grad_x = attn_n.backward(attn_grads)
            self._order_cross_stream_for_comp(dispatch_grads)
            self._order_cross_stream_for_comp(combine_grads)

        return grad_x

    def _backward_entry(self, entry, grad_h):
        """Dispatch backward by entry type."""
        tag, nodes = entry
        if tag == 'layer2':
            return self._backward_layer(nodes, grad_h)
        elif tag == 'layer4':
            return self._backward_layer_4node(nodes, grad_h)
        else:
            raise ValueError(f"Unknown entry tag: {tag}")

    def _merged_step(self, bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h):
        """2-phase overlap for dense layers."""
        _, bwd_nodes = bwd_entry
        bwd_attn, bwd_body = bwd_nodes
        bwd_released = False

        try:
            fwd_nodes = self._create_nodes(fwd_lc, fwd_event)
            fwd_attn, fwd_body = fwd_nodes

            # Ensure all intermediate ops run on COMP stream, not default stream.
            with torch.cuda.stream(get_comp_stream()):
                body_grads = bwd_body.backward(grad_h)
                fwd_attn_out = fwd_attn.forward((fwd_h,))

                grad_x = bwd_attn.backward(body_grads)
                self._release_entry_state(bwd_entry)
                bwd_released = True
                del body_grads
                fwd_h_out = fwd_body.forward(fwd_attn_out)

                if fwd_attn.checkpoint:
                    fwd_attn.output = None
                if fwd_body.checkpoint:
                    fwd_body.output = None

            return fwd_h_out, grad_x, ('layer2', fwd_nodes)
        finally:
            if not bwd_released:
                self._release_entry_state(bwd_entry)

    def _merged_step_4phase(self, bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h):
        """4-phase overlap for MoE layers.

        Each phase interleaves COMP and COMM operations from different
        micro-batches. Within each phase, COMP and COMM run in PARALLEL on
        the GPU because we skip the per-node shared-event protocol (which
        would serialize all operations). Cross-stream ordering is expressed
        with per-dependency events instead of phase-level bidirectional
        barriers, so each stream waits only for the producer it consumes.

        Phase 1: f_attn_router(COMP) || b_combine(COMM)   [combine = pure comm]
        Phase 2: b_expert    (COMP) || f_dispatch (COMM)   [expert includes shared expert bwd]
        Phase 3: f_expert    (COMP) || b_dispatch (COMM)   [expert includes shared expert fwd]
        Phase 4: b_attn      (COMP) || f_combine  (COMM)

        Shared expert is merged into expert_fn (COMP stream), keeping
        combine_fn (COMM stream) as pure communication.
        """
        _, bwd_nodes = bwd_entry
        bwd_attn, bwd_dispatch, bwd_expert, bwd_combine = bwd_nodes
        fwd_nodes = self._create_nodes_4(fwd_lc, fwd_event)
        fwd_attn, fwd_dispatch, fwd_expert, fwd_combine = fwd_nodes

        pool = self._async_pool  # None in sequential mode

        # --- Skip per-node event protocol during merged step ---
        # The default shared event serializes all nodes (COMP waits for COMM
        # and vice versa), preventing within-phase parallelism. By setting
        # _skip_event=True, each node still switches to its own stream, but
        # does not wait/record the shared event. Cross-stream dependencies are
        # enforced explicitly at the consumer side below.
        all_nodes = (*bwd_nodes, *fwd_nodes)
        for n in all_nodes:
            n._skip_event = True

        # Ensure grad addition runs on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        # ScheduleNode calls internally switch to their own stream and restore on exit.
        with torch.cuda.stream(get_comp_stream()):
            # Initial sync: COMP→COMM so COMM can read grad_h (from loss_bwd on COMP)
            self._sync_comp_to_comm()

            # Phase 1: COMM(b_combine) || COMP(f_attn_router)
            # b_combine is pure communication (shared expert merged into expert).
            if pool is not None:
                fut_combine = pool.submit(bwd_combine.backward, grad_h)
                fwd_attn_out = fwd_attn.forward((fwd_h,))
                combine_grads = fut_combine.result()
            else:
                combine_grads = bwd_combine.backward(grad_h)
                fwd_attn_out = fwd_attn.forward((fwd_h,))
            combine_grads = self._collect_combine_grads(bwd_combine, combine_grads)
            fwd_h_residual, fwd_h_ln, fwd_routing_probs = fwd_attn_out[:3]
            self._prepare_combine_state(fwd_lc, h_residual=fwd_h_residual)
            self._prepare_loss_aux_tensors(fwd_attn, fwd_lc, fwd_attn_out)

            # Dependencies for Phase 2:
            # - f_dispatch(COMM) consumes f_attn_router(COMP) outputs.
            # - b_expert(COMP) consumes b_combine(COMM) gradients.
            self._record_comp_dependency(slot=0)
            self._record_comm_dependency(slot=0)

            # Phase 2: COMM(f_dispatch) || COMP(b_expert)
            # expert backward includes shared expert backward (both on COMP).
            # combine_grads: (grad_expert_outs, grad_h_residual, grad_shared_expert_out)
            if pool is not None:
                self._comm_waits_for_comp_dependency(slot=0)
                fut_dispatch = pool.submit(fwd_dispatch.forward, (fwd_h_ln, fwd_routing_probs))
                self._comp_waits_for_comm_dependency(slot=0)
                expert_grads = bwd_expert.backward((combine_grads[0], combine_grads[2]))
                fwd_dispatch_out = fut_dispatch.result()
            else:
                self._comm_waits_for_comp_dependency(slot=0)
                fwd_dispatch_out = fwd_dispatch.forward((fwd_h_ln, fwd_routing_probs))
                self._comp_waits_for_comm_dependency(slot=0)
                expert_grads = bwd_expert.backward((combine_grads[0], combine_grads[2]))
            # expert_grads: (grad_sorted_tokens, grad_sorted_probs, grad_h_ln)

            # Dependencies for Phase 3:
            # - f_expert(COMP) consumes f_dispatch(COMM) outputs.
            # - b_dispatch(COMM) consumes b_expert(COMP) gradients.
            self._record_comm_dependency(slot=1)
            self._record_comp_dependency(slot=1)

            if self.early_attn_memory_release:
                # Run backward dispatch and attention before later forward
                # expert/combine allocations. This lowers peak memory at the
                # cost of Phase-3/4 overlap.
                self._comm_waits_for_comp_dependency(slot=1)
                dispatch_grads = self._dispatch_backward_with_optional_wgrad(
                    bwd_dispatch, bwd_expert, expert_grads, pool)

                self._record_comm_dependency(slot=2)
                self._comp_waits_for_comm_dependency(slot=2)
                grad_h_ln_total = dispatch_grads[0] + expert_grads[2]
                attn_grads = self._attn_backward_grads(
                    bwd_attn, combine_grads[1], grad_h_ln_total, dispatch_grads[1])
                grad_x = bwd_attn.backward(attn_grads)
                self._order_cross_stream_for_comp(dispatch_grads)
                self._order_cross_stream_for_comp(combine_grads)
                self._release_entry_state(bwd_entry)
                del combine_grads, expert_grads, dispatch_grads, attn_grads, grad_h_ln_total

                fwd_expert_result = fwd_expert.forward((*fwd_dispatch_out, fwd_h_ln))
                fwd_expert_out, fwd_shared_expert_out = fwd_expert_result
                self._prepare_combine_state(
                    fwd_lc, shared_expert_out=fwd_shared_expert_out)
                self._record_comp_dependency(slot=2)
                self._comm_waits_for_comp_dependency(slot=2)
                fwd_h_out = fwd_combine.forward((fwd_expert_out,))

                self._restore_loss_aux_step_data(fwd_attn, fwd_lc)

                for n in fwd_nodes:
                    if n.checkpoint:
                        n.output = None

                self._sync_comm_to_comp()

                for n in fwd_nodes:
                    n._skip_event = False

                return fwd_h_out, grad_x, ('layer4', fwd_nodes)

            # Phase 3: when delayed wgrad is active, overlap routed expert
            # wgrad on COMP with dispatch backward on COMM. Otherwise keep the
            # original f_expert(COMP) || b_dispatch(COMM) overlap.
            if self._has_delayed_expert_wgrad(bwd_expert):
                self._comm_waits_for_comp_dependency(slot=1)
                dispatch_grads = self._dispatch_backward_with_optional_wgrad(
                    bwd_dispatch, bwd_expert, expert_grads, pool)
                self._comp_waits_for_comm_dependency(slot=1)
                fwd_expert_result = fwd_expert.forward((*fwd_dispatch_out, fwd_h_ln))
            elif pool is not None:
                self._comm_waits_for_comp_dependency(slot=1)
                fut_dispatch_bwd = pool.submit(bwd_dispatch.backward, (expert_grads[0], expert_grads[1]))
                self._comp_waits_for_comm_dependency(slot=1)
                fwd_expert_result = fwd_expert.forward((*fwd_dispatch_out, fwd_h_ln))
                dispatch_grads = fut_dispatch_bwd.result()
                bwd_expert._release_attr('wgrad_func')
            else:
                self._comm_waits_for_comp_dependency(slot=1)
                dispatch_grads = bwd_dispatch.backward((expert_grads[0], expert_grads[1]))
                bwd_expert._release_attr('wgrad_func')
                self._comp_waits_for_comm_dependency(slot=1)
                fwd_expert_result = fwd_expert.forward((*fwd_dispatch_out, fwd_h_ln))
            fwd_expert_out, fwd_shared_expert_out = fwd_expert_result
            self._prepare_combine_state(fwd_lc, shared_expert_out=fwd_shared_expert_out)

            # Dependencies for Phase 4:
            # - f_combine(COMM) consumes f_expert(COMP) outputs/state.
            # - b_attn(COMP) consumes b_dispatch(COMM) gradients.
            self._record_comp_dependency(slot=2)
            self._record_comm_dependency(slot=2)

            # Phase 4: COMM(f_combine) || COMP(b_attn)
            if pool is not None:
                self._comm_waits_for_comp_dependency(slot=2)
                fut_combine_fwd = pool.submit(fwd_combine.forward, (fwd_expert_out,))
                self._comp_waits_for_comm_dependency(slot=2)
                grad_h_ln_total = dispatch_grads[0] + expert_grads[2]
                attn_grads = self._attn_backward_grads(
                    bwd_attn, combine_grads[1], grad_h_ln_total, dispatch_grads[1])
                grad_x = bwd_attn.backward(attn_grads)
                self._order_cross_stream_for_comp(dispatch_grads)
                self._order_cross_stream_for_comp(combine_grads)
                self._release_entry_state(bwd_entry)
                del combine_grads, expert_grads, dispatch_grads, attn_grads, grad_h_ln_total
                fwd_h_out = fut_combine_fwd.result()
            else:
                self._comm_waits_for_comp_dependency(slot=2)
                fwd_h_out = fwd_combine.forward((fwd_expert_out,))
                self._comp_waits_for_comm_dependency(slot=2)
                grad_h_ln_total = dispatch_grads[0] + expert_grads[2]
                attn_grads = self._attn_backward_grads(
                    bwd_attn, combine_grads[1], grad_h_ln_total, dispatch_grads[1])
                grad_x = bwd_attn.backward(attn_grads)
                self._order_cross_stream_for_comp(dispatch_grads)
                self._order_cross_stream_for_comp(combine_grads)
                self._release_entry_state(bwd_entry)
                del combine_grads, expert_grads, dispatch_grads, attn_grads, grad_h_ln_total

            self._restore_loss_aux_step_data(fwd_attn, fwd_lc)

            for n in fwd_nodes:
                if n.checkpoint:
                    n.output = None

            # Sync COMM→COMP: fwd_combine ran on COMM stream producing fwd_h_out.
            # The next merged step's fwd_attn will consume fwd_h_out on COMP stream,
            # so COMP must wait for COMM to finish.
            self._sync_comm_to_comp()

        # Restore event protocol for sequential backward use (cooldown)
        for n in fwd_nodes:
            n._skip_event = False

        return fwd_h_out, grad_x, ('layer4', fwd_nodes)

    def _merged_step_general(self, bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h):
        """Dispatch to 4-phase or 2-phase merged step based on layer types."""
        bwd_tag, _ = bwd_entry

        if self._use_4node and bwd_tag == 'layer4' and fwd_lc.is_moe:
            try:
                return self._merged_step_4phase(bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h)
            except BaseException:
                self._release_entry_state(bwd_entry)
                raise
        elif bwd_tag == 'layer2':
            if fwd_lc.is_moe:
                raise NotImplementedError(
                    "Merging dense backward with MoE forward is unsupported. "
                    "The MoE forward must run through the 4-node schedule.")
            return self._merged_step(bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h)
        else:
            with nnscaler.sync_grad_when(False):
                grad_x = self._backward_entry_and_release(bwd_entry, grad_h)
            if fwd_lc.is_moe and self._use_4node:
                fwd_h_out, fwd_entry = self._forward_single_layer_4node(fwd_h, fwd_lc, fwd_event)
            else:
                fwd_h_out, fwd_entry = self._forward_single_layer(fwd_h, fwd_lc, fwd_event)
            return fwd_h_out, grad_x, fwd_entry
