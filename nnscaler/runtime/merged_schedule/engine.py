"""
MoE FWD-BWD overlap scheduling engine.

Provides ScheduleNode (CUDA stream/event scheduling primitive) and
MergedScheduler (merged forward-backward with communication/computation overlap).

4-Phase Overlap (MoE layers):
  Phase 1: f_attn_router(COMP) || b_combine(COMM)
  Phase 2: b_expert (COMP) || f_dispatch   (COMM)
  Phase 3: f_expert (COMP) || b_dispatch(COMM)
  Phase 4: b_attn   (COMP) || f_combine    (COMM)

2-Phase Overlap (dense layers):
  Phase 1: body_bwd(prev)[COMP] || attn_fwd(next)[COMP]
  Phase 2: attn_bwd(prev)[COMP] || body_fwd(next)[COMP]

MoE layers use 4 ScheduleNodes (attn_router, dispatch, expert, combine)
alternating COMP/COMM streams for true communication/computation overlap.
Dense layers use 2 ScheduleNodes (attn, ffn) both on COMP stream.
"""

import logging
import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.autograd import Variable

import nnscaler


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


def _make_viewless(t):
    """Ensure tensor has its own storage view (avoids grad-engine pitfalls)."""
    if isinstance(t, torch.Tensor) and t._base is not None:
        return t.clone()
    return t


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


def _record_tensors_stream(value, stream):
    """Tell the CUDA allocator when tensors are consumed on another stream."""
    for tensor in _iter_tensors(value):
        if not tensor.is_cuda:
            continue
        producer_stream = _get_tensor_stream(tensor)
        if producer_stream is not None and _same_stream(producer_stream, stream):
            continue
        tensor.record_stream(stream)


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
        self.detached = tuple()
        self.before_detached = tuple()

    def forward(self, inputs=()):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(*inputs)

    def _forward(self, *inputs):
        with self._stream_ctx(f"{self.name} fwd"):
            _record_tensors_stream(inputs, self.stream)
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

    def detach(self, tensor):
        """Detach a tensor for layer state while preserving its backward edge."""
        if tensor is None:
            return None
        tensor = _make_viewless(tensor)
        detached = tensor.detach()
        detached.requires_grad = tensor.requires_grad
        _copy_tensor_stream(tensor, detached)
        self.before_detached = self.before_detached + (tensor,)
        self.detached = self.detached + (detached,)
        return detached

    def backward(self, output_grad, retain_graph=False):
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(*output_grad, retain_graph=retain_graph)

    def _backward(self, *output_grad, retain_graph=False):
        with self._stream_ctx(f"{self.name} bwd"):
            _record_tensors_stream(output_grad, self.stream)
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
                self._append_detached_backward_tensors(tensor_outputs, tensor_grads)
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
                self._append_detached_backward_tensors(tensor_outputs, tensor_grads)
                if tensor_outputs:
                    self.backward_func(tuple(tensor_outputs), tuple(tensor_grads),
                                       retain_graph=retain_graph)

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

    def _append_detached_backward_tensors(self, tensor_outputs, tensor_grads):
        for before, detached in zip(self.before_detached, self.detached):
            if not isinstance(before, torch.Tensor) or not before.requires_grad:
                continue
            grad = detached.grad if isinstance(detached, torch.Tensor) else None
            if grad is None:
                continue
            _record_tensors_stream(grad, self.stream)
            tensor_outputs.append(before)
            tensor_grads.append(grad)

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
                yield
        finally:
            if name:
                torch.cuda.nvtx.range_pop()
            if not self._skip_event:
                self.event.record(self.stream)

    def _release(self):
        self.inputs = None
        self.output = None
        self.detached = tuple()
        self.before_detached = tuple()


@dataclass
class LayerCallables:
    """User-provided callable description for each layer step."""
    # MoE 4-node mode:
    attn_fn: Callable = None          # (h) -> residual, dispatch inputs, aux, shared
    dispatch_fn: Callable = None      # dispatch communication only
    expert_fn: Callable = None        # dispatch postprocess + routed expert + combine preprocess
    combine_fn: Callable = None       # (expert_outs) -> h_out; residual/shared state in step_data

    # Dense-only:
    body_fn: Callable = None          # (h) -> h_out (wraps FFN)

    is_moe: bool = False
    step_data: dict = field(default_factory=dict)

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
                 early_attn_memory_release=False):
        self.parallel_module = parallel_module
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self._use_4node = True
        self.early_attn_memory_release = early_attn_memory_release

        # Async overlap: launch COMM and COMP ops from separate CPU threads
        # so their kernel launches can interleave, enabling true GPU overlap.
        # Controlled by ASYNC_4PHASE env var (default=1, set 0 to disable).
        _async_on = os.environ.get('ASYNC_4PHASE', '1') not in ('0', '')
        self._async_pool = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix='moe-overlap')
            if _async_on else None
        )

        # Pre-allocate CUDA events for cross-stream synchronization.
        # Using separate events per barrier slot avoids re-recording while
        # a previous wait may still be pending (undefined behavior per CUDA spec).
        # 3 barrier slots in _merged_step_4phase, each needs comp + comm events
        self._barrier_comp_evts = [torch.cuda.Event() for _ in range(3)]
        self._barrier_comm_evts = [torch.cuda.Event() for _ in range(3)]
        # 2 uni-directional sync events (comp->comm and comm->comp)
        self._sync_c2m_evt = torch.cuda.Event()
        self._sync_m2c_evt = torch.cuda.Event()

        self._timing_enabled = (
            os.environ.get('NNSCALER_MOE_OVERLAP_TIMING', '0') not in ('0', '')
        )
        self._timing_log_every = max(
            1, int(os.environ.get('NNSCALER_MOE_OVERLAP_TIMING_EVERY', '1')))
        self._timing_warmup = max(
            0, int(os.environ.get('NNSCALER_MOE_OVERLAP_TIMING_WARMUP', '0')))
        self._timing_rank_filter = os.environ.get(
            'NNSCALER_MOE_OVERLAP_TIMING_RANKS', '0')
        self._timing_run_idx = 0
        self._timing_current_records = None
        self._timing_pending_records = []

    def shutdown(self):
        """Release scheduler-owned CPU worker resources."""
        self._timing_flush(force=True)
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

    @staticmethod
    def _timing_percentile(values, pct):
        if not values:
            return 0.0
        values = sorted(values)
        index = max(0, min(len(values) - 1, math.ceil(len(values) * pct / 100.0) - 1))
        return values[index]

    @staticmethod
    def _timing_rank():
        dist = torch.distributed
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0

    def _timing_should_log_rank(self):
        rank_filter = self._timing_rank_filter.strip().lower()
        if rank_filter in ('all', '*'):
            return True
        rank = self._timing_rank()
        try:
            return rank in {int(item) for item in rank_filter.split(',') if item.strip()}
        except ValueError:
            return rank == 0

    def _timing_begin_run(self):
        if not self._timing_enabled:
            self._timing_current_records = None
            return
        self._timing_run_idx += 1
        if self._timing_run_idx <= self._timing_warmup:
            self._timing_current_records = None
            return
        if not self._timing_should_log_rank():
            self._timing_current_records = None
            return
        self._timing_current_records = []

    def _timing_record_start(self, name):
        if self._timing_current_records is None:
            return None
        record = {
            'name': name,
            'wall_comp_start': torch.cuda.Event(enable_timing=True),
            'wall_comp_end': torch.cuda.Event(enable_timing=True),
            'wall_comm_start': torch.cuda.Event(enable_timing=True),
            'wall_comm_end': torch.cuda.Event(enable_timing=True),
            'comp_start': None,
            'comp_end': None,
            'comm_start': None,
            'comm_end': None,
        }
        record['wall_comp_start'].record(get_comp_stream())
        record['wall_comm_start'].record(get_comm_stream())
        return record

    def _timing_record_end(self, record):
        if record is None:
            return
        record['wall_comp_end'].record(get_comp_stream())
        record['wall_comm_end'].record(get_comm_stream())
        self._timing_current_records.append(record)

    def _timing_branch_start(self, record, branch):
        if record is None:
            return
        stream = get_comp_stream() if branch == 'comp' else get_comm_stream()
        record[f'{branch}_start'] = torch.cuda.Event(enable_timing=True)
        record[f'{branch}_end'] = torch.cuda.Event(enable_timing=True)
        record[f'{branch}_start'].record(stream)

    def _timing_branch_end(self, record, branch):
        if record is None:
            return
        stream = get_comp_stream() if branch == 'comp' else get_comm_stream()
        record[f'{branch}_end'].record(stream)

    def _timing_call(self, record, branch, func, *args):
        if record is None:
            return func(*args)
        self._timing_branch_start(record, branch)
        try:
            return func(*args)
        finally:
            self._timing_branch_end(record, branch)

    def _timing_finish_run(self):
        records = self._timing_current_records
        self._timing_current_records = None
        if not records:
            return
        self._timing_pending_records.extend(records)
        completed_runs = self._timing_run_idx - self._timing_warmup
        if completed_runs > 0 and completed_runs % self._timing_log_every == 0:
            self._timing_flush()

    def _timing_flush(self, force=False):
        if not self._timing_enabled or not self._timing_pending_records:
            return
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            self._timing_pending_records.clear()
            return

        torch.cuda.synchronize()
        by_name = defaultdict(lambda: defaultdict(list))
        for record in self._timing_pending_records:
            wall_comp_ms = record['wall_comp_start'].elapsed_time(record['wall_comp_end'])
            wall_comm_ms = record['wall_comm_start'].elapsed_time(record['wall_comm_end'])
            if record['comp_start'] is not None:
                comp_ms = record['comp_start'].elapsed_time(record['comp_end'])
            else:
                comp_ms = wall_comp_ms
            if record['comm_start'] is not None:
                comm_ms = record['comm_start'].elapsed_time(record['comm_end'])
            else:
                comm_ms = wall_comm_ms
            by_name[record['name']]['comp'].append(comp_ms)
            by_name[record['name']]['comm'].append(comm_ms)
            by_name[record['name']]['wall'].append(max(wall_comp_ms, wall_comm_ms))
            by_name[record['name']]['imbalance'].append(abs(comp_ms - comm_ms))

        rank = self._timing_rank()
        lines = [
            f"[NNSCALER_MOE_OVERLAP_TIMING rank={rank} run={self._timing_run_idx} "
            f"records={len(self._timing_pending_records)}]",
            "name                         n  comp_p50  comp_p95  comm_p50  comm_p95  wall_p50  wall_p95  imbalance_p50  ms",
        ]
        for name in sorted(by_name):
            stats = by_name[name]
            n = len(stats['wall'])
            lines.append(
                f"{name:<28} {n:4d} "
                f"{self._timing_percentile(stats['comp'], 50):9.3f} "
                f"{self._timing_percentile(stats['comp'], 95):9.3f} "
                f"{self._timing_percentile(stats['comm'], 50):9.3f} "
                f"{self._timing_percentile(stats['comm'], 95):9.3f} "
                f"{self._timing_percentile(stats['wall'], 50):9.3f} "
                f"{self._timing_percentile(stats['wall'], 95):9.3f} "
                f"{self._timing_percentile(stats['imbalance'], 50):13.3f}"
            )
        print('\n'.join(lines), flush=True)
        self._timing_pending_records.clear()

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
        self._timing_begin_run()

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
        _mark_tensors_stream(h0, get_comp_stream())
        _embed_h_list = [h0]  # Save embedding outputs for backward

        lc_list_0 = []
        for si in range(num_steps):
            lc_list_0.append(layer_callables_fn(si, samples[0]))

        h0, all_nodes_0, rmaps_0, eprobs_0 = self._forward_all_layers(
            h0, lc_list_0, events[0])

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
            _mark_tensors_stream(fwd_h, get_comm_stream())
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
                    fwd_idx += 1
                    continue

                bwd_entry = prev_all_nodes[bwd_idx]
                if isinstance(bwd_entry, tuple) and len(bwd_entry) == 2 and bwd_entry[0] == 'special':
                    bwd_lc = bwd_entry[1]
                    with nnscaler.sync_grad_when(False):
                        with torch.cuda.stream(get_comp_stream()):
                            grad_h = bwd_lc.special_backward(grad_h)
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
                    fwd_routing_maps.append(fwd_lc.step_data.get('routing_map'))
                    fwd_expert_probs.append(fwd_lc.step_data.get('gate_scores'))

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
                    fwd_routing_maps.append(fwd_lc.step_data.get('routing_map'))
                    fwd_expert_probs.append(fwd_lc.step_data.get('gate_scores'))
                fwd_idx += 1

            while bwd_idx >= 0:
                bwd_entry = prev_all_nodes[bwd_idx]
                if isinstance(bwd_entry, tuple) and len(bwd_entry) == 2 and bwd_entry[0] == 'special':
                    bwd_lc = bwd_entry[1]
                    with nnscaler.sync_grad_when(False):
                        with torch.cuda.stream(get_comp_stream()):
                            grad_h = bwd_lc.special_backward(grad_h)
                    prev_all_nodes[bwd_idx] = None
                    bwd_idx -= 1
                    continue
                if bwd_entry is None:
                    bwd_idx -= 1
                    continue
                with nnscaler.sync_grad_when(False):
                    grad_h = self._backward_entry(bwd_entry, grad_h)
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
                    lc = entry[1]
                    with torch.cuda.stream(get_comp_stream()):
                        grad_h = lc.special_backward(grad_h)
                    prev_all_nodes[i] = None
                    continue
                if entry is None:
                    continue
                grad_h = self._backward_entry(entry, grad_h)
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
        self._timing_finish_run()

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

        combine_node = ScheduleNode(
            lc.combine_fn, comm_stream, event,
            free_input=True, name="combine", checkpoint=False)
        combine_node.step_data = lc.step_data

        return (attn_node, dispatch_node, expert_node, combine_node)

    def _sync_comm_to_comp(self):
        self._sync_m2c_evt.record(get_comm_stream())
        get_comp_stream().wait_event(self._sync_m2c_evt)

    def _sync_comp_to_comm(self):
        self._sync_c2m_evt.record(get_comp_stream())
        get_comm_stream().wait_event(self._sync_c2m_evt)

    def _cross_stream_barrier(self, slot=0):
        """Bidirectional sync: each stream waits for the other's current position.

        Records events on both streams simultaneously, then makes each wait
        for the other. Uses pre-allocated events indexed by slot to avoid
        event creation overhead and re-recording conflicts.
        """
        comp_evt = self._barrier_comp_evts[slot]
        comm_evt = self._barrier_comm_evts[slot]
        comp_evt.record(get_comp_stream())
        comm_evt.record(get_comm_stream())
        get_comm_stream().wait_event(comp_evt)
        get_comp_stream().wait_event(comm_evt)

    def _order_cross_stream_for_comp(self, tensors):
        """Compatibility hook; ScheduleNode records actual cross-stream uses."""
        return None

    @staticmethod
    def _replace_node_output_indexes(node, replacements):
        if not node.output_is_tuple() or node.output is None:
            return
        outputs = list(node.output)
        for index, value in replacements.items():
            if -len(outputs) <= index < len(outputs):
                outputs[index] = value
        node.output = tuple(outputs)

    def _prepare_combine_state(self, attn_node, lc, h_residual=None, shared_expert_out=None):
        replacements = {}
        if h_residual is not None:
            lc.step_data['_combine_residual'] = attn_node.detach(h_residual)
            replacements[0] = None
        if shared_expert_out is not None:
            lc.step_data['_combine_shared_expert_out'] = attn_node.detach(shared_expert_out)
            replacements[-1] = None
        self._replace_node_output_indexes(attn_node, replacements)

    def _prepare_dispatch_state(self, dispatch_node, lc, dispatch_out):
        if not isinstance(dispatch_out, tuple):
            return dispatch_out
        if len(dispatch_out) < 2:
            return dispatch_out[0] if len(dispatch_out) == 1 else dispatch_out

        dispatched_tokens, dispatched_probs = dispatch_out[:2]
        if isinstance(dispatched_probs, torch.Tensor):
            lc.step_data['_dispatched_probs'] = dispatch_node.detach(dispatched_probs)
            replacements = {1: None}
            self._replace_node_output_indexes(dispatch_node, replacements)
        return dispatched_tokens

    def _collect_combine_grads(self, combine_node, combine_grads):
        """Collect grads from combine's detached layer-state tensors."""
        if isinstance(combine_grads, tuple) and len(combine_grads) == 3:
            return combine_grads

        grad_expert_out = combine_grads[0] if isinstance(combine_grads, tuple) else combine_grads
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

        return grad_expert_out, grad_residual, grad_shared

    def _prepare_loss_aux_tensors(self, attn_node, lc, attn_out):
        aux_tensors = lc.step_data.get('_loss_aux_tensors')
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
                "(h_residual, dispatch_tokens, dispatch_probs).")

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

        for key, value in list(lc.step_data.items()):
            replaced, changed = self._replace_tensors_by_identity(value, replacements)
            if changed:
                lc.step_data[key] = replaced
        lc.step_data['_loss_aux_tensors'] = tuple(grad_receivers)
        attn_node.loss_aux_tensors = tuple(grad_receivers)

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
    def _attn_backward_grads(attn_node, grad_residual, grad_dispatch_tokens,
                             grad_dispatch_probs, grad_shared=None):
        if not attn_node.output_is_tuple():
            return grad_residual

        output_arity = attn_node.output_arity()
        grads = [grad_residual, grad_dispatch_tokens, grad_dispatch_probs]
        aux_tensors = getattr(attn_node, 'loss_aux_tensors', ())
        aux_count = max(0, output_arity - 4, len(aux_tensors))
        for idx in range(aux_count):
            tensor = aux_tensors[idx] if idx < len(aux_tensors) else None
            if isinstance(tensor, torch.Tensor):
                grads.append(tensor.grad)
            else:
                grads.append(None)
        grads.append(grad_shared)
        while len(grads) < output_arity:
            grads.append(None)
        return tuple(grads[:output_arity])

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
                routing_maps.append(lc.step_data.get('routing_map'))
                expert_probs.append(lc.step_data.get('gate_scores'))

            all_nodes.append(entry)

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

        attn_fn owns router/preprocess/shared expert. dispatch_fn is the
        communication node, expert_fn produces combine-ready expert output,
        and combine_fn reads detached residual/shared state from step_data.
        """
        nodes = self._create_nodes_4(lc, event)
        attn_n, dispatch_n, expert_n, combine_n = nodes

        attn_out = attn_n.forward((h,))
        h_residual, dispatch_tokens, dispatch_probs = attn_out[:3]
        shared_expert_out = attn_out[-1] if len(attn_out) > 3 else None
        self._prepare_combine_state(
            attn_n, lc, h_residual=h_residual, shared_expert_out=shared_expert_out)
        self._prepare_loss_aux_tensors(attn_n, lc, attn_out)
        dispatch_out = dispatch_n.forward((dispatch_tokens, dispatch_probs))
        del attn_out, h_residual, shared_expert_out, dispatch_tokens, dispatch_probs
        dispatch_out = self._prepare_dispatch_state(dispatch_n, lc, dispatch_out)
        expert_out = expert_n.forward(dispatch_out)
        h_out = combine_n.forward((expert_out,))

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

        Shared expert backward is part of attn_node because shared forward runs
        in attn_fn; expert backward only returns dispatch output gradients.
        """
        attn_n, dispatch_n, expert_n, combine_n = nodes

        # Ensure all intermediate ops run on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        with torch.cuda.stream(get_comp_stream()):
            self._sync_comp_to_comm()

            # combine_grads: (grad_expert_outs, grad_h_residual, grad_shared_expert_out)
            combine_grads = combine_n.backward(grad_h)
            combine_grads = self._collect_combine_grads(combine_n, combine_grads)
            # expert_grads: grad for dispatch output tokens; detached probs grad
            # is collected from dispatch_node.detach() during dispatch backward.
            expert_grads = expert_n.backward(combine_grads[0])
            if not isinstance(expert_grads, tuple):
                expert_grads = (expert_grads,)
            dispatch_grads = dispatch_n.backward(expert_grads)
            if not isinstance(dispatch_grads, tuple):
                dispatch_grads = (dispatch_grads,)

            self._sync_comm_to_comp()

            attn_grads = self._attn_backward_grads(
                attn_n, combine_grads[1], dispatch_grads[0], dispatch_grads[1],
                combine_grads[2])
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

        fwd_nodes = self._create_nodes(fwd_lc, fwd_event)
        fwd_attn, fwd_body = fwd_nodes

        # Ensure all intermediate ops run on COMP stream, not default stream.
        with torch.cuda.stream(get_comp_stream()):
            body_grads = bwd_body.backward(grad_h)
            fwd_attn_out = fwd_attn.forward((fwd_h,))

            grad_x = bwd_attn.backward(body_grads)
            fwd_h_out = fwd_body.forward(fwd_attn_out)

            if fwd_attn.checkpoint:
                fwd_attn.output = None
            if fwd_body.checkpoint:
                fwd_body.output = None

        return fwd_h_out, grad_x, ('layer2', fwd_nodes)

    def _merged_step_4phase(self, bwd_nodes, fwd_lc, fwd_event, grad_h, fwd_h):
        """4-phase overlap for MoE layers.

        Each phase interleaves COMP and COMM operations from different
        micro-batches. Within each phase, COMP and COMM run in PARALLEL on
        the GPU because we skip the per-node shared-event protocol (which
        would serialize all operations). Instead, we use explicit cross-stream
        barriers at phase boundaries to enforce data dependencies.

        Phase 1: f_attn_router(COMP) || b_combine(COMM)   [attn includes shared fwd]
        Phase 2: b_expert    (COMP) || f_dispatch (COMM)   [dispatch = communication]
        Phase 3: f_expert    (COMP) || b_dispatch (COMM)   [expert excludes shared]
        Phase 4: b_attn      (COMP) || f_combine  (COMM)
        """
        bwd_attn, bwd_dispatch, bwd_expert, bwd_combine = bwd_nodes
        fwd_nodes = self._create_nodes_4(fwd_lc, fwd_event)
        fwd_attn, fwd_dispatch, fwd_expert, fwd_combine = fwd_nodes

        pool = self._async_pool  # None in sequential mode

        # --- Skip per-node event protocol during merged step ---
        # The default shared event serializes all nodes (COMP waits for COMM
        # and vice versa), preventing within-phase parallelism. By setting
        # _skip_event=True, each node still switches to its own stream, but
        # does not wait/record the shared event. Cross-stream dependencies are
        # enforced explicitly via _cross_stream_barrier() at phase boundaries.
        all_nodes = (*bwd_nodes, *fwd_nodes)
        for n in all_nodes:
            n._skip_event = True

        # Ensure grad addition runs on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        # ScheduleNode calls internally switch to their own stream and restore on exit.
        with torch.cuda.stream(get_comp_stream()):
            # Initial sync: COMP→COMM so COMM can read grad_h (from loss_bwd on COMP)
            self._sync_comp_to_comm()

            # Phase 1: COMM(b_combine) || COMP(f_attn_router/shared)
            timing_record = self._timing_record_start('phase1_f_attn_b_combine')
            if pool is not None:
                fut_combine = pool.submit(
                    self._timing_call, timing_record, 'comm', bwd_combine.backward, grad_h)
                fwd_attn_out = self._timing_call(
                    timing_record, 'comp', fwd_attn.forward, (fwd_h,))
                combine_grads = fut_combine.result()
            else:
                combine_grads = self._timing_call(
                    timing_record, 'comm', bwd_combine.backward, grad_h)
                fwd_attn_out = self._timing_call(
                    timing_record, 'comp', fwd_attn.forward, (fwd_h,))
            self._timing_record_end(timing_record)
            combine_grads = self._collect_combine_grads(bwd_combine, combine_grads)
            fwd_h_residual, fwd_dispatch_tokens, fwd_dispatch_probs = fwd_attn_out[:3]
            fwd_shared_expert_out = fwd_attn_out[-1] if len(fwd_attn_out) > 3 else None
            self._prepare_combine_state(
                fwd_attn,
                fwd_lc,
                h_residual=fwd_h_residual,
                shared_expert_out=fwd_shared_expert_out,
            )
            self._prepare_loss_aux_tensors(fwd_attn, fwd_lc, fwd_attn_out)
            del fwd_attn_out, fwd_h_residual, fwd_shared_expert_out

            # Phase boundary: Phase 2 COMP needs COMM output (combine_grads),
            # Phase 2 COMM needs COMP output (attn_out + precomputed metadata)
            timing_record = self._timing_record_start('barrier0_p1_to_p2')
            self._cross_stream_barrier(slot=0)
            self._timing_record_end(timing_record)

            # Phase 2: COMM(f_dispatch) || COMP(b_expert)
            # combine_grads: (grad_expert_outs, grad_h_residual, grad_shared_expert_out)
            timing_record = self._timing_record_start('phase2_b_expert_f_dispatch')
            if pool is not None:
                fut_dispatch = pool.submit(
                    self._timing_call, timing_record, 'comm',
                    fwd_dispatch.forward, (fwd_dispatch_tokens, fwd_dispatch_probs))
                expert_grads = self._timing_call(
                    timing_record, 'comp', bwd_expert.backward, combine_grads[0])
                fwd_dispatch_out = fut_dispatch.result()
            else:
                fwd_dispatch_out = self._timing_call(
                    timing_record, 'comm', fwd_dispatch.forward,
                    (fwd_dispatch_tokens, fwd_dispatch_probs))
                expert_grads = self._timing_call(
                    timing_record, 'comp', bwd_expert.backward, combine_grads[0])
            self._timing_record_end(timing_record)
            if not isinstance(expert_grads, tuple):
                expert_grads = (expert_grads,)
            # expert_grads: grad for dispatch output tokens; detached probs grad
            # is collected from dispatch_node.detach() during dispatch backward.
            fwd_dispatch_out = self._prepare_dispatch_state(
                fwd_dispatch, fwd_lc, fwd_dispatch_out)
            del fwd_dispatch_tokens, fwd_dispatch_probs

            # Phase boundary: Phase 3 COMP needs COMM output (dispatch_out),
            # Phase 3 COMM needs COMP output (expert_grads)
            timing_record = self._timing_record_start('barrier1_p2_to_p3')
            self._cross_stream_barrier(slot=1)
            self._timing_record_end(timing_record)

            if self.early_attn_memory_release:
                # Run backward dispatch and attention before later forward
                # expert/combine allocations. This lowers peak memory at the
                # cost of Phase-3/4 overlap.
                timing_record = self._timing_record_start('phase3_early_b_dispatch_b_attn')
                if pool is not None:
                    fut_dispatch_bwd = pool.submit(
                        self._timing_call, timing_record, 'comm',
                        bwd_dispatch.backward, expert_grads)
                    dispatch_grads = fut_dispatch_bwd.result()
                else:
                    dispatch_grads = self._timing_call(
                        timing_record, 'comm', bwd_dispatch.backward, expert_grads)
                if not isinstance(dispatch_grads, tuple):
                    dispatch_grads = (dispatch_grads,)

                self._sync_comm_to_comp()
                attn_grads = self._attn_backward_grads(
                    bwd_attn, combine_grads[1], dispatch_grads[0], dispatch_grads[1],
                    combine_grads[2])
                grad_x = self._timing_call(
                    timing_record, 'comp', bwd_attn.backward, attn_grads)
                self._order_cross_stream_for_comp(dispatch_grads)
                self._order_cross_stream_for_comp(combine_grads)
                self._timing_record_end(timing_record)

                timing_record = self._timing_record_start('phase4_early_f_expert_f_combine')
                fwd_expert_out = self._timing_call(
                    timing_record, 'comp', fwd_expert.forward, fwd_dispatch_out)
                self._sync_comp_to_comm()
                fwd_h_out = self._timing_call(
                    timing_record, 'comm', fwd_combine.forward, (fwd_expert_out,))
                self._timing_record_end(timing_record)

                for n in fwd_nodes:
                    if n.checkpoint:
                        n.output = None

                self._sync_comm_to_comp()

                for n in fwd_nodes:
                    n._skip_event = False

                return fwd_h_out, grad_x, ('layer4', fwd_nodes)

            # Phase 3: COMM(b_dispatch) || COMP(f_expert)
            timing_record = self._timing_record_start('phase3_f_expert_b_dispatch')
            if pool is not None:
                fut_dispatch_bwd = pool.submit(
                    self._timing_call, timing_record, 'comm',
                    bwd_dispatch.backward, expert_grads)
                fwd_expert_out = self._timing_call(
                    timing_record, 'comp', fwd_expert.forward, fwd_dispatch_out)
                dispatch_grads = fut_dispatch_bwd.result()
            else:
                dispatch_grads = self._timing_call(
                    timing_record, 'comm', bwd_dispatch.backward, expert_grads)
                fwd_expert_out = self._timing_call(
                    timing_record, 'comp', fwd_expert.forward, fwd_dispatch_out)
            self._timing_record_end(timing_record)
            if not isinstance(dispatch_grads, tuple):
                dispatch_grads = (dispatch_grads,)

            # Phase boundary: Phase 4 COMP needs COMM output (dispatch_grads),
            # Phase 4 COMM needs COMP output (combine-ready expert_out)
            timing_record = self._timing_record_start('barrier2_p3_to_p4')
            self._cross_stream_barrier(slot=2)
            self._timing_record_end(timing_record)

            # Phase 4: COMM(f_combine) || COMP(b_attn)
            timing_record = self._timing_record_start('phase4_b_attn_f_combine')
            if pool is not None:
                # Launch combine on the host before attention backward.  The
                # combine node has free_input=True and releases fwd_expert_out
                # immediately after its COMM-stream kernels are enqueued.  If
                # the worker thread is merely submitted here, no-timing runs can
                # let b_attn enqueue first and hit a higher transient peak; CUDA
                # event timing accidentally hid that by slowing the main thread.
                fwd_h_out = self._timing_call(
                    timing_record, 'comm', fwd_combine.forward, (fwd_expert_out,))
                attn_grads = self._attn_backward_grads(
                    bwd_attn, combine_grads[1], dispatch_grads[0], dispatch_grads[1],
                    combine_grads[2])
                grad_x = self._timing_call(
                    timing_record, 'comp', bwd_attn.backward, attn_grads)
                self._order_cross_stream_for_comp(dispatch_grads)
                self._order_cross_stream_for_comp(combine_grads)
            else:
                fwd_h_out = self._timing_call(
                    timing_record, 'comm', fwd_combine.forward, (fwd_expert_out,))
                attn_grads = self._attn_backward_grads(
                    bwd_attn, combine_grads[1], dispatch_grads[0], dispatch_grads[1],
                    combine_grads[2])
                grad_x = self._timing_call(
                    timing_record, 'comp', bwd_attn.backward, attn_grads)
                self._order_cross_stream_for_comp(dispatch_grads)
                self._order_cross_stream_for_comp(combine_grads)
            self._timing_record_end(timing_record)

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
        bwd_tag, bwd_nodes = bwd_entry

        if self._use_4node and bwd_tag == 'layer4' and fwd_lc.is_moe:
            return self._merged_step_4phase(bwd_nodes, fwd_lc, fwd_event, grad_h, fwd_h)
        elif bwd_tag == 'layer2':
            if fwd_lc.is_moe:
                raise NotImplementedError(
                    "Merging dense backward with MoE forward is unsupported. "
                    "The MoE forward must run through the 4-node schedule.")
            return self._merged_step(bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h)
        else:
            with nnscaler.sync_grad_when(False):
                grad_x = self._backward_entry(bwd_entry, grad_h)
            if fwd_lc.is_moe and self._use_4node:
                fwd_h_out, fwd_entry = self._forward_single_layer_4node(fwd_h, fwd_lc, fwd_event)
            else:
                fwd_h_out, fwd_entry = self._forward_single_layer(fwd_h, fwd_lc, fwd_event)
            return fwd_h_out, grad_x, fwd_entry
