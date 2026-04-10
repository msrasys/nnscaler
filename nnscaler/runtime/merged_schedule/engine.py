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
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.autograd import Variable

import nnscaler

from .utils import sanitize_grad

# Suppress AccumulateGrad stream mismatch warning in dual-stream mode.
if hasattr(torch.autograd.graph, 'set_warn_on_accumulate_grad_stream_mismatch'):
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

_logger = logging.getLogger(__name__)

_COMP_STREAM = None   # computation stream (default stream)
_COMM_STREAM = None   # communication stream (side stream)
_IN_RECOMPUTE = False  # Flag: True when inside ScheduleNode checkpoint recompute
_SEQUENTIAL_MODE = False  # Flag: skip stream/event ops in sequential mode
_PROFILE_OVERLAP = os.environ.get('PROFILE_OVERLAP', '0') not in ('0', '')
_profile_data = []  # collected per merged-step timing records
_TIMING_BREAKDOWN = os.environ.get('TIMING_BREAKDOWN', '0') not in ('0', '')
_LAYER_CHECKSUM = bool(os.environ.get('LAYER_CHECKSUM'))



def _log_checksum(tag, tensor):
    """Log tensor checksum for precision debugging."""
    if not _LAYER_CHECKSUM:
        return
    with torch.no_grad():
        t = tensor.float()
        s = t.sum().item()
        n = t.norm().item()
        mx = t.abs().max().item()
    _logger.info(f"[CHECKSUM] {tag} sum={s:.15e} norm={n:.15e} max={mx:.15e}")



class _TimingCtx:
    """Lightweight timing context for run() breakdown using CUDA events.

    Uses CUDA events with enable_timing=True instead of torch.cuda.synchronize()
    + perf_counter. This avoids ~15 device-wide syncs per step that distort
    profiling results and make overlap debugging impossible.

    Only one synchronize happens: inside report(), after all work is done.

    Usage:
        tb = _TimingCtx(enabled)
        tb.start()
        ... code ...
        tb.mark('phase_name')  # records a CUDA event (no sync)
        ... more code ...
        tb.mark('next_phase')
        tb.report()             # single sync + elapsed_time() for all segments
    """

    def __init__(self, enabled):
        self.enabled = enabled
        self._events = []       # [(name, start_event, end_event)]
        self._last_event = None

    def start(self):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._last_event = ev

    def mark(self, name):
        if not self.enabled:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self._events.append((name, self._last_event, ev))
        self._last_event = ev

    def report(self, label="TIMING BREAKDOWN"):
        if not self.enabled or not self._events:
            return
        torch.cuda.synchronize()
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank != 0:
            return
        records = []
        for name, start_ev, end_ev in self._events:
            ms = start_ev.elapsed_time(end_ev)
            records.append((name, ms / 1000.0))
        total = sum(t for _, t in records)
        lines = [f"=== {label} ==="]
        for name, t in records:
            pct = t / total * 100 if total > 0 else 0
            lines.append(f"  {name:30s}: {t:8.3f}s  ({pct:5.1f}%)")
        lines.append(f"  {'TOTAL':30s}: {total:8.3f}s")
        _logger.info("\n".join(lines))


def _profile_report():
    """Log overlap profiling summary. Called after torch.cuda.synchronize()."""
    if not _profile_data:
        return
    import torch.distributed as dist
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        _profile_data.clear()
        return

    sequential = _profile_data[0].get('sequential', False)
    phase_names = ['P1(b_combine||f_attn)', 'P2(b_expert||f_dispatch)',
                   'P3(b_dispatch||f_expert)', 'P4(b_attn||f_combine)']

    # Compute timings from stored events
    records = []
    for raw in _profile_data:
        ce, me = raw['ce'], raw['me']
        phases = []
        for pi in range(4):
            comp_ms = ce[pi].elapsed_time(ce[pi + 1])
            if sequential:
                # Both streams are the same; comp_ms == serialized total for this phase
                phases.append({'total_ms': comp_ms})
            else:
                comm_ms = me[pi].elapsed_time(me[pi + 1])
                wall_ms = max(comp_ms, comm_ms)
                phases.append({'comp_ms': comp_ms, 'comm_ms': comm_ms, 'wall_ms': wall_ms})
        if sequential:
            total = sum(p['total_ms'] for p in phases)
            records.append({'total_ms': total, 'phases': phases})
        else:
            total_comp = sum(p['comp_ms'] for p in phases)
            total_comm = sum(p['comm_ms'] for p in phases)
            total_wall = max(ce[0].elapsed_time(ce[4]), me[0].elapsed_time(me[4]))
            records.append({
                'wall_ms': total_wall, 'comp_ms': total_comp, 'comm_ms': total_comm,
                'phases': phases
            })

    n = len(records)

    if sequential:
        avg_total = sum(r['total_ms'] for r in records) / n
        _logger.info(
            f"[PROFILE-SEQ] merged step (n={n}): total={avg_total:.2f}ms (serialized, no overlap)"
        )
        for pi in range(4):
            avg_phase = sum(r['phases'][pi]['total_ms'] for r in records) / n
            _logger.info(f"  {phase_names[pi]}: total={avg_phase:.2f}ms")
    else:
        total_wall = sum(r['wall_ms'] for r in records)
        total_comp = sum(r['comp_ms'] for r in records)
        total_comm = sum(r['comm_ms'] for r in records)
        avg_wall = total_wall / n
        avg_comp = total_comp / n
        avg_comm = total_comm / n
        avg_overlap_pct = (avg_comp + avg_comm - avg_wall) / avg_wall * 100 if avg_wall > 0 else 0
        _logger.info(
            f"[PROFILE] 4-phase merged step (n={n}): "
            f"wall={avg_wall:.2f}ms, comp={avg_comp:.2f}ms, comm={avg_comm:.2f}ms, "
            f"measured_overlap={avg_overlap_pct:.1f}%"
        )
        for pi in range(4):
            comp_times = [r['phases'][pi]['comp_ms'] for r in records]
            comm_times = [r['phases'][pi]['comm_ms'] for r in records]
            wall_times = [r['phases'][pi]['wall_ms'] for r in records]
            ac = sum(comp_times) / n
            am = sum(comm_times) / n
            aw = sum(wall_times) / n
            op = (ac + am - aw) / aw * 100 if aw > 0 else 0
            _logger.info(
                f"  {phase_names[pi]}: wall={aw:.2f}ms comp={ac:.2f}ms comm={am:.2f}ms measured_overlap={op:.1f}%"
            )
    _profile_data.clear()


def set_streams(sequential_mode=False):
    """Initialize global COMP/COMM streams.
    In sequential_mode both streams are the default stream (no overlap).
    In overlap mode BOTH streams are non-default so that CUDA's legacy
    default-stream implicit sync does not serialise them.
    """
    global _COMP_STREAM, _COMM_STREAM, _SEQUENTIAL_MODE
    _SEQUENTIAL_MODE = sequential_mode
    if sequential_mode:
        _COMP_STREAM = torch.cuda.default_stream()
        _COMM_STREAM = _COMP_STREAM
    else:
        _COMP_STREAM = torch.cuda.Stream()
        # DEBUG: use same stream for both to test if dual-stream is the issue
        if os.environ.get('DEBUG_SINGLE_STREAM'):
            _COMM_STREAM = _COMP_STREAM
        else:
            _COMM_STREAM = torch.cuda.Stream()


def get_comp_stream():
    assert _COMP_STREAM is not None, "call set_streams() first"
    return _COMP_STREAM


def get_comm_stream():
    assert _COMM_STREAM is not None, "call set_streams() first"
    return _COMM_STREAM


def _make_viewless(t):
    """Ensure tensor has its own storage view (avoids grad-engine pitfalls)."""
    if isinstance(t, torch.Tensor) and t._base is not None:
        return t.clone()
    return t


def _detach_tensor(t):
    """Detach tensor, mark requires_grad, and ensure own storage view."""
    if t is None:
        return None
    d = _make_viewless(t).detach()
    d.requires_grad = t.requires_grad
    return d


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
        self._skip_event = False  # When True, skip event wait/record in _stream_ctx

    def forward(self, inputs=()):
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        return self._forward(*inputs)

    def _forward(self, *inputs):
        if _SEQUENTIAL_MODE:
            self.inputs = []
            for inp in inputs:
                if inp is None:
                    self.inputs.append(None)
                elif isinstance(inp, torch.Tensor):
                    d = inp.detach()
                    d.requires_grad = inp.requires_grad
                    self.inputs.append(d)
                else:
                    self.inputs.append(inp)

            if self.checkpoint:
                with torch.no_grad():
                    data = self.forward_func(*self.inputs)
                if not isinstance(data, tuple):
                    data = data.detach().requires_grad_(True) if isinstance(data, torch.Tensor) else data
                else:
                    data = tuple(
                        e.detach().requires_grad_(True) if isinstance(e, torch.Tensor) else e
                        for e in data
                    )
            else:
                data = self.forward_func(*self.inputs)

            self.output = data
            return self.output

        # Record cross-stream tensor usage for CUDA caching allocator safety.
        # Inputs may have been produced on a different stream; tell the allocator
        # that this stream will also read from their underlying storage.
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                inp.record_stream(self.stream)

        with self._stream_ctx(f"{self.name} fwd"):
            self.inputs = []
            for inp in inputs:
                if inp is None:
                    self.inputs.append(None)
                elif isinstance(inp, torch.Tensor):
                    d = inp.detach()
                    d.requires_grad = inp.requires_grad
                    self.inputs.append(d)
                else:
                    self.inputs.append(inp)

            if self.checkpoint:
                with torch.no_grad():
                    data = self.forward_func(*self.inputs)
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

            self.output = data

        if self.free_input:
            for inp in inputs:
                if inp is not None:
                    inp.record_stream(self.stream)
                    inp.untyped_storage().resize_(0)

        return self.output

    def get_output(self):
        return self.output

    def backward(self, output_grad, retain_graph=False):
        if not isinstance(output_grad, tuple):
            output_grad = (output_grad,)
        return self._backward(*output_grad, retain_graph=retain_graph)

    def _backward(self, *output_grad, retain_graph=False):
        global _IN_RECOMPUTE
        if _SEQUENTIAL_MODE:
            if self.checkpoint:
                _IN_RECOMPUTE = True
                try:
                    recomputed = self.forward_func(*self.inputs)
                finally:
                    _IN_RECOMPUTE = False
                if not isinstance(recomputed, tuple):
                    recomputed = (recomputed,)
                tensor_outputs = []
                tensor_grads = []
                for i, out in enumerate(recomputed):
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        g = output_grad[i] if i < len(output_grad) else None
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
                        g = output_grad[i] if i < len(output_grad) else None
                        tensor_outputs.append(out)
                        tensor_grads.append(g)
                if tensor_outputs:
                    self.backward_func(tuple(tensor_outputs), tuple(tensor_grads),
                                       retain_graph=retain_graph)
            grads = self.get_grad()
            self._release()
            return grads

        # Record cross-stream tensor usage for output_grad tensors.
        for g in output_grad:
            if isinstance(g, torch.Tensor):
                g.record_stream(self.stream)

        with self._stream_ctx(f"{self.name} bwd"):
            if self.checkpoint:
                _IN_RECOMPUTE = True
                try:
                    recomputed = self.forward_func(*self.inputs)
                finally:
                    _IN_RECOMPUTE = False
                if not isinstance(recomputed, tuple):
                    recomputed = (recomputed,)

                tensor_outputs = []
                tensor_grads = []
                for i, out in enumerate(recomputed):
                    if isinstance(out, torch.Tensor) and out.requires_grad:
                        g = output_grad[i] if i < len(output_grad) else None
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
                        g = output_grad[i] if i < len(output_grad) else None
                        tensor_outputs.append(out)
                        tensor_grads.append(g)
                if tensor_outputs:
                    self.backward_func(tuple(tensor_outputs), tuple(tensor_grads),
                                       retain_graph=retain_graph)

        grads = self.get_grad()
        self._release()
        return grads

    def get_grad(self):
        grad = tuple(e.grad if e is not None else None for e in self.inputs)
        return grad[0] if len(grad) == 1 else grad

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

    @contextmanager
    def _stream_ctx(self, name=None):
        if _SEQUENTIAL_MODE:
            if name:
                torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                if name:
                    torch.cuda.nvtx.range_pop()
        else:
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


class NoopScheduleNode:
    """Transparent pass-through: forward returns inputs, backward returns output_grad."""

    def __init__(self, name="noop"):
        self.name = name

    def forward(self, inputs=()):
        return inputs

    def backward(self, output_grad):
        return output_grad

    def get_output(self):
        return None

    def get_grad(self):
        return None


@dataclass
class LayerCallables:
    """User-provided callable description for each layer step."""
    # MoE 4-node mode:
    attn_fn: Callable = None          # (h) -> (h, h_ln) for MoE, (h) -> h for dense
    dispatch_fn: Callable = None      # (h_ln) -> (sorted_tokens, sorted_probs)
    expert_fn: Callable = None        # (sorted_tokens, sorted_probs) -> expert_outs
    combine_fn: Callable = None       # (expert_outs, h, h_ln) -> h_out

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
                 sequential_mode=False, grad_clamp_value=0,
                 use_checkpoint=False):
        self.parallel_module = parallel_module
        self.num_layers = num_layers
        self.sequential_mode = sequential_mode
        self.grad_clamp_value = grad_clamp_value
        self.use_checkpoint = use_checkpoint
        self._use_4node = True

        # Async overlap: launch COMM and COMP ops from separate CPU threads
        # so their kernel launches can interleave, enabling true GPU overlap.
        # Controlled by ASYNC_4PHASE env var (default=1, set 0 to disable).
        _async_on = (not sequential_mode
                     and os.environ.get('ASYNC_4PHASE', '1') not in ('0', ''))
        self._async_pool = ThreadPoolExecutor(max_workers=1) if _async_on else None

        # Pre-allocate CUDA events for cross-stream synchronization.
        # Using separate events per barrier slot avoids re-recording while
        # a previous wait may still be pending (undefined behavior per CUDA spec).
        if not sequential_mode:
            # 3 barrier slots in _merged_step_4phase, each needs comp + comm events
            self._barrier_comp_evts = [torch.cuda.Event() for _ in range(3)]
            self._barrier_comm_evts = [torch.cuda.Event() for _ in range(3)]
            # 2 uni-directional sync events (comp->comm and comm->comp)
            self._sync_c2m_evt = torch.cuda.Event()
            self._sync_m2c_evt = torch.cuda.Event()
        else:
            self._barrier_comp_evts = None
            self._barrier_comm_evts = None
            self._sync_c2m_evt = None
            self._sync_m2c_evt = None

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

        tb = _TimingCtx(_TIMING_BREAKDOWN)
        tb.start()

        # Ensure COMP/COMM streams see all prior default-stream work
        # (optimizer.step, zero_grad from the previous training step).
        # Non-blocking streams do NOT auto-synchronize with the default stream,
        # so without this explicit sync, the forward pass might read stale
        # parameters or see non-zeroed gradients.
        if not self.sequential_mode:
            default_done = torch.cuda.Event()
            default_done.record(torch.cuda.default_stream())
            get_comp_stream().wait_event(default_done)
            get_comm_stream().wait_event(default_done)

        num_steps = self.num_layers
        events = [torch.cuda.Event() for _ in range(num_mbs)]
        results = [None] * num_mbs

        # Layer-level profiler: removed (synchronize-based profiling destroys
        # dual-stream overlap; use PROFILE_OVERLAP or nsys instead)

        _logger.debug("Warmup: forward mb0")
        with torch.cuda.stream(get_comp_stream()):
            h0 = embed_fn(samples[0])
        _log_checksum("merged embed mb=0", h0)
        tb.mark("warmup_embed")

        lc_list_0 = []
        for si in range(num_steps):
            lc_list_0.append(layer_callables_fn(si, samples[0]))
        tb.mark("warmup_lc_create")

        h0, all_nodes_0, rmaps_0, eprobs_0 = self._forward_all_layers(
            h0, lc_list_0, events[0])
        tb.mark("warmup_fwd_layers")

        # Sync COMM→COMP: last forward node may be on COMM (MoE combine),
        # but loss_node runs on COMP with a fresh event (no wait).
        self._sync_comm_to_comp()

        loss_node_0, output_info_0 = loss_fn(h0, samples[0], rmaps_0, eprobs_0)
        loss_0 = loss_node_0.forward((h0,))
        results[0] = output_info_0['output_tuple']

        with torch.cuda.stream(get_comp_stream()):
            loss_grad = torch.ones_like(loss_0)
        del h0, rmaps_0, eprobs_0, output_info_0
        tb.mark("warmup_loss")

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

            # Loss backward on COMP — overlaps with embed on COMM
            with nnscaler.sync_grad_when(False):
                grad_h = prev_loss_node.backward(loss_grad)

            prev_loss_node._release()
            tb.mark(f"mb{mb_i}_loss_bwd")

            # Create layer callables (CPU work, overlaps with GPU)
            fwd_lc_list = []
            for si in range(num_steps):
                fwd_lc_list.append(layer_callables_fn(si, fwd_sample))
            tb.mark(f"mb{mb_i}_embed_lc")

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
                            grad_h = sanitize_grad(grad_h, self.grad_clamp_value)
                    prev_all_nodes[bwd_idx] = None
                    bwd_idx -= 1
                    continue
                if bwd_entry is None:
                    bwd_idx -= 1
                    continue

                with nnscaler.sync_grad_when(False):
                    fwd_h, grad_h, fwd_entry = self._merged_step_general(
                        bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h,
                        fwd_layer_idx=fwd_idx, bwd_layer_idx=bwd_idx)

                prev_all_nodes[bwd_idx] = None

                fwd_all_nodes[fwd_idx] = fwd_entry

                if fwd_lc.is_moe:
                    fwd_routing_maps.append(fwd_lc.step_data.get('routing_map'))
                    fwd_expert_probs.append(fwd_lc.step_data.get('gate_scores'))

                fwd_idx += 1
                bwd_idx -= 1

            tb.mark(f"mb{mb_i}_merged_loop")

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
                            grad_h = sanitize_grad(grad_h, self.grad_clamp_value)
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
            tb.mark(f"mb{mb_i}_remaining_fwd_bwd")

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
            tb.mark(f"mb{mb_i}_loss_fwd")

        _logger.debug(f"Cooldown: backward mb{num_mbs-1}")
        with nnscaler.sync_grad_when(False):
            grad_h = prev_loss_node.backward(loss_grad)
            for i in reversed(range(num_steps)):
                entry = prev_all_nodes[i]
                if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == 'special':
                    lc = entry[1]
                    with torch.cuda.stream(get_comp_stream()):
                        grad_h = lc.special_backward(grad_h)
                        grad_h = sanitize_grad(grad_h, self.grad_clamp_value)
                    continue
                if entry is None:
                    continue
                grad_h = self._backward_entry(entry, grad_h)
        tb.mark("cooldown_bwd")

        if _PROFILE_OVERLAP:
            # Profile needs host-visible event completion for elapsed_time().
            torch.cuda.synchronize()
            _profile_report()
        elif not self.sequential_mode:
            # Use stream-event sync instead of device-wide synchronize:
            # Make default stream wait for COMP/COMM to finish, without blocking host.
            comp_done = torch.cuda.Event()
            comm_done = torch.cuda.Event()
            comp_done.record(get_comp_stream())
            comm_done.record(get_comm_stream())
            torch.cuda.default_stream().wait_event(comp_done)
            torch.cuda.default_stream().wait_event(comm_done)
        else:
            torch.cuda.synchronize()
        tb.mark("cuda_sync")

        for i in range(len(results)):
            if results[i] is not None:
                results[i] = tuple(
                    t.detach() if isinstance(t, torch.Tensor) else t
                    for t in results[i]
                )

        del prev_all_nodes, prev_loss_node


        tb.report("MERGED SCHEDULER BREAKDOWN")

        return results

    def _create_nodes(self, lc, event):
        """Create 2 ScheduleNodes for a layer (dense or MoE fallback)."""
        comp_stream = get_comp_stream()

        attn_node = ScheduleNode(
            lc.attn_fn, comp_stream, event,
            name="attn", checkpoint=self.use_checkpoint)

        if lc.is_moe:
            def combined_body_fn(h, h_ln, routing_probs):
                sorted_tokens, sorted_probs = lc.dispatch_fn(h_ln, routing_probs)
                expert_outs = lc.expert_fn(sorted_tokens, sorted_probs)
                h_out = lc.combine_fn(expert_outs, h, h_ln)
                return h_out

            body_node = ScheduleNode(
                combined_body_fn, comp_stream, event,
                name="moe_body", checkpoint=self.use_checkpoint)
        else:
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
            name="combine", checkpoint=False)

        return (attn_node, dispatch_node, expert_node, combine_node)

    def _sync_comm_to_comp(self):
        if self.sequential_mode:
            return
        self._sync_m2c_evt.record(get_comm_stream())
        get_comp_stream().wait_event(self._sync_m2c_evt)

    def _sync_comp_to_comm(self):
        if self.sequential_mode:
            return
        self._sync_c2m_evt.record(get_comp_stream())
        get_comm_stream().wait_event(self._sync_c2m_evt)

    def _cross_stream_barrier(self, slot=0):
        """Bidirectional sync: each stream waits for the other's current position.

        Records events on both streams simultaneously, then makes each wait
        for the other. Uses pre-allocated events indexed by slot to avoid
        event creation overhead and re-recording conflicts.
        """
        if self.sequential_mode:
            return
        comp_evt = self._barrier_comp_evts[slot]
        comm_evt = self._barrier_comm_evts[slot]
        comp_evt.record(get_comp_stream())
        comm_evt.record(get_comm_stream())
        get_comm_stream().wait_event(comp_evt)
        get_comp_stream().wait_event(comm_evt)

    def _record_for_comp(self, grads):
        """Record COMM-produced tensors for safe use on COMP stream."""
        if self.sequential_mode:
            return
        comp = get_comp_stream()
        if isinstance(grads, tuple):
            for t in grads:
                if isinstance(t, torch.Tensor):
                    t.record_stream(comp)
        elif isinstance(grads, torch.Tensor):
            grads.record_stream(comp)

    def _record_for_comm(self, grads):
        """Record COMP-produced tensors for safe use on COMM stream."""
        if self.sequential_mode:
            return
        comm = get_comm_stream()
        if isinstance(grads, tuple):
            for t in grads:
                if isinstance(t, torch.Tensor):
                    t.record_stream(comm)
        elif isinstance(grads, torch.Tensor):
            grads.record_stream(comm)

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

            _log_checksum(f"merged layer={si}", h)
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
        """Forward a single MoE layer through 4 nodes."""
        nodes = self._create_nodes_4(lc, event)
        attn_n, dispatch_n, expert_n, combine_n = nodes

        attn_out = attn_n.forward((h,))
        h_residual, h_ln, routing_probs = attn_out
        dispatch_out = dispatch_n.forward((h_ln, routing_probs))
        expert_out = expert_n.forward(dispatch_out)
        h_out = combine_n.forward((expert_out, h_residual, h_ln))

        for n in nodes:
            if n.checkpoint:
                n.output = None

        return h_out, ('layer4', nodes)

    def _backward_layer(self, nodes, grad_h):
        """Backward through a 2-node layer: body -> attn."""
        attn_n, body_n = nodes
        # Ensure sanitize_grad and all intermediate ops run on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        with torch.cuda.stream(get_comp_stream()):
            body_grads = body_n.backward(grad_h)
            if isinstance(body_grads, tuple):
                body_grads = tuple(sanitize_grad(g, self.grad_clamp_value) for g in body_grads)
            else:
                body_grads = sanitize_grad(body_grads, self.grad_clamp_value)

            grad_x = attn_n.backward(body_grads)
            grad_x = sanitize_grad(grad_x, self.grad_clamp_value)
        return grad_x

    def _backward_layer_4node(self, nodes, grad_h):
        """Backward through a 4-node MoE layer: combine→expert→dispatch→attn."""
        attn_n, dispatch_n, expert_n, combine_n = nodes
        # Ensure grad addition and sanitize_grad run on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        with torch.cuda.stream(get_comp_stream()):
            self._sync_comp_to_comm()

            combine_grads = combine_n.backward(grad_h)
            expert_grads = expert_n.backward(combine_grads[0])
            dispatch_grads = dispatch_n.backward(expert_grads)

            self._sync_comm_to_comp()
            # Record COMM-produced grads for COMP-side addition and attn backward.
            self._record_for_comp(dispatch_grads)
            self._record_for_comp(combine_grads)
            grad_h_ln_total = dispatch_grads[0] + combine_grads[2]

            grad_x = attn_n.backward((combine_grads[1], grad_h_ln_total, dispatch_grads[1]))
            grad_x = sanitize_grad(grad_x, self.grad_clamp_value)
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

    def _merged_step(self, bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h,
                     fwd_layer_idx=-1, bwd_layer_idx=-1):
        """2-phase overlap for dense layers."""
        bwd_tag, bwd_nodes = bwd_entry
        bwd_attn, bwd_body = bwd_nodes

        fwd_nodes = self._create_nodes(fwd_lc, fwd_event)
        fwd_attn, fwd_body = fwd_nodes

        # Ensure sanitize_grad runs on COMP stream, not default stream.
        with torch.cuda.stream(get_comp_stream()):
            body_grads = bwd_body.backward(grad_h)
            fwd_attn_out = fwd_attn.forward((fwd_h,))

            if isinstance(body_grads, tuple):
                body_grads = tuple(sanitize_grad(g, self.grad_clamp_value) for g in body_grads)
            else:
                body_grads = sanitize_grad(body_grads, self.grad_clamp_value)

            grad_x = bwd_attn.backward(body_grads)
            fwd_h_out = fwd_body.forward(fwd_attn_out)

            grad_x = sanitize_grad(grad_x, self.grad_clamp_value)

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

        Phase 1: f_attn_router(COMP) || b_combine(COMM)
        Phase 2: b_expert    (COMP) || f_dispatch (COMM)
        Phase 3: f_expert    (COMP) || b_dispatch (COMM)
        Phase 4: b_attn      (COMP) || f_combine  (COMM)

        Async overlap: COMM and COMP ops within each phase are launched from
        separate CPU threads so their kernel launches interleave, enabling
        true GPU-side overlap instead of CPU-serialized submission.
        """
        bwd_attn, bwd_dispatch, bwd_expert, bwd_combine = bwd_nodes
        fwd_nodes = self._create_nodes_4(fwd_lc, fwd_event)
        fwd_attn, fwd_dispatch, fwd_expert, fwd_combine = fwd_nodes

        pool = self._async_pool  # None in sequential mode

        profiling = bool(_PROFILE_OVERLAP)
        if profiling:
            comp_s, comm_s = get_comp_stream(), get_comm_stream()
            # 5 boundaries × 2 streams = 10 events (enable_timing for elapsed_time)
            # Boundaries: start, after_p1, after_p2, after_p3, after_p4
            ce = [torch.cuda.Event(enable_timing=True) for _ in range(5)]  # COMP
            me = [torch.cuda.Event(enable_timing=True) for _ in range(5)]  # COMM

        # --- Skip per-node event protocol during merged step ---
        # The default shared event serializes all 4 nodes (COMP waits for COMM
        # and vice versa), preventing within-phase parallelism. By setting
        # _skip_event=True, each node still switches to its own stream and
        # does record_stream for allocator safety, but does not wait/record
        # the shared event. Cross-stream dependencies are enforced explicitly
        # via _cross_stream_barrier() at phase boundaries.
        all_nodes = (*bwd_nodes, *fwd_nodes)
        for n in all_nodes:
            n._skip_event = True

        # Ensure grad addition and sanitize_grad run on COMP stream,
        # not the default stream (which has no sync with COMP/COMM in overlap mode).
        # ScheduleNode calls internally switch to their own stream and restore on exit.
        with torch.cuda.stream(get_comp_stream()):
            # Initial sync: COMP→COMM so COMM can read grad_h (from loss_bwd on COMP)
            self._sync_comp_to_comm()

            if profiling:
                ce[0].record(comp_s)
                me[0].record(comm_s)

            # Phase 1: COMM(b_combine) || COMP(f_attn_router)
            # Launch COMM backward in background thread, main thread does COMP forward.
            # torch.cuda.stream() is thread-local so each thread targets its own stream.
            # GIL is released during CUDA kernel launches (C++ layer), enabling true
            # concurrent kernel submission to different streams.
            if pool is not None:
                fut_combine = pool.submit(bwd_combine.backward, grad_h)
                fwd_attn_out = fwd_attn.forward((fwd_h,))
                combine_grads = fut_combine.result()
            else:
                combine_grads = bwd_combine.backward(grad_h)
                fwd_attn_out = fwd_attn.forward((fwd_h,))
            fwd_h_residual, fwd_h_ln, fwd_routing_probs = fwd_attn_out

            if profiling:
                ce[1].record(comp_s)
                me[1].record(comm_s)

            # Phase boundary: Phase 2 COMP needs COMM output (combine_grads),
            # Phase 2 COMM needs COMP output (attn_out + precomputed metadata)
            self._cross_stream_barrier(slot=0)

            # Phase 2: COMM(f_dispatch) || COMP(b_expert)
            if pool is not None:
                fut_dispatch = pool.submit(fwd_dispatch.forward, (fwd_h_ln, fwd_routing_probs))
                expert_grads = bwd_expert.backward(combine_grads[0])
                fwd_dispatch_out = fut_dispatch.result()
            else:
                fwd_dispatch_out = fwd_dispatch.forward((fwd_h_ln, fwd_routing_probs))
                expert_grads = bwd_expert.backward(combine_grads[0])

            if profiling:
                ce[2].record(comp_s)
                me[2].record(comm_s)

            # Phase boundary: Phase 3 COMP needs COMM output (dispatch_out),
            # Phase 3 COMM needs COMP output (expert_grads)
            self._cross_stream_barrier(slot=1)

            # Phase 3: COMM(b_dispatch) || COMP(f_expert)
            if pool is not None:
                fut_dispatch_bwd = pool.submit(bwd_dispatch.backward, expert_grads)
                fwd_expert_out = fwd_expert.forward(fwd_dispatch_out)
                dispatch_grads = fut_dispatch_bwd.result()
            else:
                dispatch_grads = bwd_dispatch.backward(expert_grads)
                fwd_expert_out = fwd_expert.forward(fwd_dispatch_out)

            if profiling:
                ce[3].record(comp_s)
                me[3].record(comm_s)

            # Phase boundary: Phase 4 COMP needs COMM output (dispatch_grads),
            # Phase 4 COMM needs COMP output (expert_out)
            self._cross_stream_barrier(slot=2)

            # Phase 4: COMM(f_combine) || COMP(b_attn)
            if pool is not None:
                fut_combine_fwd = pool.submit(fwd_combine.forward,
                                              (fwd_expert_out, fwd_h_residual, fwd_h_ln))
                # Prep for bwd_attn while combine runs in background
                self._record_for_comp(dispatch_grads)
                self._record_for_comp(combine_grads)
                grad_h_ln_total = dispatch_grads[0] + combine_grads[2]
                grad_x = bwd_attn.backward((combine_grads[1], grad_h_ln_total, dispatch_grads[1]))
                fwd_h_out = fut_combine_fwd.result()
            else:
                fwd_h_out = fwd_combine.forward((fwd_expert_out, fwd_h_residual, fwd_h_ln))
                self._record_for_comp(dispatch_grads)
                self._record_for_comp(combine_grads)
                grad_h_ln_total = dispatch_grads[0] + combine_grads[2]
                grad_x = bwd_attn.backward((combine_grads[1], grad_h_ln_total, dispatch_grads[1]))

            if profiling:
                ce[4].record(comp_s)
                me[4].record(comm_s)

            grad_x = sanitize_grad(grad_x, self.grad_clamp_value)
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

        if profiling:
            # Don't synchronize here - just store events for later analysis.
            # _profile_report() will be called after torch.cuda.synchronize() in run().
            _profile_data.append({'ce': ce, 'me': me, 'sequential': self.sequential_mode})

        return fwd_h_out, grad_x, ('layer4', fwd_nodes)

    def _merged_step_general(self, bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h,
                             fwd_layer_idx=-1, bwd_layer_idx=-1):
        """Dispatch to 4-phase or 2-phase merged step based on layer types."""
        bwd_tag, bwd_nodes = bwd_entry

        if self._use_4node and bwd_tag == 'layer4' and fwd_lc.is_moe:
            # DEBUG: force fallback to isolate 4-phase interleaving vs stream issues
            if os.environ.get('DEBUG_NO_4PHASE'):
                grad_x = self._backward_entry(bwd_entry, grad_h)
                fwd_h_out, fwd_entry = self._forward_single_layer_4node(fwd_h, fwd_lc, fwd_event)
                return fwd_h_out, grad_x, fwd_entry
            return self._merged_step_4phase(bwd_nodes, fwd_lc, fwd_event, grad_h, fwd_h)
        elif bwd_tag == 'layer2':
            return self._merged_step(bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h,
                                     fwd_layer_idx=fwd_layer_idx, bwd_layer_idx=bwd_layer_idx)
        else:
            with nnscaler.sync_grad_when(False):
                grad_x = self._backward_entry(bwd_entry, grad_h)
            if fwd_lc.is_moe and self._use_4node:
                fwd_h_out, fwd_entry = self._forward_single_layer_4node(fwd_h, fwd_lc, fwd_event)
            else:
                fwd_h_out, fwd_entry = self._forward_single_layer(fwd_h, fwd_lc, fwd_event)
            return fwd_h_out, grad_x, fwd_entry
