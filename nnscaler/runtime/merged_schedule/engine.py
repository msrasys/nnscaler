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
        # Record cross-stream tensor usage for output_grad tensors.
        for g in output_grad:
            if isinstance(g, torch.Tensor):
                g.record_stream(self.stream)

        with self._stream_ctx(f"{self.name} bwd"):
            if self.checkpoint:
                recomputed = self.forward_func(*self.inputs)
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


@dataclass
class LayerCallables:
    """User-provided callable description for each layer step."""
    # MoE 4-node mode:
    attn_fn: Callable = None          # (h) -> (h, h_ln) for MoE, (h) -> h for dense
    dispatch_fn: Callable = None      # (h_ln) -> (sorted_tokens, sorted_probs)
    expert_fn: Callable = None        # (sorted_tokens, sorted_probs) -> expert_outs
    combine_fn: Callable = None       # (expert_outs, h, shared_expert_out) -> h_out

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
                 use_checkpoint=False):
        self.parallel_module = parallel_module
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self._use_4node = True

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

    def shutdown(self):
        """Release scheduler-owned CPU worker resources."""
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
                _embed_h_list[mb_i].backward(grad_h)

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
                    continue
                if entry is None:
                    continue
                grad_h = self._backward_entry(entry, grad_h)
            # Propagate gradient through embedding graph to tok_embed weight
            _embed_h_list[-1].backward(grad_h)

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
            name="combine", checkpoint=False)

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

    def _record_for_comp(self, grads):
        """Record COMM-produced tensors for safe use on COMP stream."""
        comp = get_comp_stream()
        if isinstance(grads, tuple):
            for t in grads:
                if isinstance(t, torch.Tensor):
                    t.record_stream(comp)
        elif isinstance(grads, torch.Tensor):
            grads.record_stream(comp)

    @staticmethod
    def _attn_backward_grads(attn_node, grad_h, grad_h_ln, grad_routing):
        outputs = attn_node.get_output()
        if not isinstance(outputs, tuple):
            return grad_h

        grads = [grad_h, grad_h_ln, grad_routing]
        aux_tensors = getattr(attn_node, 'loss_aux_tensors', ())
        for tensor in aux_tensors:
            if isinstance(tensor, torch.Tensor):
                grads.append(tensor.grad)
            else:
                grads.append(None)
        while len(grads) < len(outputs):
            grads.append(None)
        return tuple(grads[:len(outputs)])

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

        expert_fn now includes shared expert: takes (sorted_tokens, sorted_probs, h_ln),
        returns (expert_outs, shared_expert_out). combine_fn takes shared_expert_out
        directly, keeping combine (COMM) as pure communication.
        """
        nodes = self._create_nodes_4(lc, event)
        attn_n, dispatch_n, expert_n, combine_n = nodes

        attn_out = attn_n.forward((h,))
        h_residual, h_ln, routing_probs = attn_out[:3]
        loss_aux_tensors = lc.step_data.get('_loss_aux_tensors')
        if loss_aux_tensors is not None:
            attn_n.loss_aux_tensors = loss_aux_tensors
        dispatch_out = dispatch_n.forward((h_ln, routing_probs))
        expert_result = expert_n.forward((*dispatch_out, h_ln))
        expert_out, shared_expert_out = expert_result
        h_out = combine_n.forward((expert_out, h_residual, shared_expert_out))

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
            # expert backward needs grads for both outputs: expert_outs and shared_expert_out
            expert_grads = expert_n.backward((combine_grads[0], combine_grads[2]))
            # expert_grads: (grad_sorted_tokens, grad_sorted_probs, grad_h_ln)
            dispatch_grads = dispatch_n.backward((expert_grads[0], expert_grads[1]))

            self._sync_comm_to_comp()
            # Record COMM-produced grads for COMP-side addition and attn backward.
            self._record_for_comp(dispatch_grads)
            self._record_for_comp(combine_grads)

            grad_h_ln_total = dispatch_grads[0] + expert_grads[2]

            attn_grads = self._attn_backward_grads(
                attn_n, combine_grads[1], grad_h_ln_total, dispatch_grads[1])
            grad_x = attn_n.backward(attn_grads)

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

        Phase 1: f_attn_router(COMP) || b_combine(COMM)   [combine = pure comm]
        Phase 2: b_expert    (COMP) || f_dispatch (COMM)   [expert includes shared expert bwd]
        Phase 3: f_expert    (COMP) || b_dispatch (COMM)   [expert includes shared expert fwd]
        Phase 4: b_attn      (COMP) || f_combine  (COMM)

        Shared expert is merged into expert_fn (COMP stream), keeping
        combine_fn (COMM stream) as pure communication.
        """
        bwd_attn, bwd_dispatch, bwd_expert, bwd_combine = bwd_nodes
        fwd_nodes = self._create_nodes_4(fwd_lc, fwd_event)
        fwd_attn, fwd_dispatch, fwd_expert, fwd_combine = fwd_nodes

        pool = self._async_pool  # None in sequential mode

        # --- Skip per-node event protocol during merged step ---
        # The default shared event serializes all nodes (COMP waits for COMM
        # and vice versa), preventing within-phase parallelism. By setting
        # _skip_event=True, each node still switches to its own stream and
        # does record_stream for allocator safety, but does not wait/record
        # the shared event. Cross-stream dependencies are enforced explicitly
        # via _cross_stream_barrier() at phase boundaries.
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
            fwd_h_residual, fwd_h_ln, fwd_routing_probs = fwd_attn_out[:3]
            loss_aux_tensors = fwd_lc.step_data.get('_loss_aux_tensors')
            if loss_aux_tensors is not None:
                fwd_attn.loss_aux_tensors = loss_aux_tensors

            # Phase boundary: Phase 2 COMP needs COMM output (combine_grads),
            # Phase 2 COMM needs COMP output (attn_out + precomputed metadata)
            self._cross_stream_barrier(slot=0)

            # Phase 2: COMM(f_dispatch) || COMP(b_expert)
            # expert backward includes shared expert backward (both on COMP).
            # combine_grads: (grad_expert_outs, grad_h_residual, grad_shared_expert_out)
            if pool is not None:
                fut_dispatch = pool.submit(fwd_dispatch.forward, (fwd_h_ln, fwd_routing_probs))
                expert_grads = bwd_expert.backward((combine_grads[0], combine_grads[2]))
                fwd_dispatch_out = fut_dispatch.result()
            else:
                fwd_dispatch_out = fwd_dispatch.forward((fwd_h_ln, fwd_routing_probs))
                expert_grads = bwd_expert.backward((combine_grads[0], combine_grads[2]))
            # expert_grads: (grad_sorted_tokens, grad_sorted_probs, grad_h_ln)

            # Phase boundary: Phase 3 COMP needs COMM output (dispatch_out),
            # Phase 3 COMM needs COMP output (expert_grads)
            self._cross_stream_barrier(slot=1)

            # Phase 3: COMM(b_dispatch) || COMP(f_expert)
            # expert forward includes shared expert (takes h_ln, returns tuple).
            if pool is not None:
                fut_dispatch_bwd = pool.submit(bwd_dispatch.backward, (expert_grads[0], expert_grads[1]))
                fwd_expert_result = fwd_expert.forward((*fwd_dispatch_out, fwd_h_ln))
                dispatch_grads = fut_dispatch_bwd.result()
            else:
                dispatch_grads = bwd_dispatch.backward((expert_grads[0], expert_grads[1]))
                fwd_expert_result = fwd_expert.forward((*fwd_dispatch_out, fwd_h_ln))
            fwd_expert_out, fwd_shared_expert_out = fwd_expert_result

            # Phase boundary: Phase 4 COMP needs COMM output (dispatch_grads),
            # Phase 4 COMM needs COMP output (expert_out + shared_expert_out)
            self._cross_stream_barrier(slot=2)

            # Phase 4: COMM(f_combine) || COMP(b_attn)
            if pool is not None:
                fut_combine_fwd = pool.submit(fwd_combine.forward,
                                              (fwd_expert_out, fwd_h_residual, fwd_shared_expert_out))
                self._record_for_comp(dispatch_grads)
                self._record_for_comp(combine_grads)
                grad_h_ln_total = dispatch_grads[0] + expert_grads[2]
                attn_grads = self._attn_backward_grads(
                    bwd_attn, combine_grads[1], grad_h_ln_total, dispatch_grads[1])
                grad_x = bwd_attn.backward(attn_grads)
                fwd_h_out = fut_combine_fwd.result()
            else:
                fwd_h_out = fwd_combine.forward((fwd_expert_out, fwd_h_residual, fwd_shared_expert_out))
                self._record_for_comp(dispatch_grads)
                self._record_for_comp(combine_grads)
                grad_h_ln_total = dispatch_grads[0] + expert_grads[2]
                attn_grads = self._attn_backward_grads(
                    bwd_attn, combine_grads[1], grad_h_ln_total, dispatch_grads[1])
                grad_x = bwd_attn.backward(attn_grads)

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
