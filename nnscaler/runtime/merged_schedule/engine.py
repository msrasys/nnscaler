"""
MoE FWD-BWD overlap scheduling engine.

Provides ScheduleNode (CUDA stream/event scheduling primitive) and
MergedScheduler (merged forward-backward with communication/computation overlap).

4-Phase Overlap (MoE layers):
  Phase 1: b_combine(COMM) || f_attn_router(COMP)
  Phase 2: b_expert (COMP) || f_dispatch   (COMM)
  Phase 3: b_dispatch(COMM)|| f_expert     (COMP)
  Phase 4: b_attn   (COMP) || f_combine    (COMM)

2-Phase Overlap (dense layers):
  Phase 1: body_bwd(prev)[COMP] || attn_fwd(next)[COMP]
  Phase 2: attn_bwd(prev)[COMP] || body_fwd(next)[COMP]

MoE layers use 4 ScheduleNodes (attn_router, dispatch, expert, combine)
alternating COMP/COMM streams for true communication/computation overlap.
Dense layers use 2 ScheduleNodes (attn, ffn) both on COMP stream.
"""

import gc
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch
from torch.autograd import Variable

import nnscaler

from .utils import sanitize_grad

# Suppress AccumulateGrad stream mismatch warning in dual-stream mode.
torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

_logger = logging.getLogger(__name__)

_COMP_STREAM = None   # computation stream (default stream)
_COMM_STREAM = None   # communication stream (side stream)
_IN_RECOMPUTE = False  # Flag: True when inside ScheduleNode checkpoint recompute
_SEQUENTIAL_MODE = False  # Flag: skip stream/event ops in sequential mode


def set_streams(sequential_mode=False):
    """Initialize global COMP/COMM streams.
    In sequential_mode both streams are the default stream (no overlap).
    """
    global _COMP_STREAM, _COMM_STREAM, _SEQUENTIAL_MODE
    _SEQUENTIAL_MODE = sequential_mode
    _COMP_STREAM = torch.cuda.default_stream()
    if sequential_mode:
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

        with self._stream_ctx(f"{self.name} fwd"):
            self.inputs = [_detach_tensor(inp) for inp in inputs]

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
                if not isinstance(data, tuple):
                    data = _make_viewless(data)
                else:
                    data = tuple(
                        _make_viewless(e) if isinstance(e, torch.Tensor) else e
                        for e in data
                    )

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
            self.event.wait(self.stream)
            if name:
                torch.cuda.nvtx.range_push(name)
            try:
                with torch.cuda.stream(self.stream):
                    yield
            finally:
                if name:
                    torch.cuda.nvtx.range_pop()
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
                 sequential_mode=False, grad_clamp_value=1e4,
                 use_checkpoint=False):
        self.parallel_module = parallel_module
        self.num_layers = num_layers
        self.sequential_mode = sequential_mode
        self.grad_clamp_value = grad_clamp_value
        self.use_checkpoint = use_checkpoint
        self._use_4node = True
        self._debug_no_interleave = False

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

        gc.collect()
        torch.cuda.empty_cache()

        num_steps = self.num_layers
        events = [torch.cuda.Event() for _ in range(num_mbs)]
        results = [None] * num_mbs

        _logger.info("Warmup: forward mb0")
        h0 = embed_fn(samples[0])

        lc_list_0 = []
        for si in range(num_steps):
            lc_list_0.append(layer_callables_fn(si, samples[0]))

        h0, all_nodes_0, rmaps_0, eprobs_0 = self._forward_all_layers(
            h0, lc_list_0, events[0])

        torch.cuda.empty_cache()

        loss_node_0, output_info_0 = loss_fn(h0, samples[0], rmaps_0, eprobs_0)
        loss_0 = loss_node_0.forward((h0,))
        results[0] = output_info_0['output_tuple']

        loss_grad = torch.ones_like(loss_0)
        del h0, rmaps_0, eprobs_0, output_info_0

        prev_all_nodes = all_nodes_0
        prev_loss_node = loss_node_0

        for mb_i in range(num_mbs - 1):
            fwd_sample = samples[mb_i + 1]
            fwd_event = events[mb_i + 1]

            _logger.info(f"[MERGED] bwd(mb{mb_i}) + fwd(mb{mb_i+1})")

            torch.cuda.empty_cache()

            with nnscaler.sync_grad_when(False):
                grad_h = prev_loss_node.backward(loss_grad)

            prev_loss_node._release()
            gc.collect()
            torch.cuda.empty_cache()

            fwd_h = embed_fn(fwd_sample)
            fwd_lc_list = []
            for si in range(num_steps):
                fwd_lc_list.append(layer_callables_fn(si, fwd_sample))

            fwd_routing_maps = []
            fwd_expert_probs = []
            fwd_all_nodes = [None] * num_steps

            bwd_idx = num_steps - 1
            fwd_idx = 0

            if self._debug_no_interleave:
                _logger.info("[DEBUG] no_interleave mode: all backward then all forward")
                with nnscaler.sync_grad_when(False):
                    for bi in reversed(range(num_steps)):
                        entry = prev_all_nodes[bi]
                        if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == 'special':
                            bwd_lc = entry[1]
                            grad_h = bwd_lc.special_backward(grad_h)
                            grad_h = sanitize_grad(grad_h, self.grad_clamp_value)
                            prev_all_nodes[bi] = None
                            continue
                        if entry is None:
                            continue
                        grad_h = self._backward_entry(entry, grad_h)
                        prev_all_nodes[bi] = None

                for fi in range(num_steps):
                    fwd_lc = fwd_lc_list[fi]
                    if fwd_lc.is_special:
                        fwd_h, special_data = fwd_lc.special_forward(fwd_h)
                        fwd_all_nodes[fi] = ('special', fwd_lc)
                        continue
                    if fwd_lc.is_moe and self._use_4node:
                        fwd_h, fwd_entry = self._forward_single_layer_4node(
                            fwd_h, fwd_lc, fwd_event)
                    else:
                        fwd_h, fwd_entry = self._forward_single_layer(
                            fwd_h, fwd_lc, fwd_event)
                    fwd_all_nodes[fi] = fwd_entry
                    if fwd_lc.is_moe:
                        fwd_routing_maps.append(fwd_lc.step_data.get('routing_map'))
                        fwd_expert_probs.append(fwd_lc.step_data.get('gate_scores'))

            else:
                while fwd_idx < num_steps and bwd_idx >= 0:
                    fwd_lc = fwd_lc_list[fwd_idx]

                    if fwd_lc.is_special:
                        fwd_h, special_data = fwd_lc.special_forward(fwd_h)
                        fwd_all_nodes[fwd_idx] = ('special', fwd_lc)
                        fwd_idx += 1
                        continue

                    bwd_entry = prev_all_nodes[bwd_idx]
                    if isinstance(bwd_entry, tuple) and len(bwd_entry) == 2 and bwd_entry[0] == 'special':
                        bwd_lc = bwd_entry[1]
                        with nnscaler.sync_grad_when(False):
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

                while fwd_idx < num_steps:
                    fwd_lc = fwd_lc_list[fwd_idx]
                    if fwd_lc.is_special:
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
            gc.collect()
            torch.cuda.empty_cache()

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
            gc.collect()
            torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        _logger.info(f"Cooldown: backward mb{num_mbs-1}")
        with nnscaler.sync_grad_when(False):
            grad_h = prev_loss_node.backward(loss_grad)
            for i in reversed(range(num_steps)):
                entry = prev_all_nodes[i]
                if isinstance(entry, tuple) and len(entry) == 2 and entry[0] == 'special':
                    lc = entry[1]
                    grad_h = lc.special_backward(grad_h)
                    grad_h = sanitize_grad(grad_h, self.grad_clamp_value)
                    continue
                if entry is None:
                    continue
                grad_h = self._backward_entry(entry, grad_h)

        torch.cuda.synchronize()

        for i in range(len(results)):
            if results[i] is not None:
                results[i] = tuple(
                    t.detach() if isinstance(t, torch.Tensor) else t
                    for t in results[i]
                )

        del prev_all_nodes, prev_loss_node
        gc.collect()
        torch.cuda.empty_cache()

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
        sync = torch.cuda.Event()
        sync.record(get_comm_stream())
        get_comp_stream().wait_event(sync)

    def _sync_comp_to_comm(self):
        if self.sequential_mode:
            return
        sync = torch.cuda.Event()
        sync.record(get_comp_stream())
        get_comm_stream().wait_event(sync)

    def _forward_all_layers(self, h, lc_list, event):
        """Warmup: forward all layers, collect nodes and routing data."""
        all_nodes = []
        routing_maps = []
        expert_probs = []

        for si, lc in enumerate(lc_list):
            if lc.is_special:
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

        self._sync_comp_to_comm()

        combine_grads = combine_n.backward(grad_h)
        expert_grads = expert_n.backward(combine_grads[0])
        dispatch_grads = dispatch_n.backward(expert_grads)

        self._sync_comm_to_comp()
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

        Phase 1: b_combine(COMM) || f_attn_router(COMP)
        Phase 2: b_expert (COMP) || f_dispatch   (COMM)
        Phase 3: b_dispatch(COMM)|| f_expert     (COMP)
        Phase 4: b_attn   (COMP) || f_combine    (COMM)
        """
        bwd_attn, bwd_dispatch, bwd_expert, bwd_combine = bwd_nodes
        fwd_nodes = self._create_nodes_4(fwd_lc, fwd_event)
        fwd_attn, fwd_dispatch, fwd_expert, fwd_combine = fwd_nodes

        self._sync_comp_to_comm()

        # Phase 1
        combine_grads = bwd_combine.backward(grad_h)
        fwd_attn_out = fwd_attn.forward((fwd_h,))
        fwd_h_residual, fwd_h_ln, fwd_routing_probs = fwd_attn_out

        # Phase 2
        expert_grads = bwd_expert.backward(combine_grads[0])
        fwd_dispatch_out = fwd_dispatch.forward((fwd_h_ln, fwd_routing_probs))

        # Phase 3
        dispatch_grads = bwd_dispatch.backward(expert_grads)
        fwd_expert_out = fwd_expert.forward(fwd_dispatch_out)

        # Phase 4 prep
        self._sync_comm_to_comp()
        grad_h_ln_total = dispatch_grads[0] + combine_grads[2]

        # Phase 4
        grad_x = bwd_attn.backward((combine_grads[1], grad_h_ln_total, dispatch_grads[1]))
        fwd_h_out = fwd_combine.forward((fwd_expert_out, fwd_h_residual, fwd_h_ln))

        grad_x = sanitize_grad(grad_x, self.grad_clamp_value)
        for n in fwd_nodes:
            if n.checkpoint:
                n.output = None
        return fwd_h_out, grad_x, ('layer4', fwd_nodes)

    def _merged_step_general(self, bwd_entry, fwd_lc, fwd_event, grad_h, fwd_h,
                             fwd_layer_idx=-1, bwd_layer_idx=-1):
        """Dispatch to 4-phase or 2-phase merged step based on layer types."""
        bwd_tag, bwd_nodes = bwd_entry

        if self._use_4node and bwd_tag == 'layer4' and fwd_lc.is_moe:
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
