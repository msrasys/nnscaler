#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Shared, real (not toy-faked) MoE-FFN transformer layer model for Step C's
end-to-end tests (``tests/parallel_module/test_phase_moe_e2e.py`` /
``test_phase_moe_multistage_e2e.py``) and gencode-precision test
(``tests/codegen/test_phase_gencode.py``).

Architecture per layer
-----------------------
- ``ATTENTION``: real multi-head self-attention (QKV projections, scaled dot
  product, output projection) -- genuine compute, no communication.
- ``MOE_DISPATCH``: a real top-1 gate (linear + softmax + argmax), a
  real, differentiable, fixed-*capacity* scatter into a
  ``[ep_size, capacity, hidden]`` buffer (overflow dropped, underflow
  zero-padded -- the standard GShard/Switch-Transformer capacity mechanism,
  keeping the all-to-all's shape static for nnScaler's compile-time
  codegen), then :func:`nnscaler.runtime.adapter.moe.moe_dispatch` (real
  all-to-all issue).
- ``EXPERT_COMPUTE``: :func:`~nnscaler.runtime.adapter.moe.moe_dispatch_wait`,
  a real local expert FFN (Linear -> SiLU -> Linear, one expert per EP rank),
  then :func:`~nnscaler.runtime.adapter.moe.moe_combine` (real all-to-all
  issue).
- ``MOE_COMBINE``: :func:`~nnscaler.runtime.adapter.moe.moe_combine_wait`,
  gather back to per-token order (weighted by the gate probability, zero for
  dropped tokens), residual add.

Each EP rank hosts exactly one expert (``num_experts == len(ep_ranks)``) --
honestly scoped (see ``nnscaler/graph/schedule/phase.py``'s "Scope"): multiple
local experts per rank would need an additional local expert-selection split,
not attempted here.

Honest scoping note: what's real, and what's replicated for simplicity
------------------------------------------------------------------------
Every op *before* :func:`~nnscaler.runtime.adapter.moe.moe_dispatch` (QKV,
attention, the gate, and the fixed-capacity scatter build) is
*replicated* (``nnscaler.policies._replica``) across each stage's
``ep_ranks``, i.e. every EP rank runs the identical computation on the
identical (replicated) input -- **not** TP-sharded per rank. This was a
deliberate simplification found necessary while building this model: the
capacity-scatter's constituent ops (``torch.argmax``, ``torch.nn.functional.one_hot``,
``Tensor.new_zeros``, ``torch.scatter_add``) are not registered nnScaler
``IRDimops`` (confirmed via the "Find unknown pytorch operation" trace-time
notice), so they have no partition-dimension algorithm nnScaler's
``_tp``/``graph.partition`` can act on; TP-sharding only the *upstream*
attention ops while replicating these downstream ones produced a genuine,
confirmed compile-time error (``IRAdapterGener.gen_activation``'s
``local_consumer_multiref``: "Detect that a full tensor is partitioned
differently on a device") -- nnScaler's ``_replica`` primitive means "run
identically on the full (replicated) tensor", not "process whatever local
shard an upstream partition happened to produce", so mixing TP-sharded
upstream ops with replica'd downstream ops on the same tensor is
structurally inconsistent, not just under-tested. Registering these ops
with proper partition annotations (mirroring how
``examples/deepseek_coder_v2_lite`` registers ``nnscaler_moe_gmm``) would
resolve this but is out of scope for Step C's IR/scheduling/communication
focus.

Similarly, the expert FFN weights (``gate``/``expert_up``/``expert_down``)
are replicated, i.e. every rank's local "expert" computes the identical
function, rather than each rank owning an independently-parameterized expert
slice (which would need the same kind of new, stacked-expert-axis weight
partitioning, mirroring ``nnscaler_moe_gmm``/``build_ep_transform_rule``).

Consequently, in these e2e tests every EP rank computes *identical* gating
on *identical* (replicated) input, so the content actually moved by
dispatch/combine happens to be identical across ranks too. This does **not**
weaken what is actually novel and being tested here: the all-to-all
*communication* itself is fully real (genuine NCCL collectives, genuine
async issue + deferred wait, genuine gradients through real autograd
Functions -- see ``nnscaler/runtime/adapter/moe.py``), and the phase
IR/scheduling/overlap machinery operates identically regardless of *what*
content is moved. The routing/capacity *logic itself* (non-uniform,
per-token-varying expert assignment, overflow drop, underflow pad) is
separately, directly unit-tested with synthetic non-uniform inputs in
``tests/runtime/test_moe_comm.py`` (CPU, no distributed launch needed),
decoupled from this distributed-replication limitation.

Phase-tagged vs. plain ("serial baseline") variants
------------------------------------------------------
:class:`PhaseMoEModel`'s ``use_phases`` flag: when ``True``, each layer calls
:func:`nnscaler.graph.schedule.phase.phase_anchor` at each phase boundary
(structurally a no-op, see that module's docstring); when ``False`` it emits
the exact same ops with no anchors at all. Both variants are byte-for-byte
the same math (anchors never touch a tensor), so comparing a phase-lowered
compile against a plain (single-segment-per-stage) compile of the
``use_phases=False`` twin is a valid, apples-to-apples numeric-equivalence
baseline.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnscaler.graph.graph import IRGraph
from nnscaler.graph.function.anchor import IRGraphAnchor
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.parallel import ComputeConfig
from nnscaler.policies import _replica
from nnscaler.graph.schedule.phase import PhaseType, PhaseAwareSched, lower_layer_to_phases, validate_phase_layout, phase_anchor
from nnscaler.graph.schedule.schedplan import StreamContext
from nnscaler.runtime.adapter.moe import moe_dispatch, moe_dispatch_wait, moe_combine, moe_combine_wait



@dataclass
class MoEConfig:
    dim: int = 16
    n_heads: int = 2
    seq_len: int = 4  # pseudo-sequence positions carved out of each sample's hidden dim
    ffn_hidden: int = 32
    capacity_factor: float = 1.0  # capacity = ceil(capacity_factor * local_tokens / num_experts)


class SelfAttention(nn.Module):
    """A real (not faked) multi-head self-attention block: QKV projection,
    scaled dot-product attention, output projection.

    Operates *within* each sample's own hidden vector: ``x: [T, dim]`` is
    reshaped to ``[T, seq_len, dim // seq_len]`` (``T`` independent samples,
    each with ``seq_len`` "positions" carved out of its own hidden dim) and
    attention runs across the ``seq_len`` axis, per sample -- never across
    ``T``. This is a deliberate modeling choice (not needed for a plain,
    single-device model) so that ``T`` -- the MoE dispatch/combine's own
    per-token axis -- can be freely, *safely* batch-sharded across the EP
    ranks (see ``phase_moe_common`` module docstring "Making dispatch/combine
    non-degenerate"): real cross-token attention would make that unsound
    (each rank would only see part of the sequence it needs to attend
    over), whereas per-sample attention has no such cross-``T`` dependency.
    Still a genuine matmul+softmax+matmul, not a faked/degenerate op.
    """

    def __init__(self, dim: int, n_heads: int, seq_len: int):
        super().__init__()
        assert dim % seq_len == 0
        self.pos_dim = dim // seq_len
        assert self.pos_dim % n_heads == 0
        self.dim = dim
        self.seq_len = seq_len
        self.n_heads = n_heads
        self.head_dim = self.pos_dim // n_heads
        # Precomputed here (plain Python float, __init__ time) rather than
        # `math.sqrt(self.head_dim)` inside forward(): the generated
        # `_train_step` module does not import `math` (a genuine codegen
        # gap found while building this model, see MoEFFN._capacity's
        # comment for the same finding) -- avoided entirely rather than
        # patched, since `head_dim` is a plain Python int constant anyway.
        self._attn_scale = self.head_dim ** (-0.5)
        self.qkv = nn.Linear(self.pos_dim, 3 * self.pos_dim, bias=False)
        self.out_proj = nn.Linear(self.pos_dim, self.pos_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, dim] (T independent samples; the MoE dispatch/combine's own
        # per-token axis -- see class docstring).
        T = x.shape[0]
        xr = x.view(T, self.seq_len, self.pos_dim)
        qkv = self.qkv(xr).view(T, self.seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [T, seq_len, n_heads, head_dim]
        q = q.permute(0, 2, 1, 3)  # [T, n_heads, seq_len, head_dim]
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self._attn_scale  # [T, n_heads, seq_len, seq_len]
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [T, n_heads, seq_len, head_dim]
        out = out.permute(0, 2, 1, 3).reshape(T, self.seq_len, self.pos_dim)
        out = self.out_proj(out)
        return out.reshape(T, self.dim)


def _capacity_scatter(x: torch.Tensor, expert_idx: torch.Tensor, num_experts: int, capacity: int):
    """Real, differentiable, fixed-capacity scatter of local tokens into a
    ``[num_experts, capacity, hidden]`` buffer (standard native
    ``scatter_add``/fancy-indexing autograd, no custom Function needed).

    Overflow (more than ``capacity`` tokens routed to one expert) is
    silently dropped (contributes zero, matching standard GShard/Switch
    capacity semantics); underflow is zero-padded.

    Returns ``(buffer [num_experts, capacity, hidden], dest_slot [T] int64)``
    -- ``dest_slot[t] == num_experts * capacity`` means token ``t`` overflowed
    and was dropped (a dedicated trash row, sliced back off before use).

    Raises:
        ValueError: if ``expert_idx``'s length does not match ``x``'s token
            count (fail-fast rather than silently mis-routing/broadcasting).
    """
    T, H = x.shape
    if expert_idx.shape != (T,):
        raise ValueError(
            f"expert_idx must have shape ({T},) to match x's token count, "
            f"got {tuple(expert_idx.shape)}"
        )
    with torch.no_grad():
        one_hot = F.one_hot(expert_idx, num_classes=num_experts).to(x.dtype)  # [T, E]
        # 0-indexed position of each token among same-expert tokens, in
        # original token order (a stable, deterministic "first come, first
        # served" capacity policy).
        running = torch.cumsum(one_hot, dim=0) - 1.0  # [T, E]
        slot = running.gather(1, expert_idx.unsqueeze(1)).squeeze(1).long()  # [T]
        keep = slot < capacity
        dest = torch.where(keep, expert_idx.long() * capacity + slot, torch.full_like(slot, num_experts * capacity))

    buffer_flat = x.new_zeros(num_experts * capacity + 1, H)
    buffer_flat = buffer_flat.scatter_add(0, dest.unsqueeze(1).expand(-1, H), x)
    buffer = buffer_flat[:num_experts * capacity].view(num_experts, capacity, H)
    return buffer, dest


def _capacity_gather(combined: torch.Tensor, dest: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_capacity_scatter`: gather each token's own expert
    output back out (zero for a dropped token, via the trash row), weighted
    by its top-1 gate probability."""
    num_experts, capacity, H = combined.shape
    flat = combined.reshape(num_experts * capacity, H)
    trash = flat.new_zeros(1, H)
    flat = torch.cat([flat, trash], dim=0)
    gathered = flat.index_select(0, dest)  # [T, H]
    return gathered * gate_weight.unsqueeze(-1)


class MoEFFN(nn.Module):
    """Capacity-based, top-1-gated, expert-parallel FFN: one expert per EP
    rank (``num_experts == len(ep_ranks)``), real dispatch/combine
    all-to-all. See module docstring for the full data flow.
    """

    def __init__(self, dim: int, ffn_hidden: int, ep_ranks: Sequence[int],
                 layer_id: int, capacity_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.ep_ranks = tuple(ep_ranks)
        self.num_experts = len(self.ep_ranks)
        self.layer_id = layer_id
        self.capacity_factor = capacity_factor
        # Represent capacity_factor as an exact integer fraction (computed
        # once, here, on the plain Python float -- never on a traced value)
        # so `_capacity` below can compute using only `*`/`//`, both of
        # which nnScaler traces as plain arithmetic ops needing no extra
        # runtime-function import. `num_local_tokens` (`T = x.shape[0]`) is
        # itself a *traced* value (a `ConcreteProxy`, not a plain Python
        # int), which does not support `math.ceil`/`round` (confirmed while
        # building this: both raised/failed -- `math.ceil` traces but the
        # generated `_train_step` module never imports `math`, a genuine
        # codegen gap out of scope to patch for Step C; `round()` outright
        # has no `ConcreteProxy.__round__`), so avoiding both entirely,
        # rather than patching either gap, is the robust fix.
        self._cf_num, self._cf_den = capacity_factor.as_integer_ratio()
        self.gate = nn.Linear(dim, self.num_experts, bias=False)
        self.expert_up = nn.Linear(dim, ffn_hidden, bias=False)
        self.expert_down = nn.Linear(ffn_hidden, dim, bias=False)

    def _capacity(self, num_local_tokens: int) -> int:
        # Integer ceiling division of (num_local_tokens * capacity_factor)
        # by num_experts, via the standard `-(-a // b)` idiom -- see
        # __init__'s comment for why this must stay pure `*`/`//` (traceable
        # even when num_local_tokens is a traced value).
        numerator = num_local_tokens * self._cf_num
        denominator = self.num_experts * self._cf_den
        return max(1, -(-numerator // denominator))

    def forward(self, x: torch.Tensor, use_phases: bool) -> torch.Tensor:
        T = x.shape[0]
        capacity = self._capacity(T)

        if use_phases:
            phase_anchor(self.layer_id, PhaseType.MOE_DISPATCH)
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_idx = torch.argmax(gate_logits, dim=-1)
        gate_weight = gate_probs.gather(1, expert_idx.unsqueeze(1)).squeeze(1)

        buffer, dest = _capacity_scatter(x, expert_idx, self.num_experts, capacity)
        channel_d = f'phase_moe_L{self.layer_id}_dispatch'
        pending = moe_dispatch(buffer, self.ep_ranks, channel=channel_d, max_outstanding=1)

        if use_phases:
            phase_anchor(self.layer_id, PhaseType.EXPERT_COMPUTE)
        dispatched = moe_dispatch_wait(pending)
        flat_in = dispatched.reshape(self.num_experts * capacity, self.dim)
        expert_out = self.expert_down(F.silu(self.expert_up(flat_in)))
        combine_buffer = expert_out.view(self.num_experts, capacity, self.dim)
        channel_c = f'phase_moe_L{self.layer_id}_combine'
        pending2 = moe_combine(combine_buffer, self.ep_ranks, channel=channel_c, max_outstanding=1)

        if use_phases:
            phase_anchor(self.layer_id, PhaseType.MOE_COMBINE)
        combined = moe_combine_wait(pending2)
        out = _capacity_gather(combined, dest, gate_weight)
        return out


class PhaseMoELayer(nn.Module):
    """One transformer layer: ATTENTION then MOE_DISPATCH/EXPERT_COMPUTE/MOE_COMBINE,
    with a residual around each sub-block (pre-norm-free, kept minimal)."""

    def __init__(self, cfg: MoEConfig, ep_ranks: Sequence[int], layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.attn = SelfAttention(cfg.dim, cfg.n_heads, cfg.seq_len)
        self.moe = MoEFFN(cfg.dim, cfg.ffn_hidden, ep_ranks, layer_id, cfg.capacity_factor)

    def forward(self, x: torch.Tensor, use_phases: bool, emit_attention_anchor: bool = True) -> torch.Tensor:
        # `emit_attention_anchor=False` for the model's very first layer:
        # its ATTENTION anchor is emitted by PhaseMoEModel BEFORE the input
        # dict is even unpacked, so it is the first node in the whole traced
        # graph (see PhaseMoEModel.forward) -- required so it lands at
        # index 0 of its own layer_nodes range for AnchorBoundary's "an
        # anchor at position 0 does not split anything off" rule (with no
        # `getitem` ahead of it to push it to a later index).
        if use_phases and emit_attention_anchor:
            phase_anchor(self.layer_id, PhaseType.ATTENTION)
        attn_out = self.attn(x) + x
        # No outer residual around the MoE block (i.e. deliberately not
        # `self.moe(attn_out) + attn_out`): `attn_out` would then be
        # referenced by two *non-adjacent* phases (MOE_DISPATCH's gating
        # input, and a residual add landing in MOE_COMBINE, three phases
        # later) -- a genuine nnScaler adapter-generation limitation found
        # while building this model (`IRAdapterGener.gen_activation`'s
        # `local_consumer_multiref` raises `AttributeError: 'IRSegment'
        # object has no attribute 'recompute'` when a multiref-requiring
        # tensor's two consumers sit in different, already-lowered phase
        # segments -- Step B's own local-segment tests never exercised this,
        # since their simple feedforward-chain model has no skip
        # connections at all). Fixing that core-file bug is out of scope
        # for Step C (a cross-cutting change to shared, heavily-used adapter
        # generation, not a phase-IR/scheduling change); documented here
        # honestly and worked around by keeping residual connections
        # same-phase-adjacent only (attention's own `+ x` above is fine: both
        # uses of `x` are within the single ATTENTION phase segment).
        moe_out = self.moe(attn_out, use_phases)
        return moe_out


class PhaseMoEModel(nn.Module):
    """``num_stages`` physical stages, each ``layers_per_stage`` MoE layers
    deep, followed by a scalar (MSE-to-target) loss -- ``use_phases``
    switches between the phase-anchored and plain ("serial baseline")
    variants of the exact same math (see module docstring).

    ``ep_ranks_per_stage``: a list of length ``num_stages``, each element the
    tuple of *global* device ranks that stage's EP group spans (also that
    stage's own physical-stage device tuple -- see phase.py's "Same physical
    stage"). A single, module-construction-time Python constant per layer,
    exactly mirroring how ``nnscaler.customized_ops.ring_attention``'s
    ``process_group``-style arguments are plain Python values, not traced
    tensors.
    """

    def __init__(self, cfg: MoEConfig, num_stages: int, layers_per_stage: int,
                 ep_ranks_per_stage: Sequence[Sequence[int]], use_phases: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_phases = use_phases
        self.num_stages = num_stages
        self.layers_per_stage = layers_per_stage
        assert len(ep_ranks_per_stage) == num_stages
        layers = []
        for sid in range(num_stages):
            for _ in range(layers_per_stage):
                layers.append(PhaseMoELayer(cfg, ep_ranks_per_stage[sid], layer_id=len(layers)))
        self.layers = nn.ModuleList(layers)

    def forward(self, data):
        # Emit layer 0's own ATTENTION anchor BEFORE unpacking `data` at all,
        # so it is the very first node in the traced graph (index 0) -- see
        # PhaseMoELayer.forward's docstring for why this matters structurally
        # (a plain `getitem`-before-first-anchor is otherwise unavoidable and
        # breaks AnchorBoundary's position-0 special case). A no-op either
        # way (real or traced) when `use_phases` is False.
        if self.use_phases:
            phase_anchor(0, PhaseType.ATTENTION)
        x = data['data']
        for i, layer in enumerate(self.layers):
            x = layer(x, self.use_phases, emit_attention_anchor=(i > 0))
        target = data['target']
        return F.mse_loss(x, target).view(1)


# ---------------------------------------------------------------------------
# PAS policy: stage the graph, (optionally) lower to phases, partition ops,
# schedule.
# ---------------------------------------------------------------------------
#
# Every real op (linear or otherwise) is replicated across its stage's
# ep_ranks -- see module docstring's "Honest scoping note" for exactly why
# (TP-sharding the batch dim would need registering the capacity-scatter's
# constituent ops with proper partition annotations, out of scope here).


def _layer_node_ranges(all_ops: List, num_stages: int, layers_per_stage: int, use_phases: bool):
    """Return, for each global layer id, the (start, end) index range (into
    `all_ops`) of that layer's forward nodes -- boundaries found via phase
    anchors (if `use_phases`) or via a fixed 5-linears-per-layer count
    otherwise (both variants trace the exact same op sequence, see module
    docstring)."""
    total_layers = num_stages * layers_per_stage
    if use_phases:
        anchor_positions = {
            n.kwargs.get('name'): i for i, n in enumerate(all_ops) if isinstance(n, IRGraphAnchor)
        }
        starts = [anchor_positions[f'__phase__{lid}:attention'] for lid in range(total_layers)]
    else:
        linear_positions = [i for i, n in enumerate(all_ops) if n.name == 'linear']
        assert len(linear_positions) == total_layers * 5
        starts = [linear_positions[lid * 5] for lid in range(total_layers)]
    ends = starts[1:] + [len(all_ops)]
    return list(zip(starts, ends))


#: Dedicated CUDA stream name for the MoE all-to-all issues, mirroring
#: ``test_combined_1f1b_pipeline_e2e.py``'s own ``_pas_multi_stream`` precedent
#: (Step A) for routing inter-segment communication onto a separate stream via
#: the existing, first-class ``StreamContext``/``op_context`` extension point
#: -- no core nnscaler code changes needed here either.
MOE_COMM_STREAM = 'moe_comm'


def _set_moe_stream_context(phase_nodes) -> None:
    """Real stream/event wiring for one layer's phases (only meaningful for
    the MoE phase sequence; a no-op, harmlessly, for a dense/ATTENTION-only
    layer): ``MOE_DISPATCH``'s forward segment (ends with the dispatch issue)
    runs on :data:`MOE_COMM_STREAM`, waiting for the default stream (its
    input buffer must already be built); ``EXPERT_COMPUTE``'s forward
    segment (starts with dispatch's wait, ends with the combine issue) runs
    on the default stream, waiting for :data:`MOE_COMM_STREAM` (dispatch's
    result must be ready) -- exactly the same ``wait_streams`` correctness
    requirement documented (and empirically load-bearing) in
    ``test_combined_1f1b_pipeline_e2e.py``'s ``_pas_multi_stream`` docstring.

    Honest scoping note: ``EXPERT_COMPUTE``'s own combine-issue therefore
    runs on the *default* stream (not `MOE_COMM_STREAM`), an asymmetry kept
    for simplicity -- one ``StreamContext`` applies to a whole segment (see
    ``nnscaler.codegen.schedule.schedule._emit_stream_context``), and
    ``EXPERT_COMPUTE`` is a single segment that both waits for dispatch and
    issues combine. This does not affect correctness (the schedule's
    program-order interleave, not stream placement, is what the phase IR's
    issue/wait channel-tracking already guarantees is safe -- see
    ``nnscaler/runtime/adapter/moe.py``), only how much of the combine
    window's overlap is genuine hardware-level stream concurrency versus
    NCCL's own internal async execution.
    """
    by_type = {pn.identity.phase_type: pn for pn in phase_nodes}
    dispatch = by_type.get(PhaseType.MOE_DISPATCH)
    expert = by_type.get(PhaseType.EXPERT_COMPUTE)
    if dispatch is None or expert is None:
        return  # dense (ATTENTION-only) layer: nothing to do
    dispatch.segment.set_op_context(
        'stream_context', StreamContext(stream=MOE_COMM_STREAM, wait_streams=['default']))
    expert.segment.set_op_context(
        'stream_context', StreamContext(stream='default', wait_streams=[MOE_COMM_STREAM]))


def make_pas(num_stages: int, layers_per_stage: int, ep_ranks_per_stage: Sequence[Sequence[int]],
             use_phases: bool):
    """Build a ``parallelize(..., pas_fn, ...)``-compatible PAS policy.

    Stages the graph into ``num_stages`` physical stages (each
    ``layers_per_stage`` MoE layers), and, per layer: if ``use_phases``,
    lowers it to its 4 phases (:func:`lower_layer_to_phases`); otherwise
    groups the whole stage into one plain segment (Step A/B-style baseline).
    Every real op is replicated (``nnscaler.policies._replica``) across its
    stage's ``ep_ranks`` (see module docstring's "Honest scoping note").
    Schedules with :meth:`PhaseAwareSched.sched_1f1b_phase_aware` (phase
    variant) or ``PredefinedSched.sched_1f1b`` (plain variant).
    """
    def pas(graph: IRGraph, config: ComputeConfig):
        from nnscaler.graph.schedule.predefined import PredefinedSched

        nmicros = config.pas_config['pipeline_nmicros']
        all_ops = [n for n in graph.nodes() if isinstance(n, IRFwOperation)]
        layer_ranges = _layer_node_ranges(all_ops, num_stages, layers_per_stage, use_phases)
        dataloaders = list(graph.select(ntype=IRDataOperation))

        # PhaseMoEModel emits layer 0's ATTENTION anchor before unpacking
        # `data` at all specifically so there are never any leading ops
        # ahead of it (see PhaseMoELayer.forward's docstring): a `getitem`
        # ahead of the first anchor would otherwise place it at a non-zero
        # index within layer 0's own layer_nodes range, defeating
        # `AnchorBoundary`'s "an anchor at position 0 does not split
        # anything off" rule (one anchor too many would count as a split).
        # For the PLAIN (non-phase) variant there is no anchor to rely on,
        # so stage 0's start is forced to 0 below instead (`graph.group()`
        # has no such index-0 sensitivity).
        first_boundary = layer_ranges[0][0]
        assert not use_phases or first_boundary == 0, (
            "expected layer 0's ATTENTION anchor to be the very first node "
            f"(got {first_boundary} leading node(s)); PhaseMoEModel.forward "
            "must emit it before unpacking `data`"
        )

        for sid in range(num_stages):
            ep_ranks = list(ep_ranks_per_stage[sid])
            stage_layer_ids = range(sid * layers_per_stage, (sid + 1) * layers_per_stage)
            stage_start = layer_ranges[stage_layer_ids[0]][0]
            stage_end = layer_ranges[stage_layer_ids[-1]][1]
            if sid == 0 and not use_phases:
                stage_start = 0  # fold leading ops directly into stage 0's group

            per_layer_nodes = {lid: all_ops[layer_ranges[lid][0]:layer_ranges[lid][1]] for lid in stage_layer_ids}

            if use_phases:
                for lid in stage_layer_ids:
                    phase_nodes = lower_layer_to_phases(graph, per_layer_nodes[lid], layer_id=lid)
                    for pn in phase_nodes:
                        for node in pn.segment.nodes():
                            _replica(graph, node, devs=ep_ranks)
                    _set_moe_stream_context(phase_nodes)
            else:
                stage_nodes = all_ops[stage_start:stage_end]
                graph.group(stage_nodes)
                for node in stage_nodes:
                    _replica(graph, node, devs=ep_ranks)

        for dl in dataloaders:
            _replica(graph, dl, devs=list(range(config.plan_ngpus)))

        if use_phases:
            validate_phase_layout(graph, num_stages)
            PhaseAwareSched.sched_1f1b_phase_aware(graph, nmicros, num_stages)
        else:
            PredefinedSched.sched_1f1b(graph, nmicros, num_stages)
        return graph

    return pas
