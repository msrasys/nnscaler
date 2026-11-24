# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Compose selected checkpoint regions using resolved PAS layouts."""

from collections import Counter
from dataclasses import dataclass, replace
from itertools import groupby
from types import MappingProxyType
from typing import Mapping, Optional

from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir import IRSubTensor


def get_recompute_span(nodes):
    """Return the effective [start, end) used by ``IRGraph.recompute``.

    Leading/trailing ops without a gradient-bearing activation input remain
    outside checkpoints. Composition must not absorb them just because they
    sit between two selected groups. Keep this rule shared with graph.recompute.
    """
    def participates(node):
        return any(isinstance(t, IRSubTensor) and not t.is_attr() and t.grad is not None
                   for t in node.inputs())

    start, end = 0, len(nodes)
    while start < end and not participates(nodes[start]):
        start += 1
    while end > start and not participates(nodes[end - 1]):
        end -= 1
    return start, end


@dataclass(frozen=True)
class _PlanLayouts:
    combined: Optional[Mapping]
    outputs: Optional[Mapping]


def _add_layout(layouts, key, layout):
    return layouts.setdefault(key, layout) == layout


def _collect_plan_layouts(plan, nodes, degree):
    """Snapshot all ranks before staging; input conflicts do not erase outputs."""
    if nodes is None or len(nodes) != degree:
        return _PlanLayouts(None, None)
    combined, outputs = {}, {}
    combined_valid = outputs_valid = True
    for rank, node in enumerate(nodes):
        for tensor in node.oobjs():
            if isinstance(tensor, IRSubTensor):
                key = (rank, tensor.parent)
                layout = (tensor.indmap, tensor.valmap, False)
                if not _add_layout(outputs, key, layout):
                    outputs_valid = False
                if not _add_layout(combined, key, layout):
                    combined_valid = False
        # Repeated argument slots can carry different gradient reductions.
        slots = [(tensor, index) for index, tensor in enumerate(node.inputs())]
        direct = {id(tensor) for tensor, _ in slots}
        slots += [(tensor, None) for tensor in node.iobjs() if id(tensor) not in direct]
        for tensor, input_index in slots:
            if isinstance(tensor, IRSubTensor):
                key = (rank, tensor.parent)
                grad_reduce = False
                if (degree > 1 and plan.partition is not None
                        and tensor.requires_grad and not tensor.splitdims()
                        and input_index is not None):
                    grad_reduce = not node.ignore_grad_reduce(input_idx=input_index)
                layout = (tensor.indmap, tensor.valmap, grad_reduce)
                if not _add_layout(combined, key, layout):
                    combined_valid = False
    return _PlanLayouts(
        MappingProxyType(combined) if combined_valid and outputs_valid else None,
        MappingProxyType(outputs) if outputs_valid else None,
    )


def _resolve_plan_layouts(plan, degree):
    nodes = None
    if plan.partition is None:
        nodes = [plan.op] * degree
    elif plan.partition != 'auto' and isinstance(plan.op, IRDimops):
        nodes = plan.op.algorithm('dim').instantiate(
            idx=plan.partition.input, dim=plan.partition.dim, num=degree)
    return _collect_plan_layouts(plan, nodes, degree)


def merge_recompute_regions(plans, *, partition_degree=1,
                            hooks_allow_recompute=None, _resolved_layouts=None,
                            region_key=None):
    """Return copied plans and a report after composing maximal legal chains.

    Existing selection supplies the replay contract, including opaque ops.
    Preserve edge trimming, known effects, hooks, stage/offload boundaries and
    actual layouts, including external producers. No unselected op is absorbed.
    Plans must be resolved and graph.recompute not yet applied. External uses
    remain outputs; composition does not guarantee lower peak memory.

    ``fn`` supplies its immutable layout cache; standalone callers resolve each
    operation once. The cache is local to this pass, before staging changes IR.
    """
    if (isinstance(partition_degree, bool) or not isinstance(partition_degree, int)
            or partition_degree < 1):
        raise ValueError('partition_degree must be a positive integer')
    plans = [replace(plan) for plan in plans]
    layout_cache = dict(_resolved_layouts or {})

    def layout_for(plan):
        if plan.op not in layout_cache:
            layout_cache[plan.op] = _resolve_plan_layouts(plan, partition_degree)
        return layout_cache[plan.op]

    regions = [list(group) for _, group in groupby(plans, key=lambda p: p.recompute_id)]
    seen = set()
    for region in regions:
        rid = region[0].recompute_id
        if rid < -1:
            raise ValueError('Invalid recompute_id')
        if rid != -1:
            if rid in seen:
                raise ValueError('Nodes in a recompute group must be continuous')
            seen.add(rid)
            if len({p.stage_id for p in region}) != 1:
                raise ValueError('All ops in a recompute group must be in the same stage')
        if any(p.op.recompute is not None for p in region):
            raise ValueError('Merge OpPlans before graph.recompute is applied')

    trimmed = []
    for region in regions:
        if region[0].recompute_id == -1:
            continue
        start, end = get_recompute_span([p.op for p in region])
        for plan in region[:start] + region[end:]:
            plan.recompute_id = -1
            trimmed.append(plan.op.cid)
    regions = [list(group) for _, group in groupby(plans, key=lambda p: p.recompute_id)]
    effective_groups = {p.recompute_id for p in plans if p.recompute_id != -1}

    # Keep every producer, including those outside either candidate region.
    producers = {}
    for plan in plans:
        for tensor in plan.op.oobjs():
            if isinstance(tensor, IRSubTensor) and not tensor.is_attr():
                producers.setdefault(tensor.parent, []).append(plan)

    blocked = Counter()
    merged = []
    previous = None

    def describe(region):
        if region[0].recompute_id == -1:
            return None, 'not_selected'
        if region[0].stage_id < 0:
            return None, 'unresolved_stage'
        offload = {p.offload_id != -1 or p.op.offload is not None for p in region}
        if len(offload) != 1:
            return None, 'mixed_offload'
        layouts, inputs, outputs = {}, set(), set()
        region_plans = {id(plan) for plan in region}
        for plan in region:
            op = plan.op
            if getattr(op, 'recompute_barriers', ()):
                return None, 'effect'
            name = (op.signature or '').rsplit('.', 1)[-1]
            if (name.endswith('_') or op.kwargs.get('inplace', False)
                    or op.kwargs.get('out') is not None
                    or name in ('setitem', 'setattr', 'delitem', 'delattr')
                    or (name in ('iadd', 'isub', 'imul', 'itruediv', 'ifloordiv',
                                 'imod', 'ipow', 'imatmul', 'iand', 'ior', 'ixor',
                                 'ilshift', 'irshift')
                        and any(isinstance(t, IRSubTensor) for t in op.iobjs()))):
                return None, 'mutation'
            if plan.pre_hook is not None or plan.post_hook is not None:
                if (hooks_allow_recompute is None or not hooks_allow_recompute(
                        plan.pre_hook, plan.post_hook)):
                    return None, 'hook'
            local = layout_for(plan).combined
            if local is None or any(k in layouts and layouts[k] != v for k, v in local.items()):
                return None, 'layout'
            layouts.update(local)
            inputs.update(t.parent for t in op.iobjs()
                          if isinstance(t, IRSubTensor) and not t.is_attr())
            outputs.update(t.parent for t in op.oobjs() if isinstance(t, IRSubTensor))
        # An external conversion may already sit at the previous checkpoint's
        # tail; composing regions can make that adapter run during replay.
        for tensor in inputs:
            for producer in producers.get(tensor, ()):
                if id(producer) in region_plans:
                    continue
                produced = layout_for(producer).outputs
                if produced is None or any(
                        produced.get((rank, tensor)) != layouts.get((rank, tensor))
                        for rank in range(partition_degree)):
                    return None, 'layout'
        return dict(plans=region, layouts=layouts, inputs=inputs, outputs=outputs,
                    offload=offload.pop(),
                    scheduling_key=region_key([p.op for p in region]) if region_key else None), None

    for region in regions:
        current, reason = describe(region)
        if current is None:
            blocked[reason] += 1
            previous = None
            continue
        if previous is not None:
            reason = None
            if previous['plans'][0].stage_id != region[0].stage_id:
                reason = 'stage'
            elif previous['offload'] != current['offload']:
                reason = 'offload'
            elif previous['scheduling_key'] != current['scheduling_key']:
                reason = 'scheduling'
            elif not previous['outputs'] & current['inputs']:
                reason = 'no_dependency'
            elif any(k in previous['layouts'] and previous['layouts'][k] != v
                     for k, v in current['layouts'].items()):
                reason = 'layout'
            if reason is None:
                leader = previous['plans'][0].recompute_id
                merged.append((leader, region[0].recompute_id))
                for plan in region:
                    plan.recompute_id = leader
                for key in ('layouts', 'inputs', 'outputs'):
                    previous[key].update(current[key])
                previous['plans'].extend(region)
                continue
            blocked[reason] += 1
        previous = current
    return plans, dict(groups_before=len(effective_groups),
                       groups_after=len(effective_groups) - len(merged),
                       merged_boundaries=merged, barriers=dict(blocked),
                       selected_groups=len(seen), trimmed_edge_ops=trimmed)
