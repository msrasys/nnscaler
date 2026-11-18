"""Select a later async boundary for replay of an existing checkpoint."""
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatchcase

from nnscaler.ir.cten import IRTensor
from .module_streams import checkpoint_stream, normalize_module_stream_rules


@dataclass
class CheckpointPrefetchPlan:
    gid: int
    key: str
    anchor: object
    rule: object


def plan_checkpoint_prefetch(nodes, rules):
    rules = [r for r in normalize_module_stream_rules(rules) if r.recompute_prefetch_on]
    if len({r.recompute_prefetch_bytes for r in rules}) > 1:
        raise ValueError('Checkpoint prefetch rules must share one per-device budget')
    groups = defaultdict(list)
    positions = {node: i for i, node in enumerate(nodes)}
    producers = {value: node for node in nodes for value in node.oobjs()}
    for node in nodes:
        if node.recompute is not None:
            groups[node.recompute].append(node)
    plans, selected = [], set()
    for rule in rules:
        for gid, group in groups.items():
            if checkpoint_stream(group, (rule,)) is None:
                continue
            if positions[group[-1]] - positions[group[0]] + 1 != len(group):
                raise ValueError('Checkpoint prefetch requires one contiguous replay region')
            if gid in selected:
                raise ValueError('A checkpoint may have only one prefetch anchor')
            sample = next(node for node in group if node.module_stack)
            stack = list(sample.module_stack)
            path = next(path for path in reversed(stack) if fnmatchcase(path, rule.module))
            parent = stack[stack.index(path) - 1] if stack.index(path) else ''
            call = (sample.module_call_stack or {}).get(parent)
            anchors = [node for node in nodes if node not in group
                       and fnmatchcase(node.signature, rule.recompute_prefetch_on)
                       and (not parent or parent in (node.module_stack or {}))
                       and (call is None or (node.module_call_stack or {}).get(parent) == call)]
            if len(anchors) != 1:
                raise ValueError(f'{path}: expected one checkpoint prefetch anchor, found {len(anchors)}')
            anchor = anchors[0]
            if anchor.recompute is not None or positions[anchor] <= positions[group[-1]]:
                raise ValueError(f'{path}: prefetch anchor must follow the complete checkpoint')
            ancestors, pending = set(), [anchor]
            while pending:
                current = pending.pop()
                for value in current.iobjs():
                    if not isinstance(value, IRTensor) or not value.requires_grad or value.is_attr():
                        continue
                    producer = producers.get(value)
                    if producer is not None and producer not in ancestors:
                        ancestors.add(producer)
                        pending.append(producer)
            if not ancestors.intersection(group):
                raise ValueError(f'{path}: prefetch anchor must depend on this checkpoint')
            plans.append(CheckpointPrefetchPlan(gid, f'_checkpoint_prefetch_{len(plans)}', anchor, rule))
            selected.add(gid)
    return plans
