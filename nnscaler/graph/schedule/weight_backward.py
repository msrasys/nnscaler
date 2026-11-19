# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Plan module-local weight backward without changing the pipeline schedule."""
from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatchcase

from nnscaler.ir.cten import IRTensor
from .module_streams import normalize_module_stream_rules, _unsafe, _local_adapter


@dataclass
class WeightBackwardPlan:
    key: str
    stream: str
    anchor: object
    nodes: tuple
    parameters: tuple
    independent: bool = False
    control_output: object = None


def plan_weight_backward(nodes, rules):
    rules = [rule for rule in normalize_module_stream_rules(rules) if rule.weight_backward_on]
    if not rules:
        return []
    positions = {node: i for i, node in enumerate(nodes)}
    producers = {value: node for node in nodes for value in node.oobjs()}
    plans, selected = [], set()
    for rule in rules:
        invocations = defaultdict(list)
        for node in nodes:
            matches = [path for path in (node.module_stack or {}) if fnmatchcase(path, rule.module)]
            if matches:
                path = matches[-1]
                invocations[(path, (node.module_call_stack or {}).get(path))].append(node)
        for (path, _), group in invocations.items():
            if any(node in selected or (_unsafe(node) and not _local_adapter(node)) for node in group):
                raise ValueError(f'{path}: weight backward requires nonoverlapping pure modules')
            parameters = tuple(dict.fromkeys(value for node in group for value in node.iobjs()
                                             if isinstance(value, IRTensor) and value.is_param() and value.requires_grad))
            if not parameters:
                continue
            stack = list(group[0].module_stack)
            parent = stack[stack.index(path) - 1] if stack.index(path) else ''
            call = (group[0].module_call_stack or {}).get(parent)
            anchors = [node for node in nodes if node not in group
                       and fnmatchcase(node.signature, rule.weight_backward_on)
                       and (not parent or parent in (node.module_stack or {}))
                       and (call is None or (node.module_call_stack or {}).get(parent) == call)]
            if len(anchors) != 1:
                raise ValueError(f'{path}: expected one weight backward anchor, found {len(anchors)}')
            anchor = anchors[0]
            if anchor.recompute is not None or positions[anchor] >= positions[group[0]]:
                raise ValueError(f'{path}: weight backward anchor must precede the module outside checkpoints')
            ancestors, pending = set(), list(group)
            while pending:
                node = pending.pop()
                for value in node.iobjs():
                    if not isinstance(value, IRTensor) or not value.requires_grad or value.is_attr():
                        continue
                    producer = producers.get(value)
                    if producer is not None and producer not in ancestors:
                        ancestors.add(producer)
                        pending.append(producer)
            independent = rule.weight_backward_independent
            control_output = None
            if independent:
                if any(node.recompute is not None for node in group):
                    raise ValueError(f'{path}: independent weight backward requires an eager module')
                control_output = next((value for value in anchor.oobjs()
                                       if isinstance(value, IRTensor) and value.requires_grad), None)
                if control_output is None:
                    raise ValueError(f'{path}: independent weight backward requires a differentiable anchor output')
                # Every parameter-consuming operation must lie downstream of
                # a differentiable activation at this module's boundary. The
                # runtime puts registration controls on those inputs.
                group_set = set(group)
                reachable = {value for node in group for value in node.iobjs()
                             if isinstance(value, IRTensor) and value.requires_grad and not value.is_attr()
                             and producers.get(value) not in group_set}
                for node in group:
                    active = any(value in reachable for value in node.iobjs())
                    if any(isinstance(value, IRTensor) and value.is_param() and value.requires_grad
                           for value in node.iobjs()) and not active:
                        raise ValueError(f'{path}: independent weight backward needs differentiable activation paths')
                    if active:
                        reachable.update(value for value in node.oobjs()
                                         if isinstance(value, IRTensor) and value.requires_grad)
            elif anchor not in ancestors:
                raise ValueError(f'{path}: module must have a differentiable data dependency on its anchor')
            key = f'_weight_backward_{len(plans)}'
            plans.append(WeightBackwardPlan(key, rule.stream, anchor, tuple(group), parameters,
                                            independent=independent, control_output=control_output))
            selected.update(group)
    return plans
