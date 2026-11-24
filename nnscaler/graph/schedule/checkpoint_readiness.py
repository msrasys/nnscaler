# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bind checkpoint recomputation to a downstream backend's input readiness."""

from collections import defaultdict
from fnmatch import fnmatchcase

from nnscaler.ir.cten import IRTensor

from .module_streams import checkpoint_stream, normalize_module_stream_rules


def plan_checkpoint_readiness(nodes, outputs, rules):
    rules = [rule for rule in normalize_module_stream_rules(rules) if rule.recompute_ready_on]
    if not rules:
        return {}
    groups, users = defaultdict(list), defaultdict(set)
    producers = {value: node for node in nodes for value in node.oobjs()}
    positions = {node: index for index, node in enumerate(nodes)}
    for node in nodes:
        if node.recompute is not None:
            groups[node.recompute].append(node)
        for value in node.iobjs():
            users[value].add(node)
    result = {}
    for rule in rules:
        for gid, group in groups.items():
            if checkpoint_stream(group, (rule,)) is None:
                continue
            sample = next(node for node in group if node.module_stack)
            stack = list(sample.module_stack)
            path = next(path for path in reversed(stack) if fnmatchcase(path, rule.module))
            parents = list(reversed(stack[:stack.index(path)]))
            if not parents or parents[-1] != '':
                parents.append('')
            anchors = []
            # A selected compute child can live inside a checkpoint wrapper;
            # its asynchronous consumer then belongs to an enclosing module.
            # Keep invocation identity and the sole-consumer proof below.
            for parent in parents:
                call = (sample.module_call_stack or {}).get(parent)
                anchors = [node for node in nodes if node not in group
                           and fnmatchcase(node.signature, rule.recompute_ready_on)
                           and (not parent or parent in (node.module_stack or {}))
                           and (call is None or (node.module_call_stack or {}).get(parent) == call)]
                if anchors:
                    break
            if len(anchors) != 1:
                raise ValueError(f'{path}: expected one recompute readiness anchor, found {len(anchors)}')
            anchor = anchors[0]
            if anchor.recompute is not None or positions[anchor] <= positions[group[-1]]:
                raise ValueError(f'{path}: recompute readiness anchor must follow the whole checkpoint')
            # A different differentiable consumer could request recomputation
            # before the selected backend's backward has recorded readiness.
            # Restrict to a direct, unique consumer instead of risking a wait
            # cycle or silently using an event from another invocation.
            group_set = set(group)
            # Materialization can place an input multiref inside a checkpoint.
            # Its pass-through outputs have no saved tensors and cannot trigger
            # recomputation. Do not exempt a multiref of a computed activation.
            passthrough = {value for node in group if node.name == 'multiref'
                           and all(producers.get(arg) not in group_set for arg in node.iobjs())
                           for value in node.oobjs()}
            gradients = {value for node in group for value in node.oobjs()
                         if isinstance(value, IRTensor) and value.requires_grad
                         and value not in passthrough
                         and (value in outputs or users[value] - group_set)}
            if not gradients or any(value in outputs or users[value] - group_set != {anchor}
                                    for value in gradients):
                raise ValueError(f'{path}: readiness anchor must be the sole direct gradient consumer')
            if gid in result and result[gid] is not anchor:
                raise ValueError(f'{path}: conflicting recompute readiness anchors')
            result[gid] = anchor
    return result
