# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bind cooperative dW regions to an earlier asynchronous backward anchor."""

from fnmatch import fnmatchcase

from nnscaler.ir.cten import IRTensor
from .module_streams import normalize_module_stream_rules


def plan_weight_backward(nodes, rules):
    positions = {node: i for i, node in enumerate(nodes)}
    producers = {value: node for node in nodes for value in node.oobjs()}
    result = {}
    for rule in normalize_module_stream_rules(rules):
        if not (rule.weight_backward_only and rule.backward_on):
            continue
        for node in nodes:
            stack = list(node.module_stack or {})
            matches = [path for path in stack if fnmatchcase(path, rule.module)]
            if not matches or not any(fnmatchcase(node.signature, p) for p in rule.operators):
                continue
            if not any(isinstance(v, IRTensor) and v.requires_grad for v in node.oobjs()):
                continue
            if not any(isinstance(v, IRTensor) and v.is_param() and v.requires_grad for v in node.iobjs()):
                continue
            path = matches[-1]
            parents = list(reversed(stack[:stack.index(path) + 1]))
            if not parents or parents[-1] != '':
                parents.append('')
            candidates = []
            for parent in parents:
                call = (node.module_call_stack or {}).get(parent)
                candidates = [anchor for anchor in nodes if anchor is not node
                              and fnmatchcase(anchor.signature, rule.backward_on)
                              and (not parent or parent in (anchor.module_stack or {}))
                              and (call is None or (anchor.module_call_stack or {}).get(parent) == call)]
                if candidates:
                    break
            if len(candidates) != 1:
                raise ValueError(f'{path}: expected one weight backward anchor, found {len(candidates)}')
            anchor = candidates[0]
            first = next(n for n in nodes if n.recompute == node.recompute) if node.recompute is not None else node
            if anchor.recompute is not None or positions[anchor] >= positions[first]:
                raise ValueError(f'{path}: weight backward anchor must precede the whole compute/checkpoint region')
            if not any(isinstance(v, IRTensor) and v.requires_grad and not v.is_attr() for v in anchor.iobjs()):
                raise ValueError(f'{path}: weight backward anchor needs a differentiable input')
            pending = [node]
            ancestors = set()
            while pending:
                current = pending.pop()
                for value in current.iobjs():
                    producer = producers.get(value)
                    if producer is not None and producer not in ancestors:
                        ancestors.add(producer)
                        pending.append(producer)
            if anchor not in ancestors:
                raise ValueError(f'{path}: weight backward anchor must be an upstream dependency')
            if node in result and result[node] != (anchor, rule.stream):
                raise ValueError(f'{path}: conflicting weight backward anchors')
            result[node] = (anchor, rule.stream)
    return result
