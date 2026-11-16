# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Opt-in, dependency-checked scheduling of independent module branches.

Rules select *pure* module invocations by qualified-name glob. No model-specific
operators are recognized here. Scheduling is local to a materialized segment;
pipeline stages, collectives and checkpoint boundaries retain their order.
"""

from collections import defaultdict
from dataclasses import dataclass
from fnmatch import fnmatchcase
import json
import os

from nnscaler.ir.operator import IRFwOperation


@dataclass(frozen=True)
class ModuleStreamRule:
    module: str
    stream: str = 'module_overlap'
    launch_before: str | None = None
    split_before: str | None = None
    resume_before: str | None = None

    def __post_init__(self):
        for name in ('module', 'stream'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('launch_before', 'split_before', 'resume_before'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f'{name} must be a nonempty string or None')
        if self.stream == 'default':
            raise ValueError('Module streams must use a distinct, named stream')
        if (self.split_before is None) != (self.resume_before is None):
            raise ValueError('split_before and resume_before must be specified together')


def module_stream_rules_from_env():
    """Resolve once into ComputeConfig so cache reuse includes the actual rules."""
    return normalize_module_stream_rules(json.loads(os.environ.get('NNSCALER_MODULE_STREAMS', '[]')))


def normalize_module_stream_rules(rules):
    if not isinstance(rules, (list, tuple)):
        raise ValueError('module_streams must be a list of module stream rules')
    return tuple(rule if isinstance(rule, ModuleStreamRule) else ModuleStreamRule(**rule) for rule in rules)


@dataclass
class ModuleStreamRegion:
    key: str
    stream: str
    nodes: list
    inputs: tuple
    outputs: tuple


def _unsafe(node):
    name = node.signature.rsplit('.', 1)[-1]
    return (not isinstance(node, IRFwOperation)
            or node.kwargs.get('inplace', False)
            or (name.endswith('_') and not name.startswith('__'))
            or name in ('__setitem__', 'setitem', 'dropout', 'batch_norm', 'rand',
                        'randn', 'rand_like', 'randn_like', 'bernoulli', 'multinomial'))


def plan_module_streams(nodes, outputs, rules):
    """Return reordered nodes, side regions and consumer joins; never mutate IR.

    A launch anchor is an operator signature glob in the selected module's
    enclosing invocation. ``split_before`` is a relative child-module glob.
    Known mutations/random operators and invalid data dependencies fail closed.
    The caller explicitly opts in to reordering pure custom operators.
    """
    nodes = list(nodes)
    if not rules:
        return nodes, {}, {}
    positions = {node: i for i, node in enumerate(nodes)}
    producers = {obj: node for node in nodes for obj in node.oobjs()}
    consumers = defaultdict(list)
    for node in nodes:
        for obj in node.iobjs():
            consumers[obj].append(node)
    graph_outputs = set(outputs)
    regions, placements, selected = {}, defaultdict(list), set()

    for rule in normalize_module_stream_rules(rules):
        invocations = defaultdict(list)
        scopes = {}
        for node in nodes:
            stack = list((node.module_stack or {}).keys())
            matches = [path for path in stack if fnmatchcase(path, rule.module)]
            if not matches:
                continue
            path = matches[-1]
            calls = node.module_call_stack or {}
            key = (path, calls.get(path))
            invocations[key].append(node)
            parent = stack[stack.index(path) - 1] if stack.index(path) else ''
            scopes[key] = (parent, calls.get(parent))

        for invocation, branch in invocations.items():
            path, _ = invocation
            if any(n in selected for n in branch):
                raise ValueError(f'Overlapping module stream rules for {path}')
            if any(_unsafe(n) for n in branch):
                raise ValueError(f'Module stream rule for {path} includes a mutation or unsupported operation')
            chunks = [branch]
            if rule.split_before is not None:
                split = next((i for i, n in enumerate(branch) if any(
                    fnmatchcase(p[len(path) + 1:], rule.split_before)
                    for p in (n.module_stack or {}) if p.startswith(path + '.'))), None)
                if split is None or split == 0:
                    raise ValueError(f'No nonempty split at {rule.split_before!r} in {path}')
                chunks = [branch[:split], branch[split:]]

            parent, call = scopes[invocation]
            for chunk, anchor in zip(chunks, (rule.launch_before, rule.resume_before)):
                destination = positions[chunk[0]]
                if anchor:
                    candidates = [n for n in nodes if n not in branch
                                  and fnmatchcase(n.signature, anchor)
                                  and (not parent or parent in (n.module_stack or {}))
                                  and (call is None or (n.module_call_stack or {}).get(parent) == call)]
                    if len(candidates) != 1:
                        raise ValueError(f'{path}: expected one launch anchor {anchor!r}, found {len(candidates)}')
                    destination = positions[candidates[0]]
                if destination > positions[chunk[0]]:
                    raise ValueError(f'{path}: module stream launch must not move computation later')
                if any(_unsafe(n) for n in nodes[destination:positions[chunk[0]]]
                       if n.signature.rsplit('.', 1)[-1] != 'multiref'
                       and not n.signature.startswith('nnscaler.runtime.adapter')):
                    # Most IR adapters are not ordinary operations. Cross-stream
                    # movement across those cannot preserve collective ordering.
                    raise ValueError(f'{path}: cannot move a module across a mutation or unsupported operation')
                chunk_set = set(chunk)
                gids = {n.recompute for n in chunk if n.recompute is not None}
                if any(n.recompute in gids and n not in chunk_set for n in nodes):
                    raise ValueError(f'{path}: module stream split crosses a recompute region')
                if (destination > 0 and nodes[destination].recompute is not None
                        and nodes[destination].recompute == nodes[destination - 1].recompute):
                    raise ValueError(f'{path}: launch anchor is inside a recompute region')
                inputs = tuple(dict.fromkeys(obj for n in chunk for obj in n.iobjs()
                                             if producers.get(obj) not in chunk_set))
                region_outputs = tuple(dict.fromkeys(obj for n in chunk for obj in n.oobjs()
                                                     if obj in graph_outputs or any(
                                                         c not in chunk_set for c in consumers[obj])))
                key = f'module_stream_{len(regions)}'
                region = ModuleStreamRegion(key, rule.stream, chunk, inputs, region_outputs)
                regions[key] = region
                placements[destination].append(region)
                selected.update(chunk)

    ordered = []
    for i, node in enumerate(nodes):
        for region in placements[i]:
            ordered.extend(region.nodes)
        if node not in selected:
            ordered.append(node)
    new_positions = {node: i for i, node in enumerate(ordered)}
    for node in ordered:
        for obj in node.iobjs():
            producer = producers.get(obj)
            if producer is not None and producer != node and new_positions[producer] >= new_positions[node]:
                raise ValueError(f'Module stream launch violates dependency {producer.signature} -> {node.signature}')

    # A stream join is needed at the first consumer outside each region. Later
    # consumers are ordered by the caller stream; same-stream regions need none.
    owners = {n: r for r in regions.values() for n in r.nodes}
    joins = defaultdict(list)
    for region in regions.values():
        for node in ordered:
            consumer_region = owners.get(node)
            if consumer_region is region:
                continue
            if any(producers.get(obj) in region.nodes for obj in node.iobjs()):
                if consumer_region is None or consumer_region.stream != region.stream:
                    target = node if consumer_region is None else consumer_region.nodes[0]
                    joins[target].append(region.key)
                    break
    return ordered, regions, dict(joins)
