# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Opt-in, dependency-checked scheduling of independent module branches.

Rules select *pure* module invocations by qualified-name glob. No model-specific
operators are recognized here. Scheduling is local to a materialized segment;
pipeline stages, collectives and checkpoint boundaries retain their order.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from fnmatch import fnmatchcase
import json
import os

from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.cten import IRObject
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import (
    IdentityPrim, SelectPrim, MergeDimPrim, SumPrim, ChunkPrim, VChunkPrim, SplitAllGatherPrim,
)


@dataclass(frozen=True)
class ModuleStreamRule:
    module: str
    stream: str = 'module_overlap'
    launch_before: str | None = None
    split_before: str | None = None
    resume_before: str | None = None
    launch_after: str | None = None
    resume_after: str | None = None
    host_concurrency: bool = False
    async_anchor_copies: bool = False
    recompute_only: bool = False
    launch_on: str | None = None
    resume_on: str | None = None
    overlap_backward: bool = False
    launch_phase: str = 'submitted'
    kernel_graphs: bool = False
    recompute_ready_on: str | None = None
    weight_backward_on: str | None = None
    backward_on: str | None = None
    # Persist the generated-code ABI, including in pickled configurations.
    # A default factory leaves old pickles detectably missing the field.
    _codegen_version: int = field(default_factory=lambda: 14, init=False, repr=False)

    def __post_init__(self):
        if self.weight_backward_on is not None:
            if not isinstance(self.weight_backward_on, str) or not self.weight_backward_on:
                raise ValueError('weight_backward_on must be a nonempty signature pattern')
            if (self.recompute_only or self.launch_before or self.launch_after or self.launch_on
                    or self.split_before or self.resume_before or self.resume_after or self.resume_on
                    or self.host_concurrency or self.async_anchor_copies or self.overlap_backward or self.backward_on
                    or self.kernel_graphs or self.launch_phase != 'submitted'):
                raise ValueError('Weight backward rules cannot also move forward computation')
        if self.recompute_ready_on is not None and not self.recompute_only:
            raise ValueError('recompute_ready_on requires a recompute_only rule')
        if type(self.kernel_graphs) is not bool:
            raise ValueError('kernel_graphs must be a boolean')
        if self.kernel_graphs and self.launch_phase != 'input_ready':
            raise ValueError('kernel_graphs requires an input_ready backend launch point')
        if self.launch_phase not in ('submitted', 'input_ready'):
            raise ValueError('launch_phase must be submitted or input_ready')
        if self.launch_phase == 'input_ready' and (not self.launch_on
                or (self.split_before and not self.resume_on)):
            raise ValueError('input_ready requires launch_on/resume_on backend boundaries')
        if type(self.recompute_only) is not bool:
            raise ValueError('recompute_only must be a boolean')
        if self.recompute_only and (self.launch_before or self.launch_after or self.split_before
                                    or self.resume_before or self.resume_after
                                    or self.host_concurrency or self.async_anchor_copies
                                    or self.launch_on or self.resume_on or self.overlap_backward):
            raise ValueError('Recompute-only rules cannot also move forward computation')
        if type(self.host_concurrency) is not bool:
            raise ValueError('host_concurrency must be a boolean')
        if type(self.async_anchor_copies) is not bool:
            raise ValueError('async_anchor_copies must be a boolean')
        if self.async_anchor_copies and not self.launch_after:
            raise ValueError('async_anchor_copies requires launch_after')
        for name in ('module', 'stream'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'{name} must be a nonempty string')
        for name in ('launch_before', 'split_before', 'resume_before', 'launch_after', 'resume_after',
                     'launch_on', 'resume_on', 'recompute_ready_on', 'backward_on'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f'{name} must be a nonempty string or None')
        if self.stream == 'default':
            raise ValueError('Module streams must use a distinct, named stream')
        if (sum(bool(x) for x in (self.launch_before, self.launch_after, self.launch_on)) > 1
                or sum(bool(x) for x in (self.resume_before, self.resume_after, self.resume_on)) > 1):
            raise ValueError('Use either before or after for each launch anchor')
        if (self.split_before is None) != (self.resume_before is None and self.resume_after is None and self.resume_on is None):
            raise ValueError('split_before and a resume anchor must be specified together')
        if type(self.overlap_backward) is not bool:
            raise ValueError('overlap_backward must be a boolean')
        if self.launch_on or self.resume_on:
            if self.host_concurrency or self.async_anchor_copies:
                raise ValueError('Explicit launch points cannot use host concurrency or implicit async copies')
        if self.overlap_backward and (not (self.launch_on or self.launch_before)
                or (self.split_before and not (self.resume_on or self.resume_before))
                or self.host_concurrency):
            raise ValueError('Backward overlap requires synchronous host submission before or at each anchor')
        if self.backward_on is not None and (not self.overlap_backward or not self.launch_before
                or (self.split_before and not self.resume_before) or self.host_concurrency):
            raise ValueError('backward_on requires eager launch_before/resume_before with overlap_backward')


def module_stream_rules_from_env():
    """Resolve once into ComputeConfig so cache reuse includes the actual rules."""
    return normalize_module_stream_rules(json.loads(os.environ.get('NNSCALER_MODULE_STREAMS', '[]')))


def normalize_async_copy_ops(patterns):
    if not isinstance(patterns, (list, tuple)) or any(not isinstance(p, str) or not p for p in patterns):
        raise ValueError('async_copy_ops must be a list of nonempty operator signature patterns')
    return tuple(patterns)


def async_copy_ops_from_env():
    return normalize_async_copy_ops(json.loads(os.environ.get('NNSCALER_ASYNC_COPY_OPS', '[]')))


def normalize_module_stream_rules(rules):
    if not isinstance(rules, (list, tuple)):
        raise ValueError('module_streams must be a list of module stream rules')
    def normalize(rule):
        if isinstance(rule, ModuleStreamRule):
            return rule
        values = dict(rule)
        values.pop('_codegen_version', None)  # asdict/YAML round trips use the current compiler
        return ModuleStreamRule(**values)
    return tuple(normalize(rule) for rule in rules)


@dataclass
class ModuleStreamRegion:
    key: str
    stream: str
    nodes: list
    inputs: tuple
    outputs: tuple
    ready_before: object = None
    host_concurrency: bool = False
    async_anchor_copies: bool = False
    caller_outputs: tuple = ()
    deferred: bool = False
    overlap_backward: bool = False
    backward_anchor: object = None
    native: bool = False
    kernel_graphs: bool = False
    retain_backward_outputs: bool = False


def _unsafe(node):
    if not isinstance(node, IRFwOperation):
        return True
    name = node.signature.rsplit('.', 1)[-1]
    return (node.kwargs.get('inplace', False)
            or (name.endswith('_') and not name.startswith('__'))
            or name in ('__setitem__', 'setitem', 'dropout', 'batch_norm', 'rand',
                        'randn', 'rand_like', 'randn_like', 'bernoulli', 'multinomial'))


def _local_adapter(node):
    # A SplitAllGather executes only a local slice in forward. Its backward
    # collective follows autograd's stream semantics, like any original op.
    return isinstance(node, IRAdapter) and all(type(prim) in (
        IdentityPrim, SelectPrim, MergeDimPrim, SumPrim, ChunkPrim, VChunkPrim, SplitAllGatherPrim,
    ) for prim in node.prims)


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
        if rule.recompute_only or rule.weight_backward_on:
            continue
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
            original_branch = set(branch)
            if any(n in selected for n in branch):
                raise ValueError(f'Overlapping module stream rules for {path}')
            if any(_unsafe(n) for n in branch):
                raise ValueError(f'Module stream rule for {path} includes a mutation or unsupported operation')
            # Materialization inserts local layout adapters with no module
            # stack. Include those used exclusively by this branch; otherwise
            # a reshape can move while its slice remains at the old location.
            branch_set = set(branch)
            changed = True
            while changed:
                changed = False
                for node in nodes:
                    if node in branch_set or node in selected:
                        continue
                    if not (_local_adapter(node) or node.name == 'multiref'):
                        continue
                    users = {c for obj in node.oobjs() for c in consumers[obj]}
                    if users and users <= branch_set and not any(obj in graph_outputs for obj in node.oobjs()):
                        branch_set.add(node)
                        changed = True
            branch = sorted(branch_set, key=positions.__getitem__)
            chunks = [branch]
            if rule.split_before is not None:
                split = next((i for i, n in enumerate(branch) if any(
                    fnmatchcase(p[len(path) + 1:], rule.split_before)
                    for p in (n.module_stack or {}) if p.startswith(path + '.'))), None)
                if split is None or split == 0:
                    raise ValueError(f'No nonempty split at {rule.split_before!r} in {path}')
                chunks = [branch[:split], branch[split:]]

            parent, call = scopes[invocation]
            backward_override = None
            if rule.backward_on is not None:
                candidates = [n for n in nodes if n not in branch
                              and fnmatchcase(n.signature, rule.backward_on)
                              and (not parent or parent in (n.module_stack or {}))
                              and (call is None or (n.module_call_stack or {}).get(parent) == call)]
                if len(candidates) != 1:
                    raise ValueError(f'{path}: expected one backward anchor {rule.backward_on!r}, found {len(candidates)}')
                backward_override = candidates[0]
                if backward_override.recompute is not None:
                    raise ValueError(f'{path}: backward anchor must remain outside checkpoints')
            anchors = ((rule.launch_before or rule.launch_after or rule.launch_on,
                        bool(rule.launch_after or rule.launch_on), bool(rule.launch_on)),
                       (rule.resume_before or rule.resume_after or rule.resume_on,
                        bool(rule.resume_after or rule.resume_on), bool(rule.resume_on)))
            for chunk, (anchor, after, deferred) in zip(chunks, anchors):
                native = deferred and rule.launch_phase == 'input_ready'
                deferred = deferred and not native
                destination = positions[chunk[0]]
                ready_before = None
                backward_anchor = backward_override
                if anchor:
                    candidates = [n for n in nodes if n not in branch
                                  and fnmatchcase(n.signature, anchor)
                                  and (not parent or parent in (n.module_stack or {}))
                                  and (call is None or (n.module_call_stack or {}).get(parent) == call)]
                    if len(candidates) != 1:
                        raise ValueError(f'{path}: expected one launch anchor {anchor!r}, found {len(candidates)}')
                    destination = positions[candidates[0]] + int(after)
                    ready_before = candidates[0] if after else None
                    if backward_override is None:
                        backward_anchor = candidates[0] if rule.overlap_backward else None
                # Already-available synthesized inputs need not move with the
                # module. Preserve their original position and other users.
                chunk = [n for n in chunk if n in original_branch or positions[n] >= destination]
                if destination > positions[chunk[0]] and not (deferred or native):
                    raise ValueError(f'{path}: module stream launch must not move computation later')
                begin, end = sorted((destination, positions[chunk[0]]))
                barriers = [n for n in nodes[begin:end] if n not in branch_set
                            if not isinstance(n, IRAdapter)
                            and n.signature.rsplit('.', 1)[-1] != 'multiref' and _unsafe(n)]
                if barriers:
                    raise ValueError(f'{path}: cannot move a module across a mutation or unsupported operation: '
                                     f'{[(type(n).__name__, n.signature) for n in barriers]}')
                # Main-stream adapters retain their original ordering. Moving
                # an independent pure branch past them is safe; dependencies on
                # their outputs are checked below. Adapters in the moved branch
                # itself remain unsupported.
                chunk_set = set(chunk)
                gids = {n.recompute for n in chunk if n.recompute is not None}
                if any(n.recompute in gids and n not in chunk_set for n in nodes):
                    raise ValueError(f'{path}: module stream split crosses a recompute region')
                if (0 < destination < len(nodes) and nodes[destination].recompute is not None
                        and nodes[destination].recompute == nodes[destination - 1].recompute):
                    raise ValueError(f'{path}: launch anchor is inside a recompute region')
                if ready_before is not None and ready_before.recompute is not None:
                    raise ValueError(f'{path}: after-launch anchor cannot be inside a recompute region')
                inputs = tuple(dict.fromkeys(obj for n in chunk for obj in n.iobjs()
                                             if producers.get(obj) not in chunk_set))
                region_outputs = tuple(dict.fromkeys(obj for n in chunk for obj in n.oobjs()
                                                     if obj in graph_outputs or any(
                                                         c not in chunk_set for c in consumers[obj])))
                if deferred or native:
                    from nnscaler.ir.cten import IRTensor
                    if not region_outputs or any(not isinstance(obj, IRTensor) for obj in (*inputs, *region_outputs)):
                        raise ValueError('Deferred module regions require tensor inputs, attributes and outputs')
                    if gids:
                        raise ValueError('Deferred module regions cannot contain checkpoint regions')
                key = f'module_stream_{len(regions)}'
                region = ModuleStreamRegion(key, rule.stream, chunk, inputs, region_outputs,
                                            ready_before, rule.host_concurrency, rule.async_anchor_copies,
                                            deferred=deferred, overlap_backward=rule.overlap_backward,
                                            backward_anchor=backward_anchor, native=native,
                                            kernel_graphs=rule.kernel_graphs,
                                            retain_backward_outputs=backward_override is not None)
                regions[key] = region
                placements[destination].append(region)
                selected.update(chunk)

    ordered = []
    for i, node in enumerate(nodes):
        for region in placements[i]:
            ordered.extend(region.nodes)
        if node not in selected:
            ordered.append(node)
    # An explicit launch point may be the segment's final operation. Keep
    # regions delayed to that boundary in the plan and its tensor lifetimes.
    for region in placements[len(nodes)]:
        ordered.extend(region.nodes)
    new_positions = {node: i for i, node in enumerate(ordered)}
    for node in ordered:
        for obj in node.iobjs():
            producer = producers.get(obj)
            if producer is not None and producer != node and new_positions[producer] >= new_positions[node]:
                raise ValueError(f'Module stream launch violates dependency {producer.signature} -> {node.signature}: {producer}')

    # A stream join is needed at the first consumer outside each region. Later
    # consumers are ordered by the caller stream; same-stream regions need none.
    owners = {n: r for r in regions.values() for n in r.nodes}
    for region in regions.values():
        if region.retain_backward_outputs:
            if region.backward_anchor in owners:
                raise ValueError('Explicit backward anchors must remain on the caller stream')
            if any(other.ready_before is region.backward_anchor and (other.deferred or other.native)
                   for other in regions.values()):
                raise ValueError('Explicit backward anchors cannot also be deferred launch points')
            if any(new_positions[n] >= new_positions[region.backward_anchor] for n in region.nodes):
                raise ValueError('Explicit backward anchor must follow all selected forward chunks')
        region.caller_outputs = tuple(obj for obj in region.outputs if obj in graph_outputs
                                      or any(node not in owners for node in consumers[obj]))
    for region in regions.values():
        if region.ready_before is None:
            continue
        if region.ready_before in owners:
            raise ValueError('After-launch anchors must remain on the caller stream')
        for obj in region.inputs:
            producer = producers.get(obj)
            if producer is None:
                continue
            owner = owners.get(producer)
            if owner is not None and owner.stream == region.stream:
                continue  # same-stream FIFO provides the dependency
            if owner is not None or new_positions[producer] >= new_positions[region.ready_before]:
                raise ValueError('After-launch module inputs must be ready before its anchor '
                                 'or produced earlier on the same module stream')
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


def checkpoint_stream(nodes, rules):
    """Select a whole checkpoint; never silently move part of a larger region."""
    result = None
    named = [node for node in nodes if node.module_stack is not None]
    for rule in normalize_module_stream_rules(rules):
        if not rule.recompute_only:
            continue
        matches = [any(fnmatchcase(path, rule.module) for path in node.module_stack) for node in named]
        if not any(matches):
            continue
        if not all(matches):
            raise ValueError('A recompute stream rule must cover the whole checkpoint region')
        local_updates = _local_dict_updates(nodes)
        unsafe = [node for node in nodes
                  if _unsafe(node) and not _local_adapter(node) and node not in local_updates]
        if unsafe:
            signatures = ', '.join(dict.fromkeys(node.signature for node in unsafe))
            raise ValueError(f'Recompute streams require pure operations and local adapters: {signatures}')
        if result is not None and result != rule.stream:
            raise ValueError('Conflicting recompute streams for one checkpoint')
        result = rule.stream
    return result


def _local_dict_updates(nodes):
    """Recognize kwargs built afresh inside a complete checkpoint invocation.

    A literal dict is emitted as a new Python dict on every execution. The
    runtime setitem returns that same dict, so a chain of direct updates also
    stays local. Neither constant IRObjects nor dictionaries reached through
    getitem establish ownership: those may alias caller state. Tensor writes,
    nested dictionary writes and arbitrary user-defined keys remain rejected.
    This proof is deliberately limited to recomputation, which keeps the
    original forward order and recreates the whole checkpoint's local state.
    """
    owned, updates = set(), set()
    for node in nodes:
        if (node.signature != 'nnscaler.runtime.function.setitem'
                or len(node.inputs()) != 3 or len(node.outputs()) != 1):
            continue
        target, key, _ = node.inputs()
        if type(key) not in (str, int):
            continue
        if type(target) is dict or (isinstance(target, IRObject) and target in owned):
            updates.add(node)
            owned.add(node.output(0))
    return updates
