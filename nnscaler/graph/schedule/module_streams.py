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
    IdentityPrim,
    SelectPrim,
    MergeDimPrim,
    SumPrim,
    ChunkPrim,
    VChunkPrim,
    SplitAllGatherPrim,
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
    recompute_only: bool = False
    overlap_backward: bool = False
    recompute_ready_on: str | None = None
    backward_on: str | None = None
    operators: str | tuple[str, ...] | None = None
    weight_backward_only: bool = False
    cuda_graph: bool = False
    _codegen_version: int = field(default_factory=lambda: 27, init=False, repr=False)

    def __post_init__(self):
        for name in ('module', 'stream'):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f'{name} must be a nonempty string')
        if self.stream == 'default':
            raise ValueError('Module streams must use a distinct, named stream')
        for name in (
            'launch_before',
            'launch_after',
            'split_before',
            'resume_before',
            'resume_after',
            'recompute_ready_on',
            'backward_on',
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f'{name} must be a nonempty string or None')
        for name in ('recompute_only', 'overlap_backward', 'weight_backward_only', 'cuda_graph'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        if (self.launch_before and self.launch_after) or (self.resume_before and self.resume_after):
            raise ValueError('Use either before or after for each launch anchor')
        if (self.split_before is None) != (self.resume_before is None and self.resume_after is None):
            raise ValueError('split_before and a resume anchor must be specified together')
        if self.operators is not None:
            patterns = (self.operators,) if isinstance(self.operators, str) else self.operators
            if (
                not isinstance(patterns, (tuple, list))
                or not patterns
                or any(not isinstance(p, str) or not p for p in patterns)
            ):
                raise ValueError('operators requires nonempty operator signature patterns')
            object.__setattr__(self, 'operators', tuple(patterns))
            if self.recompute_only or self.split_before:
                raise ValueError('Operator selection cannot split modules or select recompute regions')
        if self.recompute_ready_on is not None and not self.recompute_only:
            raise ValueError('recompute_ready_on requires a recompute_only rule')
        if self.cuda_graph and (self.recompute_only or self.weight_backward_only):
            raise ValueError('CUDA graph regions cannot also select recomputation or cooperative dW')
        if self.weight_backward_only and (
            not self.operators or self.recompute_only or self.overlap_backward
            or self.launch_before or self.launch_after or self.split_before
            or self.resume_before or self.resume_after
        ):
            raise ValueError('Weight-backward rules require cooperative operators without forward placement')
        if self.recompute_only and (
            self.launch_before
            or self.launch_after
            or self.split_before
            or self.resume_before
            or self.resume_after
            or self.overlap_backward
            or self.backward_on
        ):
            raise ValueError('Recompute-only rules cannot also move forward computation')
        if self.overlap_backward and not (self.launch_before or self.launch_after):
            raise ValueError('Backward overlap requires a launch anchor')
        if self.backward_on is not None and not (self.overlap_backward or self.weight_backward_only):
            raise ValueError('backward_on requires backward overlap or weight-backward scheduling')


def module_stream_rules_from_env():
    """Resolve once into ComputeConfig so cache reuse includes the actual rules."""
    return normalize_module_stream_rules(json.loads(os.environ.get('NNSCALER_MODULE_STREAMS', '[]')))


def normalize_multi_stream_ops(patterns):
    if not isinstance(patterns, (list, tuple)) or any(not isinstance(p, str) or not p for p in patterns):
        raise ValueError('multi_stream_ops must be a list of nonempty operator signature patterns')
    return tuple(patterns)


def multi_stream_ops_from_env():
    return normalize_multi_stream_ops(json.loads(os.environ.get('NNSCALER_MULTI_STREAM_OPS', '[]')))


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
    caller_outputs: tuple = ()
    overlap_backward: bool = False
    backward_anchor: object = None
    retain_backward_outputs: bool = False
    prioritize_backward: bool = False
    cuda_graph: bool = False


def _unsafe(node):
    if not isinstance(node, IRFwOperation):
        return True
    name = node.signature.rsplit('.', 1)[-1]
    return (
        node.kwargs.get('inplace', False)
        or (name.endswith('_') and not name.startswith('__'))
        or name
        in (
            '__setitem__',
            'setitem',
            'dropout',
            'batch_norm',
            'rand',
            'randn',
            'rand_like',
            'randn_like',
            'bernoulli',
            'multinomial',
        )
    )


def _local_adapter(node):
    # A SplitAllGather executes only a local slice in forward. Its backward
    # collective follows autograd's stream semantics, like any original op.
    return isinstance(node, IRAdapter) and all(
        type(prim)
        in (
            IdentityPrim,
            SelectPrim,
            MergeDimPrim,
            SumPrim,
            ChunkPrim,
            VChunkPrim,
            SplitAllGatherPrim,
        )
        for prim in node.prims
    )


def _local_container_read(node):
    """A constant-key projection from traced builtin metadata, without kernels.

    Only exclusively used reads join an operator branch. The usual mutation
    barriers still apply; custom containers/keys and tensor indexing are not
    inferred as movable dependencies.
    """
    if node.signature != '_operator.getitem' or len(node.inputs()) != 2:
        return False
    container, key = node.inputs()
    if not isinstance(container, IRObject) or not container.is_constant:
        return False
    value = container.value
    return (type(value) is dict and type(key) in (str, int)) or (type(value) in (list, tuple) and type(key) is int)


def plan_module_streams(nodes, outputs, rules):
    """Return reordered nodes, side regions and consumer joins; never mutate IR.

    A launch anchor is an operator signature glob in the selected module's
    enclosing invocation, or in the selected invocation itself when filtering
    by ``operators``. ``split_before`` is a relative child-module glob.
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
        if rule.recompute_only or rule.weight_backward_only:
            continue
        invocations = defaultdict(list)
        scopes = {}
        for node in nodes:
            if rule.operators is not None and not any(
                fnmatchcase(node.signature, pattern) for pattern in rule.operators
            ):
                continue
            stack = list((node.module_stack or {}).keys())
            matches = [path for path in stack if fnmatchcase(path, rule.module)]
            if not matches:
                continue
            path = matches[-1]
            calls = node.module_call_stack or {}
            key = (path, calls.get(path))
            invocations[key].append(node)
            # Operator branches have no dedicated child module. Their anchor
            # belongs to this invocation, not to an enclosing loop containing
            # other invocations with the same operator signatures.
            parent = (
                path if rule.operators is not None else (stack[stack.index(path) - 1] if stack.index(path) else '')
            )
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
                    if not (
                        _local_adapter(node)
                        or node.name == 'multiref'
                        or (rule.operators is not None and _local_container_read(node))
                    ):
                        continue
                    users = {c for obj in node.oobjs() for c in consumers[obj]}
                    if users and users <= branch_set and not any(obj in graph_outputs for obj in node.oobjs()):
                        branch_set.add(node)
                        changed = True
            branch = sorted(branch_set, key=positions.__getitem__)
            chunks = [branch]
            if rule.split_before is not None:
                split = next(
                    (
                        i
                        for i, n in enumerate(branch)
                        if any(
                            fnmatchcase(p[len(path) + 1 :], rule.split_before)
                            for p in (n.module_stack or {})
                            if p.startswith(path + '.')
                        )
                    ),
                    None,
                )
                if split is None or split == 0:
                    raise ValueError(f'No nonempty split at {rule.split_before!r} in {path}')
                chunks = [branch[:split], branch[split:]]

            parent, call = scopes[invocation]
            backward_override = None
            if rule.backward_on is not None:
                candidates = [
                    n
                    for n in nodes
                    if n not in branch
                    and fnmatchcase(n.signature, rule.backward_on)
                    and (not parent or parent in (n.module_stack or {}))
                    and (call is None or (n.module_call_stack or {}).get(parent) == call)
                ]
                if len(candidates) != 1:
                    raise ValueError(
                        f'{path}: expected one backward anchor {rule.backward_on!r}, found {len(candidates)}'
                    )
                backward_override = candidates[0]
                if backward_override.recompute is not None:
                    raise ValueError(f'{path}: backward anchor must remain outside checkpoints')
            anchors = (
                (rule.launch_before or rule.launch_after, bool(rule.launch_after)),
                (rule.resume_before or rule.resume_after, bool(rule.resume_after)),
            )
            for chunk, (anchor, after) in zip(chunks, anchors):
                destination = positions[chunk[0]]
                ready_before = None
                backward_anchor = backward_override
                if anchor:
                    candidates = [
                        n
                        for n in nodes
                        if n not in branch
                        and fnmatchcase(n.signature, anchor)
                        and (not parent or parent in (n.module_stack or {}))
                        and (call is None or (n.module_call_stack or {}).get(parent) == call)
                    ]
                    if len(candidates) != 1:
                        raise ValueError(f'{path}: expected one launch anchor {anchor!r}, found {len(candidates)}')
                    destination = positions[candidates[0]] + int(after)
                    ready_before = candidates[0] if after else None
                    if backward_override is None:
                        backward_anchor = candidates[0] if rule.overlap_backward else None
                # Already-available synthesized inputs need not move with the
                # module. Preserve their original position and other users.
                chunk = [n for n in chunk if n in original_branch or positions[n] >= destination]
                if destination > positions[chunk[0]]:
                    raise ValueError(f'{path}: module stream launch must not move computation later')
                begin, end = sorted((destination, positions[chunk[0]]))
                barriers = [
                    n
                    for n in nodes[begin:end]
                    if n not in branch_set
                    if not isinstance(n, IRAdapter) and n.signature.rsplit('.', 1)[-1] != 'multiref' and _unsafe(n)
                ]
                if barriers:
                    raise ValueError(
                        f'{path}: cannot move a module across a mutation or unsupported operation: '
                        f'{[(type(n).__name__, n.signature) for n in barriers]}'
                    )
                # Main-stream adapters retain their original ordering. Moving
                # an independent pure branch past them is safe; dependencies on
                # their outputs are checked below. Adapters in the moved branch
                # itself remain unsupported.
                chunk_set = set(chunk)
                gids = {n.recompute for n in chunk if n.recompute is not None}
                if any(n.recompute in gids and n not in chunk_set for n in nodes):
                    raise ValueError(f'{path}: module stream split crosses a recompute region')
                if (
                    0 < destination < len(nodes)
                    and nodes[destination].recompute is not None
                    and nodes[destination].recompute == nodes[destination - 1].recompute
                ):
                    raise ValueError(f'{path}: launch anchor is inside a recompute region')
                if ready_before is not None and ready_before.recompute is not None:
                    raise ValueError(f'{path}: after-launch anchor cannot be inside a recompute region')
                inputs = tuple(
                    dict.fromkeys(obj for n in chunk for obj in n.iobjs() if producers.get(obj) not in chunk_set)
                )
                region_outputs = tuple(
                    dict.fromkeys(
                        obj
                        for n in chunk
                        for obj in n.oobjs()
                        if obj in graph_outputs or any(c not in chunk_set for c in consumers[obj])
                    )
                )
                key = f'module_stream_{len(regions)}'
                region = ModuleStreamRegion(
                    key,
                    rule.stream,
                    chunk,
                    inputs,
                    region_outputs,
                    ready_before,
                    overlap_backward=rule.overlap_backward,
                    backward_anchor=backward_anchor,
                    retain_backward_outputs=backward_override is not None,
                    prioritize_backward=backward_override is not None,
                    cuda_graph=rule.cuda_graph,
                )
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
                raise ValueError(
                    f'Module stream launch violates dependency {producer.signature} -> {node.signature}: {producer}'
                )

    # A stream join is needed at the first consumer outside each region. Later
    # consumers are ordered by the caller stream; same-stream regions need none.
    owners = {n: r for r in regions.values() for n in r.nodes}
    for region in regions.values():
        if region.overlap_backward and region.backward_anchor in owners:
            raise ValueError('Backward anchors must remain on the caller stream')
        if region.retain_backward_outputs:
            if (region.ready_before is not region.backward_anchor
                    and any(new_positions[n] >= new_positions[region.backward_anchor] for n in region.nodes)):
                raise ValueError('Explicit backward anchor must follow all selected forward chunks')
        region.caller_outputs = tuple(
            obj for obj in region.outputs if obj in graph_outputs or any(node not in owners for node in consumers[obj])
        )
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
                raise ValueError(
                    'After-launch module inputs must be ready before its anchor '
                    'or produced earlier on the same module stream'
                )
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


def weight_gradient_stream(node, rules):
    """Select a cooperative operator's dW stream without moving forward work."""
    streams = {
        rule.stream for rule in rules
        if rule.weight_backward_only
        and any(fnmatchcase(path, rule.module) for path in (node.module_stack or {}))
        and any(fnmatchcase(node.signature, pattern) for pattern in rule.operators)
    }
    if len(streams) > 1:
        raise ValueError('Conflicting weight-gradient streams for an operator')
    return next(iter(streams), None)


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
        unsafe = [node for node in nodes if _unsafe(node) and not _local_adapter(node) and node not in local_updates]
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
        if (
            node.signature != 'nnscaler.runtime.function.setitem'
            or len(node.inputs()) != 3
            or len(node.outputs()) != 1
        ):
            continue
        target, key, _ = node.inputs()
        if type(key) not in (str, int):
            continue
        if type(target) is dict or (isinstance(target, IRObject) and target in owned):
            updates.add(node)
            owned.add(node.output(0))
    return updates
