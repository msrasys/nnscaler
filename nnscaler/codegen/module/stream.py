# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit dependency-checked module branches and their CUDA stream boundaries."""

from collections import defaultdict

from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.codegen.syntax.blocks import Block
from nnscaler.flags import CompileFlag
from nnscaler.graph.schedule.checkpoint_readiness import plan_checkpoint_readiness
from nnscaler.graph.schedule.module_streams import plan_module_streams
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.cten import IRTensor


def _record_parameter_streams(codegen, nodes, regions):
    streams = {node: region.stream for region in regions.values() for node in region.nodes}
    for node in nodes:
        origins = set()
        for value in node.iobjs():
            if isinstance(value, IRTensor) and value.is_param():
                origins.add(codegen.tensor_name(value))
            origins.update(codegen.module_parameter_origins.get(value, ()))
        multiref = node.name == 'multiref' and node.kwargs.get('clone_level', 0) == 0
        if not multiref:
            for name in origins:
                codegen.module_parameter_uses.setdefault(name, set()).add(streams.get(node))
        if (
            multiref
            or isinstance(node, IRAdapter)
            or node.signature.rsplit('.', 1)[-1]
            in (
                'view',
                'view_as',
                'reshape',
                'detach',
                'contiguous',
                'to',
                'float',
                'half',
                'bfloat16',
                'identity',
                'transpose',
                'permute',
                't',
            )
        ):
            for output in node.oobjs():
                codegen.module_parameter_origins[output] = origins


def emit_module_streams(codegen, segment, runtime_devid):
    nodes, regions, joins = plan_module_streams(segment.nodes(), segment.oobjs(), CompileFlag.module_streams)
    anchors = plan_checkpoint_readiness(nodes, segment.oobjs(), CompileFlag.module_streams)
    readiness = {gid: f'_checkpoint_ready_{i}' for i, gid in enumerate(anchors)}
    ready_at = defaultdict(list)
    for gid, anchor in anchors.items():
        if any(
            anchor in region.nodes or any(node.recompute == gid for node in region.nodes)
            for region in regions.values()
        ):
            raise ValueError('Recompute readiness cannot also move its checkpoint or backend forward')
        ready_at[anchor].append(readiness[gid])
    _record_parameter_streams(codegen, nodes, regions)
    if not regions and not anchors:
        return codegen._emit_segment_nodes(segment, nodes, runtime_devid)

    codegen.execplan.use_multi_streams = True
    lifetime = LifeCycle(nodes, segment.inputs(), segment.outputs())
    starts = {region.nodes[0]: region for region in regions.values()}
    owners = {node: region for region in regions.values() for node in region.nodes}
    captures, backwards = defaultdict(list), defaultdict(list)
    for region in regions.values():
        if region.ready_before is not None:
            captures[region.ready_before].append(region.key)
        if region.overlap_backward:
            backwards[region.backward_anchor].append(region)
    # A checkpoint must join before entering, even if only an internal node
    # consumes the branch output. Its local tensor names are not visible here.
    checkpoint_starts = {}
    for node in nodes:
        if node.recompute is not None:
            checkpoint_starts.setdefault(node.recompute, node)
    waits = defaultdict(list)
    for node, keys in joins.items():
        waits[checkpoint_starts.get(node.recompute, node)].extend(keys)

    def names(objects):
        return codegen.tuple_name(objects, prefix_attr='self.')

    def emit(group):
        return codegen._emit_segment_nodes(
            segment,
            group,
            runtime_devid,
            lifetime,
            release_unused=False,
            checkpoint_readiness=readiness,
        )

    body = [f'{name} = nnscaler.runtime.backward.BackwardLaunchPoint(())' for name in readiness.values()]
    unused = lifetime.release_tensors_after_line(-1)
    if unused:
        body.append(codegen.emit_release(unused))
    pending = []

    def flush():
        if pending:
            body.extend(emit(pending))
            pending.clear()

    for node in nodes:
        if node in owners and node not in starts:
            continue
        if node in captures or node in waits:
            flush()
            body.extend(f'_module_streams.capture({key!r})' for key in captures.get(node, ()))
            body.extend(f'_module_streams.join({key!r})' for key in waits.get(node, ()))
        if node in backwards:
            flush()
            anchored = backwards[node]
            dependencies = [obj for region in anchored for obj in region.outputs if obj.requires_grad]
            if dependencies:
                value = next(
                    (
                        obj
                        for obj in node.iobjs()
                        if isinstance(obj, IRTensor) and obj.requires_grad and not obj.is_attr()
                    ),
                    None,
                )
                if value is None:
                    raise ValueError(f'Backward overlap anchor {node.signature} has no differentiable input')
                keys = tuple(region.key for region in anchored)
                parts, retained = [], []
                for region in anchored:
                    if region.retain_backward_outputs:
                        name = f'_{region.key}_backward_outputs'
                        retained.append(name)
                        parts.append(f'*{name}')
                    else:
                        parts.extend(
                            codegen.tensor_name(obj, prefix_attr='self.')
                            for obj in region.outputs
                            if obj.requires_grad
                        )
                dependencies = '(' + ', '.join(parts) + ',)'
                value = codegen.tensor_name(value)
                body.append(f'{value} = _module_streams.after_backward({value}, {dependencies}, {keys!r})')
                contexts = [
                    f'_module_streams.backward_point({keys!r})',
                    *ready_at.get(node, ()),
                ]
                with Block(f'with {", ".join(contexts)}:') as block:
                    block.insert_body(emit([node]))
                body.extend(block.code)
                if retained:
                    body.append('del ' + ', '.join(retained))
                continue
        if node in starts:
            flush()
            region = starts[node]
            if region.retain_backward_outputs:
                body.append(
                    f'_{region.key}_backward_boundaries = '
                    f'nnscaler.runtime.backward.backward_boundaries({names(region.inputs)})'
                )
            preserve = ', preserve_accumulators=True' if region.overlap_backward else ''
            with Block(
                f'with _module_streams.fork({region.key!r}, {region.stream!r}, ' f'{names(region.inputs)}{preserve}):'
            ) as block:
                block.insert_body(emit(region.nodes))
            body.extend(block.code)
            caller = region.caller_outputs
            producer = tuple(obj for obj in region.outputs if obj not in caller)
            if caller:
                body.append(f'_module_streams.publish({names(caller)})')
            if producer:
                body.append(f'_module_streams.publish({names(producer)}, stream={region.stream!r})')
            if region.retain_backward_outputs:
                dependencies = tuple(obj for obj in region.outputs if obj.requires_grad)
                body.append(
                    f'nnscaler.runtime.backward.prioritize_backward('
                    f'{names(dependencies)}, _{region.key}_backward_boundaries)'
                )
                body.append(f'del _{region.key}_backward_boundaries')
                body.append(f'_{region.key}_backward_outputs = {names(dependencies)}')
        elif node in ready_at:
            flush()
            with Block(f'with {", ".join(ready_at[node])}:') as block:
                block.insert_body(emit([node]))
            body.extend(block.code)
        elif node not in owners:
            pending.append(node)
    flush()
    with Block('with nnscaler.runtime.stream.ModuleStreams() as _module_streams:') as block:
        block.insert_body(body)
    return block.code
