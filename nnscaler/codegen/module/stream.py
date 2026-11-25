# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Emit dependency-checked module branches and their CUDA stream boundaries."""

import ast
import copy
from collections import defaultdict

from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.codegen.syntax.blocks import Block
from nnscaler.flags import CompileFlag
from nnscaler.graph.schedule.checkpoint_readiness import plan_checkpoint_readiness
from nnscaler.graph.schedule.module_streams import plan_module_streams, weight_gradient_stream
from nnscaler.graph.schedule.weight_backward import plan_weight_backward
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
            gradient_stream = weight_gradient_stream(node, CompileFlag.module_streams)
            for name in origins:
                codegen.module_parameter_uses.setdefault(name, set()).add(gradient_stream or streams.get(node))
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
    weight_plans = plan_weight_backward(nodes, CompileFlag.module_streams)
    weight_names = {node: f'_weight_region_{i}' for i, node in enumerate(weight_plans)}
    weight_anchors = defaultdict(list)
    for node, (anchor, stream) in weight_plans.items():
        if anchor.signature not in codegen.async_op_signatures:
            raise ValueError('Weight backward anchors require a registered async operator')
        if any(node in region.nodes or anchor in region.nodes for region in regions.values()):
            raise ValueError('Weight backward anchors and cooperative operators cannot also move forward streams')
        weight_anchors[anchor].append(node)
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
    if not regions and not anchors and not weight_plans:
        return codegen._emit_segment_nodes(segment, nodes, runtime_devid)

    codegen.execplan.mark_module_streams()
    deferred = defaultdict(list)
    for region in regions.values():
        anchor = region.ready_before
        if anchor is not None and anchor.signature in codegen.async_op_signatures:
            deferred[anchor].append(region)
        elif anchor is not None and region.overlap_backward and region.backward_anchor is anchor:
            raise ValueError('Backward overlap after launch requires a registered async operator')
    finish_after, pending_names = {}, {}
    lifetime_nodes = list(nodes)
    for index, (anchor, after_regions) in enumerate(deferred.items()):
        after_nodes = [node for region in after_regions for node in region.nodes]
        position = nodes.index(anchor)
        if nodes[position + 1:position + 1 + len(after_nodes)] != after_nodes:
            raise ValueError('Async after-launch regions must immediately follow their anchor')
        lifetime_nodes.remove(anchor)
        lifetime_nodes.insert(lifetime_nodes.index(after_nodes[-1]) + 1, anchor)
        finish_after[after_regions[-1].key] = anchor
        pending_names[anchor] = f'_async_operator_{index}'
    for region in regions.values():
        if region.overlap_backward and region.backward_anchor in deferred:
            # A before-anchor chunk can be consumed by an after-anchor suffix
            # before binding the control edge. Keep its graph output explicitly.
            region.retain_backward_outputs = True
    lifetime = LifeCycle(lifetime_nodes, segment.inputs(), segment.outputs())
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
            weight_regions=weight_names,
        )

    body = [f'{name} = nnscaler.runtime.backward.BackwardLaunchPoint(())' for name in readiness.values()]
    unused = lifetime.release_tensors_after_line(-1)
    if unused:
        body.append(codegen.emit_release(unused))
    pending = []

    async_codes = {}

    def async_code(node, *, submit):
        if node not in async_codes:
            # Preserve ordinary argument emission, pre/post hooks, grad/autocast
            # contexts, metadata unpacking and lifetime releases exactly once.
            tree = ast.parse('\n'.join(emit([node])))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and ast.unparse(n.func) == node.signature]
            if len(calls) != 1:
                raise ValueError('Async operators require one direct emitted operator call')
            call = calls[0]
            if any(isinstance(n, ast.Call) for arg in [*call.args, *[k.value for k in call.keywords]]
                   for n in ast.walk(arg)):
                raise ValueError('Async operator arguments must be values, not nested function calls')
            name = pending_names[node]
            start = copy.deepcopy(call)
            start.args.insert(0, start.func)
            start.func = ast.parse('_module_streams.submit', mode='eval').body
            statement = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())], value=start)
            call.func = ast.Attribute(value=ast.Name(id=name, ctx=ast.Load()), attr='complete', ctx=ast.Load())

            def split(statements):
                for i, current in enumerate(statements):
                    if call in ast.walk(current):
                        if isinstance(current, ast.With):
                            before, after = split(current.body)
                            return (
                                statements[:i] + [ast.With(items=copy.deepcopy(current.items), body=before)],
                                [ast.With(items=copy.deepcopy(current.items), body=after)] + statements[i + 1:],
                            )
                        if isinstance(current, (ast.Assign, ast.Expr)):
                            return statements[:i] + [statement], statements[i:]
                        raise ValueError('Async operator emission must be an assignment in ordinary contexts')
                raise ValueError('Async operator call was not found')

            before, after = split(tree.body)
            async_codes[node] = tuple(
                ast.unparse(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))).splitlines()
                for body in (before, after)
            )
        return async_codes[node][0 if submit else 1]

    def emit_anchor(node, *, prepared=False):
        anchored = backwards.get(node, ())
        dependencies = [obj for region in anchored for obj in region.outputs if obj.requires_grad]
        retained = []
        contexts = list(ready_at.get(node, ()))
        for weight_node in weight_anchors.get(node, ()):
            name = weight_names[weight_node]
            _, stream = weight_plans[weight_node]
            parameters = tuple(dict.fromkeys(v for v in weight_node.iobjs()
                               if isinstance(v, IRTensor) and v.is_param() and v.requires_grad))
            body.append(f'{name} = nnscaler.WeightGradientAccumulator({stream!r}, '
                        f'{names(parameters)}, defer_until_backward=True)')
            body.append(f'_weight_regions.append({name})')
            value = next(v for v in node.iobjs()
                         if isinstance(v, IRTensor) and v.requires_grad and not v.is_attr())
            value = codegen.tensor_name(value)
            body.append(f'{value} = {name}.after_backward({value})')
            contexts.append(f'{name}.backward_point()')
        if dependencies:
            value = next((obj for obj in node.iobjs()
                          if isinstance(obj, IRTensor) and obj.requires_grad and not obj.is_attr()), None)
            if value is None:
                raise ValueError(f'Backward overlap anchor {node.signature} has no differentiable input')
            keys = tuple(region.key for region in anchored)
            parts = []
            for region in anchored:
                if region.retain_backward_outputs:
                    name = f'_{region.key}_backward_outputs'
                    retained.append(name)
                    parts.append(f'*{name}')
                else:
                    parts.extend(codegen.tensor_name(obj, prefix_attr='self.')
                                 for obj in region.outputs if obj.requires_grad)
            dependencies = '(' + ', '.join(parts) + ',)'
            value = codegen.tensor_name(value)
            body.append(f'{value} = _module_streams.after_backward({value}, {dependencies}, {keys!r})')
            contexts.insert(0, f'_module_streams.backward_point({keys!r})')
        codes = async_code(node, submit=False) if prepared else emit([node])
        if contexts:
            with Block(f'with {", ".join(contexts)}:') as block:
                block.insert_body(codes)
            body.extend(block.code)
        else:
            body.extend(codes)
        if prepared:
            body.append(f'del {pending_names[node]}')
        if retained:
            body.append('del ' + ', '.join(retained))

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
        if node in deferred:
            flush()
            body.extend(async_code(node, submit=True))
            continue
        if node in backwards or node in weight_anchors:
            flush()
            emit_anchor(node)
            continue
        if node in starts:
            flush()
            region = starts[node]
            if region.prioritize_backward:
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
                if region.prioritize_backward:
                    body.append(
                        f'nnscaler.runtime.backward.prioritize_backward('
                        f'{names(dependencies)}, _{region.key}_backward_boundaries)'
                    )
                    body.append(f'del _{region.key}_backward_boundaries')
                body.append(f'_{region.key}_backward_outputs = {names(dependencies)}')
            if region.key in finish_after:
                emit_anchor(finish_after[region.key], prepared=True)
        elif node in ready_at:
            flush()
            emit_anchor(node)
        elif node not in owners:
            pending.append(node)
    flush()
    if regions:
        with Block('with nnscaler.runtime.stream.ModuleStreams() as _module_streams:') as block:
            block.insert_body(body)
        body = block.code
    if weight_plans:
        with Block('with nnscaler.runtime.weight_accumulator.weight_backward_scopes() as _weight_regions:') as block:
            block.insert_body(body)
        body = block.code
    return body
