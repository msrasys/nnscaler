# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extract a pure tensor region without retaining its owning module in closures."""

import ast
from dataclasses import replace
from fnmatch import fnmatchcase

from nnscaler.flags import CompileFlag
from nnscaler.ir.cten import IRTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import IdentityPrim, SelectPrim, ChunkPrim, VChunkPrim, SplitAllGatherPrim
from nnscaler.graph.schedule.module_streams import weight_gradient_stream, _local_container_read, _local_adapter


def split_graph_metadata(region):
    """Read structural metadata on the host before a tensor-only graph call."""
    if all(isinstance(value, IRTensor) for value in region.inputs):
        return [], region
    producers = {value: node for node in region.nodes for value in node.oobjs()}
    host = set()
    for node in region.nodes:
        metadata_alias = node.name == 'multiref' and node.kwargs.get('clone_level', 0) == 0
        local_view = isinstance(node, IRAdapter) and all(
            type(primitive) in (IdentityPrim, SelectPrim, ChunkPrim, VChunkPrim, SplitAllGatherPrim)
            for primitive in node.prims)
        if (_local_container_read(node) or metadata_alias or local_view) and all(
            value not in producers or producers[value] in host for value in node.iobjs()
        ):
            host.add(node)
    if not host or len(host) == len(region.nodes):
        return [], region
    nodes = [node for node in region.nodes if node not in host]
    produced = {value for node in nodes for value in node.oobjs()}
    inputs = tuple(dict.fromkeys(value for node in nodes for value in node.iobjs()
                                 if value not in produced))
    return [node for node in region.nodes if node in host], replace(region, nodes=nodes, inputs=inputs)


def emit_graph_region(codegen, segment, region, codes, lifetime, *, prepared=None):
    if CompileFlag.use_zero > 1:
        raise ValueError('Module CUDA graphs require stable parameter storage (ZeRO 0/1)')
    if not region.outputs or any(not isinstance(v, IRTensor) for v in (*region.inputs, *region.outputs)):
        raise ValueError('Module CUDA graph regions require tensor-only boundaries')
    for node in region.nodes:
        if ((not isinstance(node, IRFwOperation) and not (region.cuda_graph == 'forward' and _local_adapter(node)))
                or node.recompute is not None
                or node.pre_hook is not None or node.post_hook is not None
                or node.signature in codegen.async_op_signatures
                or any(fnmatchcase(node.signature, p) for p in CompileFlag.multi_stream_ops)
                or weight_gradient_stream(node, CompileFlag.module_streams)):
            raise ValueError('Module CUDA graphs require pure operators without checkpoints, hooks or internal streams: '
                             f'{type(node).__name__} {node.signature}')

    arguments = [f'_graph_input_{i}' for i in range(len(region.inputs))]
    replacements = {codegen.tensor_name(value, prefix_attr='self.'): name
                    for value, name in zip(region.inputs, arguments)}

    class Inputs(ast.NodeTransformer):
        def visit_Attribute(self, node):
            name = replacements.get(ast.unparse(node))
            return ast.copy_location(ast.Name(id=name, ctx=node.ctx), node) if name else self.generic_visit(node)

        def visit_Name(self, node):
            return ast.copy_location(ast.Name(id=replacements.get(node.id, node.id), ctx=node.ctx), node)

    output_names = codegen.tuple_name(region.outputs, prefix_attr='self.')
    tree = Inputs().visit(ast.parse('\n'.join([*codes, f'return {output_names}'])))
    if any(isinstance(n, ast.Name) and n.id == 'self' for n in ast.walk(tree)):
        raise ValueError('Module CUDA graphs cannot capture hidden module state')
    name = f'_graph_{region.key}'
    function = ast.FunctionDef(name=name,
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in arguments], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=tree.body, decorator_list=[], type_params=[])
    code = ast.unparse(ast.fix_missing_locations(function)).splitlines()
    parameters = tuple(i for i, value in enumerate(region.inputs) if value.is_param())
    key = (codegen.node_name(segment), region.key)
    inputs = codegen.tuple_name(region.inputs, prefix_attr='self.')
    preparation = f', prepared={prepared}' if prepared is not None else ''
    code.append(f'{output_names} = nnscaler.runtime.module_graph.run_module_graph('
                f'self, {key!r}, {name}, {inputs}, {parameters!r}, capture_backward={region.cuda_graph is True!r}{preparation})')
    if prepared is not None:
        code.append(f'del {prepared}')
    released = [v for v in region.inputs
                if lifetime.releasable_after_line(v, lifetime.get_line(region.nodes[-1]) + 1)]
    if released:
        code.append(codegen.emit_release(released))
    return code
