# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Extract a pure tensor region without retaining its owning module in closures."""

import ast
from fnmatch import fnmatchcase

from nnscaler.flags import CompileFlag
from nnscaler.ir.cten import IRTensor
from nnscaler.ir.operator import IRFwOperation
from nnscaler.graph.schedule.module_streams import weight_gradient_stream


def emit_graph_region(codegen, segment, region, codes, lifetime, *, prepared=None):
    if CompileFlag.use_zero > 1:
        raise ValueError('Module CUDA graphs require stable parameter storage (ZeRO 0/1)')
    if not region.outputs or any(not isinstance(v, IRTensor) for v in (*region.inputs, *region.outputs)):
        raise ValueError('Module CUDA graph regions require tensor-only boundaries')
    for node in region.nodes:
        if (not isinstance(node, IRFwOperation) or node.recompute is not None
                or node.pre_hook is not None or node.post_hook is not None
                or node.signature in codegen.async_op_signatures
                or any(fnmatchcase(node.signature, p) for p in CompileFlag.multi_stream_ops)
                or weight_gradient_stream(node, CompileFlag.module_streams)):
            raise ValueError('Module CUDA graphs require pure operators without checkpoints, hooks or internal streams')

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
