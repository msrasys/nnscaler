# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast

import pytest
import torch
import torch.utils.checkpoint as ckpt

from nnscaler.codegen.module.module import ModuleCodeGen
from nnscaler.codegen.emit import FuncEmission
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.cten import IRObject
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor


@pytest.mark.parametrize('whole_group', [False, True])
@pytest.mark.parametrize('unused_tensor', [False, True])
@pytest.mark.parametrize('returned', [False, True])
def test_unused_outer_input_stays_out_of_recompute_scope(whole_group, unused_tensor, returned):
    def tensor(name):
        return IRFullTensor((4,), name=name, dtype=torch.float64, requires_grad=True).tosub()

    x, middle, result = [tensor(name) for name in ('x', 'middle', 'result')]
    unused = tensor('unused') if unused_tensor else IRObject('unused')
    nodes = []
    for name, value, output in [('sin', x, middle), ('cos', middle, result)]:
        node = IRFwOperation(name, 'torch.' + name, [value], 1)
        node.set_output(0, output)
        node.device = (0,)
        node.recompute = 0 if whole_group or name == 'sin' else None
        nodes.append(node)
    outputs = [result, unused] if returned else [result]
    segment = IRSegment(nodes, [x, unused], outputs)
    emitter = ModuleCodeGen.__new__(ModuleCodeGen)
    FuncEmission.__init__(emitter)
    emitter.plan_ndevs = emitter.runtime_ndevs = 1
    lines = emitter.emit_segment(segment, 0)
    source = '\n'.join([
        'def run(' + ', '.join(emitter.tensor_name(v) for v in segment.inputs()) + '):',
        *['    ' + line for line in lines],
        '    return ' + ', '.join(emitter.tensor_name(v) for v in outputs),
    ])
    namespace = dict(torch=torch, ckpt=ckpt)
    exec(source, namespace)
    value = torch.randn(4, dtype=torch.float64, requires_grad=True)
    extra = torch.randn(4) if unused_tensor else {'tensor': torch.randn(4)}
    actual = namespace['run'](value, extra)
    if returned:
        actual, kept = actual
        assert kept is extra
    expected = value.sin().cos()
    actual.sum().backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(value.grad, -value.sin().sin() * value.cos())
    nested = next(node for node in ast.walk(ast.parse(source))
                  if isinstance(node, ast.FunctionDef) and node.name == 'recompute')
    name = emitter.tensor_name(unused)
    assert all(not isinstance(node, ast.Name) or node.id != name for node in ast.walk(nested))
