# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import weakref

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


@pytest.mark.parametrize('metadata', [False, True])
def test_final_recompute_consumer_releases_the_outer_input(metadata):
    def tensor(name):
        return IRFullTensor((4,), name=name, dtype=torch.float64).tosub()

    x, a, b, result = [tensor(name) for name in ('x', 'a', 'b', 'result')]
    extra = IRObject('extra') if metadata else tensor('extra')
    nodes = []
    for signature, inputs, output, group in [('torch.sin', [x], a, 0),
                                             ('consume', [a, extra], b, 0),
                                             ('later', [b], result, None)]:
        node = IRFwOperation(signature, signature, inputs, 1)
        node.set_output(0, output)
        node.device, node.recompute = (0,), group
        nodes.append(node)
    segment = IRSegment(nodes, [x, extra], [result])
    emitter = ModuleCodeGen.__new__(ModuleCodeGen)
    FuncEmission.__init__(emitter)
    emitter.plan_ndevs = emitter.runtime_ndevs = 1
    lines = emitter.emit_segment(segment, 0)
    payload = torch.ones(4, dtype=torch.float64)
    reference = weakref.ref(payload)
    arguments = [{'tensor': payload} if metadata else payload]
    del payload

    def later(value):
        assert reference() is None
        return value.cos()

    namespace = dict(torch=torch, ckpt=ckpt, later=later,
                     consume=lambda value, extra: value + (extra['tensor'] if metadata else extra))
    source = '\n'.join([
        'def run(' + ', '.join(emitter.tensor_name(v) for v in segment.inputs()) + '):',
        *['    ' + line for line in lines],
        '    return ' + emitter.tensor_name(result),
    ])
    exec(source, namespace)
    with torch.no_grad():
        actual = namespace['run'](torch.ones(4, dtype=torch.float64), arguments.pop())
    torch.testing.assert_close(actual, (torch.ones_like(actual).sin() + 1).cos())
