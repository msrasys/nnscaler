#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch

from nnscaler.codegen.emit import FuncEmission
from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.codegen.module.module import ModuleCodeGen
from nnscaler.graph.segment import IRSegment
from nnscaler.ir import ModelSpec
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import AllReducePrim
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor


def _tensor(name):
    return IRFullTensor(
        (2, 2),
        name=name,
        requires_grad=True,
        dtype=torch.float32,
    ).tosub()


def _node(name, input, *, model_spec=None, op_context=None):
    node = IRFwOperation(name, 'torch.neg', [input], 1)
    node.set_output(0, _tensor(f'{name}_output'))
    node.device = 0
    node.model_spec = model_spec
    node.op_context = op_context
    return node


def _op_context(*, no_grad=False):
    return {
        'grad_mode': {
            'grad_mode': not no_grad,
            'no_grad_mode': no_grad,
            'inference_mode': False,
        },
        'autocast_info': {
            'nesting': 0,
            'cache_enabled': True,
            'cpu_enabled': False,
            'cpu_dtype': torch.bfloat16,
            'cuda_enabled': False,
            'cuda_dtype': torch.float16,
        },
    }


def _codegen():
    codegen = object.__new__(ModuleCodeGen)
    FuncEmission.__init__(codegen)
    codegen.devices = (0,)
    codegen.runtime_ndevs = 1
    return codegen


def _emit_nodes(nodes):
    lifecycle = LifeCycle(nodes, [nodes[0].input(0)], [nodes[-1].output(0)])
    return '\n'.join(_codegen()._emit_nodes(nodes, lifecycle, 0))


def test_codegen_groups_adjacent_operations_by_model_spec():
    spec = ModelSpec('mlp', 'model.layers.0.mlp', 'model.py:30-40')
    first = _node('first', _tensor('input'), model_spec=spec)
    second = _node('second', first.output(0), model_spec=spec)

    code = _emit_nodes([first, second])

    assert code.count('with ct.component(') == 1
    assert "model_fqn='model.layers.0.mlp'" in code
    assert "model_site='model.py:30-40'" in code
    assert 'process_scope=False' in code
    assert 'launch_site=' not in code
    assert code.index('torch.neg') < code.rindex('torch.neg')


def test_codegen_splits_model_ranges_at_operation_context_boundaries():
    spec = ModelSpec('attention', 'model.layers.0.attn', 'model.py:10-20')
    first = _node(
        'first', _tensor('input'), model_spec=spec, op_context=_op_context()
    )
    second = _node(
        'second', first.output(0), model_spec=spec,
        op_context=_op_context(no_grad=True),
    )
    third = _node(
        'third', second.output(0), model_spec=spec,
        op_context=_op_context(no_grad=True),
    )

    code = _emit_nodes([first, second, third])

    assert code.count('with ct.component(') == 2
    assert code.count('with torch.no_grad():') == 1
    assert 'with torch.no_grad():\n    with ct.component(' in code


def test_codegen_does_not_wrap_unresolved_operations():
    spec = ModelSpec('normalization', 'model.norm', 'model.py:50')
    first = _node('first', _tensor('input'), model_spec=spec)
    unresolved = _node('unresolved', first.output(0))
    third = _node('third', unresolved.output(0), model_spec=spec)

    code = _emit_nodes([first, unresolved, third])

    assert code.count('with ct.component(') == 2
    unresolved_line = next(line for line in code.splitlines() if 'unresolved_output' in line)
    assert not unresolved_line.startswith('    ')


def test_codegen_keeps_communication_range_outside_model_ranges():
    spec = ModelSpec('mlp', 'model.mlp', 'model.py:30-40')
    first = _node('first', _tensor('input'), model_spec=spec)
    communicated = _tensor('communicated')
    adapter = IRAdapter([first.output(0)], [communicated])
    adapter.device = 0
    communicated.cell = adapter
    adapter.prims = [
        AllReducePrim([first.output(0)], [communicated], ranks=[0])
    ]
    second = _node('second', communicated, model_spec=spec)

    code = _emit_nodes([first, adapter, second])

    assert code.count('with ct.component(') == 2
    assert code.count('with ct.range(') == 1
    communication_range = next(
        line for line in code.splitlines() if 'with ct.range(' in line
    )
    assert not communication_range.startswith('    ')


def test_recompute_body_groups_model_spec_operations():
    spec = ModelSpec('moe', 'model.layers.0.moe', 'model.py:60-80')
    first = _node('first', _tensor('input'), model_spec=spec)
    second = _node('second', first.output(0), model_spec=spec)
    first.recompute = 1
    second.recompute = 1
    segment = IRSegment(
        [first, second],
        [first.input(0)],
        [second.output(0)],
    )

    code = '\n'.join(_codegen().emit_segment(segment, 0))

    assert 'def recompute(' in code
    assert code.count('with ct.component(') == 1
    assert 'ckpt.checkpoint(recompute,' in code
