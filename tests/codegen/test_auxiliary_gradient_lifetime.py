# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from types import SimpleNamespace

import pytest
import torch

from nnscaler.codegen.lifecycle import LifeCycle
from nnscaler.codegen.schedule.schedule import ScheduleCodeGen
from nnscaler.graph.segment import IRSegment
from nnscaler.ir.cten import IRCell
from nnscaler.ir.operator import IRFwOperation, IRBpOperation
from nnscaler.ir.tensor import IRFullTensor


@pytest.mark.parametrize(('terminal', 'produced'), [(False, False), (True, False), (True, True)])
def test_only_unseeded_terminal_gradients_use_inline_zeros(terminal, produced):
    def tensor(name):
        result = IRFullTensor((2,), name=name, requires_grad=True, dtype=torch.float32).tosub()
        result.grad = result.parent.grad.tosub()
        return result

    source, output = tensor('source'), tensor('output')
    forward_op = IRFwOperation('sin', 'torch.sin', [source], 1)
    forward_op.set_output(0, output)
    backward_op = IRBpOperation([output.grad], [source.grad])
    forward = IRSegment([forward_op], [source], [output])
    backward = IRSegment([backward_op], [output.grad], [source.grad])
    IRCell.make_pair(forward, backward)
    outputs = [output] if terminal else []
    inputs = [source, output.grad] if produced else [source]
    lifetime = LifeCycle([forward, backward], inputs, outputs)
    inline_zero = terminal and not produced
    assert (output.grad in lifetime.lifetime) != inline_zero
    codegen = ScheduleCodeGen.__new__(ScheduleCodeGen)
    codegen.execplan = SimpleNamespace(outputs=lambda: outputs)
    code = '\n'.join(codegen.emit_node(backward, produced_tids={obj.tid for obj in inputs}))
    assert ('torch.zeros_like(' in code) == inline_zero
    if not inline_zero:
        assert codegen.tensor_name(output.grad) in code
