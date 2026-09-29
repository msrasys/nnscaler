#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler.execplan.execplan import ExeReuseCell
from nnscaler.ir.cten import IRCell
from nnscaler.ir.tensor import IRFullTensor


@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.parametrize('spatial_partitioned', [False, True])
def test_reuse_dispatch_spatial_partition(monkeypatch, requires_grad, spatial_partitioned):
    original_tensors = []
    dispatched_tensors = []
    reused_tensors = []
    for index, name in enumerate(('input', 'output')):
        parent = IRFullTensor((8,), name=name, dtype=torch.float32, requires_grad=requires_grad)
        original = parent.tosub()
        indmap = ((index * 4, (index + 1) * 4),) if spatial_partitioned else original.indmap
        dispatched = parent.select(indmap, original.valmap)
        reused = parent.like().tosub()
        if requires_grad:
            original.grad = parent.grad.tosub()
            # Spatially narrowed forward tensors can still have partial gradients.
            grad_valmap = (1 - index, 2) if spatial_partitioned else (0, 1)
            dispatched.grad = parent.grad.select(indmap, grad_valmap)
            reused.grad = reused.parent.grad.tosub()
        original_tensors.append(original)
        dispatched_tensors.append(dispatched)
        reused_tensors.append(reused)

    cell = IRCell('cell', 'cell', 1, 1)
    cell.set_input(0, original_tensors[0])
    cell.set_output(0, original_tensors[1])
    cell.device = (0, 1)
    dispatched_cell = IRCell('cell', 'cell', 1, 1)
    dispatched_cell.set_input(0, dispatched_tensors[0])
    dispatched_cell.set_output(0, dispatched_tensors[1])
    dispatched_cell.device = 0
    monkeypatch.setattr(cell, 'dispatch', lambda device: dispatched_cell)

    reuse = ExeReuseCell(cell, [reused_tensors[0]], [reused_tensors[1]])
    actual = reuse.dispatch(0)
    assert actual.cell is dispatched_cell
    assert actual is reuse.dispatch(0)
    for tensor, template, source in zip(
        (actual.input(0), actual.output(0)), dispatched_tensors, reused_tensors
    ):
        assert tensor.parent is source.parent
        assert tensor.shape == template.shape
        assert source.shape == (8,)
        assert source.indmap == ((0, 8),)
        assert tensor.indmap == template.indmap
        assert tensor.valmap == template.valmap
        assert tensor.tid == source.parent.select(template.indmap, template.valmap).tid
        assert source.valmap == (0, 1)
        if not spatial_partitioned:
            assert tensor.tid == source.tid
        if requires_grad:
            assert tensor.grad.parent is source.grad.parent
            assert tensor.grad.indmap == template.grad.indmap
            assert tensor.grad.valmap == template.grad.valmap
            assert source.grad.indmap == ((0, 8),)
            assert source.grad.valmap == (0, 1)
        else:
            assert tensor.grad is None
