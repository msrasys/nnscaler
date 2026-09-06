#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from unittest.mock import Mock

import torch

from nnscaler.autodist.cost_database import CostDatabase
from nnscaler.autodist.cube_operator import CubeOperator
from nnscaler.autodist.op_partition import OpPartition
from nnscaler.graph.function.dimops import IRDimops
from nnscaler.ir.cten import IRObject
from nnscaler.ir.tensor import IRFullTensor


def _source_op(x, signature='test.source'):
    return IRDimops(
        _source_op, 'source', signature, ['a -> ?, a'], [x])


def _destination_op(metadata, x, signature='test.destination'):
    return IRDimops(
        _destination_op, 'destination', signature, ['?, a -> a'],
        [metadata, x])


def test_estimate_comm_cost_ignores_non_tensor_edges(monkeypatch):
    metadata = IRObject(value='metadata')
    activation = IRFullTensor(
        (8,), dtype=torch.float32, requires_grad=True).tosub()

    src_cell = _source_op(activation)
    src_cell.set_output(0, metadata)
    src_cell.set_output(1, activation)
    dst_cell = _destination_op(metadata, activation)
    dst_cell.set_output(0, IRFullTensor(
        (8,), dtype=torch.float32, requires_grad=True).tosub())

    src_partition = OpPartition(
        (-1,), (2,), CubeOperator(src_cell))
    dst_partition = OpPartition(
        ('a',), (2,), CubeOperator(dst_cell))

    database = CostDatabase.__new__(CostDatabase)
    primitive_to_cost = Mock(return_value=0.5)
    monkeypatch.setattr(database, 'primitive_to_cost', primitive_to_cost)

    assert database.estimate_comm_cost(src_partition, dst_partition, True) == 0.0
    primitive_to_cost.assert_not_called()
    assert database.estimate_comm_cost(src_partition, dst_partition, False) == 0.5
    primitive_to_cost.assert_called_once_with(
        2, activation.byte_size(), 'all gather')
