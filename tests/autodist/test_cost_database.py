#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from nnscaler.autodist.cost_database import CostDatabase
from nnscaler.graph.function.dimops import DimopSplit
from nnscaler.ir.cten import IRObject, IRTensor


class _Rule:

    def inputs(self):
        return [DimopSplit.R(), DimopSplit.R()]


class _DimAlgorithm:

    def infer(self, *_args):
        return _Rule()


def test_estimate_comm_cost_ignores_non_tensor_edges(monkeypatch):
    metadata = IRObject(value='metadata')
    activation = IRTensor(shape=(8,), dtype=torch.float32, requires_grad=True)

    src_cell = SimpleNamespace(outputs=lambda: [metadata, activation])
    dst_cell = SimpleNamespace(
        inputs=lambda: [metadata, activation],
        algorithm=lambda _tag: _DimAlgorithm(),
    )
    src_operator = SimpleNamespace(
        ir_cell=src_cell,
        dim_id2pos=lambda _dim: (-1, -1),
    )
    dst_operator = SimpleNamespace(
        ir_cell=dst_cell,
        dim_id2pos=lambda _dim: (0, 0),
    )
    src_partition = SimpleNamespace(
        operator=src_operator,
        partition_dims=(-1,),
        partition_nums=(2,),
    )
    dst_partition = SimpleNamespace(
        operator=dst_operator,
        partition_dims=('dim',),
        partition_nums=(2,),
    )

    database = CostDatabase.__new__(CostDatabase)
    primitive_to_cost = Mock(return_value=0.5)
    monkeypatch.setattr(database, 'primitive_to_cost', primitive_to_cost)

    assert database.estimate_comm_cost(src_partition, dst_partition, True) == 0.0
    primitive_to_cost.assert_not_called()
    assert database.estimate_comm_cost(src_partition, dst_partition, False) == 0.5
    primitive_to_cost.assert_called_once_with(
        2, activation.byte_size(), 'all reduce')
