#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.ir.adapter.prim import (
    AllGatherPrim, ChunkPrim, RVGatherPrim, ReduceScatterPrim,
)
from nnscaler.ir.cten import IR
from nnscaler.ir.tensor import IRFullTensor


def _set_device(tensor, device):
    return IR.set_object_device(tensor, device)


def test_allgather_preserves_logical_partition_rank_order():
    full = IRFullTensor((16,))
    devices = [0, 2, 1, 3]
    inputs = [
        _set_device(full.select(((index * 4, (index + 1) * 4),), (0, 1)), device)
        for index, device in enumerate(devices)
    ]
    outputs = [_set_device(full.tosub(), device) for device in devices]

    prim = AllGatherPrim(inputs, outputs, dim=0)

    assert prim.kwargs['ranks'] == (0, 2, 1, 3)


def test_chunk_preserves_logical_partition_rank_order():
    full = IRFullTensor((16,))
    devices = [0, 2, 1, 3]
    inputs = [_set_device(full.tosub(), device) for device in devices]
    outputs = [
        _set_device(full.select(((index * 4, (index + 1) * 4),), (0, 1)), device)
        for index, device in enumerate(devices)
    ]

    prim = ChunkPrim(inputs, outputs, dim=0)

    assert prim.kwargs['ranks'] == (0, 2, 1, 3)


def test_rvgather_uses_input_devices_as_sources():
    full = IRFullTensor((16,))
    inputs = [
        _set_device(full.tosub(), 2),
        _set_device(full.tosub(), 1),
    ]
    output = _set_device(full.tosub(), 0)

    prim = RVGatherPrim(inputs, [output])

    assert prim.kwargs['srcs'] == (2, 1)
    assert prim.kwargs['dst'] == 0


def test_reduce_scatter_preserves_output_partition_order_after_dispatch():
    full = IRFullTensor((16,))
    devices = [0, 2, 1, 3]
    inputs = [
        _set_device(full.select(((0, 16),), (rank, 4)), rank)
        for rank in range(4)
    ]
    outputs = [
        _set_device(full.select(((index * 4, (index + 1) * 4),), (0, 1)), device)
        for index, device in enumerate(devices)
    ]

    prim = ReduceScatterPrim(inputs, outputs, dim=0)

    assert prim.kwargs['ranks'] == tuple(devices)
    for rank in devices:
        local = prim.dispatch(rank)
        assert local.kwargs['ranks'] == tuple(devices)
        assert local.output(0).indmap[0] == (
            devices.index(rank) * 4, (devices.index(rank) + 1) * 4,
        )
