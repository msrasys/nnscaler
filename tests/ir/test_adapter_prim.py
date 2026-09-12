#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest

from nnscaler.ir.adapter.prim import AllGatherPrim, ChunkPrim, RVGatherPrim
from nnscaler.ir.cten import IR
from nnscaler.ir.tensor import IRFullTensor


def _set_device(tensor, device):
    return IR.set_object_device(tensor, device)


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


@pytest.mark.parametrize('explicit_ranks', [None, (1, 2)])
def test_gather_and_chunk_preserve_layout_order(explicit_ranks):
    full = IRFullTensor((16,))
    shards = [
        _set_device(full.select(((0, 8),), (0, 1)), 2),
        _set_device(full.select(((8, 16),), (0, 1)), 1),
    ]
    replicas = [_set_device(full.tosub(), rank) for rank in (2, 1)]
    kwargs = {} if explicit_ranks is None else {'ranks': explicit_ranks}
    for prim in (AllGatherPrim(shards, replicas, 0, **kwargs),
                 ChunkPrim(replicas, shards, 0, **kwargs)):
        assert prim.kwargs['ranks'] == (explicit_ranks or (2, 1))
