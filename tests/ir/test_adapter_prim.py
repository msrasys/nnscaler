#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import inspect
import pytest

from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter import prim as primitives
from nnscaler.runtime.adapter import collectives

from nnscaler.ir.adapter.prim import AllGatherPrim, AllToAllPrim, ChunkPrim, ReduceScatterPrim, RVGatherPrim
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
                 ReduceScatterPrim(replicas, shards, 0, **kwargs),
                 ChunkPrim(replicas, shards, 0, **kwargs)):
        assert prim.kwargs['ranks'] == (explicit_ranks or (2, 1))


def test_all_to_all_preserves_layout_order():
    full = IRFullTensor((8, 8))
    inputs = [_set_device(full.select(((i * 4, (i + 1) * 4), (0, 8)), (0, 1)), rank)
              for i, rank in enumerate((2, 1))]
    outputs = [_set_device(full.select(((0, 8), (i * 4, (i + 1) * 4)), (0, 1)), rank)
               for i, rank in enumerate((2, 1))]
    assert AllToAllPrim(inputs, outputs, 0, 1).kwargs['ranks'] == (2, 1)


@pytest.mark.parametrize('primitive', [primitives.AllToAllPrim, primitives.AllToAllAllToAllPrim])
@pytest.mark.parametrize('explicit_ranks', [False, True])
def test_all_to_all_keeps_independent_layouts_through_dispatch_and_scaling(primitive, explicit_ranks):
    from nnscaler.codegen import ModuleCodeGen

    full = IRFullTensor((8, 8))
    inputs = [_set_device(full.select(((i * 2, (i + 1) * 2), (0, 8)), (0, 1)), rank)
              for i, rank in enumerate((2, 0, 3, 1))]
    outputs = [_set_device(full.select(((0, 8), (i * 2, (i + 1) * 2)), (0, 1)), rank)
               for i, rank in enumerate((1, 3, 0, 2))]
    prim = primitive(inputs, outputs, 0, 1, **({'ranks': (2, 0, 3, 1)} if explicit_ranks else {}))
    assert prim.kwargs['ranks'] == (2, 0, 3, 1)
    assert prim.kwargs['dsts'] == (1, 3, 0, 2)
    for rank in range(4):
        assert prim.dispatch(rank).kwargs == prim.kwargs

    adapter = IRAdapter(inputs, outputs)
    adapter.prims = [prim]
    generator = object.__new__(ModuleCodeGen)
    generator.devices = (0, 1, 2, 3)
    generator.runtime_ndevs = 8
    generator.enable_dp = True
    scaled = generator.scale(adapter, 4).prims[0]
    assert tuple(scaled.kwargs['ranks']) == (6, 4, 7, 5)
    assert tuple(scaled.kwargs['dsts']) == (5, 7, 4, 6)


@pytest.mark.parametrize('primitive', [
    primitives.AllGatherPrim, primitives.AllGatherReduceScatterPrim, primitives.AllGatherSplitPrim,
    primitives.ReduceScatterPrim, primitives.ReduceScatterAllGatherPrim,
    primitives.ChunkPrim, primitives.SplitAllGatherPrim,
])
def test_one_sided_shard_order_does_not_follow_replica_order(primitive):
    full = IRFullTensor((8,))
    shards = [_set_device(full.select(((i * 2, (i + 1) * 2),), (0, 1)), rank)
              for i, rank in enumerate((2, 0, 3, 1))]
    replicas = [_set_device(full.tosub(), rank) for rank in (1, 3, 0, 2)]
    if primitive in (primitives.AllGatherPrim, primitives.AllGatherReduceScatterPrim, primitives.AllGatherSplitPrim):
        prim = primitive(shards, replicas, 0)
    else:
        prim = primitive(replicas, shards, 0)
    assert prim.kwargs['ranks'] == (2, 0, 3, 1)


@pytest.mark.parametrize('dim', [0, 1])
@pytest.mark.parametrize('ranks', [None, (7,), (7, 2), (8, 2, 5), (2, 0, 3, 1)])
def test_split_allgather_volume_matches_backward_allgather(dim, ranks):
    full = IRFullTensor((12, 12))
    ndevs = len(ranks) if ranks is not None else 4
    replicas = [full.tosub() for _ in range(ndevs)]
    shards = full.tosub().split_dim(dim, ndevs)
    if ranks is not None:
        replicas = [_set_device(tensor, rank) for tensor, rank in zip(replicas, sorted(ranks))]
        shards = [_set_device(tensor, rank) for tensor, rank in zip(shards, ranks)]

    backward = AllGatherPrim(shards, replicas, dim)
    fused = primitives.SplitAllGatherPrim(replicas, shards, dim)
    assert fused.volume() == backward.volume()
    if ranks is not None:
        assert fused.kwargs['ranks'] == ranks
        for rank in ranks:
            # Dispatch retains one input/output but still communicates with the whole group.
            assert fused.dispatch(rank).volume() == backward.volume()


def test_split_allgather_fusion_preserves_backward_volume():
    from nnscaler.codegen import ModuleCodeGen
    from nnscaler.execplan.planpass.fusion import DiffFusion
    from nnscaler.ir.cten import IRCell

    full = IRFullTensor((32,), requires_grad=True)
    ranks = (2, 0, 3, 1)
    replicas = [_set_device(full.tosub(), rank) for rank in sorted(ranks)]
    shards = [_set_device(tensor, rank) for tensor, rank in zip(full.tosub().split_dim(0, 4), ranks)]
    grad_shards = [_set_device(full.grad.select(t.indmap, t.valmap), t.device[0]) for t in shards]
    grad_replicas = [_set_device(full.grad.tosub(), t.device[0]) for t in replicas]
    forward = IRAdapter(replicas, shards)
    forward.prims = [ChunkPrim(replicas, shards, 0)]
    backward = IRAdapter(grad_shards, grad_replicas)
    backward.prims = [AllGatherPrim(grad_shards, grad_replicas, 0)]
    IRCell.make_pair(forward, backward)
    expected = backward.prims[0].volume()
    assert expected == 24

    generator = object.__new__(ModuleCodeGen)
    generator.devices = (0, 1, 2, 3)
    generator.runtime_ndevs = 8
    generator.enable_dp = True
    for rank in ranks:
        # ExecutionPlan dispatches adapters before DiffFusion runs.
        local = forward.dispatch(rank)
        assert DiffFusion.nnfuse(local)
        fused = local.prims[0]
        assert fused.signature == 'nnscaler.runtime.adapter.nn.split_allgather'
        assert fused.kwargs['ranks'] == ranks
        assert fused.volume() == expected
        scaled = generator.scale(local, rank + 4).prims[0]
        assert tuple(scaled.kwargs['ranks']) == tuple(r + 4 for r in ranks)
        assert scaled.volume() == expected


@pytest.mark.parametrize('primitive,runtime', [
    (primitives.RVScatterPrim, collectives.rvscatter),
    (primitives.RVGatherPrim, collectives.rvgather),
])
def test_value_collective_layout_metadata_matches_runtime(primitive, runtime):
    full = IRFullTensor((12,))
    root = _set_device(full.tosub(), 0)
    parts = [_set_device(full.select(((0, 12),), (index, 3)), rank)
             for index, rank in enumerate((3, 1, 2))]
    prim = primitive([root], parts) if primitive is primitives.RVScatterPrim else primitive(parts, [root])
    for rank in range(4):
        local = prim.dispatch(rank)
        inspect.signature(runtime).bind(None, **local.kwargs)
        assert local.kwargs['ranks'] == prim.kwargs['ranks']
        assert set(local.kwargs['ranks']) == {0, 1, 2, 3}
    key = 'dsts' if primitive is primitives.RVScatterPrim else 'srcs'
    assert prim.kwargs[key] == (3, 1, 2)
