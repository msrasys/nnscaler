#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from collections import Counter

import pytest

from nnscaler.flags import CompileFlag
from nnscaler.graph.gener.concurrent import ConcurrentGener
from nnscaler.ir.adapter.prim import ObjectBroadcastPrim, ObjectMovePrim
from nnscaler.ir.cten import IR, IRObject
from nnscaler.ir.tensor import IRFullTensor


def _placements(obj, ranks):
    return [IR.copy_and_set_object_device(obj, rank) for rank in ranks]


def test_object_adapter_distributes_complete_replicas():
    obj = IRObject('metadata')
    adapter = ConcurrentGener.gen_objects(
        _placements(obj, [0, 4]), _placements(obj, range(8)),
    )
    groups = {prim.kwargs['src']: set(prim.device) for prim in adapter.prims}
    assert groups == {0: {0, 1, 2, 3}, 4: {4, 5, 6, 7}}


def test_object_adapter_uses_move_for_one_remote_consumer():
    obj = IRObject('metadata')
    adapter = ConcurrentGener.gen_objects(
        _placements(obj, [2]), _placements(obj, [1, 2]),
    )
    assert len(adapter.prims) == 1
    assert adapter.prims[0].signature == 'nnscaler.runtime.adapter.move_object'
    assert adapter.prims[0].kwargs['src'] == 2
    assert adapter.prims[0].kwargs['dst'] == 1


def test_object_adapter_skips_local_consumers():
    obj = IRObject('metadata')
    assert ConcurrentGener.gen_objects(
        _placements(obj, [0, 1]), _placements(obj, [1]),
    ) is None


@pytest.mark.parametrize('producer_ranks,consumer_ranks', [
    ([0], [1, 2]),
    ([0, 2], [1, 3, 4]),
    ([0, 1, 2], [3, 4, 5, 6, 7]),
    ([0, 4], list(range(8))),
    ([4, 0], [7, 1, 6, 2, 5, 3]),
    ([0, 1, 2], [2, 3]),
    ([0, 0], [0, 1, 1, 2]),
])
@pytest.mark.parametrize('disable_fusion', [False, True])
def test_object_adapter_covers_remote_consumers(monkeypatch, producer_ranks, consumer_ranks, disable_fusion):
    monkeypatch.setattr(CompileFlag, 'disable_comm_fusion', disable_fusion)
    obj = IRObject('metadata')
    producers, consumers = _placements(obj, producer_ranks), _placements(obj, consumer_ranks)
    adapter = ConcurrentGener.gen_objects(producers, consumers)
    assert list(adapter.inputs()) == producers
    assert list(adapter.outputs()) == consumers
    assert adapter.mirror is None
    received = Counter()
    for prim in adapter.prims:
        assert isinstance(prim, (ObjectMovePrim, ObjectBroadcastPrim))
        if disable_fusion:
            assert isinstance(prim, ObjectMovePrim)
        source = prim.kwargs['src']
        assert source in producer_ranks
        received.update(output.device[0] for output in prim.outputs() if output.device[0] != source)
    assert received == Counter({rank: 1 for rank in set(consumer_ranks) - set(producer_ranks)})


def test_broadcast_source_is_not_an_adapter_output(monkeypatch):
    monkeypatch.setattr(CompileFlag, 'disable_comm_fusion', False)
    obj = IRObject('metadata')
    adapter = ConcurrentGener.gen_objects(_placements(obj, [0]), _placements(obj, [1, 2]))
    assert isinstance(adapter.prims[0], ObjectBroadcastPrim)
    assert set(adapter.prims[0].device) == {0, 1, 2}
    assert {output.device[0] for output in adapter.outputs()} == {1, 2}
    assert not adapter.dispatch(0).outputs()
    assert adapter.dispatch(0).prims


@pytest.mark.parametrize('producer_ranks,consumer_ranks', [([], [0]), ([0], []), ([0], [0])])
def test_object_adapter_without_transfers(producer_ranks, consumer_ranks):
    obj = IRObject('metadata')
    assert ConcurrentGener.gen_objects(_placements(obj, producer_ranks), _placements(obj, consumer_ranks)) is None


def test_object_adapter_requires_replicas_of_one_object():
    with pytest.raises(ValueError, match='replicas of one non-tensor IRObject'):
        ConcurrentGener.gen_objects(_placements(IRObject('a'), [0]), _placements(IRObject('b'), [1]))
    tensor = IRFullTensor((4,)).tosub()
    with pytest.raises(ValueError, match='replicas of one non-tensor IRObject'):
        ConcurrentGener.gen_objects(_placements(tensor, [0]), _placements(tensor, [1]))


def test_object_adapter_requires_expanded_devices():
    obj = IRObject('metadata')
    producer = IR.copy_and_set_object_device(obj, (0, 1))
    with pytest.raises(ValueError, match='one device per object placement'):
        ConcurrentGener.gen_objects([producer], _placements(obj, [2]))
