#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.graph.gener.gen import IRAdapterGener
from nnscaler.ir.cten import IR, IRObject


def _placements(obj, ranks):
    return [IR.copy_and_set_object_device(obj, rank) for rank in ranks]


def test_object_adapter_keeps_replica_groups_together():
    obj = IRObject('metadata')
    adapter = IRAdapterGener.gen_object_adapter(
        _placements(obj, [0, 4]), _placements(obj, range(8)),
    )
    groups = {prim.kwargs['src']: set(prim.device) for prim in adapter.prims}
    assert groups == {0: {0, 1, 2, 3}, 4: {4, 5, 6, 7}}


def test_object_adapter_uses_move_for_one_remote_consumer():
    obj = IRObject('metadata')
    adapter = IRAdapterGener.gen_object_adapter(
        _placements(obj, [2]), _placements(obj, [1, 2]),
    )
    assert len(adapter.prims) == 1
    assert adapter.prims[0].signature == 'nnscaler.runtime.adapter.move_object'
    assert adapter.prims[0].kwargs['src'] == 2
    assert adapter.prims[0].kwargs['dst'] == 1


def test_object_adapter_skips_local_consumers():
    obj = IRObject('metadata')
    assert IRAdapterGener.gen_object_adapter(
        _placements(obj, [0, 1]), _placements(obj, [1]),
    ) is None
