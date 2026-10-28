# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from types import SimpleNamespace

from nnscaler.codegen import ModuleCodeGen
from nnscaler.ir.adapter import IRAdapter
from nnscaler.ir.adapter.prim import MovePrim, ObjectMovePrim, RDScatterPrim, RDGatherPrim, RVScatterPrim
from nnscaler.ir.cten import IR, IRObject
from nnscaler.ir.tensor import IRFullTensor


def test_all_p2p_primitives_declare_normalized_scaled_pairs():
    full = IRFullTensor((12,))
    tensor = lambda rank: IR.set_object_device(full.tosub(), rank)
    object_ = lambda rank: IR.set_object_device(IRObject(name='meta', value={'scale': 7}), rank)
    prims = [
        MovePrim([tensor(3)], [tensor(2)]),
        ObjectMovePrim([object_(3)], [object_(1)]),
        RDScatterPrim([tensor(2)], [tensor(0), tensor(1)], dim=0),
        RDGatherPrim([tensor(3), tensor(0)], [tensor(1)], dim=0),
        # A value scatter is a group broadcast, not a set of P2P transfers.
        RVScatterPrim([tensor(0)], [tensor(3)]),
    ]
    adapter = IRAdapter([], [])
    adapter.prims = prims
    generator = object.__new__(ModuleCodeGen)
    generator.plan_ndevs = 4
    generator.runtime_ndevs = 8
    generator.execplan = SimpleNamespace(graph=SimpleNamespace(select=lambda ntype: [adapter]))
    expected = {(2, 3), (1, 3), (0, 2), (1, 2), (0, 1)}
    assert generator.get_p2p_pairs() == sorted(expected | {(a + 4, b + 4) for a, b in expected})
    generator.model_init_statements = []
    generator.comm_groups = []
    generator.emit_comm_groups()
    assert f'self.init_p2p_groups(pairs={generator.get_p2p_pairs()})' in generator.model_init_statements
