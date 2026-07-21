#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.ir.adapter.prim import MovePrim, RDGatherPrim, RDScatterPrim
from nnscaler.ir.cten import IR
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.profiler.chronotrigger import p2p_trace_spec, primitive_trace_spec


def _on_device(tensor, device):
    return IR.set_object_device(tensor, device)


def test_p2p_endpoints_share_entity_and_have_opposite_directions():
    kwargs = {"src": 0, "dst": 2}

    sender = p2p_trace_spec(0, "adapter10916", "move", kwargs, [16083])
    receiver = p2p_trace_spec(2, "adapter10916", "move", kwargs, [16083])

    assert sender.kind == "SEND"
    assert sender.peer == 2
    assert receiver.kind == "RECV"
    assert receiver.peer == 0
    assert sender.entity == receiver.entity == "move:0->2:t16083"


def test_p2p_direction_is_collective_when_rank_is_not_an_endpoint():
    spec = p2p_trace_spec(1, "adapter7", "move", {"src": 0, "dst": 2})

    assert spec.kind == "COLLECTIVE"
    assert spec.peer is None


def test_fanout_keeps_direction_without_inventing_peer():
    sender = p2p_trace_spec(0, "adapter8", "rdscatter", {"src": 0, "dsts": [1, 2]})
    receiver = p2p_trace_spec(2, "adapter8", "rdscatter", {"src": 0, "dsts": [1, 2]})

    assert sender.kind == "SEND"
    assert sender.peer is None
    assert receiver.kind == "RECV"
    assert receiver.peer == 0
    assert sender.entity == receiver.entity


def test_move_primitive_endpoints_normalize_parent_identity():
    full = IRFullTensor((8,))
    source = _on_device(full.tosub(), 0)
    destination = _on_device(full.tosub(), 1)
    prim = MovePrim([source], [destination])

    sender = primitive_trace_spec(prim.dispatch(0), 0, "adapter-left", 2)
    receiver = primitive_trace_spec(prim.dispatch(1), 1, "adapter-right", 7)

    assert sender.kind == "SEND"
    assert sender.peer == 1
    assert receiver.kind == "RECV"
    assert receiver.peer == 0
    assert sender.entity == receiver.entity == f"move:0->1:t{full.tid}"


def test_rdscatter_endpoints_share_fanout_entity_without_sender_peer():
    full = IRFullTensor((8,))
    source = _on_device(full.tosub(), 0)
    destinations = [
        _on_device(full.select(((0, 4),), (0, 1)), 1),
        _on_device(full.select(((4, 8),), (0, 1)), 2),
    ]
    prim = RDScatterPrim([source], destinations, dim=0)

    sender = primitive_trace_spec(prim.dispatch(0), 0, "adapter-left", 2)
    receiver = primitive_trace_spec(prim.dispatch(2), 2, "adapter-right", 7)

    assert sender.kind == "SEND"
    assert sender.peer is None
    assert receiver.kind == "RECV"
    assert receiver.peer == 0
    assert sender.entity == receiver.entity == f"rdscatter:0->1,2:t{full.tid}"


def test_rdgather_endpoints_share_fanin_entity_without_receiver_peer():
    full = IRFullTensor((8,))
    sources = [
        _on_device(full.select(((0, 4),), (0, 1)), 1),
        _on_device(full.select(((4, 8),), (0, 1)), 2),
    ]
    destination = _on_device(full.tosub(), 0)
    prim = RDGatherPrim(sources, [destination], dim=0)

    sender = primitive_trace_spec(prim.dispatch(1), 1, "adapter-left", 2)
    receiver = primitive_trace_spec(prim.dispatch(0), 0, "adapter-right", 7)

    assert sender.kind == "SEND"
    assert sender.peer == 0
    assert receiver.kind == "RECV"
    assert receiver.peer is None
    assert sender.entity == receiver.entity == f"rdgather:1,2->0:t{full.tid}"
