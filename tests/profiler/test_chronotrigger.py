#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from nnscaler.profiler.chronotrigger import p2p_trace_spec


def test_p2p_endpoints_share_entity_and_have_opposite_directions():
    kwargs = {"src": 0, "dst": 2}

    sender = p2p_trace_spec(0, "adapter10916", "move", kwargs, [16083])
    receiver = p2p_trace_spec(2, "adapter10916", "move", kwargs, [16083])

    assert sender.kind == "SEND"
    assert sender.peer == 2
    assert receiver.kind == "RECV"
    assert receiver.peer == 0
    assert sender.entity == receiver.entity == "adapter10916:move:0->2:t16083"


def test_p2p_direction_is_generic_when_rank_is_not_an_endpoint():
    spec = p2p_trace_spec(1, "adapter7", "move", {"src": 0, "dst": 2})

    assert spec.kind == "COMM"
    assert spec.peer is None


def test_fanout_keeps_direction_without_inventing_peer():
    sender = p2p_trace_spec(0, "adapter8", "rdscatter", {"src": 0, "dsts": [1, 2]})
    receiver = p2p_trace_spec(2, "adapter8", "rdscatter", {"src": 0, "dsts": [1, 2]})

    assert sender.kind == "SEND"
    assert sender.peer is None
    assert receiver.kind == "RECV"
    assert receiver.peer == 0
    assert sender.entity == receiver.entity
