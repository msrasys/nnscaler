# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from types import SimpleNamespace

import pytest
import torch

from nnscaler.runtime.adapter import collectives
from nnscaler.runtime.executor import AsyncCommHandler


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(rank=3, dedicated=True, calls=[], waits=0)

    def pair(src, dst):
        low, high = sorted((src, dst))
        if not state.dedicated:
            return None, None, None
        return (low, high), int(src == high), int(dst == high)

    def transmit(kind, tensor, peer=None, **kwargs):
        state.calls.append((kind, peer, kwargs, tensor))

        def wait():
            state.waits += 1

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(collectives, 'DeviceGroup', lambda: SimpleNamespace(get_p2p_group=pair))
    monkeypatch.setattr(collectives, 'CudaTimer', lambda: SimpleNamespace(start=lambda **kw: None, stop=lambda **kw: None))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda: state.rank)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 'cpu')
    for name in ['send', 'recv', 'isend', 'irecv']:
        monkeypatch.setattr(torch.distributed, name, lambda tensor, peer=None, name=name, **kw: transmit(name, tensor, peer, **kw))
    monkeypatch.setattr(torch.distributed, 'send_object_list', lambda objects, **kw: transmit('send_object_list', objects, **kw))
    payload = collectives._serialize_object({'source': 3})

    def recv_object(objects, **kwargs):
        objects[0] = payload
        transmit('recv_object_list', objects, **kwargs)

    monkeypatch.setattr(torch.distributed, 'recv_object_list', recv_object)
    AsyncCommHandler().clear()
    yield state
    AsyncCommHandler().clear()


@pytest.mark.parametrize('dedicated', [False, True])
@pytest.mark.parametrize('rank', [3, 1])
def test_object_move_uses_dedicated_pair_with_global_object_peer(backend, dedicated, rank):
    backend.rank, backend.dedicated = rank, dedicated
    value = {'source': 3}
    result = collectives.move_object(value if rank == 3 else None, 3, 1)
    assert result == value
    kind, _, kwargs, _ = backend.calls[0]
    assert kind == ('send_object_list' if rank == 3 else 'recv_object_list')
    assert kwargs.get('group') == ((1, 3) if dedicated else None)
    assert kwargs['dst' if rank == 3 else 'src'] == (1 if rank == 3 else 3)


@pytest.mark.parametrize('dedicated', [False, True])
@pytest.mark.parametrize('async_op', [False, True])
@pytest.mark.parametrize('rank', [3, 2, 0])
def test_rdscatter_uses_pair_and_correct_local_peer(backend, dedicated, async_op, rank):
    backend.rank, backend.dedicated = rank, dedicated
    full = torch.arange(12).reshape(4, 3)
    result = collectives.rdscatter(full if rank == 3 else None, full.shape, full.dtype,
                                   dim=0, src=3, dsts=(2, 0), async_op=async_op)
    destinations = (2, 0) if rank == 3 else (rank,)
    for (_, peer, kwargs, tensor), dst in zip(backend.calls, destinations):
        if dedicated:
            assert peer is None
            assert kwargs['group'] == (dst, 3)
            assert kwargs['group_dst' if rank == 3 else 'group_src'] == (0 if rank == 3 else 1)
        else:
            assert peer == (dst if rank == 3 else 3)
            assert not kwargs
        if rank == 3:
            torch.testing.assert_close(tensor, full.chunk(2)[(2, 0).index(dst)])
    assert len(backend.calls) == len(destinations)
    if async_op:
        if rank == 3:
            AsyncCommHandler().drain_sends()
        else:
            AsyncCommHandler().wait(result)
        assert backend.waits == len(destinations)
    AsyncCommHandler().check_clear()
