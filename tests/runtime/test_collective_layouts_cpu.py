# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU transport oracle; real NCCL coverage lives in test_collective_layouts."""
from types import SimpleNamespace

import pytest
import torch

from nnscaler.runtime.adapter import collectives
from nnscaler.runtime.executor import AsyncCommHandler


@pytest.fixture(autouse=True)
def isolated_handler(monkeypatch):
    AsyncCommHandler().clear()
    monkeypatch.setattr(collectives, 'CudaTimer', lambda: SimpleNamespace(
        start=lambda **kw: None, stop=lambda **kw: None))
    yield
    AsyncCommHandler().clear()


@pytest.mark.parametrize('single', [False, True])
@pytest.mark.parametrize('async_op', [False, True])
@pytest.mark.parametrize('idim,odim', [(0, 1), (1, 0)])
@pytest.mark.parametrize('ranks,dsts', [
    ((0, 1), None),
    ((2, 0, 3, 1), None),
    ((2, 0, 3, 1), (1, 3, 0, 2)),
    ((3, 0, 2), (2, 3, 0)),
])
def test_alltoall_translates_shards_and_waits_for_received_data(
        monkeypatch, single, async_op, idim, odim, ranks, dsts):
    destinations = ranks if dsts is None else dsts
    backend_ranks = sorted(ranks)
    rank = ranks[0]
    count = len(ranks)
    full = torch.arange(6 * count * count, dtype=torch.float64).reshape(2 * count, 3 * count)
    local = full.chunk(count, dim=idim)[ranks.index(rank)].T.contiguous().T
    expected = full.chunk(count, dim=odim)[destinations.index(rank)]
    monkeypatch.setattr(collectives, 'DeviceGroup', lambda: SimpleNamespace(get_group=lambda ranks: None))
    monkeypatch.setattr(torch.distributed, 'get_process_group_ranks', lambda group: backend_ranks)
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda group=None: rank)
    waits = []

    def exchange(outputs, inputs, group=None, async_op=False):
        send = local.chunk(count, dim=odim)
        for output_rank, sent in zip(backend_ranks, inputs):
            torch.testing.assert_close(sent, send[destinations.index(output_rank)])

        def finish():
            waits.append(True)
            for output, source_rank in zip(outputs, backend_ranks):
                expected_piece = full.chunk(count, dim=idim)[ranks.index(source_rank)].chunk(
                    count, dim=odim)[destinations.index(rank)]
                output.copy_(expected_piece)

        if async_op:
            return SimpleNamespace(wait=finish)
        finish()

    def exchange_single(output, tensor, group=None, async_op=False):
        # The single-buffer backend always partitions axis 0.
        inputs = [part.transpose(0, odim) if odim else part for part in tensor.chunk(count)]
        outputs = [part.transpose(0, odim) if odim else part for part in output.chunk(count)]
        return exchange(outputs, inputs, group, async_op)

    monkeypatch.setattr(torch.distributed, 'all_to_all', exchange)
    monkeypatch.setattr(torch.distributed, 'all_to_all_single', exchange_single)
    operation = collectives.all_to_all_single if single else collectives.all_to_all
    kwargs = {} if dsts is None else {'dsts': dsts}
    handle = operation(local, idim, odim, ranks, async_op=async_op, **kwargs)
    if async_op:
        assert not waits, 'transport must remain asynchronous until its result is consumed'
        actual = AsyncCommHandler().wait(handle)
    else:
        actual = handle
    assert len(waits) == 1
    torch.testing.assert_close(actual, expected)
    AsyncCommHandler().check_clear()


@pytest.mark.parametrize('operation', ['all_gather', 'reduce_scatter'])
def test_world_rank_lookup_uses_explicit_group_for_older_pytorch(monkeypatch, operation):
    world = object()
    monkeypatch.setattr(torch.distributed, 'group', SimpleNamespace(WORLD=world))
    monkeypatch.setattr(collectives, 'DeviceGroup', lambda: SimpleNamespace(get_group=lambda ranks: None))
    monkeypatch.setattr(torch.distributed, 'get_rank', lambda group=None: 0)

    def group_ranks(group):
        assert group is world
        return [0, 1]

    monkeypatch.setattr(torch.distributed, 'get_process_group_ranks', group_ranks)
    monkeypatch.setattr(torch.distributed, 'all_gather', lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, 'reduce_scatter', lambda *args, **kwargs: None)
    getattr(collectives, operation)(torch.ones(4), 0, (1, 0))
