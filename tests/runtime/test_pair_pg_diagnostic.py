#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Opt-in all-rank pair-process-group P2P diagnostic.

This is deliberately not wired into ``GlobalCommSchedule``.  It isolates the
process-group creation and group-local-peer mechanics needed to decide whether
pair PGs are viable independently of a dynamic pipeline schedule:

    NN_SCALER_RUN_PAIR_PG_DIAGNOSTIC=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
      pytest -s -q tests/runtime/test_pair_pg_diagnostic.py

Every rank creates every group in the same sorted order.  For each ordinal it
posts all local sends/receives before waiting, verifies asymmetric payloads,
and ends at a world barrier.  The PP4 topology intentionally gives ranks 1 and
2 two different pair groups.
"""
import os

import pytest
import torch
import torch.distributed as dist

from ..launch_torchrun import launch_torchrun, clone_to_cpu


RUN_DIAGNOSTIC = os.environ.get('NN_SCALER_RUN_PAIR_PG_DIAGNOSTIC') == '1'
PAIR_TOPOLOGIES = (
    ((0, 2), (1, 3)),
    ((0, 1), (1, 2), (2, 3)),
)
N_ORDINALS = 20


def _init_distributed():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.set_default_device(f'cuda:{rank}')


def _diagnostic_worker(pairs):
    _init_distributed()
    rank = dist.get_rank()
    pairs = tuple(sorted(tuple(sorted(pair)) for pair in pairs))

    # This loop is intentionally executed by every world rank, including
    # non-members, in one deterministic order.
    groups = {pair: dist.new_group(ranks=list(pair)) for pair in pairs}
    dist.barrier()

    received = []
    for ordinal in range(N_ORDINALS):
        works = []
        expected = []
        # Post every local operation first. P2POp uses group_peer, not a
        # world-rank peer, which is the important pair-PG API distinction.
        for pair in pairs:
            source, destination = pair if ordinal % 2 == 0 else (pair[1], pair[0])
            if rank not in pair:
                continue
            group = groups[pair]
            group_source = dist.get_group_rank(group, source)
            group_destination = dist.get_group_rank(group, destination)
            payload_value = float(10_000 * ordinal + 100 * source + destination)
            if rank == source:
                payload = torch.full((4,), payload_value, device=torch.cuda.current_device())
                ops = [dist.P2POp(dist.isend, payload, group=group, group_peer=group_destination)]
            else:
                assert rank == destination
                payload = torch.empty((4,), device=torch.cuda.current_device())
                ops = [dist.P2POp(dist.irecv, payload, group=group, group_peer=group_source)]
                expected.append((pair, payload, payload_value))
            works.extend(dist.batch_isend_irecv(ops))

        for work in works:
            work.wait()
        for pair, payload, payload_value in expected:
            assert torch.equal(payload, torch.full_like(payload, payload_value)), (
                rank, ordinal, pair, payload, payload_value
            )
            received.append((ordinal, pair, clone_to_cpu(payload)))
        # A diagnostic ordinal boundary: this is intentionally conservative
        # and prevents this test from being mistaken for a production overlap
        # implementation.
        dist.barrier()
    return {'rank': rank, 'received': received}


@pytest.mark.skipif(not RUN_DIAGNOSTIC, reason='set NN_SCALER_RUN_PAIR_PG_DIAGNOSTIC=1')
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires >= 4 GPUs')
@pytest.mark.parametrize('pairs', PAIR_TOPOLOGIES)
def test_pair_pg_all_rank_precreation_and_group_peers(pairs):
    outputs = launch_torchrun(4, _diagnostic_worker, pairs)
    assert outputs and len(outputs) == 4
    for rank in range(4):
        result = outputs[rank]
        assert result['rank'] == rank
        assert len(result['received']) == (N_ORDINALS // 2) * sum(rank in pair for pair in pairs)
