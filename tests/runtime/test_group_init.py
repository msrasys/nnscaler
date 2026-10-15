# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from nnscaler.runtime import group_init
from nnscaler.runtime.device import _DeviceGroup


def test_partition_groups_are_disjoint_complete_and_order_independent():
    groups = [(0, 1), (2, 3), (0, 2), (1, 3), (0,), (1,), (2,), (3,)]
    batches = group_init.partition_groups(groups)
    assert batches == group_init.partition_groups(list(reversed(groups)) + groups)
    assert len(batches) == 3
    assert {ranks for batch in batches for ranks in batch} == set(groups)
    for batch in batches:
        ranks = [rank for group in batch for rank in group]
        assert len(ranks) == len(set(ranks))


def test_production_layout_batches():
    world, scale, stage = 1584, 48, 16
    frontend = [tuple(range(i, i + scale)) for i in range(0, world, scale)]
    frontend += [tuple(range(i, i + stage)) for i in range(0, world, stage)]
    assert len(group_init.partition_groups(frontend)) == 2
    assert len(group_init.partition_groups([(i,) for i in range(world)])) == 1
    nodes = [tuple(range(i, i + 8)) for i in range(0, world, 8)]
    assert len(group_init.partition_groups(nodes)) == 1
    pairs = [(base + lane + a * stage, base + lane + b * stage)
             for base in range(0, world, scale) for lane in range(stage)
             for a, b in [(0, 1), (0, 2), (1, 2)]]
    assert len(group_init.partition_groups(pairs)) == 3


def fake_split(monkeypatch, rank, world_size):
    calls = []
    monkeypatch.setattr(group_init, '_MIN_GROUPS_PER_ROUND', 1)
    monkeypatch.setattr(group_init, '_split_device', lambda: torch.device('cpu'))
    monkeypatch.setattr(dist, 'all_reduce', lambda *a, **kw: None)

    def split(ranks, *, timeout, use_local_synchronization):
        assert use_local_synchronization and rank in ranks
        calls.append(ranks)
        return tuple(ranks)

    monkeypatch.setattr(dist, 'new_group', split)
    return calls


@pytest.mark.parametrize('rank', range(6))
def test_partial_rounds_keep_names_balanced_without_merging_pairs(monkeypatch, rank):
    calls = fake_split(monkeypatch, rank, 6)
    pairs = [(0, 2), (0, 4), (1, 3), (1, 5), (2, 4), (3, 5)]
    extra = []
    result = group_init.create_groups(pairs, rank=rank, world_size=6,
                                      timeout=timedelta(seconds=30), auxiliary_groups=extra)
    assert len(calls) == 3
    assert extra == [(rank,)]
    for pair in pairs:
        assert result[pair] == (pair if rank in pair else dist.GroupMember.NON_GROUP_MEMBER)


def test_different_existing_histories_use_original_collective_path(monkeypatch):
    calls = fake_split(monkeypatch, 0, 4)
    new_groups = []

    def disagreement(value, **kwargs):
        value[0] = 10
        value[1] = -9

    monkeypatch.setattr(dist, 'all_reduce', disagreement)
    monkeypatch.setattr(dist, 'new_group', lambda ranks, **kw: new_groups.append(ranks) or tuple(ranks))
    result = group_init.create_groups([(0, 1), (2, 3)], rank=0, world_size=4,
                                      timeout=timedelta(seconds=30), auxiliary_groups=[])
    assert not calls
    assert new_groups == [[0, 1], [2, 3]]
    assert result[(0, 1)] == (0, 1)


def test_different_plans_fail_before_any_split(monkeypatch):
    calls = fake_split(monkeypatch, 0, 4)

    def disagreement(value, **kwargs):
        value[2] = 10
        value[3] = -9

    monkeypatch.setattr(dist, 'all_reduce', disagreement)
    with pytest.raises(ValueError, match='different pending'):
        group_init.create_groups([(0, 1), (2, 3)], rank=0, world_size=4,
                                 timeout=timedelta(seconds=30), auxiliary_groups=[])
    assert not calls


def test_unsupported_backend_keeps_new_group(monkeypatch):
    monkeypatch.setattr(group_init, '_split_device', lambda: None)
    calls = []
    monkeypatch.setattr(dist, 'new_group', lambda ranks, **kw: calls.append(ranks) or tuple(ranks))
    result = group_init.create_groups([(0, 1), (2, 3)], rank=0, world_size=4,
                                      timeout=timedelta(seconds=30), auxiliary_groups=[])
    assert calls == [[0, 1], [2, 3]]
    assert result[(2, 3)] == (2, 3)


def test_device_group_reuses_memberships_and_world(monkeypatch):
    manager = object.__new__(_DeviceGroup)
    manager.rank, manager.world_size = 0, 4
    manager.groups = {'1111': None}
    manager._split_auxiliary_groups = []
    calls = fake_split(monkeypatch, 0, 4)
    manager.init_groups([[0, 1], [2, 3], [1, 0], [0, 1, 2, 3]])
    assert len(calls) == 1
    assert manager.get_group([1, 0]) == (0, 1)
    assert manager.get_group([0, 1, 2, 3]) is None
    manager.init_groups([[2, 3], [0, 1]])
    assert len(calls) == 1


@pytest.mark.parametrize('ranks', [[], [0, 0], [-1], [4]])
def test_invalid_memberships_are_rejected(ranks):
    manager = object.__new__(_DeviceGroup)
    manager.world_size = 4
    with pytest.raises(ValueError):
        manager.init_groups([ranks])


def test_sync_group_caches_topology_and_does_not_expose_mutable_ranks(monkeypatch):
    manager = object.__new__(_DeviceGroup)
    manager.rank, manager.world_size = 3, 6
    manager.groups = {'111111': None}
    manager._split_auxiliary_groups = []
    manager._sync_groups = {}
    calls = fake_split(monkeypatch, 3, 6)
    ranks, group = manager.get_sync_group(2, 6)
    assert ranks == [1, 3, 5] and group == (1, 3, 5)
    ranks.clear()
    assert manager.get_sync_group(2, 6) == ([1, 3, 5], group)
    assert len(calls) == 1
    assert manager.get_sync_group(1, 6) == (list(range(6)), None)
    assert len(calls) == 1


def test_p2p_groups_keep_separate_communicators(monkeypatch):
    manager = object.__new__(_DeviceGroup)
    manager.rank, manager.world_size = 0, 4
    manager.use_p2p_groups = True
    manager.groups, manager.p2p_groups = {'1111': None}, {}
    manager._split_auxiliary_groups = []
    calls = fake_split(monkeypatch, 0, 4)
    monkeypatch.setattr(dist, 'barrier', lambda: None)
    manager.init_groups([[0, 1], [2, 3]])
    manager.init_p2p_groups([(1, 0), (2, 3), (0, 0)])
    assert len(calls) == 2
    assert manager.get_p2p_group(1, 0) == ((0, 1), 1, 0)
    manager.init_p2p_groups([(0, 1), (3, 2)])
    assert len(calls) == 2


def test_single_partial_group_keeps_registration_counts_balanced(monkeypatch):
    calls = fake_split(monkeypatch, 3, 4)
    extra = []
    group_init.create_groups([(0, 2)], rank=3, world_size=4,
                             timeout=timedelta(seconds=30), auxiliary_groups=extra)
    assert calls == [[3]]
    assert extra == [(3,)]


def test_creation_failure_propagates(monkeypatch):
    fake_split(monkeypatch, 0, 4)
    calls = []
    def fail(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('backend split failure')
    monkeypatch.setattr(dist, 'new_group', fail)
    with pytest.raises(RuntimeError, match='backend split failure'):
        group_init.create_groups([(0, 1), (2, 3)], rank=0, world_size=4,
                                 timeout=timedelta(seconds=30), auxiliary_groups=[])
    assert len(calls) == 1 and calls[0]['use_local_synchronization']


def test_small_fanout_uses_original_path(monkeypatch):
    calls = []
    monkeypatch.setattr(dist, 'new_group', lambda ranks, **kw: calls.append(kw) or tuple(ranks))
    group_init.create_groups([(0, 1), (2, 3)], rank=0, world_size=4,
                             timeout=timedelta(seconds=30), auxiliary_groups=[])
    assert len(calls) == 2
    assert all('use_local_synchronization' not in call for call in calls)


def test_nccl_color_collision_uses_original_path(monkeypatch):
    fake_split(monkeypatch, 0, 4)
    monkeypatch.setattr(group_init.c10d, '_process_group_color', lambda ranks: 42)
    calls = []
    monkeypatch.setattr(dist, 'new_group', lambda ranks, **kw: calls.append(kw) or tuple(ranks))
    group_init.create_groups([(0, 1), (2, 3)], rank=0, world_size=4,
                             timeout=timedelta(seconds=30), auxiliary_groups=[])
    assert len(calls) == 2
    assert all('use_local_synchronization' not in call for call in calls)


@pytest.mark.parametrize('bound,backend,supports,eligible', [
    (True, 'nccl', True, True),
    (False, 'nccl', True, False),
    (True, 'gloo', True, False),
    (True, 'nccl', None, False),
])
def test_only_bound_splittable_nccl_uses_batching(monkeypatch, bound, backend, supports, eligible):
    device = torch.device('cuda', 0)
    native = SimpleNamespace() if supports is None else SimpleNamespace(supports_splitting=supports)
    parent = SimpleNamespace(bound_device_id=device if bound else None, _get_backend=lambda _: native)
    monkeypatch.setattr(dist, 'group', SimpleNamespace(WORLD=parent))
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_backend', lambda _: backend)
    assert group_init._split_device() == (device if eligible else None)
