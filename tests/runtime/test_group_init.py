# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from nnscaler.runtime import group_init
from nnscaler.runtime.device import _DeviceGroup

TIMEOUT = timedelta(seconds=30)


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


def fake_backend(monkeypatch, rank):
    calls = []
    monkeypatch.setattr(group_init, '_split_context', lambda: (torch.device('cpu'), True))
    monkeypatch.setattr(dist, 'all_reduce', lambda *a, **kw: None)
    monkeypatch.setattr(dist, 'get_rank', lambda: rank)
    monkeypatch.setattr(dist, 'barrier', lambda: None)

    def split(*, split_ranks, timeout):
        calls.append(('split', split_ranks))
        return next((tuple(ranks) for ranks in split_ranks if rank in ranks), None)

    def ordinary(ranks, *, timeout):
        calls.append(('ordinary', ranks))
        return tuple(ranks) if rank in ranks else dist.GroupMember.NON_GROUP_MEMBER

    monkeypatch.setattr(dist, 'split_group', split, raising=False)
    monkeypatch.setattr(dist, 'new_group', ordinary)
    return calls


@pytest.mark.parametrize('rank', range(6))
def test_sparse_rounds_do_not_add_groups_and_normalize_nonmembers(monkeypatch, rank):
    calls = fake_backend(monkeypatch, rank)
    pairs = [(0, 2), (0, 4), (1, 3), (1, 5), (2, 4), (3, 5)]
    result = group_init.create_groups(pairs, timeout=TIMEOUT)
    assert len(calls) == 3 and all(kind == 'split' for kind, _ in calls)
    assert {tuple(ranks) for _, batch in calls for ranks in batch} == set(pairs)
    for pair in pairs:
        assert result[pair] == (pair if rank in pair else dist.GroupMember.NON_GROUP_MEMBER)


def test_one_sparse_group_uses_ordinary_creation(monkeypatch):
    calls = fake_backend(monkeypatch, 3)
    result = group_init.create_groups([(0, 2)], timeout=TIMEOUT)
    assert calls == [('ordinary', [0, 2])]
    assert result[(0, 2)] == dist.GroupMember.NON_GROUP_MEMBER


def test_all_ranks_fall_back_if_one_backend_cannot_split(monkeypatch):
    calls = fake_backend(monkeypatch, 0)
    monkeypatch.setattr(dist, 'all_reduce', lambda value, **kw: value.__setitem__(0, 0))
    group_init.create_groups([(0, 1), (2, 3)], timeout=TIMEOUT)
    assert calls == [('ordinary', [0, 1]), ('ordinary', [2, 3])]


def test_different_plans_fail_before_creation(monkeypatch):
    calls = fake_backend(monkeypatch, 0)
    monkeypatch.setattr(dist, 'all_reduce', lambda value, **kw: value.__setitem__(1, value[1] - 1))
    with pytest.raises(ValueError, match='different pending'):
        group_init.create_groups([(0, 1), (2, 3)], timeout=TIMEOUT)
    assert not calls


def test_unbound_backend_keeps_ordinary_creation(monkeypatch):
    calls = fake_backend(monkeypatch, 0)
    monkeypatch.setattr(group_init, '_split_context', lambda: None)
    group_init.create_groups([(0, 1), (2, 3)], timeout=TIMEOUT)
    assert calls == [('ordinary', [0, 1]), ('ordinary', [2, 3])]


def manager(rank=0, world=4):
    value = object.__new__(_DeviceGroup)
    value.rank, value.world_size = rank, world
    value.groups = {'1' * world: None}
    value._owned_groups = []
    value.p2p_groups = {}
    value.use_p2p_groups = True
    return value


def test_cache_reuses_memberships_and_world(monkeypatch):
    device = manager()
    calls = fake_backend(monkeypatch, 0)
    groups = device.get_groups([[2, 3], [0, 1], [1, 0], [0, 1, 2, 3]])
    assert groups == [dist.GroupMember.NON_GROUP_MEMBER, (0, 1), (0, 1), None]
    assert groups[1] is groups[2]
    assert len(calls) == 1
    assert device.get_group([1, 0]) == (0, 1)
    assert device.get_group([0, 1, 2, 3]) is None
    assert device.init_groups([[2, 3], [0, 1]]) is None
    assert device.get_groups([]) == []
    assert len(calls) == 1


@pytest.mark.parametrize('ranks', [[], [0, 0], [-1], [4]])
def test_invalid_memberships_are_rejected(ranks):
    with pytest.raises(ValueError):
        manager().init_groups([ranks])


def test_sync_group_reuses_handles_and_returns_fresh_ranks(monkeypatch):
    device = manager(3, 6)
    calls = fake_backend(monkeypatch, 3)
    ranks, group = device.get_sync_group(2, 6)
    assert ranks == [1, 3, 5] and group == (1, 3, 5)
    ranks.clear()
    assert device.get_sync_group(2, 6) == ([1, 3, 5], group)
    assert len(calls) == 1
    assert device.get_sync_group(1, 6) == (list(range(6)), None)


def test_p2p_groups_have_separate_cache(monkeypatch):
    device = manager()
    calls = fake_backend(monkeypatch, 0)
    device.init_groups([[0, 1], [2, 3]])
    device.init_p2p_groups([(1, 0), (2, 3), (0, 0)])
    assert len(calls) == 2
    assert device.get_p2p_group(1, 0) == ((0, 1), 1, 0)
    assert device.get_p2p_group(0, 1)[0] is not device.get_group([0, 1])
    assert len(device._owned_groups) == 2
    device.init_p2p_groups([(0, 1), (3, 2)])
    assert len(calls) == 2
    assert len(device._owned_groups) == 2


def test_backend_errors_are_not_hidden_by_retry(monkeypatch):
    fake_backend(monkeypatch, 0)
    def fail(**kwargs):
        raise RuntimeError('backend split failure')
    monkeypatch.setattr(dist, 'split_group', fail)
    with pytest.raises(RuntimeError, match='backend split failure'):
        group_init.create_groups([(0, 1), (2, 3)], timeout=TIMEOUT)


@pytest.mark.parametrize('owns_world', [False, True])
def test_close_releases_only_owned_handles_in_reverse_order(monkeypatch, owns_world):
    device = manager()
    device._is_pg_initer = owns_world
    for key, group in [('world', None), ('first', 'first'),
                       ('nonmember', dist.GroupMember.NON_GROUP_MEMBER), ('second', 'second')]:
        device._register_group(device.groups, key, group)
    assert device.groups['nonmember'] == dist.GroupMember.NON_GROUP_MEMBER
    assert device._owned_groups == ['first', 'second']
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    closed = []
    monkeypatch.setattr(dist, 'destroy_process_group', lambda group='world': closed.append(group))
    device.close()
    assert closed == ['second', 'first'] + (['world'] if owns_world else [])
    assert not device._owned_groups and not device._is_pg_initer


def test_single_group_registration_reuses_the_handle(monkeypatch):
    device = manager()
    fake_backend(monkeypatch, 0)
    group = device.get_group([0, 1])
    assert device.get_group([1, 0]) is group
    device.get_group([2, 3])
    assert device._owned_groups == [group]
    with pytest.raises(AssertionError):
        device._register_group(device.groups, device.bitmap([0, 1]), object())
    assert device.get_group([0, 1]) is group
    assert device._owned_groups == [group]


def test_plan_agreement_uses_one_collective(monkeypatch):
    fake_backend(monkeypatch, 0)
    reductions = []
    monkeypatch.setattr(dist, 'all_reduce', lambda value, **kw: reductions.append(value))
    group_init.create_groups([(0, 1), (2, 3)], timeout=TIMEOUT)
    assert len(reductions) == 1


@pytest.mark.parametrize('bound,backend,uid,rank_map,expected', [
    (True, 'nccl', 0, [], True),
    (True, 'nccl', 7, [], False),
    (True, 'nccl', 7, [0, 1], True),
    (False, 'nccl', 0, [], None),
    (True, 'gloo', 0, [], None),
])
def test_split_capability_including_world_reinitialization(monkeypatch, bound, backend, uid, rank_map, expected):
    device = torch.device('cuda', 0)
    native = SimpleNamespace(supports_splitting=True, uid=uid,
                             options=SimpleNamespace(global_ranks_in_group=rank_map))
    parent = SimpleNamespace(bound_device_id=device if bound else None, _get_backend=lambda _: native)
    monkeypatch.setattr(dist, 'group', SimpleNamespace(WORLD=parent))
    monkeypatch.setattr(dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(dist, 'get_backend', lambda _: backend)
    monkeypatch.setattr(dist, 'split_group', lambda **kw: None, raising=False)
    assert group_init._split_context() == (None if expected is None else (device, expected))
