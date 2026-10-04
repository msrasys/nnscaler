from types import SimpleNamespace
from unittest.mock import Mock

import torch.distributed as dist
from nnscaler.runtime import group_init


def install_world(monkeypatch, rank, count, world_size=6):
    default = Mock(bound_device_id=None)
    world = SimpleNamespace(pg_map={default: ('nccl', object())}, pg_group_ranks={},
                            pg_names={object(): str(i) for i in range(count)})
    calls = []
    monkeypatch.setattr(group_init.c10d, '_world', world)
    monkeypatch.setattr(group_init.c10d, '_get_default_group', lambda: default)
    monkeypatch.setattr(dist, 'get_rank', lambda: rank)
    monkeypatch.setattr(dist, 'get_world_size', lambda: world_size)
    def create(*args, **kwargs):
        calls.append((args, kwargs))
        return Mock(), Mock()
    monkeypatch.setattr(group_init.c10d, '_new_process_group_helper', create)
    return calls, world


def test_names_independent_of_unrelated_history_and_namespace(monkeypatch):
    names = []
    for rank, history in [(0, 17), (2, 4)]:
        calls, world = install_world(monkeypatch, rank, history)
        for namespace in ['collective', 'p2p', 'collective']:
            pg = group_init.new_local_group([2, 0], namespace=namespace, timeout=1)
            assert world.pg_group_ranks[pg] == {0: 0, 2: 1}
        names.append([call[0][5] for call in calls])
    assert names[0] == names[1]
    assert len(set(names[0])) == 3


def test_nonmembers_do_not_create_a_process_group(monkeypatch):
    calls, _ = install_world(monkeypatch, 3, 6)
    result = group_init.new_local_group([0, 2], namespace='p2p', timeout=1)
    assert result == dist.GroupMember.NON_GROUP_MEMBER
    assert calls == []


def test_requested_init_barrier_is_group_local(monkeypatch):
    calls, _ = install_world(monkeypatch, 2, 4)
    barriers = []
    monkeypatch.setattr(group_init.c10d, '_is_barrier_after_init', lambda: 1)
    monkeypatch.setattr(group_init.c10d, '_store_based_barrier', lambda *a: barriers.append(a))
    group_init.new_local_group([0, 2], namespace='p2p', timeout=7)
    assert len(barriers) == 1
    assert barriers[0][0] == 2 and barriers[0][3:] == (2, 7)
    assert barriers[0][2] == calls[0][0][5]


def test_pp3_ep16_dp33_only_creates_local_memberships(monkeypatch):
    from nnscaler.runtime.device import _DeviceGroup
    size = 1584
    calls, _ = install_world(monkeypatch, 0, 1, size)
    monkeypatch.setattr(dist, 'barrier', lambda: None)
    manager = _DeviceGroup.__new__(_DeviceGroup)
    manager.world_size = size
    manager.rank = 0
    manager._local_group_init = True
    manager.groups = {'1' * size: None}
    manager.p2p_groups = {}
    manager.use_p2p_groups = True
    for rank in range(size):
        manager.get_group([rank])
    pairs = [
        (unit + stage * 16 + lane, unit + ((stage + 1) % 3) * 16 + lane)
        for unit in range(0, size, 48) for stage in range(3) for lane in range(16)
    ]
    manager.init_p2p_groups(pairs)
    assert len(manager.p2p_groups) == 1584
    assert len(calls) == 3  # one singleton and two PP neighbors for rank0
