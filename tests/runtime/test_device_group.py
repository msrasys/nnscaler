import warnings

import pytest
import torch
import torch.distributed as dist

from nnscaler import init, uninit
from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.module import CubeModule

from ..launch_torchrun import launch_torchrun


def _batched_group_initialization_worker():
    init()
    try:
        device = DeviceGroup()
        module = CubeModule()
        # The last pair overlaps both disjoint pairs and excludes ranks 0 and 3.
        ranks = [[0, 1], [2, 3], [1, 2], [1, 0], list(range(4))]
        groups = module.init_groups(ranks)
        assert groups[0] is groups[3]
        assert groups[-1] is None
        assert device.get_p2p_group(0, 1) is groups[0]
        for members, group in zip(ranks, groups):
            if device.rank in members:
                assert dist.get_world_size(group) == len(members)
                value = torch.tensor(float(device.rank), device='cuda')
                dist.all_reduce(value, group=group)
                assert value.item() == sum(members)
            else:
                assert group == dist.GroupMember.NON_GROUP_MEMBER
        cached = dict(device.groups)
        repeated = module.init_groups(ranks[::-1])
        assert all(group is previous for group, previous in zip(repeated, groups[::-1]))
        assert device.groups == cached
        assert device.get_groups([]) == []
    finally:
        uninit()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_batched_group_initialization():
    launch_torchrun(4, _batched_group_initialization_worker)


def _barrier_worker():
    init()

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(3):
                dist.barrier()

        # make sure there is no barrier() device warning
        # UserWarning: barrier(): using the device under current context. You can specify device_id in init_process_group to mute this warning.
        # return func(*args, **kwargs)
        barrier_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and 'barrier()' in str(w.message)
            and 'device_id' in str(w.message)
        ]
        assert len(barrier_warnings) == 0
    finally:
        uninit()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_barrier_warning():
    launch_torchrun(2, _barrier_worker)
