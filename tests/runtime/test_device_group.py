import warnings
from contextlib import nullcontext
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.torch_version import TorchVersion

from nnscaler import init, uninit
from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.module import CubeModule

from ..launch_torchrun import launch_torchrun


def _batched_group_initialization_worker(mock_split):
    init()
    try:
        device = DeviceGroup()
        module = CubeModule()
        # The last pair overlaps both disjoint pairs and excludes ranks 0 and 3.
        ranks = [[0, 1], [2, 3], [1, 2], [1, 0], list(range(4))]
        version = TorchVersion('2.11.0') if mock_split else torch.__version__
        split_context = patch.object(dist, 'split_group', wraps=dist.split_group) if mock_split else nullcontext()
        with patch.object(torch, '__version__', version), split_context as split_group:
            groups = module.init_groups(ranks)
            if mock_split:
                assert split_group.call_count == 2
                assert split_group.call_args_list[0].kwargs['split_ranks'] == [[0, 1], [2, 3]]
                assert split_group.call_args_list[1].kwargs['split_ranks'] == [[1, 2]]
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
@pytest.mark.parametrize('mock_split', [
    False,
    pytest.param(True, marks=pytest.mark.skipif(
        torch.__version__ < (2, 5), reason='split_group requires PyTorch 2.5 or later'
    )),
])
def test_batched_group_initialization(mock_split):
    launch_torchrun(4, _batched_group_initialization_worker, mock_split)


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
