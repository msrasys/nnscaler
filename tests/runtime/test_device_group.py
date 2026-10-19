import warnings
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.torch_version import TorchVersion

from nnscaler import init, uninit
from nnscaler.codegen.module.module import ModuleCodeGen
from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.module import CubeModule

from ..launch_torchrun import launch_torchrun
from ..utils import mock_cube_env, mock_dist


def test_batched_group_initialization():
    codegen = ModuleCodeGen.__new__(ModuleCodeGen)
    codegen.runtime_ndevs = 4
    codegen.comm_groups = [(0, 1), (1, 0), (2, 3), (1, 2), (0, 1, 2, 3)]
    codegen.model_init_statements = []
    codegen.emit_comm_groups()
    code = '\n'.join(codegen.model_init_statements)
    expected_batches = [
        [[0], [1], [2], [3]],
        [[0, 1], [2, 3]],
        [[1, 2]],
    ]
    singleton, pair, extra = object(), object(), object()

    with mock_cube_env(0, 4), mock_dist(0, 4):
        device = DeviceGroup()
        module = CubeModule()
        world = SimpleNamespace(bound_device_id=torch.device('cuda', 0))
        with patch.object(torch, '__version__', TorchVersion('2.5.0')), \
                patch.object(dist, 'get_backend', return_value='nccl'), \
                patch.object(dist, 'group', SimpleNamespace(WORLD=world)), \
                patch.object(dist, 'split_group', create=True, side_effect=[singleton, pair, None, extra]) as split, \
                patch.object(dist, 'new_group') as new:
            exec(code, {'self': module})
            assert [call.kwargs['split_ranks'] for call in split.call_args_list] == expected_batches
            assert device.get_group([0]) is singleton
            assert device.get_p2p_group(1, 0) is pair
            assert device.get_group([0, 1]) is pair
            assert device.get_group([1, 2]) == dist.GroupMember.NON_GROUP_MEMBER
            assert device.get_group([0, 1, 2, 3]) is None
            assert device.get_p2p_group(0, 3) is None
            exec(code, {'self': module})
            assert split.call_count == 3
            ranks = [[1, 3], [0, 1], [0, 2], [2, 0], [0, 1, 2, 3]]
            expected = [dist.GroupMember.NON_GROUP_MEMBER, pair, extra, extra, None]
            assert module.get_groups(ranks) == expected
            assert split.call_args.kwargs['split_ranks'] == [[0, 2], [1, 3]]
            assert device.get_groups(ranks) == expected
            assert device.get_groups([]) == []
            assert split.call_count == 4
            new.assert_not_called()


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
