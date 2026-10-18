import warnings
from datetime import timedelta
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from nnscaler import init, uninit

from ..launch_torchrun import launch_torchrun


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


def _batched_group_worker(directory):
    from nnscaler.runtime.device import DeviceGroup, uninit_device
    from nnscaler.runtime.module import CubeModule
    from nnscaler.runtime import group_init

    rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank)
    for generation in range(2):
        # Unsupported WORLD reinitialization must use the ordinary fallback.
        dist.init_process_group('nccl', rank=rank, world_size=4,
                                init_method=f'file://{Path(directory) / str(generation)}',
                                device_id=torch.device('cuda', rank), timeout=timedelta(seconds=60))
        manager = DeviceGroup()
        groups = [[0, 1], [2, 3], [0, 2]]
        registry_before = len(dist.distributed_c10d._world.pg_names)
        manager.init_groups(groups)
        assert len(dist.distributed_c10d._world.pg_names) - registry_before == 1 + int(rank in [0, 2])
        # Legacy generated code's single-group declarations remain supported.
        for member in range(4):
            CubeModule.init_group(None, [member])
        before = dict(manager.groups)
        manager.init_groups(groups)
        assert before == manager.groups
        for members in groups + [[i] for i in range(4)]:
            if rank in members:
                value = torch.tensor([rank + 1.0], device='cuda')
                dist.all_reduce(value, group=manager.get_group(members))
                assert value.item() == sum(i + 1 for i in members)
        pairs = [(0, 1), (0, 2), (1, 2), (2, 3)]
        manager.init_p2p_groups(pairs)
        if rank in [0, 1]:
            assert manager.get_p2p_group(0, 1)[0] is not manager.get_group([0, 1])
        for src, dst in pairs:
            if rank not in (src, dst):
                continue
            group, group_src, group_dst = manager.get_p2p_group(src, dst)
            assert (group_src, group_dst) == (0, 1)
            value = torch.tensor([float(src)], device='cuda')
            work = dist.isend(value, dst=dst, group=group) if rank == src else dist.irecv(value, src=src, group=group)
            work.wait()
            assert value.item() == src
        # An outside caller may leave uneven registration histories.
        dist.new_group([0, 1])
        manager.init_groups([[0, 3], [1, 2]])
        members = [0, 3] if rank in (0, 3) else [1, 2]
        value = torch.tensor([1.0], device='cuda')
        dist.all_reduce(value, group=manager.get_group(members))
        assert value.item() == 2
        dist.barrier()
        uninit_device()
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_batched_groups_and_reinitialization(tmp_path):
    launch_torchrun(4, _batched_group_worker, str(tmp_path))
