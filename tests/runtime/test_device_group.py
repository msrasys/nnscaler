import warnings

import torch.distributed as dist

from nnscaler import init, uninit

from ..launch_torchrun import launch_torchrun


def _barrier_worker():
    init()

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

    uninit()


def test_barrier_warning():
    launch_torchrun(2, _barrier_worker)
