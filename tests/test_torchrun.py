#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
import pytest
from .launch_torchrun import launch_torchrun
from .utils import multiprocessing_semaphore_available

def worker_fn():
    rank = int(os.environ["RANK"])
    return rank


@pytest.mark.skipif(
    not multiprocessing_semaphore_available(),
    reason='multiprocessing semaphores are unavailable',
)
def test_torchrun():
    outputs = launch_torchrun(2, worker_fn)
    assert outputs == {0: 0, 1: 1}
