import tempfile

import torch

from cube.parallel import parallelize, ComputeConfig

from ..launch_torchrun import launch_torchrun
from .common import CubeLinear, init_distributed, init_random, PASRandomSPMD, clear_dir_on_rank0

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x)


def _init_params_worker():
    init_distributed()
    with tempfile.TemporaryDirectory() as tempdir:
        cube_module = parallelize(
            MyModule,
            {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
            PASRandomSPMD,
            ComputeConfig(1, 1),
            dynamic_shape=True,
            cube_savedir=tempdir,
            reuse='all',
        )
        module1 = cube_module()
        module2 = cube_module()
        module3 = cube_module(init_params=False)
        assert module1.get_rank() == 0
        assert module2.get_rank() == 0
        assert module3.get_rank() == 0

        for p1, p2 in zip(module1.parameters(), module2.parameters()):
            assert torch.equal(p1, p2)

        for p1, p3 in zip(module1.parameters(), module3.parameters()):
            assert not torch.equal(p1, p3)
            assert torch.all(p3 == 0)


def test_init_params():
    if not torch.cuda.is_available():
        print('skip test_init_params due to lack of cuda devices')
        return
    launch_torchrun(1, _init_params_worker)
