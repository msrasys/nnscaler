import tempfile

import torch
import pytest

from cube.parallel import parallelize, ComputeConfig

from .common import PASData, init_distributed
from ..launch_torchrun import launch_torchrun

def _to_cube_model(module, pas, compute_config, cube_savedir):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        pas,
        compute_config,
        cube_savedir=cube_savedir
    )

class Module0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        return self.linear(x)


def _nested_module_worker():
    init_distributed()
    with tempfile.TemporaryDirectory() as tempdir:
        class Module1(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.module0 = _to_cube_model(Module0(), PASData, ComputeConfig(1, 1), cube_savedir=tempdir)

            def forward(self, x):
                return self.module0(x)

        class Module2(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.module1 = Module1()
            def forward(self, x):
                return self.module1(x)

        with pytest.raises(RuntimeError, match='CubeModule can not be nested.'):
            _to_cube_model(Module2(), PASData, ComputeConfig(1, 1), cube_savedir=tempdir)


def test_nested_module():
    if not torch.cuda.is_available():
        print('skip test_nested_module due to lack of cuda devices')
        return
    launch_torchrun(1, _nested_module_worker)
