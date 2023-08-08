import tempfile

import torch
from cube.parallel import ComputeConfig, parallel_module, CubeModule

from .common import PASData, init_distributed
from ..launch_torchrun import launch_torchrun


def _decorator_worker():
    init_distributed()
    with tempfile.TemporaryDirectory() as tempdir:
        @parallel_module(
            {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
            PASData,
            ComputeConfig(1, 1),
            dynamic_shape=True,
            cube_savedir=tempdir,
        )
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 5)

            def forward(self, x):
                return self.linear(x)
        assert issubclass(MyModule, CubeModule)
        x = MyModule()
        y = MyModule()

        MyModule()(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        # parameters from different instances will have the same value.
        for p, q in zip(x.parameters(),y.parameters()):
            assert torch.equal(p, q)


def test_decorator():
    if not torch.cuda.is_available():
        print('skip test_submodules_tp_gpu1 due to lack of cuda devices')
        return
    launch_torchrun(1, _decorator_worker)
