import tempfile

import torch
import pytest

from cube.parallel import parallelize, ComputeConfig, CubeModule

from .common import PASData, init_distributed
from ..launch_torchrun import launch_torchrun

def _to_cube_model(module, compute_config, cube_savedir, load_module):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        PASData,
        compute_config,
        dynamic_shape=True,
        cube_savedir=cube_savedir,
        load_module=load_module
    )

class Module0(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        return self.linear(x)


def _gencode_worker(tempdir):
    init_distributed()
    m = Module0()
    with pytest.raises(RuntimeError):  # config mismatch
        _to_cube_model(m, ComputeConfig(1, 1), cube_savedir=tempdir, load_module=True)


def test_codegen():
    if not torch.cuda.is_available():
        print('skip test_codegen due to lack of cuda devices')
        return
    with tempfile.TemporaryDirectory() as tempdir:
        m = Module0()
        m_new = _to_cube_model(m, ComputeConfig(2, 4), cube_savedir=tempdir, load_module=False)
        assert m_new is None
        launch_torchrun(1, _gencode_worker, tempdir)


class SliceModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:2]

def test_codegen_slice():
    if not torch.cuda.is_available():
        print('skip test_codegen_slice due to lack of cuda devices')
        return
    with tempfile.TemporaryDirectory() as tempdir:
        m_new = parallelize(
            SliceModule(),
            {'x': torch.tensor([1.0, 2.0, 3.0, 6.0])},
            PASData,
            ComputeConfig(1, 1),
            dynamic_shape=True,
            cube_savedir=tempdir,
            load_module=False
        )
        assert m_new is None
