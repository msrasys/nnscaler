import inspect
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
    """
    Test it can support modules without parameters
    """
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


class ArgsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y, *args):
        return self.linear(x) + y

def test_codegen_args():
    """
    Verify that unused args are supported by parallalize
    """
    if not torch.cuda.is_available():
        print('skip test_codegen_args due to lack of cuda devices')
        return

    with tempfile.TemporaryDirectory() as tempdir:
        # *args is not supported.
        with pytest.raises(RuntimeError):
            parallelize(
                ArgsModule(),
                {
                    'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    'y': 1.0,
                },
                PASData,
                ComputeConfig(1, 1),
                dynamic_shape=True,
                cube_savedir=tempdir,
                load_module=True
            )


class UnusedArgsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y, z=None, m=1, n=2, **kwargs):
        return self.linear(x) + m


def _gencode_unused_args_worker(tempdir):
    init_distributed()
    m_new = parallelize(
        UnusedArgsModule(),
        {
            'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            'y': torch.tensor([1, 2, 3]),
            'z': None,
            'm': 0,
            'n': None,
         },
        PASData,
        ComputeConfig(1, 1),
        dynamic_shape=True,
        cube_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl)
    assert len(args.parameters) == 6
    assert args.parameters['x'].default is inspect.Parameter.empty
    assert args.parameters['y'].default is None
    assert args.parameters['z'].default is None
    assert args.parameters['m'].default == 1
    assert args.parameters['n'].default is None
    assert args.parameters['kwargs'].default is inspect.Parameter.empty

    assert torch.equal(
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])),
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), m=1)
    )

    with pytest.raises(ValueError):
        # y must be None
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 1)

def test_codegen_unused_args():
    """
    Verify that unused args are supported by parallalize
    """
    if not torch.cuda.is_available():
        print('skip test_unused_input due to lack of cuda devices')
        return

    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_unused_args_worker, tempdir)


class UnusedArgs2Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x, y, m):
        return self.linear(x) + m


def _gencode_unused_args_worker2(tempdir):
    init_distributed()
    m_new = parallelize(
        UnusedArgs2Module(),
        {
            'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            'y': torch.tensor([1, 2, 3]),
            'm': 0
         },
        PASData,
        ComputeConfig(1, 1),
        dynamic_shape=True,
        cube_savedir=tempdir,
        load_module=True
    )
    assert m_new is not None
    args = inspect.signature(m_new._forward_impl)
    assert len(args.parameters) == 3
    assert args.parameters['x'].default is inspect.Parameter.empty
    assert args.parameters['y'].default is None
    assert args.parameters['m'].default is None

    m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), m=1)
    with pytest.raises(TypeError, match='.*must be Tensor, not NoneType.*'):
        # raise by torch.add, as m is None
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    with pytest.raises(ValueError):
        # y must be None
        m_new(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), 1)


def test_codegen_unused_args2():
    """
    Verify that unused args are supported by parallalize
    """
    if not torch.cuda.is_available():
        print('skip test_codegen_unused_args2 due to lack of cuda devices')
        return

    with tempfile.TemporaryDirectory() as tempdir:
        launch_torchrun(1, _gencode_unused_args_worker2, tempdir)
