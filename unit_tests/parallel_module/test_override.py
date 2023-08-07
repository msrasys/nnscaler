from pathlib import Path
import sys
import tempfile
import pytest
import torch
import shutil

from cube.cube import as_cube, ComputeConfig

from .common import PASData, init_distributed
from ..launch_torchrun import launch_torchrun


def _to_cube_model(module, compute_config, cube_savedir, override, instance_name):
    return as_cube(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        PASData,
        compute_config,
        dynamic_shape=True,
        override=override,
        cube_savedir=cube_savedir,
        instance_name=instance_name,
    )


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 5)

    def forward(self, x):
        return self.linear(x)


def _worker():
    init_distributed()

    with tempfile.TemporaryDirectory() as tempdir:
        # False   | empty | generate
        cmodule1 = _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, False, None)
        # False  | match | do nothing
        cmodule2 = _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, False, None)
        # true
        for (n1, v1), (n2, v2) in zip(cmodule1.named_parameters(), cmodule2.named_parameters()):
            assert n1 == n2
            assert torch.equal(v1, v2)

        cmodule3 = _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, False, 'test')
        cmodule4 = _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, False, 'test')

        for (n1, v1), (n2, v2) in zip(cmodule3.named_parameters(), cmodule4.named_parameters()):
            assert n1 == n2
            assert torch.equal(v1, v2)

        cmodule2_p = dict(cmodule2.named_parameters())
        cmodule3_p = dict(cmodule3.named_parameters())
        keys = cmodule3_p.keys()
        assert any(not torch.equal(cmodule2_p[key], cmodule3_p[key]) for key in keys)

        # True   | imported | raise error
        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, True, None)

        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, True, 'test')

        # False  | unmatch | raise error
        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule(), ComputeConfig(2, 2),tempdir, False, 'test')

        # True   | empty | generate
        cmodule1 = _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, True, 'test2')
        module_path = Path(sys.modules[cmodule1.__module__].__file__).parent
        test3_module_path = module_path.with_name('test3')
        test3_module_path.mkdir(exist_ok=True, parents=True)
        test4_module_path = module_path.with_name('test4')
        test4_module_path.mkdir(exist_ok=True, parents=True)
        for f in module_path.glob('*'):
            if f.is_file():
                shutil.copy(f, test3_module_path / f.name)
                shutil.copy(f, test4_module_path / f.name)
        # fake two gpus
        shutil.copy(test4_module_path / 'gencode0.py', test4_module_path / 'gencode1.py')

        # True   | match | generate
        cmodule2 = _to_cube_model(MyModule(), ComputeConfig(1, 1), tempdir, True, 'test3')
        cmodule2_p = dict(cmodule2.named_parameters())
        cmodule1_p = dict(cmodule1.named_parameters())
        keys = cmodule2_p.keys()
        assert any(not torch.equal(cmodule2_p[key], cmodule1_p[key]) for key in keys)

        # True   | unmatch | generate
        assert (test4_module_path / 'gencode1.py').exists()
        cmodule3 = _to_cube_model(MyModule(), ComputeConfig(1, 1), tempdir, True, 'test4')
        assert not (test4_module_path / 'gencode1.py').exists()

def test_override():
    if not torch.cuda.is_available():
        print('skip test_submodules_tp_gpu1 due to lack of cuda devices')
        return
    launch_torchrun(1, _worker)

