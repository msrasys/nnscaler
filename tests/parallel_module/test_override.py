from pathlib import Path
from time import sleep
import sys
import tempfile
import pytest
import torch
import shutil

from cube.parallel import ReuseType, parallelize, ComputeConfig

from .common import PASData, init_distributed
from ..launch_torchrun import launch_torchrun


def _to_cube_model(module, compute_config, cube_savedir, reuse, instance_name, load_module=True):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
        PASData,
        compute_config,
        reuse=reuse,
        cube_savedir=cube_savedir,
        instance_name=instance_name,
        load_module=load_module,
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
        cmodule1 = _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, ReuseType.ALL, None)
        # False  | match | do nothing
        cmodule2 = _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, ReuseType.ALL, None)
        # true
        for (n1, v1), (n2, v2) in zip(cmodule1.named_parameters(), cmodule2.named_parameters()):
            assert n1 == n2
            assert torch.equal(v1, v2)

        cmodule3 = _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, ReuseType.ALL, 'test')
        cmodule4 = _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, 'all', 'test')

        for (n1, v1), (n2, v2) in zip(cmodule3.named_parameters(), cmodule4.named_parameters()):
            assert n1 == n2
            assert torch.equal(v1, v2)

        cmodule2_p = dict(cmodule2.named_parameters())
        cmodule3_p = dict(cmodule3.named_parameters())
        keys = cmodule3_p.keys()
        assert any(not torch.equal(cmodule2_p[key], cmodule3_p[key]) for key in keys)

        # True   | imported | raise error
        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, ReuseType.NONE, None)

        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, ReuseType.NONE, 'test')

        # False  | unmatch | raise error
        with pytest.raises(RuntimeError):
            _to_cube_model(MyModule(), ComputeConfig(2, 2),tempdir, ReuseType.NONE, 'test')

        # True   | empty | generate
        cmodule1 = _to_cube_model(MyModule(), ComputeConfig(1, 1),tempdir, ReuseType.NONE, 'test2')
        module_path = Path(sys.modules[cmodule1.__module__].__file__).parent
        test3_module_path = module_path.with_name('test3')
        test3_module_path.mkdir(exist_ok=True, parents=True)
        test4_module_path = module_path.with_name('test4')
        test4_module_path.mkdir(exist_ok=True, parents=True)
        test5_module_path = module_path.with_name('test5')
        test5_module_path.mkdir(exist_ok=True, parents=True)
        for f in module_path.glob('*'):
            if f.is_file():
                shutil.copy(f, test3_module_path / f.name)
                shutil.copy(f, test4_module_path / f.name)
                shutil.copy(f, test5_module_path / f.name)
        # fake two gpus
        shutil.copy(test4_module_path / 'gencode0.py', test4_module_path / 'gencode1.py')
        shutil.copy(test5_module_path / 'gencode0.py', test5_module_path / 'gencode1.py')

        # True   | match | generate
        cmodule2 = _to_cube_model(MyModule(), ComputeConfig(1, 1), tempdir, ReuseType.NONE, 'test3')
        cmodule2_p = dict(cmodule2.named_parameters())
        cmodule1_p = dict(cmodule1.named_parameters())
        keys = cmodule2_p.keys()
        assert any(not torch.equal(cmodule2_p[key], cmodule1_p[key]) for key in keys)

        # True   | unmatch | generate
        assert (test4_module_path / 'gencode1.py').exists()
        cmodule3 = _to_cube_model(MyModule(), ComputeConfig(1, 1), tempdir, 'none', 'test4')
        assert not (test4_module_path / 'gencode1.py').exists()

        # Graph | matched | generate
        assert (test5_module_path / 'gencode1.py').exists()
        code_stat = (test5_module_path / 'gencode0.py').stat()
        graph_stat = (test5_module_path / 'graph.ckp').stat()
        args_stat = (test5_module_path / 'forward_args.pkl').stat()
        cmodule4 = _to_cube_model(MyModule(), ComputeConfig(1, 1), tempdir, 'graph', 'test5', False)
        assert not (test5_module_path / 'gencode1.py').exists()
        assert (test5_module_path / 'gencode0.py').stat().st_mtime_ns != code_stat.st_mtime_ns
        assert (test5_module_path / 'graph.ckp').stat().st_mtime_ns == graph_stat.st_mtime_ns
        assert (test5_module_path / 'forward_args.pkl').stat().st_mtime_ns == args_stat.st_mtime_ns

        code_stat = (test5_module_path / 'gencode0.py').stat()
        graph_stat = (test5_module_path / 'graph.ckp').stat()
        (test5_module_path / 'forward_args.pkl').unlink()  # remove foward_args.pkl will force to generate new code
        cmodule5 = _to_cube_model(MyModule(), ComputeConfig(1, 1), tempdir, 'graph', 'test5', False)
        assert (test5_module_path / 'gencode0.py').stat().st_mtime_ns != code_stat.st_mtime_ns
        assert (test5_module_path / 'graph.ckp').stat().st_mtime_ns != graph_stat.st_mtime_ns
        assert (test5_module_path / 'forward_args.pkl').exists()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_override():
    launch_torchrun(1, _worker)

