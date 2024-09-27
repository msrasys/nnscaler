#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import pytest

import torch

from nnscaler.parallel import _load_parallel_module_class, parallelize, ComputeConfig

from ..launch_torchrun import launch_torchrun
from .common import CubeLinear, init_distributed, init_random, clear_dir_on_rank0
from ..utils import new_empty, replace_all_device_with, mock_dist

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
            'tp',
            ComputeConfig(1, 1),
            gen_savedir=tempdir,
            reuse='match',
        )
        module1 = cube_module()
        module2 = cube_module()
        module3 = cube_module(init_params=False)
        assert module1.rank == 0
        assert module2.rank == 0
        assert module3.rank == 0

        for p1, p2 in zip(module1.parameters(), module2.parameters()):
            assert torch.equal(p1, p2)

        for p1, p3 in zip(module1.parameters(), module3.parameters()):
            assert not torch.equal(p1, p3)
            assert torch.all(p3 == 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_init_params():
    launch_torchrun(1, _init_params_worker)


class MyModule2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = CubeLinear(4, 4, bias=True)

    def forward(self, x):
        return self.linear(x)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('model_class,tp', [(MyModule2, True), (MyModule, False)])
def test_empty_weights(model_class, tp):
    # MyModule2 uses CubeLinear, so tp works
    # MyModule uses torch.nn.Linear, so tp doesn't work
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            model_class,
            {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
            'tp',
            ComputeConfig(2, 4, use_zero=True, zero_ngroups=2),
            gen_savedir=tempdir,
            reuse='match',
            load_module=False,
        )
        for i in range(4):
            module_class = _load_parallel_module_class(model_class, gen_savedir=tempdir, rank=i)
            m = new_empty(module_class)
            assert m.rank == i
            for p in m.parameters():
                assert p.device == torch.device('meta')
            for r in m.reducers:
                if tp:
                    assert r.ranks == ((0, 2) if i in (0, 2) else (1, 3))
                else:
                    assert r.ranks == (0, 1, 2, 3)
                assert len(r.buckets) == 1
                assert r.zero
                assert r.zero_ngroups == 2
                for b in r.buckets:
                    assert b._contiguous_grads.device == torch.device('meta')
                    assert b._contiguous_params.device == torch.device('meta')
