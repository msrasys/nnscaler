#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import pytest
from copy import deepcopy
from unittest.mock import patch

import torch

import nnscaler
from nnscaler.parallel import _load_parallel_module_class, parallelize, ComputeConfig, build_optimizer
from nnscaler.runtime.adapter.reducer import SINGLETON_GROUP
from nnscaler.runtime.device import _DeviceGroup

from ..launch_torchrun import launch_torchrun
from .common import CubeLinear, init_distributed, init_random
from ..utils import new_empty, replace_all_device_with, mock_dist, clear_dir_on_rank0

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
    instance_name = f'm_{tp}'
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            model_class,
            {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
            'tp',
            ComputeConfig(2, 8, use_zero=True, zero_ngroups=2),
            gen_savedir=tempdir,
            reuse='match',
            load_module=False,
            instance_name=instance_name,
        )
        for i in range(8):
            module_class = _load_parallel_module_class(model_class, gen_savedir=tempdir, instance_name=instance_name, rank=i)
            m = new_empty(module_class)
            assert m.rank == i
            for p in m.parameters():
                assert p.device == torch.device('meta')
            for r in m.reducers:
                if tp:
                    assert r.ranks == ((0, 2, 4, 6) if i in (0, 2, 4, 6) else (1, 3, 5, 7))
                else:
                    assert r.ranks == (0, 1, 2, 3, 4, 5, 6, 7)
                assert len(r.buckets) == 1
                assert r.zero
                assert r.zero_ngroups == 2
                for b in r.buckets:
                    assert b._contiguous_grads.device == torch.device('meta')
                    assert b._contiguous_params.device == torch.device('meta')


class MyModule3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = CubeLinear(8, 8, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return torch.sum(x)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('async_op', [True, False])
@pytest.mark.parametrize('zero', [1, 3])
@pytest.mark.parametrize('zero_ngroups', [1, 2])
def test_async_reducer(async_op, zero, zero_ngroups, monkeypatch):
    original_get_group = _DeviceGroup.get_group

    def get_group(self, ranks):
        assert len(ranks) > 1
        return original_get_group(self, ranks)

    monkeypatch.setattr(_DeviceGroup, 'get_group', get_group)
    instance_name = f'm_{async_op}_{zero}_{zero_ngroups}'
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            MyModule3,
            {'x': torch.randn(8, 8)},
            'dp',
            ComputeConfig(1, 2, use_zero=zero, zero_ngroups=zero_ngroups, use_end2end=True,
                          use_async_reducer=async_op,
                          trace_strategy='cpu',
                          # 1e-6to make sure one parameter per bucket
                          reducer_bucket_cap_mb=1e-6 if async_op else 0
            ),
            gen_savedir=tempdir,
            reuse='match',
            load_module=False,
            instance_name=instance_name,
        )
        for i in range(2):
            module_class = _load_parallel_module_class(MyModule3, gen_savedir=tempdir, instance_name=instance_name, rank=i)
            m = new_empty(module_class, device='cpu')
            assert m.rank == i
            assert m.runtime_version == nnscaler.__version__
            assert len(m.reducers) == 1
            assert m.reducers[0]._async == async_op
            assert m.reducers[0].zero == (zero if zero_ngroups == 1 else 0)
            if zero_ngroups == 2:
                assert m.reducers[0]._zero_subgroup is SINGLETON_GROUP
            else:
                assert m.reducers[0]._zero_crossgroup is SINGLETON_GROUP
            if async_op:
                assert len(m.reducers[0].buckets) == 2
            else:
                assert len(m.reducers[0].buckets) == 1


def _no_single_rank_groups_worker(save_dir):
    init_distributed()
    original_new_group = torch.distributed.new_group
    original_all_reduce = torch.distributed.all_reduce

    def new_group(ranks, *args, **kwargs):
        assert len(ranks) > 1
        return original_new_group(ranks, *args, **kwargs)

    def all_reduce(tensor, *args, **kwargs):
        assert torch.distributed.get_world_size(kwargs.get('group')) > 1
        return original_all_reduce(tensor, *args, **kwargs)

    # Each case specifies the ZeRO stage and number of subgroups.
    cases = [(0, 1), (1, 1), (1, 2), (1, 4), (3, 1), (3, 2), (3, 4)]
    with patch('torch.distributed.new_group', new_group), patch('torch.distributed.all_reduce', all_reduce):
        for zero, zero_ngroups in cases:
            for async_op in (False, True):
                torch.manual_seed(0)
                reference = MyModule3().cuda()
                reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
                model = parallelize(
                    deepcopy(reference),
                    {'x': torch.ones(8, 8, device='cuda')},
                    'dp',
                    ComputeConfig(
                        1, 4, use_zero=zero, zero_ngroups=zero_ngroups,
                        use_async_reducer=async_op, use_end2end=True,
                    ),
                    gen_savedir=save_dir,
                    instance_name=f'z{zero}_g{zero_ngroups}_async{async_op}',
                )
                optimizer = build_optimizer(model, torch.optim.SGD, lr=0.01)
                rank = torch.distributed.get_rank()
                for _ in range(2):
                    samples = [torch.full((8, 8), float(rank + offset), device='cuda') for offset in (1, 2)]
                    model.train_step(samples)
                    if not any(reducer.zero3 for reducer in model.reducers):
                        assert not model._forward_prefetched_params
                        assert not model._backward_prefetched_params
                    optimizer.step()
                    optimizer.zero_grad()

                    reference_optimizer.zero_grad()
                    for source_rank in range(4):
                        for offset in (1, 2):
                            reference(torch.full((8, 8), float(source_rank + offset), device='cuda')).backward()
                    reference_optimizer.step()
                    probe = torch.ones(8, 8, device='cuda')
                    torch.testing.assert_close(model.infer_step([probe])[0], reference(probe))

                if async_op:
                    # Reset must not leave the no-communication ZeRO-3 bucket marked ready.
                    with pytest.raises(RuntimeError, match='never triggered'):
                        model.reducers[0].buckets[0].sync_grads()
                    if zero == 3 and zero_ngroups == 1:
                        # A no-op cross-group still requires all accumulation steps.
                        reducer = model.reducers[0]
                        bucket = reducer.buckets[0]
                        param = bucket.params[0]
                        reducer.prefetch_param(param)
                        param.sum().backward()
                        assert 0 < bucket._async_param_cnt < bucket._async_target_cnt
                        assert bucket._async_handle is None
                        with pytest.raises(RuntimeError, match='never triggered'):
                            bucket.sync_grads()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_training_no_single_rank_groups(tmp_path):
    launch_torchrun(4, _no_single_rank_groups_worker, tmp_path)


class MyModule4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w0 = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
        self.b0 = torch.nn.Parameter(torch.randn(8, dtype=torch.float32))

        self.w1 = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float16))
        self.b1 = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))

        self.w2 = torch.nn.Parameter(torch.randn(8, 8, dtype=torch.float32))
        self.b2 = torch.nn.Parameter(torch.randn(8, dtype=torch.float16))

    def forward(self, x: torch.Tensor):
        x = self.w0 @ x + self.b0
        x = x.half()
        x = self.w1 @ x + self.b1
        x = self.w2 @ x.float()
        x = x.half() + self.b2
        return torch.sum(x).float()


@replace_all_device_with('cpu')
@pytest.mark.parametrize('async_op', [True, False])
def test_reducer_mixed_precision(async_op):
    instance_name = f'm_{async_op}'
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            MyModule4,
            {'x': torch.randn(8, 8)},
            'tp',
            ComputeConfig(2, 4, use_end2end=True,
                          use_async_reducer=async_op,
                          # a big number to make sure all parameters in one bucket
                          reducer_bucket_cap_mb=100
            ),
            gen_savedir=tempdir,
            reuse='match',
            load_module=False,
            instance_name=instance_name,
        )
        for i in range(4):
            module_class = _load_parallel_module_class(MyModule4, gen_savedir=tempdir, instance_name=instance_name, rank=i)
            m = new_empty(module_class, device='cpu')
            assert m.rank == i
            assert m.runtime_version == nnscaler.__version__
            # (intra-group + inter-group) * (float16 + float32)
            # totally 4 reducers
            assert len(m.reducers) == 4
            assert m.reducers[0]._async == async_op
