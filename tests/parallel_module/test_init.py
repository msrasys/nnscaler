#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import sys
from unittest.mock import patch
import pytest

import torch

import nnscaler
from nnscaler.parallel import _load_parallel_module_class, parallelize, ComputeConfig, ParamInitStrategy

from ..launch_torchrun import launch_torchrun
from .common import CubeLinear, init_distributed, init_random
from ..utils import new_empty, replace_all_device_with, mock_dist, mock_cube_env, mock_reducer_env, clear_dir_on_rank0

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x)


class LocalInitModule(MyModule):
    def __init__(self):
        super().__init__()
        self.register_buffer('scalar', torch.tensor(0.5), persistent=False)
        self.register_buffer('transposed', torch.arange(20.0).reshape(4, 5).t())

    def forward(self, x):
        return self.linear(x) + self.scalar + self.transposed.sum()

    @staticmethod
    def __partial__init__(attr_meta_map):
        return {
            name: torch.full(meta.sub_shape, 0.25, dtype=meta.dtype)
            for name, meta in attr_meta_map.items()
        }


@pytest.mark.parametrize('strategy', [
    ParamInitStrategy.FILE, ParamInitStrategy.RECREATE,
    ParamInitStrategy.CAPTURE, ParamInitStrategy.CUSTOM,
])
def test_param_init_config(strategy):
    config = ComputeConfig(1, 1, param_init_strategy=strategy)
    assert config.param_init_strategy == strategy
    assert nnscaler.ParamInitStrategy is ParamInitStrategy


def test_param_init_graph_config():
    configs = [
        ComputeConfig(1, 1, param_init_strategy=strategy)
        for strategy in ('recreate', 'capture', 'custom')
    ]
    assert all(config.graph_config == configs[0].graph_config for config in configs)
    assert ComputeConfig(1, 1).graph_config != configs[0].graph_config
    assert ComputeConfig(1, 1).graph_config != ComputeConfig(1, 1, param_init_seed=77).graph_config
    for config in configs:
        assert config.graph_config == ComputeConfig(
            1, 1, param_init_strategy=config.param_init_strategy, param_init_seed=77,
        ).graph_config


@pytest.mark.parametrize('strategy', [False, True, 0, 1, None, 'invalid', 'fullmodel', [], {}])
def test_param_init_config_invalid_strategy(strategy):
    with pytest.raises(ValueError, match='param_init_strategy'):
        ComputeConfig(1, 1, param_init_strategy=strategy)


@pytest.mark.parametrize('seed', [True, -1, 2 ** 32, 1.5, '1234'])
def test_param_init_config_invalid_seed(seed):
    with pytest.raises(ValueError, match='param_init_seed'):
        ComputeConfig(1, 1, param_init_seed=seed)


@pytest.mark.parametrize('source_instance', [False, True])
def test_custom_init_requires_callback(tmp_path, source_instance):
    source = MyModule() if source_instance else MyModule
    with pytest.raises(ValueError, match='__partial__init__'):
        parallelize(
            source, {'x': torch.ones(2, 4)}, 'dp',
            ComputeConfig(1, 1, param_init_strategy='custom'),
            gen_savedir=tmp_path, load_module=False,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('callback', [
    lambda self, attr_meta_map: {},
    staticmethod(lambda attr_meta_map, extra: {}),
    staticmethod(None),
])
def test_custom_init_invalid_callback(tmp_path, callback):
    with patch.object(LocalInitModule, '__partial__init__', callback):
        with pytest.raises(ValueError, match='__partial__init__'):
            parallelize(
                LocalInitModule, {'x': torch.ones(2, 4)}, 'dp',
                ComputeConfig(1, 1, param_init_strategy='custom'),
                gen_savedir=tmp_path, load_module=False,
            )


@patch('torch.cuda.is_available', lambda: False)
@replace_all_device_with('cpu', force=True)
def test_custom_init_classmethod(tmp_path):
    calls = []

    def initialize(cls, attr_meta_map):
        calls.append(cls)
        return {
            name: torch.zeros(meta.sub_shape, dtype=meta.dtype)
            for name, meta in attr_meta_map.items()
        }

    with patch.object(LocalInitModule, '__partial__init__', classmethod(initialize)), \
            mock_reducer_env(0, 1), patch('torch.distributed.barrier'), \
            patch('torch.distributed.broadcast_object_list'):
        generated = parallelize(
            LocalInitModule, {'x': torch.ones(2, 4)}, 'dp',
            ComputeConfig(1, 1, param_init_strategy='custom'),
            gen_savedir=tmp_path, instance_name='classmethod',
        )
        generated(build_buckets=False)
    assert calls == [LocalInitModule]


@patch('torch.cuda.is_available', lambda: False)
@replace_all_device_with('cpu', force=True)
@pytest.mark.parametrize('strategy', ['recreate', 'capture', 'custom'])
@pytest.mark.parametrize('source_instance', [False, True])
def test_param_init_callback_attachment(tmp_path, strategy, source_instance):
    source = LocalInitModule() if source_instance else LocalInitModule
    config = ComputeConfig(1, 1, param_init_strategy=strategy)
    with mock_reducer_env(0, 1), patch('torch.distributed.barrier'), \
            patch('torch.distributed.broadcast_object_list'):
        model_or_class = parallelize(
            source, {'x': torch.ones(2, 4)}, 'dp', config,
            gen_savedir=tmp_path, build_module_buckets=False,
            instance_name=f'{strategy}_{source_instance}',
        )
        generated = type(model_or_class) if source_instance else model_or_class
        callback_name = '_partial_init_fn' if strategy == 'custom' else '_init_module_fn'
        assert callable(getattr(generated, callback_name))
        with patch.object(LocalInitModule, '__init__', side_effect=AssertionError('full constructor')):
            if strategy == 'custom':
                model = model_or_class if source_instance else generated(init_params=True, init_module=None)
                for name, meta in model.fullmap.items():
                    expected = torch.full(meta.sub_shape, 0.25, dtype=meta.dtype)
                    assert torch.equal(getattr(model, name), expected)
        if source_instance and strategy != 'custom':
            tensors = dict(list(source.named_parameters()) + list(source.named_buffers()))
            for name, meta in model_or_class.fullmap.items():
                assert torch.equal(getattr(model_or_class, name), tensors[meta.orig_name][meta.slicers])
        assert not list(tmp_path.rglob('fullmodel.pt*'))
        assert not list(tmp_path.rglob('npbuffer.pt'))


@patch('torch.cuda.is_available', lambda: False)
def test_param_init_strategy_reload(tmp_path):
    graph_mtime = None
    for strategy in ('recreate', 'custom', 'capture', 'recreate'):
        config = ComputeConfig(1, 1, param_init_strategy=strategy, trace_strategy='cpu')
        with mock_cube_env(0, 1), mock_dist(0, 1), patch('torch.distributed.barrier'), \
                patch('torch.distributed.broadcast_object_list'):
            generated = parallelize(
                LocalInitModule, {'x': torch.ones(2, 4)}, 'dp', config,
                gen_savedir=tmp_path, instance_name='strategy_reload', reuse='moo',
            )
            try:
                assert (generated._init_module_fn is None) == (strategy == 'custom')
                assert (generated._partial_init_fn is None) == (strategy != 'custom')
                module = generated(build_buckets=False)
                assert module.compute_config.param_init_strategy == strategy
                current_mtime = (module.module_dir / 'graph.ckp').stat().st_mtime_ns
                if graph_mtime is not None:
                    assert current_mtime == graph_mtime
                graph_mtime = current_mtime
                assert not list(module.module_dir.glob('fullmodel.pt*'))
                assert not (module.module_dir / 'npbuffer.pt').exists()
                if strategy == 'custom':
                    for name, meta in module.fullmap.items():
                        assert torch.equal(getattr(module, name), torch.full(meta.sub_shape, 0.25, dtype=meta.dtype))
            finally:
                # MOO cannot replace loaded code; emulate a fresh process for the next strategy.
                sys.modules.pop(generated.__module__)


def _local_init_worker(tmp_path):
    nnscaler.init()
    for strategy in ('recreate', 'capture'):
        torch.manual_seed(1234)
        original = LocalInitModule().eval()
        expected = {
            name: tensor.detach().to(torch.bfloat16).clone()
            for name, tensor in list(original.named_parameters()) + list(original.named_buffers())
        }
        config = ComputeConfig(1, 2, param_init_strategy=strategy, param_init_seed=4321)
        sample = {'x': torch.ones(2, 4, dtype=torch.bfloat16)}
        model = parallelize(
            original, sample, 'dp', config,
            gen_savedir=tmp_path, instance_name=f'instance{strategy}',
            module_dtype=torch.bfloat16,
        )
        assert not model.training
        for attr, meta in model.fullmap.items():
            assert torch.equal(getattr(model, attr).cpu(), expected[meta.orig_name][meta.slicers])
        assert not list(model.module_dir.glob('fullmodel.pt*'))

    # Directly imported code needs an explicit source, not an implicit fallback to disk.
    parallelize(
        LocalInitModule, {'x': torch.ones(2, 4)}, 'dp',
        ComputeConfig(1, 2, param_init_strategy='recreate'),
        gen_savedir=tmp_path, instance_name='direct', load_module=False,
    )
    generated = _load_parallel_module_class(LocalInitModule, gen_savedir=tmp_path, instance_name='direct')
    with pytest.raises(RuntimeError, match='Independent initialization requires parallelize'):
        generated()
    with pytest.raises(RuntimeError, match='Independent initialization requires parallelize'):
        generated(init_params=False)
    torch.manual_seed(1234)
    source = LocalInitModule()
    model = generated(init_module=source)
    assert model.non_presistent_buffers_inited
    assert all(tensor.item() == 0.5 for tensor in model.get_non_persistent_buffers().values())
    resumed = generated(init_params=False, init_module=source)
    assert resumed.non_presistent_buffers_inited
    missing_source = LocalInitModule()
    del missing_source.transposed
    with pytest.raises(AttributeError, match='transposed'):
        generated(init_module=missing_source)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two GPUs')
def test_local_init_instance(tmp_path):
    launch_torchrun(2, _local_init_worker, tmp_path)


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
def test_async_reducer(async_op):
    instance_name = f'm_{async_op}'
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            MyModule3,
            {'x': torch.randn(8, 8)},
            'dp',
            ComputeConfig(1, 2, use_zero=True, zero_ngroups=2, use_end2end=True,
                          use_async_reducer=async_op,
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
            if async_op:
                assert len(m.reducers[0].buckets) == 2
            else:
                assert len(m.reducers[0].buckets) == 1


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
