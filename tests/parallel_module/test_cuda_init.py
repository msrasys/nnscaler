#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler import ComputeConfig, ParallelModule, parallelize
from nnscaler.runtime.initialization import DeferredInitialization
from tests.utils import mock_cube_env, mock_dist


class ModelWithBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.register_buffer('constant', torch.full((4,), 2.0), persistent=False)

    def forward(self, x):
        return x * self.weight + self.constant


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
@pytest.mark.parametrize('init_params', [False, True])
def test_parallelize_constructs_on_cuda_without_buckets(tmp_path, init_params, monkeypatch):
    """Verify source/target devices for instance, direct, and factory construction."""
    expected = torch.device('cuda', torch.cuda.current_device())
    original = ModelWithBuffer().eval()
    construction_devices = []
    materialize = DeferredInitialization.materialize

    def materialize_on_cuda(capture, tensor):
        value = materialize(capture, tensor)
        assert value.device == expected
        return value

    monkeypatch.setattr(DeferredInitialization, 'materialize', materialize_on_cuda)

    def factory():
        source = ModelWithBuffer().eval()
        construction_devices.append(source.weight.device.type)
        return source

    with mock_cube_env(0, 1), mock_dist(0, 1):
        with torch.device('cuda'):
            model = parallelize(
                original, {'x': torch.ones(4)}, 'dp',
                ComputeConfig(1, 1, trace_strategy='cpu', param_init_strategy='shard'), gen_savedir=tmp_path,
                instance_name='shard_instance',
                init_module_params=init_params, build_module_buckets=False,
            )
            assert not model.training
            generated = parallelize(
                ModelWithBuffer, {'x': torch.ones(4)}, 'dp',
                ComputeConfig(1, 1, trace_strategy='cpu', param_init_strategy='shard'), gen_savedir=tmp_path,
                instance_name='shard_factory', module_fn=factory,
            )
            assert torch.empty(0).device == expected
        assert torch.empty(0).device.type == 'cpu'
        direct_model = type(model)(init_params=init_params, build_buckets=False)
        factory_model = generated(init_params=init_params, build_buckets=False)
        assert torch.empty(0).device.type == 'cpu'
        assert construction_devices == ['cpu', 'meta']
        for instance in (model, direct_model, factory_model):
            assert all(t.device == expected for t in list(instance.parameters()) + list(instance.buffers()))
            assert instance.non_presistent_buffers_inited
            if init_params:
                assert torch.equal(next(instance.parameters()).cpu(), torch.ones(4))
            assert torch.equal(next(instance.buffers()).cpu(), torch.full((4,), 2.0))
    assert torch.empty(0).device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
def test_legacy_file_constructor_allocates_on_cuda(tmp_path, monkeypatch):
    """Reuse pre-CUDA FILE code and verify parameters/buffers are on CUDA before initialization."""
    expected = torch.device('cuda', torch.cuda.current_device())
    register_parameter = ParallelModule.register_parameter
    register_buffer = ParallelModule.register_buffer

    def check_parameter(module, name, value):
        assert value.device == expected
        return register_parameter(module, name, value)

    def check_buffer(module, name, value, persistent=True):
        assert value.device == expected
        return register_buffer(module, name, value, persistent=persistent)

    monkeypatch.setattr(ParallelModule, 'register_parameter', check_parameter)
    monkeypatch.setattr(ParallelModule, 'register_buffer', check_buffer)
    original = ModelWithBuffer().eval()
    config = ComputeConfig(1, 1, trace_strategy='cpu', param_init_strategy='file')
    with mock_cube_env(0, 1), mock_dist(0, 1):
        parallelize(
            original, {'x': torch.ones(4)}, 'dp', config,
            gen_savedir=tmp_path, instance_name='legacy_file', load_module=False,
        )
        sources = {}
        for path in tmp_path.rglob('gencode*.py'):
            code = path.read_text()
            legacy_code = code.replace(", device='cuda'", "")
            assert legacy_code != code
            path.write_text(legacy_code)
            sources[path] = legacy_code
        assert sources
        model = parallelize(
            original, {'x': torch.ones(4)}, 'dp', config,
            gen_savedir=tmp_path, instance_name='legacy_file', build_module_buckets=False,
        )
        assert model.non_presistent_buffers_inited
        assert torch.equal(next(model.parameters()).cpu(), torch.ones(4))
        assert torch.equal(next(model.buffers()).cpu(), torch.full((4,), 2.0))
        assert all(path.read_text() == code for path, code in sources.items())
    assert torch.empty(0).device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
def test_shard_initializer_restores_device_after_failure(tmp_path, monkeypatch):
    """Verify a lazy shard-hook failure propagates and restores the default CPU device."""
    expected = torch.device('cuda', torch.cuda.current_device())

    def fail(attrs):
        for attr, meta in attrs.items():
            assert torch.empty(0).device == expected
            yield attr, torch.zeros(meta.sub_shape, dtype=meta.dtype)
            assert torch.empty(0).device == expected
            raise RuntimeError('initialization failed')

    monkeypatch.setattr(ModelWithBuffer, '__shard__init__', staticmethod(fail), raising=False)
    with mock_cube_env(0, 1), mock_dist(0, 1):
        generated = parallelize(
            ModelWithBuffer, {'x': torch.ones(4)}, 'dp',
            ComputeConfig(1, 1, trace_strategy='cpu', param_init_strategy='shard'), gen_savedir=tmp_path,
        )
        with pytest.raises(RuntimeError, match='initialization failed'):
            generated(init_params=False, build_buckets=False)
    assert torch.empty(0).device.type == 'cpu'
