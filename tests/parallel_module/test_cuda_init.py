#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from contextlib import contextmanager

import pytest
import torch

from nnscaler import ComputeConfig, parallelize
from tests.utils import mock_cube_env, replace_all_device_with


class ModelWithBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.register_buffer('constant', torch.full((4,), 2.0), persistent=False)

    def forward(self, x):
        return x * self.weight + self.constant


@contextmanager
def distributed_env(tmp_path):
    with mock_cube_env(0, 1):
        torch.distributed.init_process_group(
            'gloo', init_method=(tmp_path / 'store').as_uri(), rank=0, world_size=1,
        )
        try:
            yield
        finally:
            torch.distributed.destroy_process_group()


@replace_all_device_with('cpu')
@pytest.mark.parametrize('init_params', [False, True])
def test_parallelize_constructs_on_cuda_without_buckets(tmp_path, init_params):
    expected = torch.empty(0, device='cuda').device
    original = ModelWithBuffer().eval()
    with distributed_env(tmp_path):
        model = parallelize(
            original, {'x': torch.ones(4)}, 'dp',
            ComputeConfig(1, 1, trace_strategy='cpu'), gen_savedir=tmp_path,
            init_module_params=init_params, build_module_buckets=False,
        )
        assert not model.training
        assert all(t.device == expected for t in list(model.parameters()) + list(model.buffers()))
        assert model.non_presistent_buffers_inited
        if init_params:
            torch.testing.assert_close(next(model.parameters()).cpu(), torch.ones(4))
        torch.testing.assert_close(next(model.buffers()).cpu(), torch.full((4,), 2.0))
    assert torch.empty(0).device.type == 'cpu'


@replace_all_device_with('cpu')
def test_parallelize_restores_device_after_constructor_failure(tmp_path, monkeypatch):
    expected = torch.empty(0, device='cuda').device
    with distributed_env(tmp_path):
        generated = parallelize(
            ModelWithBuffer, {'x': torch.ones(4)}, 'dp',
            ComputeConfig(1, 1, trace_strategy='cpu'), gen_savedir=tmp_path,
        )

        def fail(*args, **kwargs):
            assert torch.empty(0).device == expected
            raise RuntimeError('initialization failed')

        monkeypatch.setattr(generated, '_post_init', fail)
        with pytest.raises(RuntimeError, match='initialization failed'):
            parallelize(
                ModelWithBuffer(), {'x': torch.ones(4)}, 'dp',
                ComputeConfig(1, 1, trace_strategy='cpu'), gen_savedir=tmp_path,
                init_module_params=False, build_module_buckets=False,
            )
    assert torch.empty(0).device.type == 'cpu'
