#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler import ComputeConfig, parallelize
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
def test_parallelize_constructs_on_cuda_without_buckets(tmp_path, init_params):
    """Verify parallelize allocates on CUDA without buckets and restores non-persistent buffers and eval mode."""
    expected = torch.device('cuda', torch.cuda.current_device())
    original = ModelWithBuffer().eval()
    with mock_cube_env(0, 1), mock_dist(0, 1):
        model = parallelize(
            original, {'x': torch.ones(4)}, 'dp',
            ComputeConfig(1, 1, trace_strategy='cpu'), gen_savedir=tmp_path,
            init_module_params=init_params, build_module_buckets=False,
        )
        assert not model.training
        assert all(t.device == expected for t in list(model.parameters()) + list(model.buffers()))
        assert model.non_presistent_buffers_inited
        if init_params:
            assert torch.equal(next(model.parameters()).cpu(), torch.ones(4))
        assert torch.equal(next(model.buffers()).cpu(), torch.full((4,), 2.0))
    assert torch.empty(0).device.type == 'cpu'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
def test_parallelize_restores_device_after_constructor_failure(tmp_path, monkeypatch):
    """Verify a CUDA constructor failure propagates and restores the default CPU device."""
    expected = torch.device('cuda', torch.cuda.current_device())
    with mock_cube_env(0, 1), mock_dist(0, 1):
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
