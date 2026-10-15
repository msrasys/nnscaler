#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from types import SimpleNamespace

import pytest
import torch

from nnscaler.cli.mixed_module import ModuleParallelizeConfigAdapter
from nnscaler.parallel import ComputeConfig
from tests.utils import mock_cube_env, replace_all_device_with


class ModelWithBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.register_buffer('constant', torch.full((4,), 2.0), persistent=False)

    def forward(self, x):
        return x * self.weight + self.constant


@replace_all_device_with('cpu')
@pytest.mark.parametrize('init_params', [False, True])
def test_generated_module_placement_and_non_persistent_buffer(tmp_path, init_params):
    expected = torch.empty(0, device='cuda').device
    args = SimpleNamespace(
        model_type=ModelWithBuffer, create_model=ModelWithBuffer,
        model=SimpleNamespace(args={}), compute_config=ComputeConfig(1, 1, trace_strategy='cpu'),
        gen_savedir=tmp_path, gen_reuse='match', instance_name=None,
        broadcast_strategy='none', gen_max_workers=1, pas_policy='dp',
        precision={'param': 'fp32', 'buffer': 'fp32', 'grad': 'fp32'},
        tracing_from_weights=None, update_freq=1,
        micro_batch_size=1, global_batch_size=1,
    )
    adapter = ModuleParallelizeConfigAdapter(args)
    with mock_cube_env(0, 1):
        torch.distributed.init_process_group(
            'gloo', init_method=(tmp_path / 'store').as_uri(), rank=0, world_size=1,
        )
        try:
            model = adapter.parallelize(
                torch.ones(4), init_params=init_params, build_buckets=False,
            )
        finally:
            torch.distributed.destroy_process_group()
    assert all(t.device == expected for t in list(model.parameters()) + list(model.buffers()))
    assert model.non_presistent_buffers_inited
    if init_params:
        torch.testing.assert_close(next(model.parameters()).cpu(), torch.ones(4))
    torch.testing.assert_close(next(model.buffers()).cpu(), torch.full((4,), 2.0))
    assert torch.empty(0).device.type == 'cpu'
