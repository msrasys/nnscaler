#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler.cli.mixed_module import ModuleParallelizeConfigAdapter
from nnscaler.cli.trainer_args import DatasetConfig, ModelConfig, OptimizerConfig, TrainerArgs
from nnscaler.parallel import ComputeConfig
from tests.utils import mock_cube_env, mock_dist


class ModelWithBuffer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4))
        self.register_buffer('constant', torch.full((4,), 2.0), persistent=False)

    def forward(self, x):
        return (x * self.weight + self.constant).sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
@pytest.mark.parametrize('init_params', [False, True])
def test_generated_module_placement_and_non_persistent_buffer(tmp_path, init_params):
    """Verify CLI construction allocates on CUDA without bucket building and restores non-persistent buffers."""
    expected = torch.device('cuda', torch.cuda.current_device())
    args = TrainerArgs(
        model=ModelConfig(type=ModelWithBuffer),
        optimizer=OptimizerConfig(type=torch.optim.SGD),
        dataset=DatasetConfig(type=torch.utils.data.TensorDataset),
        compute_config=ComputeConfig(1, 1, use_end2end=True, trace_strategy='cpu'),
        gen_savedir=tmp_path, broadcast_strategy='none', pas_policy='dp',
        max_train_steps=1,
    )
    adapter = ModuleParallelizeConfigAdapter(args)
    with mock_cube_env(0, 1), mock_dist(0, 1):
        model = adapter.parallelize(
            torch.ones(4), init_params=init_params, build_buckets=False,
        )
    assert all(t.device == expected for t in list(model.parameters()) + list(model.buffers()))
    assert model.non_presistent_buffers_inited
    if init_params:
        assert torch.equal(next(model.parameters()).cpu(), torch.ones(4))
    assert torch.equal(next(model.buffers()).cpu(), torch.full((4,), 2.0))
    assert torch.empty(0).device.type == 'cpu'
