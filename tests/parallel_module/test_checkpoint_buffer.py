from pathlib import Path
import tempfile
import torch

import pytest

from cube.parallel import parallelize, ComputeConfig, merge_state_dicts, load_merged_state_dicts

from .common import PASRandomSPMD, PASData, CubeLinear, init_random, init_distributed, clear_dir_on_rank0
from ..launch_torchrun import launch_torchrun


class Net1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('buffer', torch.ones(128, 64), persistent=False)
        self.fc = torch.nn.Linear(64, 64)

    # x with shape [128, 64]
    def forward(self, x):
        return self.fc(x + self.buffer)


class Net2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('buffer', torch.ones(256, 64), persistent=False)
        self.fc = torch.nn.Linear(64, 64)

    # x with shape [256, 64]
    def forward(self, x):
        return self.fc(x + self.buffer)


def _to_cube_model(module, compute_config, cube_savedir, instance_name, input_shape):
    return parallelize(
        module,
        {'x': torch.randn(input_shape)},
        PASRandomSPMD,
        compute_config,
        cube_savedir=cube_savedir,
        instance_name=instance_name
    )


def _gpu_worker():
    init_distributed()
    compute_config = ComputeConfig(1, 1, use_zero=False)
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_ckpt') as tempdir:
        net1 = _to_cube_model(Net1(), compute_config, tempdir, 'net1', (128, 64))
        cube_state_dict = net1.state_dict()
        assert not any(key.startswith('buffer') for key in cube_state_dict)
        merged_state_dict, _ = merge_state_dicts([cube_state_dict])
        assert 'buffer' not in merged_state_dict

        net2 = Net2()
        net2.load_state_dict(merged_state_dict, strict=False) # should success

        net2 = _to_cube_model(Net2(), compute_config, tempdir, 'net2', (256, 64))
        net2.load_merged_state_dict(merged_state_dict, strict=False) # should success

        assert True


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_checkpoint_buffer():
    """
    Please note the buffer size in Net1 and Net2 are different.
    """
    launch_torchrun(1, _gpu_worker)
