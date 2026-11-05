#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import tempfile
import torch

import pytest
import torch.distributed

from nnscaler.parallel import parallelize, ComputeConfig, merge_state_dicts, load_merged_state_dict, broadcast_weights
from nnscaler.graph.parser import FxModuleParser

from .common import CubeLinear, init_random, init_distributed
from ..launch_torchrun import launch_torchrun
from ..utils import catch_log, clear_dir_on_rank0, PYTEST_RUN_ID


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


class Net3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('buffer', torch.ones(128, 64), persistent=True)
        self.fc = torch.nn.Linear(64, 64)

    # x with shape [128, 64]
    def forward(self, x):
        return self.fc(x + self.buffer)


def _to_cube_model(module, compute_config, cube_savedir, instance_name, input_shape, init_module_params=True):
    return parallelize(
        module,
        {'x': torch.randn(input_shape)},
        'tp',
        compute_config,
        gen_savedir=cube_savedir,
        instance_name=instance_name,
        init_module_params=init_module_params
    )


def _gpu_worker():
    init_distributed()
    compute_config = ComputeConfig(1, 1, use_zero=False)
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'cube_test_ckpt_{PYTEST_RUN_ID}') as tempdir:
        net1 = _to_cube_model(Net1(), compute_config, tempdir, 'net1', (128, 64))
        cube_state_dict = net1.state_dict()
        assert not any(key.startswith('buffer') for key in cube_state_dict)
        merged_state_dict, _ = merge_state_dicts([cube_state_dict])
        assert 'buffer' not in merged_state_dict

        net2 = Net2()
        net2.load_state_dict(merged_state_dict, strict=False) # should success

        from nnscaler.runtime.module import _logger
        with catch_log(_logger) as log_stream:
            net2 = _to_cube_model(Net2(), compute_config, tempdir, 'net2', (256, 64))
            net2.load_merged_state_dict(merged_state_dict, strict=True) # should success
            assert torch.equal(list(net2._buffers.values())[0], torch.ones(256, 64))

            logs = log_stream.getvalue()
            assert not 'Non-persistent buffers cannot be initialized with' in logs

        with catch_log(_logger) as log_stream:
            net2 = _to_cube_model(Net2(), compute_config, tempdir, 'net2-2', (256, 64), init_module_params=False)
            net2.load_merged_state_dict(merged_state_dict, strict=True) # should success
            # The buffer should be loaded correctly from merged_state_dict
            # even when init_module_params=False
            assert torch.equal(list(net2._buffers.values())[0], torch.ones(256, 64))

            logs = log_stream.getvalue()
            assert not 'Non-persistent buffers cannot be initialized with' in logs

        net3 = _to_cube_model(Net3(), compute_config, tempdir, 'net3', (128, 64))
        cube_state_dict = net3.state_dict()
        assert any(key.startswith('buffer') for key in cube_state_dict)
        merged_state_dict, _ = merge_state_dicts([cube_state_dict])
        assert 'buffer' in merged_state_dict

        net3 = Net3()
        net3.load_state_dict(merged_state_dict, strict=False) # should success
        assert torch.equal(net3.buffer, torch.ones(128, 64))

        with catch_log(_logger) as log_stream:
            net3 = _to_cube_model(Net3(), compute_config, tempdir, 'net3-2', (128, 64))
            net3.load_merged_state_dict(merged_state_dict, strict=True) # should success
            assert torch.equal(list(net3._buffers.values())[0], torch.ones(128, 64))

            logs = log_stream.getvalue()
            assert not 'Non-persistent buffers cannot be initialized with' in logs

        with catch_log(_logger) as log_stream:
            net3 = _to_cube_model(Net3(), compute_config, tempdir, 'net3-2', (128, 64), init_module_params=False)
            net3.load_merged_state_dict(merged_state_dict, strict=True) # should success
            assert torch.equal(list(net3._buffers.values())[0], torch.ones(128, 64))

            logs = log_stream.getvalue()
            assert not 'Non-persistent buffers cannot be initialized with' in logs


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_checkpoint_buffer():
    """
    Please note the buffer size in Net1 and Net2 are different.
    """
    launch_torchrun(1, _gpu_worker)


def _gpu_worker_broadcast():
    init_distributed()
    compute_config = ComputeConfig(1, 2, use_zero=False)
    rank = torch.distributed.get_rank()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'cube_test_ckpt_broadcast_fail_{PYTEST_RUN_ID}') as tempdir:
        net1 = _to_cube_model(Net1(), compute_config, tempdir, 'net1', (128, 64), init_module_params=False)
        broadcast_weights(net1)

        net1(torch.randn(128, 64))

        net1 = _to_cube_model(Net1(), compute_config, tempdir, 'net1-2', (128, 64),
            init_module_params=rank < 1
        )

        assert net1.non_presistent_buffers_inited
        assert torch.equal(list(net1._buffers.values())[0], torch.ones(128, 64))

        broadcast_weights(net1)
        assert net1.non_presistent_buffers_inited
        assert torch.equal(list(net1._buffers.values())[0], torch.ones(128, 64))

        net1(torch.randn(128, 64))


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_checkpoint_buffer_broadcast():
    """
    Please note the buffer size in Net1 and Net2 are different.
    """
    launch_torchrun(2, _gpu_worker_broadcast)


def _gpu_worker_npbuffer():
    """New packages restore buffer values from the common attribute index."""
    init_distributed()
    compute_config = ComputeConfig(1, 1, use_zero=False)
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'cube_test_npbuffer_{PYTEST_RUN_ID}') as tempdir:
        net1 = _to_cube_model(Net1(), compute_config, tempdir, 'net1-npb', (128, 64))
        import sys
        gen_dir = Path(sys.modules[net1.__class__.__module__].__file__).parent
        assert not (gen_dir / FxModuleParser.NON_PERSISTENT_BUFFER_FILE).exists()
        assert (gen_dir / FxModuleParser.ATTR_CONTENT_INDEX_FILE).exists()
        assert (gen_dir / FxModuleParser.ATTR_CONTENT_FILE_0).exists()

        from nnscaler.runtime.module import _logger
        with catch_log(_logger) as log_stream:
            net1_no_init = _to_cube_model(Net1(), compute_config, tempdir, 'net1-npb',
                                         (128, 64), init_module_params=False)
            assert net1_no_init.non_presistent_buffers_inited
            buffer_val = list(net1_no_init._buffers.values())[0]
            assert torch.equal(buffer_val, torch.ones(128, 64))
            logs = log_stream.getvalue()
            assert 'loading non-persistent buffers from' in logs
            assert 'loading partitioned model from' not in logs
            assert 'Non-persistent buffers cannot be initialized with' not in logs

        with catch_log(_logger) as log_stream:
            net3 = _to_cube_model(Net3(), compute_config, tempdir, 'net3-npb',
                                 (128, 64), init_module_params=False)
            assert net3.non_presistent_buffers_inited
            assert 'loading non-persistent buffers from' not in log_stream.getvalue()
        gen_dir3 = Path(sys.modules[net3.__class__.__module__].__file__).parent
        assert not (gen_dir3 / FxModuleParser.NON_PERSISTENT_BUFFER_FILE).exists()
        assert (gen_dir3 / FxModuleParser.ATTR_CONTENT_INDEX_FILE).exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_npbuffer():
    """Test the attribute-index initialization path for non-persistent buffers."""
    launch_torchrun(1, _gpu_worker_npbuffer)
