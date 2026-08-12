#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path

import pytest
import torch

from nnscaler.cli.trainer import Trainer
from tests.launch_torchrun import launch_torchrun

try:
    import dion  # noqa: F401
except ImportError:
    pytest.skip('Dion Muon not available', allow_module_level=True)


def _assert_checkpoint_dtypes(checkpoint):
    model_tensors = [
        value for value in checkpoint['model'].values()
        if torch.is_tensor(value) and value.is_floating_point()
    ]
    assert model_tensors
    assert all(tensor.dtype == torch.bfloat16 for tensor in model_tensors)

    optimizer_state = checkpoint['optimizer']['state']
    assert optimizer_state
    optimizer_tensors = [(key, value)
                         for param_state in optimizer_state.values()
                         for key, value in param_state.items()
                         if key in {'momentum', 'fp32_params'}]
    assert {key
            for key, _ in optimizer_tensors} == {
                'momentum',
                'fp32_params',
            }
    assert all(tensor.dtype == torch.float32
               for _, tensor in optimizer_tensors)


def trainer_dion_muon_worker(save_dir, config_file):
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(config_file).resolve()
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    trainer = Trainer([
        '-f', config_path,
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
    ])
    trainer.run()
    torch.distributed.barrier()

    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((ckpt_savedir / 'last').glob('*.ckpt')),
            save_dir / 'merged.pt',
        )

    torch.distributed.barrier()
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '4',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', str(save_dir / 'merged.pt'),
    ])
    trainer.run()
    torch.distributed.barrier()

    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((ckpt_savedir / 'last').glob('*.ckpt')),
            save_dir / 'result.pt',
        )

    torch.distributed.barrier()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason='lack of gpu devices',
)
def test_trainer_dion_muon_checkpoint(tmp_path):
    launch_torchrun(
        2,
        trainer_dion_muon_worker,
        tmp_path,
        'trainer_args_dion_muon.yaml',
    )

    _assert_checkpoint_dtypes(
        torch.load(tmp_path / 'merged.pt', weights_only=False))
    _assert_checkpoint_dtypes(
        torch.load(tmp_path / 'result.pt', weights_only=False))
