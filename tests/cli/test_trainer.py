from pathlib import Path

import torch
import pytest

from nnscaler.cli.trainer import Trainer
from ..launch_torchrun import launch_torchrun


def trainer_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'
    trainer = Trainer([
        '-f', config_path,
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--ckpt_save_type', 'sharded',
        '--ckpt_save_dir', str(ckpt_savedir),
    ])
    trainer.train()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer(tmp_path):
    launch_torchrun(4, trainer_worker, tmp_path)
