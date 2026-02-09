from pathlib import Path
import shutil

import torch

import pytest

from nnscaler.cli.trainer import Trainer
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_equal


try:
    from torch.optim import Muon
except ImportError:
    pytest.skip("Muon not available", allow_module_level=True)



def trainer_muon_worker(save_dir, config_file):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name(config_file).resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    # train first epoch
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '1',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
    ])
    trainer.run()
    torch.distributed.barrier()

    # create merged checkpoint
    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), ckpt_savedir / 'merged.pt')

    torch.distributed.barrier()

    # train 2nd epoch, resume from merged checkpoint
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', str(ckpt_savedir / 'merged.pt'),
    ])
    trainer.run()

    torch.distributed.barrier()

    # train 3rd epoch, resume from deduped checkpoint
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '3',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.save_type', 'sharded',
    ])
    trainer.run()

    torch.distributed.barrier()

    # train 4th epoch, resume from sharded checkpoint
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '4',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
    ])
    trainer.run()

    torch.distributed.barrier()

    ckpt1_savedir = save_dir / 'ckpt1'
    # train 4 epoch without resuming
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '4',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt1_savedir),
    ])
    trainer.run()

    torch.distributed.barrier()

    if trainer.rank == 0:
        for i in range(2):
            x = torch.load(ckpt_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            y = torch.load(ckpt1_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            for key in ['model', 'optimizer']:
                assert_equal(x[key], y[key])

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('config_file', ['trainer_args_muon.yaml', 'trainer_args_muon_hybrid.yaml'])
def test_trainer_muon_resume_correctness(tmp_path, config_file):
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file)


def param_clss_fn(param_name: str) -> tuple[int, int]:
    """
    Classify a parameter name into an optimizer index and a parameter group index.
    """
    if 'layers.1.' in param_name or 'layers.10.' in param_name:
        return 0, 0
    else:
        return 1, 0
