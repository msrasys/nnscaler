from pathlib import Path

import pytest
import torch

from nnscaler.cli.trainer import Trainer
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_equal


def mixed1_worker(save_dir, config_file):
    save_dir = Path(save_dir)
    stem = Path(config_file).stem
    config_path = str(Path(__file__).with_name(config_file).resolve())
    gen_savedir = save_dir /  f'gen_{stem}'
    ckpt_savedir = save_dir / f'ckpt_{stem}'

    # ground truth: train 6 epoches in one time with zero 0
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '6',
        '--compute_config.use_zero', '0',
        '--enable_progress_bar', False,
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
    ])
    trainer.run()
    torch.distributed.barrier()

    # train 6 epoches in 6 times, each time resume from last checkpoint
    # 1
    ckpt0_savedir = save_dir / f'ckpt0_{stem}'
    gen0_savedir = save_dir / f'gen0_{stem}'  # use a different gen_savedir for resumable dataloader
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '1',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
    ])
    trainer.run()

    torch.distributed.barrier()
    # 2
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'sharded',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
    ])
    trainer.run()

    torch.distributed.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt0_savedir / 'last').glob('*.ckpt')), ckpt0_savedir / 'merged2.pt')

    torch.distributed.barrier()
    # 3
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '3',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from.checkpoint', str(ckpt0_savedir / 'merged2.pt'),
        '--checkpoint.resume_from.save_memory', False,
    ])
    trainer.run()

    torch.distributed.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt0_savedir / 'last').glob('*.ckpt')), ckpt0_savedir / 'merged3.pt')

    torch.distributed.barrier()
    # 4
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from.checkpoint', str(ckpt0_savedir / 'merged3.pt'),
        '--checkpoint.resume_from.save_memory', True,
    ])
    trainer.run()

    torch.distributed.barrier()
    # 5
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '5',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from.checkpoint', 'last',
        '--checkpoint.resume_from.save_memory', True,
    ])
    trainer.run()

    torch.distributed.barrier()
    # 6
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '6',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen0_savedir),
        '--checkpoint.save_type', 'deduped',
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from.checkpoint', 'last',
        '--checkpoint.resume_from.save_memory', False,
    ])
    trainer.run()

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        Trainer.merge_checkpoint(list((ckpt0_savedir / 'last').glob('*.ckpt')), ckpt0_savedir / 'merged.pt')
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), ckpt_savedir / 'merged.pt')

        merged1 = torch.load(ckpt0_savedir / 'merged.pt', weights_only=False)
        merged2 = torch.load(ckpt_savedir / 'merged.pt', weights_only=False)
        assert_equal(merged1['model'], merged2['model'])
        assert_equal(merged1['optimizer'], merged2['optimizer'])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('config_file', ['trainer_args_mixed1.yaml', 'trainer_args_mixed2.yaml', 'trainer_args_mixed3.yaml'])
def test_mixed1(tmp_path, config_file):
    launch_torchrun(4, mixed1_worker, tmp_path, config_file)
