#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import shutil

import torch
import pytest
import torch.distributed

from nnscaler.cli.trainer import Trainer
from tests.parallel_module.common import assert_close, assert_equal
from ..launch_torchrun import launch_torchrun


def param_clss_fn(param_name: str) -> tuple[int, int]:
    """
    Classify a parameter name into an optimizer index and a parameter group index.
    """
    if 'layers.1.' in param_name or 'layers.10.' in param_name:
        return 0, 0
    elif 'layers.2.' in param_name or 'layers.12.' in param_name:
        return 0, 1
    else:
        return 1, 0

_lr_history = []
def on_train_step_start(trainer: 'Trainer', batches) -> None:
    _lr_history.append((
        trainer.optimizer.optimizers[0].param_groups[0]['lr'],
        trainer.optimizer.optimizers[0].param_groups[1]['lr'],
        trainer.optimizer.optimizers[1].param_groups[0]['lr'],
    ))


def trainer_worker(save_dir, use_zero):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('test_hybrid_optimizer_trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'

    _lr_history.clear()

    # train with a resume
    ckpt0_savedir = save_dir / 'ckpt0'
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '10',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--compute_config.use_zero', str(use_zero),
    ])
    trainer.run()
    torch.distributed.barrier()
    assert len(_lr_history) == 10
    assert all(x == (0.02, 0.03, 0.008) for x in _lr_history[:5])
    assert all(x == (0.04, 0.06, 0.04) for x in _lr_history[5:])

    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '20',
        '--checkpoint.resume_from', 'last',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--compute_config.use_zero', str(use_zero),
    ])
    trainer.run()
    torch.distributed.barrier()

    assert len(_lr_history) == 20
    assert all(x == (0.02, 0.03, 0.008) for x in _lr_history[:5])
    assert all(x == (0.04, 0.06, 0.04) for x in _lr_history[5:])

    _lr_history.clear()
    # train in one time
    ckpt1_savedir = save_dir / 'ckpt1'
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '20',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--compute_config.use_zero', str(use_zero),
    ])
    trainer.run()
    torch.distributed.barrier()
    assert len(_lr_history) == 20
    assert all(x == (0.02, 0.03, 0.008) for x in _lr_history[:5])
    assert all(x == (0.04, 0.06, 0.04) for x in _lr_history[5:])

    if torch.distributed.get_rank() == 0:
        for i in range(2):
            x = torch.load(ckpt0_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            y = torch.load(ckpt1_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            assert_equal(x['model'], y['model'])
            assert_equal(x['optimizer'], y['optimizer'])

    trainer = Trainer([
        '-f', config_path,
        '--compute_config.plan_ngpus', '2',
        '--pas_policy', 'tp',
        '--max_train_steps', '30',
        '--checkpoint.resume_from.checkpoint', 'last',
        '--checkpoint.resume_from.with_merged', True,
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--compute_config.use_zero', str(not use_zero),
    ])
    trainer.run()
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        r = trainer._merge_checkpoint([ckpt0_savedir / 'last' / f'{i}.ckpt' for i in range(2)])
        # should success
        assert r

    torch.distributed.barrier()

    # trainer = Trainer([
    #     '-f', config_path,
    #     '--compute_config.plan_ngpus', '1',
    #     '--max_train_steps', '40',
    #     '--checkpoint.resume_from.checkpoint', 'last',
    #     '--checkpoint.resume_from.with_merged', True,
    #     '--gen_savedir', str(gen_savedir),
    #     '--checkpoint.save_dir', str(ckpt0_savedir),
    # ])
    # trainer.run()
    # torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('use_zero', [True, False])
def test_hybrid_optimizer(tmp_path, use_zero):
    launch_torchrun(2, trainer_worker, tmp_path, use_zero)
