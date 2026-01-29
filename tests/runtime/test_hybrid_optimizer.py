#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import shutil

import torch
import pytest
import torch.distributed

from nnscaler.cli.trainer import Trainer
from nnscaler.runtime.hybrid_optimizer import ScaleDelayedOptimizerMixin
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

    # train with different config
    trainer_config = [
        '-f', config_path,
        '--compute_config.plan_ngpus', '2',
        '--pas_policy', 'tp',
        '--max_train_steps', '30',
        '--checkpoint.resume_from.checkpoint', 'last',
        '--checkpoint.resume_from.with_merged', str(True),
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--compute_config.use_zero', str(1 - use_zero),
    ]
    trainer = Trainer(trainer_config)
    trainer.run()
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        r = trainer._merge_checkpoint([ckpt0_savedir / 'last' / f'{i}.ckpt' for i in range(2)])
        # should success
        assert r

    torch.distributed.barrier()

    from subprocess import check_call as _call
    from functools import partial
    call = partial(_call, shell=True)

    if torch.distributed.get_rank() == 0:
        call(f"python -m nnscaler.cli.checkpoint distribute {ckpt1_savedir}/last {ckpt1_savedir}/sharded {' '.join(trainer_config)} --compute_config.runtime_ngpus {torch.distributed.get_world_size()}")

    torch.distributed.barrier()

    trainer = Trainer([
        '-f', config_path,
        '--compute_config.plan_ngpus', '2',
        '--pas_policy', 'tp',
        '--max_train_steps', '30',
        '--checkpoint.resume_from.checkpoint', f'{ckpt1_savedir}/sharded',
        '--checkpoint.resume_from.with_merged', str(False),
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--compute_config.use_zero', str(1 - use_zero),
    ])
    trainer.run()

    if torch.distributed.get_rank() == 0:
        for i in range(2):
            x = torch.load(ckpt0_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            y = torch.load(ckpt1_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            assert_equal(x['model'], y['model'])
            assert_equal(x['optimizer'], y['optimizer'])

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('use_zero', [0, 1])
def test_hybrid_optimizer(tmp_path, use_zero):
    launch_torchrun(2, trainer_worker, tmp_path, use_zero)


def param_clss_fn_mp(param_name: str) -> tuple[int, int]:
    """
    Classify a parameter name into an optimizer index and a parameter group index.
    """
    if 'layers.1.' in param_name or 'layers.10.' in param_name:
        return 0, 0
    else:
        return 1, 0


def trainer_worker_mp(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('test_hybrid_optimizer_trainer_args_mixed_precision.yaml').resolve())
    gen_savedir = save_dir / 'gen'

    # train with a hybrid optimizer
    ckpt0_savedir = save_dir / 'ckpt0'
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '10',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt0_savedir),
    ])
    trainer.run()
    torch.distributed.barrier()

    # resume training with hybrid optimizer
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '20',
        '--checkpoint.resume_from', 'last',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt0_savedir),
    ])
    trainer.run()
    torch.distributed.barrier()

    # train with normal optimizer
    ckpt1_savedir = save_dir / 'ckpt1'
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '20',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--optimizer.args.config!',
        '--optimizer.type', 'nnscaler.MixedPrecisionAdamW',
        '--optimizer.args.lr', '0.02',
    ])
    trainer.run()
    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        for i in range(2):
            x = torch.load(ckpt0_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            y = torch.load(ckpt1_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            assert_equal(x['model'], y['model'])
            assert_equal(x['optimizer']['state'], y['optimizer']['state'])

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_hybrid_optimizer_mp(tmp_path):
    launch_torchrun(2, trainer_worker_mp, tmp_path)



class ScaleDelayedAdamW(ScaleDelayedOptimizerMixin, torch.optim.AdamW):
    pass


def trainer_worker_mp2(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('test_hybrid_optimizer_trainer_args_mixed_precision2.yaml').resolve())
    gen_savedir = save_dir / 'gen'

    # train with a hybrid optimizer
    ckpt0_savedir = save_dir / 'ckpt0'
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '10',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt0_savedir),
    ])
    trainer.run()
    torch.distributed.barrier()

    # resume training with hybrid optimizer
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '20',
        '--checkpoint.resume_from', 'last',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt0_savedir),
    ])
    trainer.run()
    torch.distributed.barrier()

    # train with normal optimizer
    ckpt1_savedir = save_dir / 'ckpt1'
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '20',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--optimizer.args.config.optimizers.1.type', 'tests.runtime.test_hybrid_optimizer.ScaleDelayedAdamW',
    ])
    trainer.run()
    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        for i in range(2):
            x = torch.load(ckpt0_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            y = torch.load(ckpt1_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            assert_equal(x['model'], y['model'])
            assert_equal(x['optimizer']['state'], y['optimizer']['state'])

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_hybrid_optimizer_mp2(tmp_path):
    """
    Demonstrate that ScaleDelayedOptimizerMixin that is applied to existing optimizers
    are equivalent to defining new optimizers that inherit from the mixin.
    """
    launch_torchrun(2, trainer_worker_mp2, tmp_path)
