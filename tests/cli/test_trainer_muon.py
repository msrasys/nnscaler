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



def trainer_muon_worker(save_dir, config_file, name, additional_options=None):
    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name(config_file).resolve()
    work_dir = save_dir / str(name)
    gen_savedir = work_dir / 'gen'
    ckpt_savedir = work_dir / 'ckpt'
    additional_options = additional_options or []

    # train first epoch
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '1',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        *additional_options,
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
        *additional_options,
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
        *additional_options,
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
        *additional_options,
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
        *additional_options,
    ])
    trainer.run()

    torch.distributed.barrier()

    if trainer.rank == 0:
        for i in range(2):
            x = torch.load(ckpt_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            y = torch.load(ckpt1_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            for key in ['model', 'optimizer']:
                assert_equal(x[key], y[key])

    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), work_dir / 'result.pt')

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_trainer_muon_resume_correctness1(tmp_path):
    config_file = 'trainer_args_muon.yaml'
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, 'False')
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, 'True', [
        '--compute_config.zero_param_level_sharding', True,
        '--compute_config.use_zero', 1,
        '--optimizer.type', 'nnscaler.runtime.muon_optimizer.Muon',
    ])

    zero0_ckpt = torch.load(tmp_path / 'False' / 'result.pt', weights_only=False)
    zero1_ckpt = torch.load(tmp_path / 'True' / 'result.pt', weights_only=False)

    assert_equal(zero0_ckpt['model'], zero1_ckpt['model'])
    assert_equal(zero0_ckpt['optimizer']['state'], zero1_ckpt['optimizer']['state'])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_trainer_muon_resume_correctness2(tmp_path):
    config_file = 'trainer_args_muon_hybrid.yaml'
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, '1')
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, '2', [
        '--compute_config.zero_param_level_sharding', True,
        '--compute_config.use_zero', 1,
        '--optimizer.args.config.optimizers.1.type', 'nnscaler.runtime.muon_optimizer.Muon',
    ])
    launch_torchrun(2, trainer_muon_worker, tmp_path, config_file, '3', [
        '--compute_config.use_zero', 1,
        '--optimizer.args.config.optimizers.1.type', 'nnscaler.runtime.muon_optimizer.Muon',
        '--optimizer.param_clss_fn', 'tests.cli.test_trainer_muon.param_clss_fn2',
    ])

    zero0_ckpt = torch.load(tmp_path / '1' / 'result.pt', weights_only=False)
    zero1_ckpt = torch.load(tmp_path / '2' / 'result.pt', weights_only=False)
    zero2_ckpt = torch.load(tmp_path / '3' / 'result.pt', weights_only=False)

    assert_equal(zero0_ckpt['model'], zero1_ckpt['model'])
    assert_equal(zero0_ckpt['model'], zero2_ckpt['model'])
    assert_equal(zero0_ckpt['optimizer']['state'], zero1_ckpt['optimizer']['state'])
    assert_equal(zero0_ckpt['optimizer']['state'], zero2_ckpt['optimizer']['state'])


def param_clss_fn(param_name: str) -> tuple[int, int]:
    """
    Classify a parameter name into an optimizer index and a parameter group index.
    """
    if 'layers.1.' in param_name or 'layers.10.' in param_name or 'layers.2.' in param_name:
        return 0, 0
    else:
        return 1, 0


def param_clss_fn2(param_name: str) -> tuple[int, int]:
    """
    Classify a parameter name into an optimizer index and a parameter group index.
    """
    if 'layers.1.' in param_name or 'layers.10.' in param_name or 'layers.2.' in param_name:
        return 0, 0, {'zero_param_level_sharding': None}
    else:
        return 1, 0, {'zero_param_level_sharding': True}
