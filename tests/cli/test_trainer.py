#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path
import re
import shutil
from typing import Any

from mock import PropertyMock
import torch
import pytest
import torch.distributed
from unittest.mock import patch

from nnscaler import merge_state_dicts
from nnscaler.cli.serialization import Checkpointer
import nnscaler
from nnscaler.cli.trainer import Trainer, logger
from nnscaler.cli.trainer_args import AggregatedOutputs, TrainerArgs
from tests.parallel_module.common import assert_equal, assert_close
from tests.utils import catch_log, init_random, replace_all_device_with, clear_parallel_cache
from ..launch_torchrun import launch_torchrun
from .common import MixedModule, MixModuleMLP, MixModuleMLP3


def trainer_logging_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    log_savedir = save_dir / 'log'
    tb_log_savedir = log_savedir / 'tensorboard'
    wandb_log_savedir = log_savedir / 'wandb'
    # train 4 epcho in one time
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--log.0.type', 'nnscaler.cli.loggers.TensorBoardLogger',
        '--log.0.args.name', 'test-cli',
        '--log.0.args.root_dir', str(tb_log_savedir),
        '--log.1.type', 'nnscaler.cli.loggers.WandbLogger',
        '--log.1.args.name', 'test-cli',
        '--log.1.args.dir', str(wandb_log_savedir),
        '--log.1.args.project', 'nnscaler',
        '--log.1.args.mode', 'offline',
    ])
    trainer.run()

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        assert (tb_log_savedir / 'test-cli').exists()
        for tag in ['val', 'train']:
            tfevents = list((tb_log_savedir / 'test-cli' / tag).glob('events.out.tfevents.*'))
            assert len(tfevents) == 1
            assert tfevents[0].stat().st_size > 1000

        assert (wandb_log_savedir / 'wandb').exists()
        wandb_offline_dir = list((wandb_log_savedir / 'wandb').glob('offline-run-*'))
        assert len(wandb_offline_dir) == 1
        wandb_run_db = list(wandb_offline_dir[0].glob('run-*.wandb'))
        assert len(wandb_run_db) == 1
        assert wandb_run_db[0].stat().st_size > 1000


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_logging(tmp_path):
    launch_torchrun(4, trainer_logging_worker, tmp_path)


@replace_all_device_with('cpu')
def test_trainer_compile_worker(tmp_path):
    save_dir = Path(tmp_path)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    # compile only
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--run_mode', 'compile',
        '--broadcast_strategy', 'none',
    ])
    trainer.run()

    assert set([f.name for f in gen_savedir.glob('**/*.py')]) == set(['gencode0.py', 'gencode1.py', 'gencode2.py', 'gencode3.py'])
    shutil.rmtree(gen_savedir)

    # mixed compile only
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--run_mode', 'compile',
        '--broadcast_strategy', 'none',
        '--model.type', 'tests.cli.common.MixedModule',
        '--model.parallel_modules.0.type', 'tests.cli.common.MixModuleMLP2',
        '--model.parallel_modules.0.args.dim', '16',
        '--model.parallel_modules.0.args.nlayers', '16',
        '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
    ])
    trainer.run()

    assert set([f.name for f in gen_savedir.glob('**/*.py')]) == set(['gencode0.py', 'gencode1.py', 'gencode2.py', 'gencode3.py'])
    shutil.rmtree(gen_savedir)


def trainer_resume_worker(save_dir, save_type, bf16, parallel_type=0):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    optimizer_type = 'nnscaler.runtime.f16_optimizer.MixedPrecisionAdam' \
        if bf16 == 'Mixed' \
        else 'torch.optim.Adam'
    use_zero = 1 if save_type == 'sharded' else 0
    format = 'safetensors' if parallel_type % 2 else 'pt'
    rev_format = 'pt' if format == 'safetensors' else 'safetensors'

    def list_ckpt_files(dir):
        return set(dir.glob('**/*.ckpt')) | set(dir.glob('**/*.safetensors'))

    if parallel_type == 0:
        additional_args = []
    elif parallel_type == 1:
        # 1. parallelize MixModuleMLP2 (self.mlp1)
        additional_args = [
            '--model.type', 'tests.cli.common.MixedModule',
            '--model.parallel_modules.0.type', 'tests.cli.common.MixModuleMLP2',
            '--model.parallel_modules.0.args.dim', '16',
            '--model.parallel_modules.0.args.nlayers', '16',
            '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        ]
    elif parallel_type == 2:
        # 2. parallelize MixModuleMLP (self.mlp0, self.mlploss.mlp)
        additional_args = [
            '--model.type', 'tests.cli.common.MixedModule',
            '--model.parallel_modules.0.type', 'tests.cli.common.MixModuleMLP',
            '--model.parallel_modules.0.args.dim', '16',
            '--model.parallel_modules.0.args.nlayers', '16',
            '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        ]
    elif parallel_type == 3:
        # 3. parallelize MixModuleMLP and MixModuleMLP3 (self.mlp0, self.mlploss.mlp, self.mlp2)
        # We will use different compute_config for the two parallelized modules
        additional_args = [
            '--model.type', 'tests.cli.common.MixedModule',
            '--model.parallel_modules.0.type', 'tests.cli.common.MixModuleMLP',
            '--model.parallel_modules.0.args.dim', '16',
            '--model.parallel_modules.0.args.nlayers', '16',
            '--model.parallel_modules.0.compute_config.use_zero', str(use_zero),
            '--model.parallel_modules.0.compute_config.constant_folding', 'False',
            '--model.parallel_modules.0.pas_policy', 'tp',
            '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
            '--model.parallel_modules.1.type', 'tests.cli.common.MixModuleMLP3',
            '--model.parallel_modules.1.args.dim', '16',
            '--model.parallel_modules.1.args.nlayers', '16',
            '--model.parallel_modules.1.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        ]
    elif parallel_type == 4:
        # 4. parallelize MixModuleMLP and MixModuleMLPWithLoss (self.mlp0, self.mlploss)
        # Note MixModuleMLP is also a member of MixModuleMLPWithLoss
        additional_args = [
            '--model.type', 'tests.cli.common.MixedModule',
            '--model.parallel_modules.0.type', 'tests.cli.common.MixModuleMLP',
            '--model.parallel_modules.0.args.dim', '16',
            '--model.parallel_modules.0.args.nlayers', '16',
            '--model.parallel_modules.0.compute_config.use_zero', str(use_zero),
            '--model.parallel_modules.0.compute_config.constant_folding', 'False',
            '--model.parallel_modules.0.pas_policy', 'tp',
            '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
            '--model.parallel_modules.1.type', 'tests.cli.common.MixModuleMLPWithLoss',
            '--model.parallel_modules.1.args.dim', '16',
            '--model.parallel_modules.1.args.nlayers', '16',
            '--model.parallel_modules.1.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        ]
    else:
        raise ValueError(f'parallel_type {parallel_type} is not supported')

    # train 4 epcho in one time
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
        '--checkpoint.format', format,
        *additional_args,
    ])
    trainer.run()
    ckpt_files = list_ckpt_files(ckpt_savedir)
    assert len(ckpt_files)/4 == min(30, trainer.total_train_steps_per_epoch * 4) + 2 # 2 for best/last

    torch.distributed.barrier()
    # train 4 epcho two times (resume from last)
    ckpt0_savedir = save_dir / 'ckpt0'
    # first two epochs
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '2',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
        '--checkpoint.format', format,
        *additional_args,
    ])
    trainer.run()
    ckpt0_files0 = {f: f.stat().st_mtime_ns for f in list_ckpt_files(ckpt0_savedir)}
    assert len(ckpt0_files0)/4 == min(30, trainer.total_train_steps_per_epoch * 2) + 2 # 2 for best/last

    # resume from last without update max_epochs
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '2',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
        '--checkpoint.format', rev_format,
        *additional_args,
    ])
    trainer.run()
    ckpt0_files0_x = {f: f.stat().st_mtime_ns for f in list_ckpt_files(ckpt0_savedir)}
    # nothing should be updated in this case.
    assert ckpt0_files0 == ckpt0_files0_x

    # create merged checkpoint
    ckpt1_savedir = save_dir / 'ckpt1'
    ckpt1_savedir.mkdir(parents=True, exist_ok=True)
    merged_file_name = f'merged{Checkpointer.NAME_MAP[format]}'
    if trainer.rank == 0:
        Trainer.merge_checkpoint(trainer.checkpointer.list_checkpoints(ckpt0_savedir / 'last'), ckpt1_savedir / merged_file_name)

    torch.distributed.barrier()
    # continue with the last two epochs (resume for sharded/deduped checkpoint)
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.format', rev_format,
        '--checkpoint.keep_last_n_checkpoints', '30',
        *additional_args,
    ])
    trainer.run()
    left_files = {
        f: f.stat().st_mtime_ns for f in ckpt0_files0.keys()
        if f.exists() and f.parent.name not in ['last', 'best']
    }
    assert left_files  # some checkpoints are removed
    for f, s in left_files.items(): # make sure the old checkpoints are not overwritten
        assert ckpt0_files0[f] == s

    ckpt0_files1 = list_ckpt_files(ckpt0_savedir)
    assert len(ckpt0_files1)/4 == min(30, trainer.total_train_steps_per_epoch * 4) + 2 # 2 for best/last

    torch.distributed.barrier()

    # continue with the last two epochs (resume for merged)
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--compute_config.use_zero', str(use_zero),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--checkpoint.format', rev_format,
        '--checkpoint.resume_from', str(ckpt1_savedir / merged_file_name),
        '--checkpoint.keep_last_n_checkpoints', '30',
        *additional_args,
    ])
    trainer.run()
    left_files = {
        f: f.stat().st_mtime_ns for f in ckpt0_files0.keys()
        if f.exists() and f.parent.name not in ['last', 'best']
    }
    assert left_files  # some checkpoints are removed
    for f, s in left_files.items(): # make sure the old checkpoints are not overwritten
        assert ckpt0_files0[f] == s

    ckpt0_files1 = list_ckpt_files(ckpt0_savedir)
    assert len(ckpt0_files1)/4 == min(30, trainer.total_train_steps_per_epoch * 4) + 2 # 2 for best/last

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        assert {f.parent.name for f in ckpt_files} == {f.parent.name for f in ckpt0_files1}
        for i in range(4):
            x = trainer.checkpointer.load_for_rank(ckpt_savedir / 'last', i)
            y = trainer.checkpointer.load_for_rank(ckpt0_savedir / 'last', i)
            z = trainer.checkpointer.load_for_rank(ckpt1_savedir / 'last', i)
            assert_equal(x['model'], y['model'])
            assert_equal(x['optimizer'], y['optimizer'])
            assert_equal(x['lr_scheduler'], y['lr_scheduler'])
            assert_equal(x['model'], z['model'])
            assert_equal(x['optimizer'], z['optimizer'])
            assert_equal(x['lr_scheduler'], z['lr_scheduler'])

        suffix = Checkpointer.NAME_MAP[format]
        if save_type == 'deduped':
            assert (ckpt_savedir / f'last/0{suffix}').stat().st_size > (ckpt_savedir / f'last/2{suffix}').stat().st_size
            assert (ckpt_savedir / f'last/1{suffix}').stat().st_size > (ckpt_savedir / f'last/3{suffix}').stat().st_size
        else:
            assert (ckpt_savedir / f'last/0{suffix}').stat().st_size == (ckpt_savedir / f'last/2{suffix}').stat().st_size
            assert (ckpt_savedir / f'last/1{suffix}').stat().st_size == (ckpt_savedir / f'last/3{suffix}').stat().st_size


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('save_type', ['sharded', 'deduped'])
@pytest.mark.parametrize('bf16', [True, False, 'Mixed'])
def test_trainer_resume(tmp_path, save_type, bf16):
    launch_torchrun(4, trainer_resume_worker, tmp_path, save_type, bf16)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('parallel_type', [1, 2, 3, 4])
def test_trainer_resume_mixed(tmp_path, parallel_type):
    # we will parallelize the sub models in MixedModule
    # We have different ways to parallelize the sub models
    # 1. parallelize MixModuleMLP2 (self.mlp1)
    # 2. parallelize MixModuleMLP (self.mlp0, self.mlploss.mlp)
    # 3. parallelize MixModuleMLP and MixModuleMLP3 (self.mlp0, self.mlploss.mlp, self.mlp2)
    # 4. parallelize MixModuleMLP and MixModuleMLPWithLoss (self.mlp0, self.mlploss)
    #    Note MixModuleMLP is also a member of MixModuleMLPWithLoss
    launch_torchrun(4, trainer_resume_worker, tmp_path, 'deduped', True, parallel_type)


def trainer_last_checkpoint_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    for i in range (100): # make a lot of fake checkpoints
        (ckpt_savedir / f'0000-{i*15:04d}').mkdir(parents=True, exist_ok=True)

    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '1',
        '--global_batch_size', '4',  # mini_batch_size=2, update_freq=2
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
        '--val_every_n_train_steps', '1',
        '--checkpoint.every_n_train_steps', '15',
        '--checkpoint.save_dir', str(ckpt_savedir),
    ])
    trainer.run()

    torch.distributed.barrier()
    # make sure the last checkpoint is saved.
    assert (ckpt_savedir / '0000-0025' / f'{trainer.rank}.ckpt').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_trainer_last_checkpoint(tmp_path):
    launch_torchrun(1, trainer_last_checkpoint_worker, tmp_path)


_train_losses = []
_val_losses = []


def after_aggregate_train_step_outputs(trainer: 'Trainer', aggregated_outputs: 'AggregatedOutputs', train_loss: float) -> None:
    _train_losses.append(train_loss)


def after_aggregate_val_step_outputs(trainer: 'Trainer', aggregated_outputs: 'AggregatedOutputs', val_loss: float) -> None:
    _val_losses.append(val_loss)


def trainer_loss_reduction_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'

    _train_losses.clear()
    _val_losses.clear()
    trainer = Trainer([
        '-f', config_path,
        '--enable_progress_bar', 'false',
        '--max_epochs', '1',
        '--global_batch_size', '4',  # mini_batch_size=2, update_freq=2
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
        '--val_every_n_train_steps', '1',
        '--checkpoint.no_save', 'true',
        '--optimizer.loss_reduction', 'mean',
        '--hook.after_aggregate_train_step_outputs',
            'tests.cli.test_trainer.after_aggregate_train_step_outputs',
        '--hook.after_aggregate_val_step_outputs',
            'tests.cli.test_trainer.after_aggregate_val_step_outputs',
    ])
    trainer.run()

    # get a copy
    train_loss_mean = _train_losses[:]
    val_loss_mean = _val_losses[:]

    torch.distributed.barrier()

    _train_losses.clear()
    _val_losses.clear()

    trainer = Trainer([
        '-f', config_path,
        '--enable_progress_bar', 'false',
        '--max_epochs', '1',
        '--global_batch_size', '4',  # mini_batch_size=2, update_freq=2
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
        '--val_every_n_train_steps', '1',
        '--checkpoint.no_save', 'true',
        '--optimizer.loss_reduction', 'sum',
        '--hook.after_aggregate_train_step_outputs',
            'tests.cli.test_trainer.after_aggregate_train_step_outputs',
        '--hook.after_aggregate_val_step_outputs',
            'tests.cli.test_trainer.after_aggregate_val_step_outputs',
    ])
    trainer.run()
    torch.distributed.barrier()

    assert len(train_loss_mean) == len(_train_losses)
    assert len(val_loss_mean) == len(_val_losses)
    for i in range(len(train_loss_mean)):
        assert train_loss_mean[i] == _train_losses[i] / 2  # 2 is update freq

    for i in range(len(val_loss_mean)):
        assert val_loss_mean[i] == _val_losses[i]

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_trainer_loss_reduction(tmp_path):
    launch_torchrun(1, trainer_loss_reduction_worker, tmp_path)


_before_step_grads = None
def before_gnorm_clip(trainer: 'Trainer') -> None:
    global _before_step_grads
    _before_step_grads = {i: g.grad.clone() for i, g in enumerate(trainer.optimizer.param_groups[0]['params'])}


def aggregate_outputs(loss_outputs, sync_group) -> 'AggregatedOutputs':
    # loss is the first element of the output (or the only element)
    losses = [
        loss if isinstance(loss, torch.Tensor)
        else loss[0]
        for loss in loss_outputs
    ]
    loss_sum = torch.sum(torch.stack(losses), dtype=torch.float64)
    torch.distributed.all_reduce(loss_sum, group=sync_group)
    num_batches = torch.tensor(len(losses), device=torch.cuda.current_device())
    torch.distributed.all_reduce(num_batches, group=sync_group)
    num_tokens = num_batches * 2  # fake value

    return AggregatedOutputs(
        loss_sum = loss_sum.item(),
        num_batches=num_batches.item(),
        num_tokens=num_tokens.item(),
    )


def trainer_per_token_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'

    _train_losses.clear()
    _val_losses.clear()
    trainer = Trainer([
        '-f', config_path,
        '--enable_progress_bar', 'false',
        '--max_epochs', '1',
        '--global_batch_size', '8',
        '--grad_accumulation_steps', '2',
        '--gen_savedir', str(gen_savedir),
        '--max_train_steps', '1',
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--val_every_n_train_steps', '1',
        '--checkpoint.no_save', 'true',
        '--optimizer.grad_reduction', 'mean',
        '--optimizer.aggregate_outputs_fn', 'tests.cli.test_trainer.aggregate_outputs',
        '--hook.before_gnorm_clip',
            'tests.cli.test_trainer.before_gnorm_clip',
    ])
    trainer.run()

    # get a copy
    grads = _before_step_grads

    torch.distributed.barrier()

    trainer = Trainer([
        '-f', config_path,
        '--enable_progress_bar', 'false',
        '--max_epochs', '1',
        '--global_batch_size', '8',
        '--grad_accumulation_steps', '2',
        '--gen_savedir', str(gen_savedir),
        '--max_train_steps', '1',
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--val_every_n_train_steps', '1',
        '--checkpoint.no_save', 'true',
        '--optimizer.grad_reduction', 'per-token-mean',
        '--optimizer.aggregate_outputs_fn', 'tests.cli.test_trainer.aggregate_outputs',
        '--hook.before_gnorm_clip',
            'tests.cli.test_trainer.before_gnorm_clip',
    ])
    trainer.run()

    torch.distributed.barrier()

    assert set(grads.keys()) == set(_before_step_grads.keys())
    for n, p in grads.items():
        assert torch.equal(p / 2, _before_step_grads[n])

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_trainer_per_token(tmp_path):
    launch_torchrun(2, trainer_per_token_worker, tmp_path)


def test_dataset_empty_train_args():
    def _empty_train_args():
        from .common import SimpleDataset
        return SimpleDataset(10)

    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    train_args = TrainerArgs.from_cli([
        '-f', config_path,
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
    ])
    train_args.dataset.type = _empty_train_args
    train_args.dataset.train_args = {}
    train_args.dataset.val_args = {}
    assert train_args.create_dataset() is not None
    assert train_args.create_dataset('val') is None


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4 or torch.cuda.device_count() >= 8, reason='lack of gpu devices')
@pytest.mark.parametrize('use_bf16', [True, False])
@pytest.mark.parametrize('zero_ngroups', [None, '1', '2'])
def test_trainer_grad_sync_check_4gpu(tmp_path, use_bf16, zero_ngroups):
    launch_torchrun(4, trainer_grad_sync_check, tmp_path, use_bf16, zero_ngroups, '4')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 8, reason='lack of gpu devices')
@pytest.mark.parametrize('use_bf16', [True, False])
@pytest.mark.parametrize('zero_ngroups', [None, '1', '2', '4'])
def test_trainer_grad_sync_check_8gpu(tmp_path, use_bf16, zero_ngroups):
    launch_torchrun(8, trainer_grad_sync_check, tmp_path, use_bf16, zero_ngroups, '8')


def trainer_grad_sync_check(save_dir, use_bf16, zero_ngroups, runtime_ngpus):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'
    optimizer_type = 'torch.optim.Adam'
    use_zero = 0 if zero_ngroups is None else 1
    zero_ngroups = '1' if zero_ngroups is None else zero_ngroups

    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16' if use_bf16 else 'none',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '1',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', runtime_ngpus,
        '--compute_config.use_zero', str(use_zero),
        '--compute_config.zero_ngroups', zero_ngroups,
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--debug.check_gradient_sync_cross_devices', 'true',
    ])
    trainer.run()
    torch.distributed.barrier()


def trainer_correctness_worker(save_dir, parallel_type=0, async_reducer=False, hybrid_opt=False, use_zero=0):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    if parallel_type == 0:
        # parallelize the whole MixedModule
        additional_args = [
            '--model.type', 'tests.cli.common.MixedModule',
        ]
    elif parallel_type == 1:
        # 1. parallelize MixModuleMLP2 (self.mlp1)
        additional_args = [
            '--model.type', 'tests.cli.common.MixedModule',
            '--model.parallel_modules.0.type', 'tests.cli.common.MixModuleMLP2',
            '--model.parallel_modules.0.args.dim', '16',
            '--model.parallel_modules.0.args.nlayers', '16',
            '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        ]
    elif parallel_type == 2:
        # 2. parallelize MixModuleMLP (self.mlp0, self.mlploss.mlp)
        additional_args = [
            '--model.type', 'tests.cli.common.MixedModule',
            '--model.parallel_modules.0.type', 'tests.cli.common.MixModuleMLP',
            '--model.parallel_modules.0.args.dim', '16',
            '--model.parallel_modules.0.args.nlayers', '16',
            '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        ]
    elif parallel_type == 3:
        # 3. parallelize MixModuleMLP and MixModuleMLP3 (self.mlp0, self.mlploss.mlp, self.mlp2)
        # We will use different compute_config for the two parallelized modules
        additional_args = [
            '--model.type', 'tests.cli.common.MixedModule',
            '--model.parallel_modules.0.type', 'tests.cli.common.MixModuleMLP',
            '--model.parallel_modules.0.args.dim', '16',
            '--model.parallel_modules.0.args.nlayers', '16',
            '--model.parallel_modules.0.compute_config.use_zero', '0',
            '--model.parallel_modules.0.compute_config.constant_folding', 'False',
            '--model.parallel_modules.0.pas_policy', 'tp',
            '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
            '--model.parallel_modules.1.type', 'tests.cli.common.MixModuleMLP3',
            '--model.parallel_modules.1.args.dim', '16',
            '--model.parallel_modules.1.args.nlayers', '16',
            '--model.parallel_modules.1.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        ]
    elif parallel_type == 4:
        # 4. parallelize MixModuleMLP and MixModuleMLPWithLoss (self.mlp0, self.mlploss)
        # Note MixModuleMLP is also a member of MixModuleMLPWithLoss
        additional_args = [
            '--model.type', 'tests.cli.common.MixedModule',
            '--model.parallel_modules.0.type', 'tests.cli.common.MixModuleMLP',
            '--model.parallel_modules.0.args.dim', '16',
            '--model.parallel_modules.0.args.nlayers', '16',
            '--model.parallel_modules.0.compute_config.use_zero', '0',
            '--model.parallel_modules.0.compute_config.constant_folding', 'False',
            '--model.parallel_modules.0.pas_policy', 'tp',
            '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
            '--model.parallel_modules.1.type', 'tests.cli.common.MixModuleMLPWithLoss',
            '--model.parallel_modules.1.args.dim', '16',
            '--model.parallel_modules.1.args.nlayers', '16',
            '--model.parallel_modules.1.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        ]
    else:
        raise ValueError(f'parallel_type {parallel_type} is not supported')


    def param_clss_fn(param_name: str) -> tuple[int, int]:
        """
        Classify a parameter name into an optimizer index and a parameter group index.
        """
        if 'mlp0.' in param_name:
            return 0, 0
        elif 'mlp1.' in param_name:
            return 0, 1
        else:
            return 1, 0

    optimizer_config = {
        'type': 'nnscaler.HybridOptimizer',
        'param_clss_fn': param_clss_fn,
        'args': {
            'config': {
                'optimizers':[
                    {
                        'type': torch.optim.Adam,
                        'options': {
                            'lr': 0.01,
                        },
                        'param_groups': [
                            {},
                            {}
                        ],
                    },{
                        'type': torch.optim.Adam,
                        'options': {
                            'lr': 0.01
                        }
                    }
                ]
            }
        }
    }

    if hybrid_opt:
        additional_args.extend(['--optimizer!', '--optimizer', optimizer_config])

    # train 4 epcho in one time
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'fp32',
        '--max_epochs', '2',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.use_zero', str(use_zero),
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '2',
        '--compute_config.use_async_reducer', str(async_reducer),
        '--compute_config.reducer_bucket_cap_mb', '1e-6',
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '5',
        # '--model.args.dim', '16',
        # '--model.args.nlayers', '2',
        *additional_args,
    ])
    trainer.run()

    torch.distributed.barrier()

    # create merged checkpoint
    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt_savedir / 'last').glob('*.ckpt')), save_dir / 'merged.pt')
        shutil.rmtree(gen_savedir)

    clear_parallel_cache()

    torch.distributed.barrier()


def trainer_correctness_worker_aggregate(tmp_path, use_zero):
    for parallel_type in range(5):
        for async_reducer in [False, True]:
            for hybrid_opt in [True, False]:
                print(f'parallel_type={parallel_type}, async_reducer={async_reducer}, hybrid_opt={hybrid_opt}')
                save_dir = tmp_path/f'{parallel_type}-{async_reducer}-{hybrid_opt}'
                trainer_correctness_worker(save_dir, parallel_type, async_reducer, hybrid_opt, use_zero)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('use_zero', [0, 1, 3])
def test_trainer_correctness(tmp_path, use_zero):
    launch_torchrun(2, trainer_correctness_worker_aggregate, tmp_path, use_zero)
    merged_ckpts = {}
    for parallel_type in range(5):
        for async_reducer in [False, True]:
            for hybrid_opt in [True, False]:
                save_dir = tmp_path/f'{parallel_type}-{async_reducer}-{hybrid_opt}'
                merged_ckpts[(parallel_type, async_reducer, hybrid_opt)] = torch.load(save_dir/'merged.pt')

    if use_zero == 3:
        assert_fn = assert_close
    else:
        assert_fn = assert_equal

    for parallel_type in range(5):
        for async_reducer in [False, True]:
            for hybrid_opt in [True, False]:
                assert_fn(
                    merged_ckpts[(parallel_type, async_reducer, hybrid_opt)]['model'],
                    merged_ckpts[(0, False, False)]['model']
                )
                if not hybrid_opt:
                    assert_fn(
                        merged_ckpts[(parallel_type, async_reducer, hybrid_opt)]['optimizer'],
                        merged_ckpts[(0, False, False)]['optimizer']
                    )
                else:
                    # param_groups are different when using hybrid optimizer.
                    assert_fn(
                        merged_ckpts[(parallel_type, async_reducer, hybrid_opt)]['optimizer']['state'],
                        merged_ckpts[(0, False, False)]['optimizer']['state']
                    )


def tracing_from_weights_worker(tmp_path):
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())

    init_random(1141)
    mixed_module = MixedModule(16, 16, False)
    mixed_module_2 = MixedModule(16, 16)
    assert not torch.equal(mixed_module.mlp0.layers[0].weight, mixed_module_2.mlp0.layers[0].weight)
    assert not torch.equal(mixed_module.mlp2.layers[0].weight, mixed_module_2.mlp2.layers[0].weight)
    assert not torch.equal(mixed_module.mlploss.mlp.layers[0].weight, mixed_module_2.mlploss.mlp.layers[0].weight)

    tracing_weights = mixed_module.state_dict()
    tracing_from_weights = tmp_path / 'tracing_weights.pt'
    torch.save(tracing_weights, tracing_from_weights)

    def _compile(index, *additional_args):
        gen_dir = tmp_path / f'gen{index}'
        trainer = Trainer([
            '-f', config_path,
            '--gen_savedir', str(gen_dir),
            '--global_batch_size', '0',
            '--max_epochs', '-1',  # HACK: will exit without training.
            '--max_train_steps', '-1',
            '--compute_config.plan_ngpus', '1',
            '--compute_config.runtime_ngpus', '1',
            '--broadcast_strategy', 'none',
            '--model.type', 'tests.cli.common.MixedModule',
            *additional_args,
        ])
        trainer.run()
        import shutil
        shutil.rmtree(gen_dir)
        clear_parallel_cache()
        return merge_state_dicts([trainer.model.state_dict()])[0]

    model1 = _compile(1)
    model3 = _compile(3, '--tracing_from_weights', str(tracing_from_weights))
    model2 = _compile(2)


    assert_equal(model1, model2)
    assert_equal(model1, dict(**mixed_module_2.state_dict()))
    assert_equal(model3, dict(**tracing_weights))

    # parallelize MixModuleMLP2 and MixModuleMLP3 (self.mlp1, self.mlp2)
    # We will use different compute_config for the two parallelized modules
    additional_args = [
        '--model.parallel_modules.0.type', 'tests.cli.common.MixModuleMLP2',
        '--model.parallel_modules.0.args.dim', '16',
        '--model.parallel_modules.0.args.nlayers', '16',
        '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
        '--model.parallel_modules.1.type', 'tests.cli.common.MixModuleMLP3',
        '--model.parallel_modules.1.args.dim', '16',
        '--model.parallel_modules.1.args.nlayers', '16',
        '--model.parallel_modules.1.forward_args_gen_fn', 'tests.cli.common.forward_args_gen_fn',
    ]
    model4 = _compile(4, *additional_args)
    assert_equal(model4, dict(**mixed_module_2.state_dict()))

    model5 = _compile(5,
        '--tracing_from_weights', str(tracing_from_weights),
        '--model.parallel_modules.0.tracing_from_weights_prefix', 'mlp1',
        '--model.parallel_modules.1.tracing_from_weights_prefix', 'mlp2',
        *additional_args
    )
    for key in tracing_weights:
        if key.startswith('mlp1') or key.startswith('mlp2'):
            assert torch.equal(model5[key], tracing_weights[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_tracing_from_weights(tmp_path):
    launch_torchrun(1, tracing_from_weights_worker, tmp_path)


def trainer_resumable_dataloader(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    config_path_streaming = str(Path(__file__).with_name('trainer_args_streaming.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    optimizer_type = 'nnscaler.runtime.f16_optimizer.MixedPrecisionAdam'
    use_zero = True
    save_type = 'deduped'

    # ground truth: train 4 epcho in one time
    trainer = Trainer([
        '-f', config_path,
        '--precision', 'bf16',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--enable_progress_bar', False,
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', 2,
        '--compute_config.runtime_ngpus', 4,
        '--compute_config.use_zero', use_zero,
        '--dataset_sampler.train_args.shuffle', False,
        '--dataloader.train_args.shuffle', False,
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', 30,
    ])
    trainer.run()
    assert not trainer.dataloader_resumed

    # train 4 epcho in one time
    ckpt0_savedir = save_dir / 'ckpt0'
    gen_savedir = save_dir / 'gen0'  # use a different gen_savedir for resumable dataloader
    trainer = Trainer([
        '-f', config_path_streaming,
        '--precision', 'bf16',
        '--optimizer.type', optimizer_type,
        '--max_epochs', '4',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt0_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
    ])
    trainer.run()
    assert not trainer.dataloader_resumed

    torch.distributed.barrier()
    # train 4 epcho two times (resume from last)
    ckpt1_savedir = save_dir / 'ckpt1'
    # first 5 steps
    trainer = Trainer([
        '-f', config_path_streaming,
        '--precision', 'bf16',
        '--optimizer.type', optimizer_type,
        '--max_train_steps', '5',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
    ])
    trainer.run()
    assert not trainer.dataloader_resumed
    ckpt1_files0 = {f: f.stat().st_mtime_ns for f in ckpt1_savedir.glob('**/*.ckpt')}

    # resume from last without update max_epochs
    trainer = Trainer([
        '-f', config_path_streaming,
        '--precision', 'bf16',
        '--optimizer.type', optimizer_type,
        '--max_train_steps', '5',
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
    ])
    trainer.run()
    assert trainer.dataloader_resumed
    ckpt1_files0_x = {f: f.stat().st_mtime_ns for f in ckpt1_savedir.glob('**/*.ckpt')}
    # nothing should be updated in this case.
    assert ckpt1_files0 == ckpt1_files0_x

    # create merged checkpoint
    ckpt2_savedir = save_dir / 'ckpt2'
    ckpt2_savedir.mkdir(parents=True, exist_ok=True)
    if trainer.rank == 0:
        Trainer.merge_checkpoint(list((ckpt1_savedir / 'last').glob('*.ckpt')), ckpt2_savedir / 'merged.pt')
        merged_state_dict = torch.load(ckpt2_savedir / 'merged.pt')
        assert 'dataloader' in merged_state_dict
        assert isinstance(merged_state_dict['dataloader'], list)
        assert len(merged_state_dict['dataloader']) == merged_state_dict['train_args']['compute_config']['runtime_ngpus']
        merged_state_dict.pop('dataloader')  # remove the merged_state_dict from the cache
        torch.save(merged_state_dict, ckpt2_savedir / 'merged2.pt')

    torch.distributed.barrier()
    # resume for sharded/deduped checkpoint
    trainer = Trainer([
        '-f', config_path_streaming,
        '--precision', 'bf16',
        '--optimizer.type', optimizer_type,
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--checkpoint.resume_from', 'last',
        '--checkpoint.keep_last_n_checkpoints', '30',
    ])
    trainer.run()
    assert trainer.dataloader_resumed

    torch.distributed.barrier()

    # resume for merged
    with catch_log(logger) as log:
        trainer = Trainer([
            '-f', config_path_streaming,
            '--precision', 'bf16',
            '--optimizer.type', optimizer_type,
            '--enable_progress_bar', 'false',
            '--gen_savedir', str(gen_savedir),
            '--checkpoint.save_type', save_type,
            '--checkpoint.save_dir', str(ckpt2_savedir),
            '--checkpoint.resume_from.checkpoint', str(ckpt2_savedir / 'merged.pt'),
            '--checkpoint.resume_from.save_memory', False,
            '--checkpoint.keep_last_n_checkpoints', '30',
        ])
        trainer.run()
        assert trainer.dataloader_resumed
        assert 'Broadcasting trimmed checkpoint to all ranks.' not in log.getvalue()  # no warning about dataloader states

    torch.distributed.barrier()


    ckpt2_1_savedir = save_dir / 'ckpt2_1'
    ckpt2_1_savedir.mkdir(parents=True, exist_ok=True)
    # resume for merged
    with catch_log(logger) as log:
        trainer = Trainer([
            '-f', config_path_streaming,
            '--precision', 'bf16',
            '--optimizer.type', optimizer_type,
            '--enable_progress_bar', 'false',
            '--gen_savedir', str(gen_savedir),
            '--checkpoint.save_type', save_type,
            '--checkpoint.save_dir', str(ckpt2_1_savedir),
            '--checkpoint.resume_from.checkpoint', str(ckpt2_savedir / 'merged.pt'),
            '--checkpoint.resume_from.save_memory', True,
            '--checkpoint.keep_last_n_checkpoints', '30',
        ])
        trainer.run()
        assert trainer.dataloader_resumed
        assert 'Broadcasting trimmed checkpoint to all ranks.' in log.getvalue()  # no warning about dataloader states

    torch.distributed.barrier()

    # resume for merged without dataloader states
    ckpt3_savedir = save_dir / 'ckpt3'
    trainer = Trainer([
        '-f', config_path_streaming,
        '--precision', 'bf16',
        '--optimizer.type', optimizer_type,
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_type', save_type,
        '--checkpoint.save_dir', str(ckpt3_savedir),
        '--checkpoint.resume_from', str(ckpt2_savedir / 'merged2.pt'),
        '--checkpoint.keep_last_n_checkpoints', '30',
    ])
    trainer.run()
    assert not trainer.dataloader_resumed

    # resume for auto-merged checkpoint
    ckpt4_savedir = save_dir / 'ckpt4'
    with catch_log(logger) as log:
        trainer = Trainer([
            '-f', config_path_streaming,
            '--precision', 'bf16',
            '--optimizer.type', optimizer_type,
            '--enable_progress_bar', 'false',
            '--gen_savedir', str(gen_savedir),
            '--checkpoint.save_type', save_type,
            '--checkpoint.save_dir', str(ckpt4_savedir),
            '--checkpoint.resume_from.checkpoint', str(ckpt1_savedir / '0002-0035'),
            '--checkpoint.resume_from.with_merged', True,
            '--checkpoint.resume_from.save_memory', False,
            '--checkpoint.keep_last_n_checkpoints', '30',
        ])
        trainer.run()
        assert trainer.dataloader_resumed
        assert 'Broadcasting merged checkpoint to all ranks.' in log.getvalue()  # no warning about dataloader states

    # resume from auto-merged with save_memory
    ckpt5_savedir = save_dir / 'ckpt5'
    with catch_log(logger) as log:
        trainer = Trainer([
            '-f', config_path_streaming,
            '--precision', 'bf16',
            '--optimizer.type', optimizer_type,
            '--enable_progress_bar', 'false',
            '--gen_savedir', str(gen_savedir),
            '--checkpoint.save_type', save_type,
            '--checkpoint.save_dir', str(ckpt5_savedir),
            '--checkpoint.resume_from.checkpoint', str(ckpt1_savedir / '0002-0035'),
            '--checkpoint.resume_from.with_merged', True,
            '--checkpoint.resume_from.save_memory', True,
            '--checkpoint.keep_last_n_checkpoints', '30',
        ])
        trainer.run()
        assert trainer.dataloader_resumed
        assert 'Broadcasting trimmed checkpoint to all ranks.' in log.getvalue()  # no warning about dataloader states


    if torch.distributed.get_rank() == 0:
        for i in range(4):
            g = torch.load(ckpt_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            x = torch.load(ckpt0_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            y = torch.load(ckpt1_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            z = torch.load(ckpt2_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            z_1 = torch.load(ckpt2_1_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            w = torch.load(ckpt3_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            v = torch.load(ckpt4_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            u = torch.load(ckpt5_savedir / 'last' / f'{i}.ckpt', weights_only=False)
            assert 'dataloader' not in g
            assert 'dataloader' in x
            for key in ['model', 'optimizer', 'lr_scheduler', 'dataloader']:
                assert_equal(x[key], y[key])
                assert_equal(x[key], z[key])
                assert_equal(x[key], w[key])
                assert_equal(x[key], v[key])
                assert_equal(x[key], u[key])
                assert_equal(x[key], z_1[key])
                if key != 'dataloader':
                    assert_equal(g[key], x[key])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_resumable_dataloader(tmp_path):
    launch_torchrun(4, trainer_resumable_dataloader, tmp_path)


@replace_all_device_with('cpu')
def test_trainer_dynamic_worker(tmp_path):

    def check_match(code_dir: Path, should_exist: bool):
        gencode_files = list(code_dir.glob('**/*.py'))
        assert set(f.name for f in gencode_files) == set(['gencode0.py', 'gencode1.py', 'gencode2.py', 'gencode3.py'])
        for gencode_file in gencode_files:
            filecontent = gencode_file.read_text()
            matches = re.findall(r'B, T, C = x\.size\(\)', filecontent)
            if should_exist:
                assert matches
            else:
                assert not matches

        shutil.rmtree(code_dir)

    save_dir = Path(tmp_path)
    config_path = str(Path(__file__).with_name('trainer_args_csa.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    # compile only
    trainer = Trainer([
        '-f', config_path,
        '--vars.dynamic_dims', '[1]',
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--run_mode', 'compile',
        '--broadcast_strategy', 'none',
    ])
    trainer.run()
    check_match(gen_savedir, should_exist=True)

    gen_savedir = save_dir / 'gen0'
    # compile only
    trainer = Trainer([
        '-f', config_path,
        '--vars.dynamic_dims', '[]',
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--run_mode', 'compile',
        '--broadcast_strategy', 'none',
    ])
    trainer.run()
    check_match(gen_savedir, should_exist=False)

    # mixed compile
    gen_savedir = save_dir / 'gen1'
    # compile only
    trainer = Trainer([
        '-f', config_path,
        '--vars.dynamic_dims', '[1]',
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--run_mode', 'compile',
        '--broadcast_strategy', 'none',

        '--model.parallel_modules.0.type', 'tests.cli.common.CausalSelfAttention',
        '--model.parallel_modules.0.args.n_embd', '$(model.args.n_embd)',
        '--model.parallel_modules.0.args.n_head', '$(model.args.n_head)',
        '--model.parallel_modules.0.args.dropout', '$(model.args.dropout)',
        '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.csa_forward_args_gen_fn',
        '--model.parallel_modules.0.forward_args_post_process_fn', 'tests.cli.common.post_csa_forward_args_gen_fn',
    ])
    trainer.run()
    check_match(gen_savedir, should_exist=True)

    # mixed compile
    gen_savedir = save_dir / 'gen2'
    # compile only
    trainer = Trainer([
        '-f', config_path,
        '--vars.dynamic_dims', '[]',
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--compute_config.plan_ngpus', '2',
        '--compute_config.runtime_ngpus', '4',
        '--checkpoint.no_save', 'true',
        '--run_mode', 'compile',
        '--broadcast_strategy', 'none',

        '--model.parallel_modules.0.type', 'tests.cli.common.CausalSelfAttention',
        '--model.parallel_modules.0.args.n_embd', '$(model.args.n_embd)',
        '--model.parallel_modules.0.args.n_head', '$(model.args.n_head)',
        '--model.parallel_modules.0.args.dropout', '$(model.args.dropout)',
        '--model.parallel_modules.0.forward_args_gen_fn', 'tests.cli.common.csa_forward_args_gen_fn',
        '--model.parallel_modules.0.forward_args_post_process_fn', 'tests.cli.common.post_csa_forward_args_gen_fn',
    ])
    trainer.run()
    check_match(gen_savedir, should_exist=False)


def trainer_zero3(dim, save_dir, plan_ngpus, runtime_ngpus):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())

    zero_ngroups = runtime_ngpus // plan_ngpus // 2
    if zero_ngroups < 1:
        zero_ngroups = 1
    policy = 'dp' if plan_ngpus == 1 else 'tp'

    gen3_savedir = save_dir / 'gen3'
    ckpt3_savedir = save_dir / 'ckpt3'
    # train 1 epcho in one time with zero3
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '5',
        '--vars.dim', f'{dim}',
        '--gen_savedir', str(gen3_savedir),
        '--compute_config.plan_ngpus', f'{plan_ngpus}',
        '--compute_config.runtime_ngpus', f'{runtime_ngpus}',
        '--compute_config.zero_ngroups', f'{zero_ngroups}',
        '--compute_config.use_zero', '3',
        '--compute_config.pas_config.enable_random_replicated', 'True',
        '--checkpoint.save_dir', str(ckpt3_savedir),
        '--pas_policy', f'{policy}',
        '--checkpoint.save_type', 'sharded',
    ])
    trainer.run()

    torch.distributed.barrier()

    # load from sharded
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '10',
        '--vars.dim', f'{dim}',
        '--gen_savedir', str(gen3_savedir),
        '--compute_config.plan_ngpus', f'{plan_ngpus}',
        '--compute_config.runtime_ngpus', f'{runtime_ngpus}',
        '--compute_config.zero_ngroups', f'{zero_ngroups}',
        '--compute_config.use_zero', '3',
        '--compute_config.pas_config.enable_random_replicated', 'True',
        '--checkpoint.save_dir', str(ckpt3_savedir),
        '--checkpoint.resume_from', 'last',
        '--pas_policy', f'{policy}',
        '--checkpoint.save_type', 'deduped',
    ])
    trainer.run()

    # load from deduped
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '15',
        '--vars.dim', f'{dim}',
        '--gen_savedir', str(gen3_savedir),
        '--compute_config.plan_ngpus', f'{plan_ngpus}',
        '--compute_config.runtime_ngpus', f'{runtime_ngpus}',
        '--compute_config.zero_ngroups', f'{zero_ngroups}',
        '--compute_config.use_zero', '3',
        '--compute_config.pas_config.enable_random_replicated', 'True',
        '--checkpoint.save_dir', str(ckpt3_savedir),
        '--checkpoint.resume_from', 'last',
        '--pas_policy', f'{policy}',
        '--checkpoint.save_type', 'deduped',
    ])
    trainer.run()

    if torch.distributed.get_rank() == 0:
        Trainer.merge_checkpoint(list((ckpt3_savedir / 'last').glob('*.ckpt')), ckpt3_savedir / 'merged.pt')

    torch.distributed.barrier()

    # load from merged (from deduped)
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '20',
        '--vars.dim', f'{dim}',
        '--gen_savedir', str(gen3_savedir),
        '--compute_config.plan_ngpus', f'{plan_ngpus}',
        '--compute_config.runtime_ngpus', f'{runtime_ngpus}',
        '--compute_config.zero_ngroups', f'{zero_ngroups}',
        '--compute_config.use_zero', '3',
        '--compute_config.pas_config.enable_random_replicated', 'True',
        '--checkpoint.save_dir', str(ckpt3_savedir),
        '--checkpoint.resume_from', str(ckpt3_savedir / 'merged.pt'),
        '--pas_policy', f'{policy}',
        '--checkpoint.save_type', 'sharded',
    ])
    trainer.run()

    if torch.distributed.get_rank() == 0:
        Trainer.merge_checkpoint(list((ckpt3_savedir / 'last').glob('*.ckpt')), ckpt3_savedir / 'merged.pt')

        with (
            patch('nnscaler.ComputeConfig.module_dedup_group_size', new_callable=PropertyMock) as mock_dgs,
            patch('nnscaler.ComputeConfig.optimizer_dedup_group_size', new_callable=PropertyMock) as mock_dgs2
        ):
            # to mock the case where we have duplicated data in merging
            mock_dgs.return_value = runtime_ngpus
            mock_dgs2.return_value = runtime_ngpus

            Trainer.merge_checkpoint(list((ckpt3_savedir / 'last').glob('*.ckpt')), ckpt3_savedir / 'merged2.pt')
            zero3_merged_state_dict2 = torch.load(ckpt3_savedir / 'merged2.pt')
            zero3_merged_state_dict = torch.load(ckpt3_savedir / 'merged.pt')
            assert_equal(zero3_merged_state_dict, zero3_merged_state_dict2)

    torch.distributed.barrier()

    # load from merged (from sharded)
    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '25',
        '--vars.dim', f'{dim}',
        '--gen_savedir', str(gen3_savedir),
        '--compute_config.plan_ngpus', f'{plan_ngpus}',
        '--compute_config.runtime_ngpus', f'{runtime_ngpus}',
        '--compute_config.zero_ngroups', f'{zero_ngroups}',
        '--compute_config.use_zero', '3',
        '--compute_config.pas_config.enable_random_replicated', 'True',
        '--checkpoint.save_dir', str(ckpt3_savedir),
        '--checkpoint.resume_from', str(ckpt3_savedir / 'merged.pt'),
        '--pas_policy', f'{policy}',
        '--checkpoint.save_type', 'deduped',
    ])
    trainer.run()

    torch.distributed.barrier()

    gen1_savedir = save_dir / 'gen1'
    ckpt1_savedir = save_dir / 'ckpt1'

    trainer = Trainer([
        '-f', config_path,
        '--max_train_steps', '25',
        '--vars.dim', f'{dim}',
        '--gen_savedir', str(gen1_savedir),
        '--compute_config.plan_ngpus', f'{plan_ngpus}',
        '--compute_config.runtime_ngpus', f'{runtime_ngpus}',
        '--compute_config.zero_ngroups', f'{zero_ngroups}',
        '--compute_config.use_zero', '1',
        '--compute_config.pas_config.enable_random_replicated', 'True',
        '--checkpoint.save_dir', str(ckpt1_savedir),
        '--pas_policy', f'{policy}',
        '--checkpoint.save_type', 'sharded',
    ])
    trainer.run()

    torch.distributed.barrier()

    if torch.distributed.get_rank() == 0:
        Trainer.merge_checkpoint(list((ckpt1_savedir / 'last').glob('*.ckpt')), ckpt1_savedir / 'merged.pt')
        zero1_merged_state_dict = torch.load(ckpt1_savedir / 'merged.pt')
        Trainer.merge_checkpoint(list((ckpt3_savedir / 'last').glob('*.ckpt')), ckpt3_savedir / 'merged.pt')
        zero3_merged_state_dict = torch.load(ckpt3_savedir / 'merged.pt')
        assert_equal(zero1_merged_state_dict['model'], zero3_merged_state_dict['model'])
        assert_equal(zero1_merged_state_dict['optimizer'], zero3_merged_state_dict['optimizer'])

    torch.distributed.barrier()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_trainer_zero3(tmp_path):
    launch_torchrun(2, trainer_zero3, 16, tmp_path, 1, 2)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_zero3_tp(tmp_path):
    launch_torchrun(4, trainer_zero3, 16, tmp_path, 2, 4)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_trainer_zero3_ngroup(tmp_path):
    # dim that needs padding
    launch_torchrun(4, trainer_zero3, 13, tmp_path, 1, 4)


def trainer_checkpointer_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'
    from nnscaler.cli import register_format

    load_triggered = False

    class TestFormat:
        name: str = 'test_format'
        suffix: str = '.testpt'

        @classmethod
        def save(cls, obj: Any, f: Path) -> None:
            obj['test'] = True
            return torch.save(obj, f)

        @classmethod
        def load(cls, f: str | Path, *, device='cpu') -> Any:
            x = torch.load(f, map_location=device, weights_only=False)
            assert x['test'] is True
            nonlocal load_triggered
            load_triggered = True
            return x

    register_format(TestFormat)

    # train 1 epcho in one time
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '1',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.format', 'test_format',
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
    ])
    trainer.run()

    files0 = list(ckpt_savedir.glob('**/*.testpt'))
    assert files0, 'No checkpoint files saved with custom format.'

    # train 1 epcho in one time
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--checkpoint.format', 'test_format',
        '--checkpoint.resume_from', 'last',
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
    ])
    trainer.run()
    assert load_triggered, 'Custom load function not triggered when resuming.'

    files1 = list(ckpt_savedir.glob('**/*.testpt'))
    assert len(files1) > len(files0), 'Checkpoint files not updated after resuming.'
    assert all(f in files1 for f in files0), 'Some checkpoint files missing after resuming.'
    assert files1, 'No checkpoint files saved with custom format.'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_custom_checkpointer(tmp_path):
    launch_torchrun(1, trainer_checkpointer_worker, tmp_path)


def trainer_pas_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    ckpt_savedir = save_dir / 'ckpt'

    from nnscaler.policies import pas_dp
    from nnscaler.cli import TrainerArgs
    called = False

    def custom_pas(graph, cfg):
        nonlocal called
        called = True
        return pas_dp(graph, cfg)

    args = TrainerArgs.from_cli([
        '-f', config_path,
        '--max_epochs', '1',
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
    ])
    args.pas_policy = custom_pas
    # train 1 epcho in one time
    trainer = Trainer(train_args=args)
    trainer.run()

    assert called, 'Custom PAS policy not called.'

    gen_savedir = save_dir / 'gen2'
    # train 1 epcho in one time
    trainer = Trainer([
        '-f', config_path,
        '--max_epochs', '2',
        '--pas-policy', 'nnscaler.policies.pas_dp',  # use full qualified name of pas policy
        '--gen_savedir', str(gen_savedir),
        '--checkpoint.save_dir', str(ckpt_savedir),
        '--compute_config.plan_ngpus', '1',
        '--compute_config.runtime_ngpus', '1',
    ])
    trainer.run()

    assert True


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_custom_pas(tmp_path):
    launch_torchrun(1, trainer_pas_worker, tmp_path)
