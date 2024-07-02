import os
from pathlib import Path
import math

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.fabric.utilities.cloud_io import _load as pl_load

import pytest
from unittest.mock import Mock, patch

import nnscaler
from nnscaler.parallel import ComputeConfig
from nnscaler.integration.lightning.pytorch import NnScalerStrategy, NnScalerPrecision
import nnscaler.runtime

from ....launch_torchrun import launch_torchrun
from ....utils import init_random
from ....parallel_module.common import assert_close, assert_equal
from .simple_datamodules import ClassifDataModule
from .simple_models import BoringModel, ClassificationModel, ClassificationModelWithLRScheduler


def fit_worker(tmp_path):
    dm = ClassifDataModule()
    model = ClassificationModel()
    compute_config=ComputeConfig(1, 1)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        accelerator="gpu", devices=1,
        gradient_clip_val=None,
        strategy=NnScalerStrategy(compute_config=compute_config, pas_policy='tp', gen_savedir=tmp_path),
        plugins=[NnScalerPrecision('32-true')]
    )
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=ClassifDataModule())


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_multi_gpu_model_only(tmp_path):
    launch_torchrun(1, fit_worker, tmp_path)


def ckpt_path_epoch_restored_worker(tmp_path):
    """Verify resuming from checkpoint runs the right number of epochs."""

    class TestModel(BoringModel):
        # Model that tracks epochs and batches seen
        num_epochs_end_seen = 0
        num_batches_seen = 0
        num_on_load_checkpoint_called = 0

        def on_train_epoch_end(self):
            self.num_epochs_end_seen += 1

        def on_train_batch_start(self, *_):
            self.num_batches_seen += 1

        def on_load_checkpoint(self, _):
            self.num_on_load_checkpoint_called += 1

    model = TestModel()
    max_epochs = 2
    compute_config=ComputeConfig(2, 2)
    trainer = Trainer(
        max_epochs=max_epochs,
        limit_train_batches=0.65,
        limit_val_batches=1,
        callbacks=ModelCheckpoint(dirpath=tmp_path, save_top_k=-1),
        default_root_dir=tmp_path,
        val_check_interval=1.0,
        enable_progress_bar=False,
        logger=False,
        enable_model_summary=False,
        strategy=NnScalerStrategy(compute_config=compute_config, pas_policy='tp', gen_savedir=tmp_path),
        plugins=[NnScalerPrecision('32-true')]
    )
    trainer.fit(model)

    assert model.num_epochs_end_seen == max_epochs
    assert model.num_batches_seen == trainer.num_training_batches * max_epochs == trainer.global_step
    assert model.num_on_load_checkpoint_called == 0

    checkpoints = sorted(list(set(Path(trainer.checkpoint_callback.dirpath).glob("*.ckpt"))))

    assert len(checkpoints) == max_epochs
    for ckpt in checkpoints:
        model = TestModel()
        state = pl_load(ckpt / '0.pt')
        # Resume training
        trainer = Trainer(
            default_root_dir=tmp_path, max_epochs=2, enable_progress_bar=False,
            strategy=NnScalerStrategy(
                compute_config=compute_config,
                pas_policy='tp',
                gen_savedir=tmp_path
            ),
            plugins=[NnScalerPrecision('32-true')]
        )
        trainer.fit(model, ckpt_path=ckpt)
        assert state["global_step"] + model.num_batches_seen == trainer.global_step
        assert model.num_on_load_checkpoint_called == 1


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_ckpt_path_epoch_restored(tmp_path):
    launch_torchrun(2, ckpt_path_epoch_restored_worker, tmp_path)


def trainer_accumulate_grad_batches_zero_grad(tmp_path, accumulate_grad_batches):
    with patch("torch.optim.SGD.zero_grad") as sgd_zero_grad:
        model = BoringModel()
        trainer = Trainer(
            num_nodes=1,
            devices=2,
            default_root_dir=tmp_path,
            num_sanity_val_steps=0,
            limit_train_batches=20,
            limit_val_batches=1,
            max_epochs=1,
            enable_model_summary=False,
            accumulate_grad_batches=accumulate_grad_batches,
            strategy=NnScalerStrategy(compute_config=ComputeConfig(1, 2), pas_policy='tp', gen_savedir=tmp_path),
            plugins=[NnScalerPrecision('32-true')]
        )
        assert trainer.accumulate_grad_batches == accumulate_grad_batches
        trainer.fit(model)
        assert sgd_zero_grad.call_count == math.ceil(trainer.limit_train_batches / accumulate_grad_batches)


@pytest.mark.parametrize("accumulate_grad_batches", [1, 2, 3])
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_trainer_accumulate_grad_batches_zero_grad(tmp_path, accumulate_grad_batches):
    launch_torchrun(2, trainer_accumulate_grad_batches_zero_grad, tmp_path, accumulate_grad_batches)


def correctnes_worker_nnscaler(tmp_path, gradient_clip_val, with_lr_scheduler,
    precision='32-true',
    with_tp=False, with_empty_scaler=False
):
    init_random()
    dm = ClassifDataModule()
    init_random()
    if with_lr_scheduler:
        model = ClassificationModelWithLRScheduler()
    else:
        model = ClassificationModel()
    if with_tp:
        compute_config=ComputeConfig(2, 4)
        policy = 'tp'
        devices = 4
    else:
        compute_config=ComputeConfig(1, 2)
        policy = 'dp'
        devices = 2
    scaler = None
    if with_empty_scaler or precision == '16-mixed':
        scaler = torch.cuda.amp.GradScaler(enabled=(precision == '16-mixed'))
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        accelerator="gpu", devices=devices,
        gradient_clip_val=gradient_clip_val,
        strategy=NnScalerStrategy(
            compute_config=compute_config, pas_policy=policy, gen_savedir=tmp_path,
            instance_name=policy
        ),
        plugins=[NnScalerPrecision(precision, scaler=scaler)]
    )
    trainer.fit(model, datamodule=dm)
    return model.update_history, model.nnscaler_pmodule.fullmap


def correctnes_worker_nnscaler_checkpoint(tmp_path, gradient_clip_val, with_lr_scheduler,
    precision='32-true',
    with_tp=False, with_empty_scaler=False
):
    init_random()
    dm = ClassifDataModule()
    init_random()
    if with_lr_scheduler:
        model = ClassificationModelWithLRScheduler()
        state_dict_type = 'sharded'
    else:
        model = ClassificationModel()
        state_dict_type = 'deduped'
    if with_tp:
        compute_config=ComputeConfig(2, 4)
        policy = 'tp'
        devices = 4
    else:
        compute_config=ComputeConfig(1, 2)
        policy = 'dp'
        devices = 2
    scaler = None
    if with_empty_scaler or precision == '16-mixed':
        scaler = torch.cuda.amp.GradScaler(enabled=(precision == '16-mixed'))
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=1,
        callbacks=[ModelCheckpoint(dirpath=tmp_path, save_top_k=1, save_last=True)],
        accelerator="gpu", devices=devices,
        gradient_clip_val=gradient_clip_val,
        strategy=NnScalerStrategy(
            compute_config=compute_config, pas_policy=policy, gen_savedir=tmp_path,
            instance_name=policy + '_resume',
            state_dict_type=state_dict_type
        ),
        plugins=[NnScalerPrecision(precision, scaler=scaler)]
    )
    trainer.fit(model, datamodule=dm)

    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        callbacks=[ModelCheckpoint(dirpath=tmp_path, save_top_k=1, save_last=True)],
        accelerator="gpu", devices=devices,
        gradient_clip_val=gradient_clip_val,
        strategy=NnScalerStrategy(
            compute_config=compute_config, pas_policy=policy, gen_savedir=tmp_path,
            instance_name=policy + '_resume',
            state_dict_type=state_dict_type
        ),
        plugins=[NnScalerPrecision(precision, scaler=scaler)]
    )
    trainer.fit(model, datamodule=dm, ckpt_path='last')
    return model.update_history, model.nnscaler_pmodule.fullmap


def correctnes_worker_ddp(tmp_path, gradient_clip_val, with_lr_scheduler, precision='32-true'):
    init_random()
    dm = ClassifDataModule()
    init_random()
    if with_lr_scheduler:
        model = ClassificationModelWithLRScheduler()
    else:
        model = ClassificationModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        precision=precision,
        max_epochs=2,
        accelerator="gpu", devices=2,
        gradient_clip_val=gradient_clip_val,
        strategy='ddp',
    )
    trainer.fit(model, datamodule=dm)
    return model.update_history


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize("gradient_clip_val", [None, 0.1])  # 0.1 is chosen to only clip the first update
@pytest.mark.parametrize("with_lr_scheduler", [False, True])
def test_correctness(tmp_path, gradient_clip_val, with_lr_scheduler):
    def _merge_results(returns):
        results = [returns[i][0] for i in range(len(returns))]
        fullmaps = [returns[i][1] for i in range(len(returns))]
        weight_results = []
        grad_results = []
        for i in range(len(results[0])):
            weight_results.append(
                nnscaler.runtime.module.ParallelModule.merge_state_dicts(
                    fullmaps,
                    [result[i][1] for result in results]
                )[0]
            )
            grad_results.append(
                nnscaler.runtime.module.ParallelModule.merge_state_dicts(
                    fullmaps,
                    [result[i][0] for result in results]
                )[0]
            )
        return weight_results, grad_results

    # Test 16-mixed with and without gradient clipping
    # when gradient clipping is on, the following check will fail
    # TODO: fix the test when gradient clipping is on
    if not gradient_clip_val:
        ddp_results = launch_torchrun(2, correctnes_worker_ddp, tmp_path, gradient_clip_val, with_lr_scheduler, '16-mixed')

        nnscaler_returns = launch_torchrun(2, correctnes_worker_nnscaler, tmp_path, gradient_clip_val, with_lr_scheduler, '16-mixed', False, True)
        nnscaler_merged_weight_results_fp16, nnscaler_merged_grad_results_fp16 = _merge_results(nnscaler_returns)

        for i in range(len(ddp_results[0])):
            assert_close(nnscaler_merged_weight_results_fp16[i], ddp_results[0][i][1])
            assert_close(nnscaler_merged_grad_results_fp16[i], ddp_results[0][i][0])
            assert_equal(ddp_results[1][i], ddp_results[0][i])

    nnscaler_returns_ckpt = launch_torchrun(2, correctnes_worker_nnscaler_checkpoint, tmp_path, gradient_clip_val, with_lr_scheduler)
    nnscaler_merged_weight_results_ckpt, nnscaler_merged_grad_results_ckpt = _merge_results(nnscaler_returns_ckpt)

    nnscaler_returns = launch_torchrun(2, correctnes_worker_nnscaler, tmp_path, gradient_clip_val, with_lr_scheduler)
    nnscaler_merged_weight_results, nnscaler_merged_grad_results = _merge_results(nnscaler_returns)

    nnscaler_returns = launch_torchrun(2, correctnes_worker_nnscaler, tmp_path, gradient_clip_val, with_lr_scheduler, '32-true', False, True)
    nnscaler_merged_weight_results_scaler, nnscaler_merged_grad_results_scaler = _merge_results(nnscaler_returns)

    assert len(nnscaler_merged_weight_results) == len(nnscaler_merged_weight_results_ckpt)
    assert len(nnscaler_merged_weight_results) == len(nnscaler_merged_weight_results_scaler)

    assert len(nnscaler_merged_grad_results) == len(nnscaler_merged_grad_results_ckpt)
    assert len(nnscaler_merged_grad_results) == len(nnscaler_merged_grad_results_scaler)

    for i in range(len(nnscaler_merged_weight_results_scaler)):
        assert_equal(nnscaler_merged_weight_results[i], nnscaler_merged_weight_results_scaler[i])
        assert_equal(nnscaler_merged_weight_results[i], nnscaler_merged_weight_results_ckpt[i])
        assert_equal(nnscaler_merged_grad_results[i], nnscaler_merged_grad_results_scaler[i])
        assert_equal(nnscaler_merged_grad_results[i], nnscaler_merged_grad_results_ckpt[i])

    ddp_results = launch_torchrun(2, correctnes_worker_ddp, tmp_path, gradient_clip_val, with_lr_scheduler)
    for i in range(len(ddp_results[0])):
        assert_close(nnscaler_merged_weight_results[i], ddp_results[0][i][1])
        assert_close(nnscaler_merged_grad_results[i], ddp_results[0][i][0])
        assert_equal(ddp_results[1][i], ddp_results[0][i])

    if torch.cuda.device_count() >= 4:
        nnscaler_returns = launch_torchrun(4, correctnes_worker_nnscaler, tmp_path, gradient_clip_val, with_lr_scheduler, '32-true', True)
        nnscaler_merged_weight_results, nnscaler_merged_grad_results = _merge_results(nnscaler_returns)

        for i in range(len(ddp_results[0])):
            assert_close(nnscaler_merged_weight_results[i], ddp_results[0][i][1])
            assert_close(nnscaler_merged_grad_results[i], ddp_results[0][i][0])
