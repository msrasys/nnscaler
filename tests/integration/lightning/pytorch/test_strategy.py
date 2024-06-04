import os
from pathlib import Path
import math

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Timer
from lightning.fabric.utilities.cloud_io import _load as pl_load

import pytest
from unittest.mock import Mock, patch

from nnscaler.parallel import ComputeConfig
from nnscaler.integration.lightning.pytorch import NnScalerStrategy, NnScalerPrecision

from ....launch_torchrun import launch_torchrun
from .simple_datamodules import ClassifDataModule
from .simple_models import BoringModel, ClassificationModel


def fit_worker(tmp_path):
    dm = ClassifDataModule()
    model = ClassificationModel()
    compute_config=ComputeConfig(2, 2)
    trainer = Trainer(
        default_root_dir=tmp_path,
        max_epochs=2,
        accelerator="gpu", devices=2,
        gradient_clip_val=2.0,
        strategy=NnScalerStrategy(compute_config=compute_config, pas_policy='tp', gen_savedir=tmp_path),
        plugins=[NnScalerPrecision('32-true')]
    )
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=ClassifDataModule())


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_multi_gpu_model_only(tmp_path):
    launch_torchrun(2, fit_worker, tmp_path)


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
