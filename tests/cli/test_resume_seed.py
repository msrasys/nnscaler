import os
import pytest
import torch
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import *


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
def test_resume_seed():
    _set_envs({
        # required by deterministic
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',

        # fake torchrun environment, check https://pytorch.org/docs/stable/elastic/run.html#environment-variables
        'LOCAL_RANK': 0,
        'RANK': 0,
        'GROUP_RANK': 0,
        'LOCAL_WORLD_SIZE': 1,
        'WORLD_SIZE': 1,
        'MASTER_ADDR': 'localhost',
        'MASTER_PORT': 29470,
        'TORCHELASTIC_RUN_ID': 'UT',
    })

    torch.use_deterministic_algorithms(True)

    # compile separately because run multiple trainers in one process will confuse `gen_reuse`
    _compile()

    _test_resume_seed(steps_per_epoch=100, max_steps=20, resume_at=10)

    _test_resume_seed(steps_per_epoch=5, max_steps=20, resume_at=10)

    _restore_envs()


def _test_resume_seed(steps_per_epoch, max_steps, resume_at):
    # no resume
    model_1 = _train(steps_per_epoch, max_train_steps=max_steps, resume_from=None)
    weight_1 = next(model_1.parameters()).data

    # resume
    _train(steps_per_epoch, max_train_steps=resume_at, resume_from=None)
    model_2 = _train(steps_per_epoch, max_train_steps=max_steps, resume_from='last')
    weight_2 = next(model_2.parameters()).data

    assert torch.equal(weight_1, weight_2)

    ## resume without resuming seeds
    _train(steps_per_epoch, max_train_steps=resume_at, resume_from=None)
    _remove_rng_states()
    model_3 = _train(steps_per_epoch, max_train_steps=max_steps, resume_from='last')
    weight_3 = next(model_3.parameters()).data

    assert not torch.equal(weight_1, weight_3)


def _compile():
    trainer_args = TrainerArgs(
        compute_config=ComputeConfig(plan_ngpus=1, runtime_ngpus=1, use_end2end=True),
        gen_reuse='override',
        run_mode='compile',
        model=ModelConfig(type=Model),
        optimizer=OptimizerConfig(type=torch.optim.AdamW),
        dataset=DatasetConfig(type=RandomDataset, train_args={'length': 100}),
        max_train_steps=1,
        enable_progress_bar=False,
        seed=0,
    )
    trainer = Trainer(train_args=trainer_args)
    trainer.run()


def _train(steps_per_epoch, max_train_steps, resume_from):
    trainer_args = TrainerArgs(
        compute_config=ComputeConfig(plan_ngpus=1, runtime_ngpus=1, use_end2end=True),
        model=ModelConfig(type=Model),
        optimizer=OptimizerConfig(type=torch.optim.AdamW),
        dataset=DatasetConfig(type=RandomDataset, train_args={'length': steps_per_epoch}),
        checkpoint=CheckpointConfig(resume_from=resume_from),
        max_train_steps=max_train_steps,
        enable_progress_bar=False,
        seed=0,
    )
    trainer = Trainer(train_args=trainer_args)
    trainer.run()
    return trainer.model


def _remove_rng_states():
    ckpt_path = 'checkpoints/last/0.ckpt'
    ckpt = torch.load(ckpt_path, weights_only=False)
    ckpt['rng_states'] = None
    torch.save(ckpt, ckpt_path)


_backup_envs = {}

def _set_envs(envs):
    _backup_envs.clear()
    for key, value in envs.items():
        _backup_envs[key] = os.environ.get(key, None)
        os.environ[key] = str(value)

def _restore_envs():
    for key, value in _backup_envs.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 10)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, data):
        x = data['x']
        x = self.linear(x)
        x = self.dropout(x)
        return torch.nn.functional.cross_entropy(x, data['y'])


class RandomDataset:
    def __init__(self, length):
        self.length = length

    def __getitem__(self, i):
        return {
            'x': torch.rand(100),
            'y': torch.randint(10, tuple()),
        }

    def __len__(self):
        return self.length
