#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from nnscaler.cli.trainer import Trainer
from nnscaler.policies import OpPartition, OpPlan, get_pas_ops
from nnscaler.runtime.module import ExtraState, ParallelModule

from ..launch_torchrun import launch_torchrun
from ..parallel_module.common import assert_equal


class RetracedModel(nn.Module):
    def __init__(self, extra_op=False):
        super().__init__()
        self.extra_op = extra_op
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.register_buffer('offset', torch.ones(4))

    def forward(self, data):
        x = data['data']
        # this can change the compute graph
        # which further affects the ids of parameters in tracing.
        if self.extra_op:
            x = x + 0.0
        output = self.fc2(nn.functional.relu(self.fc1(x + self.offset)))
        return nn.functional.mse_loss(output, data['target'])


def stable_tp(graph, _config):
    # The default TP policy is random; an extra op can change its shard choices.
    for node in get_pas_ops(graph):
        partition = OpPartition(input=1, dim=0) if node.name == 'linear' else None
        yield OpPlan(node, partition=partition)


def capture_resume_state(trainer, checkpoint):
    trainer.resumed_model_state = copy.deepcopy(trainer.model.state_dict())
    trainer.resumed_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    trainer.resumed_train_steps = trainer.train_status.finished_train_steps


def _assert_model_state_equal(expected, actual):
    expected_meta = ExtraState(**expected[ParallelModule.EXTRA_STATE_KEY])
    actual_meta = ExtraState(**actual[ParallelModule.EXTRA_STATE_KEY])
    actual_names = {meta.orig_name: name for name, meta in actual_meta.param_area_map.items()}
    assert len(actual_names) == len(actual_meta.param_area_map) == len(expected_meta.param_area_map)
    for name, meta in expected_meta.param_area_map.items():
        actual_name = actual_names[meta.orig_name]
        assert replace(meta, tid=0) == replace(actual_meta.param_area_map[actual_name], tid=0)
        assert_equal(expected[name], actual[actual_name])


def _assert_optimizer_state_equal(expected, actual):
    # Extra state keeps the current model's extra_op argument after re-tracing.
    assert_equal(expected['state'], actual['state'])
    assert_equal(expected['param_groups'], actual['param_groups'])


def _train(savedir, plan_ngpus, use_zero, save_type, instance_name, max_steps, *, extra_op=False, resume_from=None):
    argv = [
        '--model.type', 'tests.cli.test_checkpoint_name_adapter.RetracedModel',
        '--model.args.extra_op', str(extra_op).lower(),
        '--dataset.type', 'tests.cli.common.SimpleDataset',
        '--dataset.train_args.dim', '4',
        '--dataset.train_args.size', '16',
        '--optimizer.type', 'torch.optim.Adam',
        '--optimizer.args.lr', '0.01',
        '--pas_policy', 'tests.cli.test_checkpoint_name_adapter.stable_tp',
        '--compute_config.plan_ngpus', str(plan_ngpus),
        '--compute_config.runtime_ngpus', '2',
        '--compute_config.use_zero', str(use_zero),
        '--compute_config.use_end2end', 'true',
        '--gen_savedir', str(savedir / 'gen'),
        '--instance_name', instance_name,
        '--micro_batch_size', '2',
        '--max_train_steps', str(max_steps),
        '--enable_progress_bar', 'false',
        '--seed', '0',
        '--checkpoint.save_dir', str(savedir / instance_name),
        '--checkpoint.save_type', save_type,
        '--checkpoint.every_n_train_steps', '1',
        '--hook.after_load_checkpoint', 'tests.cli.test_checkpoint_name_adapter.capture_resume_state',
    ]
    if resume_from is not None:
        argv.extend([
            '--checkpoint.resume_from.checkpoint', str(resume_from),
            '--checkpoint.resume_from.with_merged', 'false',
        ])
    trainer = Trainer(argv)
    trainer.run()
    return trainer


def _retrace_worker(savedir, plan_ngpus, use_zero, save_type):
    savedir = Path(savedir)
    # train 2 steps in total
    reference = _train(savedir, plan_ngpus, use_zero, save_type, 'reference', 2)
    # firstly train 1 step and will resume it later
    interrupted = _train(savedir, plan_ngpus, use_zero, save_type, 'old', 1)
    old_state = copy.deepcopy(interrupted.model.state_dict())
    old_optimizer_state = copy.deepcopy(interrupted.optimizer.state_dict())

    resumed = _train(
        savedir, plan_ngpus, use_zero, save_type, 'new', 2,
        extra_op=True, resume_from=savedir / 'old' / 'last',
    )
    assert set(interrupted.model._fullmap) != set(resumed.model._fullmap), \
        'Re-tracing must actually change generated names'
    assert resumed.resumed_train_steps == 1
    _assert_model_state_equal(old_state, resumed.resumed_model_state)
    _assert_optimizer_state_equal(old_optimizer_state, resumed.resumed_optimizer_state)
    assert resumed.train_status.finished_train_steps == 2
    _assert_model_state_equal(reference.model.state_dict(), resumed.model.state_dict())
    _assert_optimizer_state_equal(reference.optimizer.state_dict(), resumed.optimizer.state_dict())
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
# Cases record plan GPU count and ZeRO level.
@pytest.mark.parametrize('plan_ngpus,use_zero', [(1, 0), (2, 1), (1, 3)])
@pytest.mark.parametrize('save_type', ['sharded', 'deduped'])
def test_load_retraced_sharded_and_deduped(tmp_path, plan_ngpus, use_zero, save_type):
    launch_torchrun(2, _retrace_worker, str(tmp_path), plan_ngpus, use_zero, save_type)
