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
from nnscaler.runtime.module import ExtraState, ParallelModule, Zero3AttrMeta

from ..launch_torchrun import launch_torchrun
from ..parallel_module.common import assert_equal


class RetracedModel(nn.Module):
    def __init__(self, extra_op=False):
        super().__init__()
        self.extra_op = extra_op
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('offset', torch.ones(4))
        self.register_buffer('scalar_offset', torch.tensor(0.5))

    def forward(self, data):
        x = data['data']
        # this can change the compute graph
        # which further affects the ids of parameters in tracing.
        if self.extra_op:
            x = x + 0.0
        output = self.fc2(nn.functional.relu(self.fc1(x + self.offset)))
        output = output * self.scale + self.scalar_offset
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
        if not meta.shape:
            assert meta.slicers == actual_meta.param_area_map[actual_name].slicers == ()
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
    if use_zero == 3:
        metadata = ExtraState(**old_state[ParallelModule.EXTRA_STATE_KEY])
        assert metadata.zero3_param_metadata
        assert all(isinstance(meta, Zero3AttrMeta) for meta in metadata.zero3_param_metadata.values())
        if interrupted.rank == 0:
            checkpointer = interrupted.checkpointer
            merged_file = savedir / 'zero3_model.ckpt'
            Trainer.merge_checkpoint(
                checkpointer.list_checkpoints(savedir / 'old' / 'last'), merged_file,
                model_only=True, checkpointer=checkpointer,
            )
            merged = checkpointer.load(merged_file)
            eager = RetracedModel()
            eager.load_state_dict(merged['model'])
            for name, meta in metadata.param_area_map.items():
                expected = merged['model'][meta.orig_name]
                actual = old_state[name].cpu()
                zero3_meta = metadata.zero3_param_metadata.get(name)
                if zero3_meta is not None:
                    expected = expected.view(-1)[zero3_meta.start:zero3_meta.end]
                    actual = actual[:zero3_meta.end - zero3_meta.start]
                assert_equal(expected, actual)
        torch.distributed.barrier()

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


def _scalar_checkpoint_worker(savedir, save_type):
    savedir = Path(savedir)
    trainer = _train(savedir, 2, 1, save_type, 'original', 1)
    old_state = copy.deepcopy(trainer.model.state_dict())
    old_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    legacy_dir = savedir / 'legacy'

    if trainer.rank == 0:
        checkpointer = trainer.checkpointer
        checkpoint_files = checkpointer.list_checkpoints(savedir / 'original' / 'last')
        assert len(checkpoint_files) == 2
        merged_file = savedir / 'merged.ckpt'
        Trainer.merge_checkpoint(checkpoint_files, merged_file, checkpointer=checkpointer)
        merged = checkpointer.load(merged_file)
        eager = RetracedModel()
        eager.load_state_dict(merged['model'])
        optimizer = torch.optim.Adam(eager.parameters())
        optimizer.load_state_dict(merged['optimizer'])
        for name in ('scale', 'scalar_offset'):
            local_name = next(
                key for key, meta in trainer.model.fullmap.items() if meta.orig_name == name
            )
            assert merged['model'][name].shape == torch.Size([])
            assert_equal(merged['model'][name], getattr(trainer.model, local_name).detach().cpu())
        for key in ('exp_avg', 'exp_avg_sq'):
            assert optimizer.state[eager.scale][key].shape == torch.Size([])

        legacy_dir.mkdir()
        for path in checkpoint_files:
            checkpoint = checkpointer.load(path)
            metadata = checkpoint['model'][ParallelModule.EXTRA_STATE_KEY]
            for meta in metadata['param_area_map'].values():
                if not meta['shape']:
                    meta['slicers'] = Ellipsis
            checkpointer.save_for_rank(checkpoint, legacy_dir, metadata['rank'])
        checkpointer.flush()
        legacy_merged_file = savedir / 'legacy_merged.ckpt'
        Trainer.merge_checkpoint(
            checkpointer.list_checkpoints(legacy_dir), legacy_merged_file, checkpointer=checkpointer,
        )
        legacy_merged = checkpointer.load(legacy_merged_file)
        assert_equal(legacy_merged['model'], merged['model'])
        assert_equal(legacy_merged['optimizer'], merged['optimizer'])

    torch.distributed.barrier()
    resumed = _train(
        savedir, 2, 1, save_type, 'legacy_resume', 2, extra_op=True, resume_from=legacy_dir,
    )
    _assert_model_state_equal(old_state, resumed.resumed_model_state)
    _assert_optimizer_state_equal(old_optimizer_state, resumed.resumed_optimizer_state)
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('save_type', ['sharded', 'deduped'])
def test_load_and_merge_scalar_checkpoint(tmp_path, save_type):
    launch_torchrun(2, _scalar_checkpoint_worker, str(tmp_path), save_type)
