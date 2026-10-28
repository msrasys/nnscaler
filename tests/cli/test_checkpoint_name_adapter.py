# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Retraced sharded/deduped restore in fresh distributed worker processes."""
import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from nnscaler.cli.trainer import Trainer
from nnscaler.policies import OpPartition, OpPlan, get_pas_ops
from nnscaler.runtime.module import ExtraState, ParallelModule
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
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
    max_error = 0.0
    for name, meta in expected_meta.param_area_map.items():
        actual_name = actual_names[meta.orig_name]
        if not meta.shape:
            assert meta.slicers == actual_meta.param_area_map[actual_name].slicers == ()
        assert replace(meta, tid=0) == replace(actual_meta.param_area_map[actual_name], tid=0)
        if expected[name].numel():
            max_error = max(max_error, float((expected[name].double() - actual[actual_name].double()).abs().max()))
        assert_equal(expected[name], actual[actual_name])
    return max_error


def _assert_optimizer_state_equal(expected, actual):
    # Extra state keeps the current model's extra_op argument after re-tracing.
    assert_equal(expected['state'], actual['state'])
    assert_equal(expected['param_groups'], actual['param_groups'])
    errors = [float((value.double() - actual['state'][index][key].double()).abs().max())
              for index, state in expected['state'].items()
              for key, value in state.items() if isinstance(value, torch.Tensor) and value.numel()]
    return max(errors, default=0.0)


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


def _checkpoint_restart_worker(savedir, plan_ngpus, use_zero, save_type, phase, legacy_scalars):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    savedir = Path(savedir)
    resumed = phase == 'resumed'
    resume_path = savedir / ('legacy' if legacy_scalars else 'old/last') if resumed else None
    trainer = _train(savedir, plan_ngpus, use_zero, save_type, phase,
                     1 if phase == 'old' else 2, extra_op=resumed, resume_from=resume_path)
    result = {'model': trainer.model.state_dict(), 'optimizer': trainer.optimizer.state_dict(),
              'steps': trainer.train_status.finished_train_steps}
    if resumed:
        result.update(resumed_model=trainer.resumed_model_state,
                      resumed_optimizer=trainer.resumed_optimizer_state,
                      resumed_steps=trainer.resumed_train_steps)
    torch.save(clone_to_cpu_recursively(result), savedir / f'{phase}-rank{trainer.rank}.pt')
    torch.distributed.barrier()
    if phase == 'old' and trainer.rank == 0:
        checkpointer = trainer.checkpointer
        paths = checkpointer.list_checkpoints(savedir / 'old/last')
        merged_file = savedir / 'merged.ckpt'
        Trainer.merge_checkpoint(paths, merged_file, checkpointer=checkpointer)
        merged = checkpointer.load(merged_file)
        eager = RetracedModel()
        eager.load_state_dict(merged['model'])
        optimizer = torch.optim.Adam(eager.parameters())
        optimizer.load_state_dict(merged['optimizer'])
        assert merged['model']['scale'].shape == merged['model']['scalar_offset'].shape == torch.Size([])
        assert optimizer.state[eager.scale]['exp_avg'].shape == torch.Size([])
        if legacy_scalars:
            legacy_dir = savedir / 'legacy'
            legacy_dir.mkdir()
            for path in paths:
                state = checkpointer.load(path)
                metadata = state['model'][ParallelModule.EXTRA_STATE_KEY]
                for meta in metadata['param_area_map'].values():
                    if not meta['shape']:
                        meta['slicers'] = Ellipsis
                checkpointer.save_for_rank(state, legacy_dir, metadata['rank'])
            checkpointer.flush()
            Trainer.merge_checkpoint(checkpointer.list_checkpoints(legacy_dir),
                                     savedir / 'legacy_merged.ckpt', checkpointer=checkpointer)
            legacy_merged = checkpointer.load(savedir / 'legacy_merged.ckpt')
            assert_equal(legacy_merged['model'], merged['model'])
            assert_equal(legacy_merged['optimizer'], merged['optimizer'])
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('plan_ngpus,use_zero,legacy_scalars', [(1, 0, False), (2, 1, False), (1, 3, False), (2, 1, True)])
@pytest.mark.parametrize('save_type', ['sharded', 'deduped'])
def test_retraced_checkpoint_survives_process_restart(tmp_path, plan_ngpus, use_zero, legacy_scalars, save_type):
    # Each call creates and joins a fresh pair of worker processes. In
    # particular no initialized model or optimizer survives old -> resumed.
    for phase in ['reference', 'old', 'resumed']:
        launch_torchrun(2, _checkpoint_restart_worker, str(tmp_path), plan_ngpus,
                       use_zero, save_type, phase, legacy_scalars)
    for rank in range(2):
        reference, old, resumed = [torch.load(tmp_path / f'{phase}-rank{rank}.pt',
                                             map_location='cpu', weights_only=False)
                                   for phase in ['reference', 'old', 'resumed']]
        assert resumed['resumed_steps'] == old['steps'] == 1
        assert resumed['steps'] == reference['steps'] == 2
        assert set(old['model']) != set(resumed['model']), 'Re-tracing must change generated names'
        errors = {
            'restore_model_max_abs': _assert_model_state_equal(old['model'], resumed['resumed_model']),
            'restore_optimizer_max_abs': _assert_optimizer_state_equal(old['optimizer'], resumed['resumed_optimizer']),
            'continued_model_max_abs': _assert_model_state_equal(reference['model'], resumed['model']),
            'continued_optimizer_max_abs': _assert_optimizer_state_equal(reference['optimizer'], resumed['optimizer']),
        }
        print(json.dumps(dict(rank=rank, plan_ngpus=plan_ngpus, use_zero=use_zero,
                              save_type=save_type, legacy_scalars=legacy_scalars,
                              precision='float32', tf32=False, seed=0,
                              atol=0.0, rtol=0.0, **errors)), flush=True)

