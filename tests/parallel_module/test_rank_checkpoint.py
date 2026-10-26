#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
import weakref
from unittest.mock import patch

import pytest
import torch

from nnscaler import (
    ComputeConfig,
    build_optimizer,
    deduped_state_dict,
    load_from_rank_state_dicts,
    load_merged_state_dict,
    merge_state_dicts,
    parallelize,
)
from nnscaler.flags import RuntimeFlag
from nnscaler.runtime import checkpoint
from nnscaler.runtime.module import ParallelModule

from .common import assert_equal, init_distributed
from .test_checkpoint import End2EndMLP, pipeline_dummy_data
from ..launch_torchrun import clone_to_cpu_recursively, launch_torchrun
from ..utils import mock_reducer_env


def _snapshot(model, optimizer):
    return clone_to_cpu_recursively([model.state_dict(), optimizer.state_dict()])


class CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 4)
        self.second = torch.nn.Linear(4, 4)
        self.second.weight = self.first.weight
        self.register_buffer('offset', torch.arange(4.))
        self.register_buffer('cache', torch.ones(4), persistent=False)

    def forward(self, x):
        return (self.second(self.first(x)) + self.offset + self.cache).sum()


def test_load_from_rank_state_dicts_cpu(tmp_path, monkeypatch):
    # build_optimizer sets these process-wide runtime flags; restore them for later tests.
    for flag in ('skip_reducer', 'skip_zero_grad'):
        monkeypatch.setattr(RuntimeFlag, flag, getattr(RuntimeFlag, flag))
    with mock_reducer_env(0, 1), patch('torch.distributed.barrier'), \
            patch('torch.distributed.broadcast_object_list'):
        source = parallelize(
            CheckpointModel(), {'x': torch.ones(2, 4)}, 'tp',
            ComputeConfig(1, 1, trace_strategy='cpu'),
            gen_savedir=tmp_path,
        )
        source_opt = build_optimizer(source, torch.optim.Adam, lr=0.01)
        # Adam state is absent for all but the first parameter.
        next(source.parameters()).grad = torch.ones_like(next(source.parameters()))
        source_opt.step()
        models, optimizers = [source.state_dict()], [source_opt.state_dict()]
        snapshot = copy.deepcopy([models, optimizers])
        full_model, full_opt = merge_state_dicts(models, optimizers)
        target = type(source)()
        target_opt = build_optimizer(target, torch.optim.Adam, lr=0.03)
        load_merged_state_dict(target, full_model, target_opt, full_opt, device='cpu')
        expected = _snapshot(target, target_opt)
        for param in target.parameters():
            param.data.zero_()
        target_opt.state.clear()

        refs = []
        gather_region = checkpoint._gather_region

        def gather_one(region, shards):
            out = gather_region(region, shards)
            assert list(out.shape) == [s.stop - s.start for s in region]
            refs.append(weakref.ref(out))
            return out

        with patch.object(checkpoint, '_gather_region', side_effect=gather_one):
            load_from_rank_state_dicts(target, models, target_opt, optimizers, device='cpu')
        # Regions are the target-local shards; nothing keeps them alive after loading.
        assert refs and all(ref() is None for ref in refs)
        assert_equal(expected, _snapshot(target, target_opt))
        assert_equal(snapshot, [models, optimizers])

        plain = CheckpointModel()
        load_from_rank_state_dicts(plain, models, device='cpu')
        assert_equal(full_model, plain.state_dict())
        missing = copy.deepcopy(models)
        missing[0].pop(next(iter(source.named_parameters()))[0])
        with pytest.raises(RuntimeError, match='Missing key'):
            load_from_rank_state_dicts(target, missing, device='cpu')
        with pytest.raises(ValueError, match='Duplicate rank'):
            load_from_rank_state_dicts(target, models * 2, device='cpu')
        empty_optimizers = copy.deepcopy(optimizers)
        empty_optimizers[0]['state'].clear()
        load_from_rank_state_dicts(target, models, target_opt, empty_optimizers, device='cpu')
        assert not target_opt.state


def test_gather_region():
    full = torch.arange(6 * 8, dtype=torch.float32).view(6, 8)
    box = (slice(0, 6), slice(0, 8))
    # Column shards (box shaped), row shards flattened in ZeRO-3 style, and a scalar.
    column_shards = [checkpoint._Shard((slice(0, 6), slice(c, c + 4)), None, full[:, c:c + 4].clone()) for c in (0, 4)]
    flat_shards = [checkpoint._Shard(box, (lo, hi), full.reshape(-1)[lo:hi].clone()) for lo, hi in ((0, 20), (20, 40), (40, 48))]
    for shards in (column_shards, flat_shards):
        for target in ((slice(1, 5), slice(2, 7)), (slice(0, 3), slice(0, 8)), (slice(3, 6), slice(5, 8))):
            assert_equal(checkpoint._gather_region(target, shards), full[target])
    scalar = torch.tensor(3.0)
    assert_equal(checkpoint._gather_region((), [checkpoint._Shard((), None, scalar)]), scalar)
    assert_equal(checkpoint._gather_region((), [checkpoint._Shard((), (0, 1), scalar.reshape(1))]), scalar)
    # Overlapping shards are allowed; a gap is not, even if the element count matches.
    overlapping = [checkpoint._Shard(box, None, full.clone())] + column_shards
    assert_equal(checkpoint._gather_region(box, overlapping), full)
    with pytest.raises(ValueError, match='Incomplete checkpoint shards'):
        checkpoint._gather_region(box, [column_shards[0], column_shards[0]])


class MixedCheckpointModel(torch.nn.Module):
    def __init__(self, config, savedir, instance_name):
        super().__init__()
        self.pre = torch.nn.Linear(4, 4)
        self.block = parallelize(
            torch.nn.Linear(4, 4), {'input': torch.randn(2, 4, device='cuda')},
            'tp', config, gen_savedir=savedir, instance_name=instance_name,
        )
        self.post = torch.nn.Linear(4, 1)

    def forward(self, x):
        return self.post(self.block(self.pre(x))).sum()


def _load_rank_checkpoint_worker(tmp_path, source_zero, target_zero, mixed):
    init_distributed()
    source_config = ComputeConfig(2, 4, use_zero=source_zero, use_end2end=not mixed)
    if mixed:
        source = MixedCheckpointModel(source_config, tmp_path, 'source').cuda()
        source_samples = torch.randn(2, 4, device='cuda')
    else:
        source = parallelize(
            End2EndMLP(), {'data': pipeline_dummy_data()}, 'tp', source_config,
            gen_savedir=tmp_path, instance_name='source',
        ).cuda()
        source_samples = [pipeline_dummy_data(), pipeline_dummy_data()]
    source_opt = build_optimizer(source, torch.optim.Adam, compute_config=source_config, lr=0.01)
    if mixed:
        source(source_samples).backward()
    else:
        source.train_step(source_samples)
    source_opt.step()
    source_model_state, source_optimizer_state = deduped_state_dict(source, source_opt)
    checkpoints = [None] * 4
    torch.distributed.all_gather_object(
        checkpoints,
        clone_to_cpu_recursively({'model': source_model_state, 'optimizer': source_optimizer_state}),
    )
    models = [state['model'] for state in reversed(checkpoints)]
    optimizers = [state['optimizer'] for state in reversed(checkpoints)]
    snapshot = copy.deepcopy([models, optimizers])
    full_model, full_opt = merge_state_dicts(models, optimizers)

    target_config = ComputeConfig(
        1 if mixed else 4, 4, use_zero=target_zero, use_end2end=not mixed,
        pas_config={} if mixed else {
            'pipeline_nmicros': 2, 'pipeline_nstages': 2, 'pipeline_scheduler': '1f1b',
        },
    )
    if mixed:
        target = MixedCheckpointModel(target_config, tmp_path, 'target').cuda()
        samples = torch.randn(2, 4, device='cuda')
    else:
        target = parallelize(
            End2EndMLP(), {'data': pipeline_dummy_data()}, 'hybrid', target_config,
            gen_savedir=tmp_path, instance_name='target',
        ).cuda()
        samples = [pipeline_dummy_data(), pipeline_dummy_data()]
    target_opt = build_optimizer(target, torch.optim.Adam, compute_config=target_config, lr=0.03)
    # The reference loader may share CPU step storage with the merged checkpoint.
    load_merged_state_dict(target, full_model, target_opt, copy.deepcopy(full_opt))
    expected = _snapshot(target, target_opt)

    def step():
        target_opt.zero_grad()
        if mixed:
            target(samples).backward()
        else:
            target.train_step(samples)
        target_opt.step()

    step()
    expected_next = _snapshot(target, target_opt)
    target_opt.zero_grad()
    for param in target.parameters():
        param.data.zero_()
    target_opt.state.clear()
    load_from_rank_state_dicts(target, models, target_opt, optimizers)
    assert_equal(expected, _snapshot(target, target_opt))
    assert_equal(snapshot, [models, optimizers])
    step()
    assert_equal(expected_next, _snapshot(target, target_opt))
    assert_equal(snapshot, [models, optimizers])
    if source_zero == 0 and not mixed:
        incomplete = copy.deepcopy(optimizers)
        for state in incomplete:
            if state[ParallelModule.EXTRA_STATE_KEY]['rank'] == 1:
                state['state'].clear()
        with pytest.raises(ValueError, match='Incomplete checkpoint shards'):
            load_from_rank_state_dicts(target, models, target_opt, incomplete)
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
# Cover ZeRO 0/1/3 on both sides and both directions of ZeRO on non-parallel parameters.
@pytest.mark.parametrize('source_zero,target_zero,mixed', [
    (0, 1, False), (3, 0, False),
    (0, 3, True), (1, 0, True),
])
def test_load_from_rank_state_dicts(tmp_path, source_zero, target_zero, mixed):
    launch_torchrun(4, _load_rank_checkpoint_worker, tmp_path, source_zero, target_zero, mixed)
