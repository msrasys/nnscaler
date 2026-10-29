#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from pathlib import Path

import pytest
import torch

from nnscaler.cli.trainer import Trainer
from nnscaler.policies import OpPlan, get_layer_index, get_pas_ops

from tests.cli.common import MLP
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_equal
from tests.parallel_module.test_gencode import _gencode_contains


_LAYERS_PER_STAGE = 8


def _pipeline_policy(graph, *, cpu_offload: bool, recompute: bool = False):
    last_stage_id = 0
    for node in get_pas_ops(graph):
        offload_id = -1
        recompute_id = -1
        if torch.nn.modules.linear.Linear in node.module_class_chain:
            layer_idx = get_layer_index(node.fqn)
            last_stage_id = layer_idx // _LAYERS_PER_STAGE
            if cpu_offload:
                offload_id = layer_idx
            if recompute:
                recompute_id = layer_idx
        yield OpPlan(
            node,
            stage_id=last_stage_id,
            offload_id=offload_id,
            recompute_id=recompute_id,
            partition=None,
        )


def pipeline_zero3_policy(graph, cfg):
    return _pipeline_policy(graph, cpu_offload=False)


def pipeline_zero3_cpu_offload_policy(graph, cfg):
    return _pipeline_policy(graph, cpu_offload=True)


def pipeline_zero3_recompute_offload_policy(graph, cfg):
    """Recompute every linear layer and offload the inputs its checkpoint keeps."""
    return _pipeline_policy(graph, cpu_offload=True, recompute=True)


def _trainer_worker(save_dir, run_name, policy, offload_env):
    """Train 10 steps; ``offload_env`` (``CPU_OFFLOADING_*``) selects the offload configuration."""
    import os

    import nnscaler.runtime.cpu_offloading as cpu_offloading
    from nnscaler.runtime.cpu_offload import OffloadManager, offload_stats

    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name('trainer_args_pipeline.yaml').resolve()
    gen_savedir = save_dir / run_name / 'gen'
    checkpoint_dir = save_dir / run_name / 'checkpoints'
    merged_checkpoint = save_dir / f'{run_name}.pt'
    instance_name = f'instance_{run_name}'

    offloaded_tensors = 0
    group_ids = set()
    original_context = cpu_offloading.CPUOffloadContext

    class RecordingContext(original_context):
        def __init__(self, module, group_id=None, manager=None):
            super().__init__(module, group_id=group_id, manager=manager)
            group_ids.add(group_id)

        def _pack(self, tensor):
            nonlocal offloaded_tensors
            packed = super()._pack(tensor)
            if isinstance(packed, cpu_offloading._OffloadedTensor):
                offloaded_tensors += 1
            return packed

    saved_env = {key: os.environ.get(key) for key in (offload_env or {})}
    if offload_env is not None:
        cpu_offloading.CPUOffloadContext = RecordingContext
        os.environ.update(offload_env)
    OffloadManager.reset()

    try:
        trainer = Trainer([
            '-f', str(config_path),
            '--instance_name', instance_name,
            '--max_train_steps', '10',
            '--gen_savedir', str(gen_savedir),
            '--checkpoint.save_dir', str(checkpoint_dir),
            '--checkpoint.save_type', 'sharded',
            '--compute_config.use_zero', '3',
            '--pas_policy', policy,
        ])
        trainer.run()
        stats = offload_stats(reset=False)
    finally:
        cpu_offloading.CPUOffloadContext = original_context
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        OffloadManager.reset()

    assert trainer.model.use_scheduler
    assert trainer.model.nmicros_per_scheduler_step == 4
    assert trainer.model.compute_config.use_zero == 3
    assert trainer.model.compute_config.pas_config['pipeline_nstages'] == 2
    assert all(reducer.zero3 for reducer in trainer.model.reducers)

    total_offloaded = torch.tensor([offloaded_tensors], device='cuda')
    torch.distributed.all_reduce(total_offloaded)
    if offload_env is not None:
        # with the default ``min_backward_distance`` the last stage offloads
        # nothing (its backward follows each forward right away)
        assert total_offloaded.item() > 0 and None not in group_ids and len(group_ids) > 1
        if offload_env.get('CPU_OFFLOADING_MIN_BACKWARD_DISTANCE') == '0':
            assert offloaded_tensors > 0
        assert stats['pending_contexts'] == 0 and stats['pending_bytes'] == 0
        assert _gencode_contains(
            gen_savedir,
            MLP,
            trainer.rank,
            r'with self\.cpu_offloading_hooks\(\d+\):',
            instance_name=instance_name,
        )
    else:
        assert total_offloaded.item() == 0

    if trainer.rank == 0:
        checkpoints = list((checkpoint_dir / 'last').glob('*.ckpt'))
        assert len(checkpoints) == 4
        Trainer.merge_checkpoint(checkpoints, merged_checkpoint)

    torch.distributed.barrier()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 4,
    reason='lack of gpu devices',
)
def test_trainer_pipeline_zero3_cpu_offload(tmp_path):
    module_name = 'tests.cli.test_trainer_pipeline_zero3_offload'
    launch_torchrun(
        4,
        _trainer_worker,
        tmp_path,
        'baseline',
        f'{module_name}.pipeline_zero3_policy',
        None,
    )

    baseline = torch.load(tmp_path / 'baseline.pt', weights_only=False)
    # the model is tiny: offload everything (no size threshold) and exercise the
    # transport knobs that change how bytes move
    base_env = {'CPU_OFFLOADING_MIN_BYTES': '0', 'CPU_OFFLOADING_NUMA_MEMBIND': '0'}
    for run_name, env, policy in (
        ('cpu_offload', {}, 'pipeline_zero3_cpu_offload_policy'),
        ('cpu_offload_chunked_window', {'CPU_OFFLOADING_MIN_BACKWARD_DISTANCE': '0',
                                        'CPU_OFFLOADING_COPY_CHUNK_MB': '0.001',
                                        'CPU_OFFLOADING_MAX_INFLIGHT_COPIES': '1',
                                        'CPU_OFFLOADING_PREFETCH_GROUPS': '0',
                                        'CPU_OFFLOADING_PREFETCH_MB': '0.01'},
         'pipeline_zero3_cpu_offload_policy'),
        ('recompute_offload', {'CPU_OFFLOADING_MIN_BACKWARD_DISTANCE': '0'},
         'pipeline_zero3_recompute_offload_policy'),
    ):
        launch_torchrun(
            4,
            _trainer_worker,
            tmp_path,
            run_name,
            f'{module_name}.{policy}',
            {**base_env, **env},
        )
        cpu_offload = torch.load(tmp_path / f'{run_name}.pt', weights_only=False)
        assert_equal(cpu_offload['model'], baseline['model'])
        assert_equal(cpu_offload['optimizer'], baseline['optimizer'])
