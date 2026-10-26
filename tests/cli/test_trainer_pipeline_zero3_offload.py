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


def _pipeline_policy(graph, *, cpu_offload: bool):
    last_stage_id = 0
    for node in get_pas_ops(graph):
        offload_id = -1
        if torch.nn.modules.linear.Linear in node.module_class_chain:
            layer_idx = get_layer_index(node.fqn)
            last_stage_id = layer_idx // _LAYERS_PER_STAGE
            if cpu_offload:
                offload_id = layer_idx
        yield OpPlan(
            node,
            stage_id=last_stage_id,
            offload_id=offload_id,
            partition=None,
        )


def pipeline_zero3_policy(graph, cfg):
    return _pipeline_policy(graph, cpu_offload=False)


def pipeline_zero3_cpu_offload_policy(graph, cfg):
    return _pipeline_policy(graph, cpu_offload=True)


def pipeline_zero3_combined_policy(graph, cfg):
    for plan in _pipeline_policy(graph, cpu_offload=True):
        if plan.offload_id >= 0:
            plan.offload_id //= 2
            plan.recompute_id = plan.offload_id
        yield plan


def _trainer_worker(save_dir, run_name, policy, prefetch_level):
    import nnscaler.runtime.cpu_offloading as cpu_offloading

    save_dir = Path(save_dir)
    config_path = Path(__file__).with_name('trainer_args_pipeline.yaml').resolve()
    gen_savedir = save_dir / run_name / 'gen'
    checkpoint_dir = save_dir / run_name / 'checkpoints'
    merged_checkpoint = save_dir / f'{run_name}.pt'
    instance_name = f'instance_{run_name}'

    offloaded_tensors = 0
    original_context = cpu_offloading.CPUOffloadContext

    class RecordingContext(original_context):
        def _pack(self, tensor):
            nonlocal offloaded_tensors
            packed = super()._pack(tensor)
            if isinstance(packed, cpu_offloading._SavedTensor):
                offloaded_tensors += 1
            return packed

    cpu_offloading.CPUOffloadContext = RecordingContext
    try:
        trainer = Trainer([
            '-f', str(config_path),
            '--instance_name', instance_name,
            '--max_train_steps', '10',
            '--gen_savedir', str(gen_savedir),
            '--checkpoint.save_dir', str(checkpoint_dir),
            '--checkpoint.save_type', 'sharded',
            '--compute_config.use_zero', '3',
            '--compute_config.cpu_offload.min_tensor_bytes', '0',
            '--compute_config.cpu_offload.prefetch_groups', str(prefetch_level or 0),
            '--pas_policy', policy,
        ])
        trainer.run()
    finally:
        cpu_offloading.CPUOffloadContext = original_context

    assert trainer.model.use_scheduler
    assert trainer.model.nmicros_per_scheduler_step == 4
    assert trainer.model.compute_config.use_zero == 3
    assert trainer.model.compute_config.pas_config['pipeline_nstages'] == 2
    assert all(reducer.zero3 for reducer in trainer.model.reducers)

    if prefetch_level is not None:
        assert offloaded_tensors > 0
        assert trainer.model.activation_offload_stats()['live_cpu_bytes'] == 0
        assert _gencode_contains(
            gen_savedir,
            MLP,
            trainer.rank,
            r'with self\.cpu_offloading_hooks\((?:tail=True)?\) as _offload:',
            instance_name=instance_name,
        )
    else:
        assert offloaded_tensors == 0

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
    for run_name, prefetch_level, policy_name in (
        ('cpu_offload_demand', 0, 'pipeline_zero3_cpu_offload_policy'),
        ('cpu_offload_prefetch', 1, 'pipeline_zero3_cpu_offload_policy'),
        ('cpu_offload_combined', 1, 'pipeline_zero3_combined_policy'),
    ):
        launch_torchrun(
            4,
            _trainer_worker,
            tmp_path,
            run_name,
            f'{module_name}.{policy_name}',
            prefetch_level,
        )
        cpu_offload = torch.load(tmp_path / f'{run_name}.pt', weights_only=False)
        assert_equal(cpu_offload['model'], baseline['model'])
        assert_equal(cpu_offload['optimizer'], baseline['optimizer'])
