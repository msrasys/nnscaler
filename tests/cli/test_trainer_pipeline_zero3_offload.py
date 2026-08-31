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


def _trainer_worker(save_dir, run_name, policy, expect_cpu_offload):
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
            if isinstance(packed, cpu_offloading._OffloadedTensor):
                offloaded_tensors += 1
            return packed

    if expect_cpu_offload:
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

    if expect_cpu_offload:
        assert offloaded_tensors > 0
        assert _gencode_contains(
            gen_savedir,
            MLP,
            trainer.rank,
            r'with self\.cpu_offloading_hooks\(\):',
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
        False,
    )
    launch_torchrun(
        4,
        _trainer_worker,
        tmp_path,
        'cpu_offload',
        f'{module_name}.pipeline_zero3_cpu_offload_policy',
        True,
    )

    baseline = torch.load(tmp_path / 'baseline.pt', weights_only=False)
    cpu_offload = torch.load(tmp_path / 'cpu_offload.pt', weights_only=False)
    assert_equal(cpu_offload['model'], baseline['model'])
    assert_equal(cpu_offload['optimizer'], baseline['optimizer'])
