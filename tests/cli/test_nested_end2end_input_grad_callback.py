#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Resume an outer backward inside a nested end-to-end pipeline schedule.

The layout of ``test_nested_end2end``: an eager wrapper owns a "vision" tower
and a "language" model compiled as an end-to-end pipeline.  Here the vision
tower is replicated without collectives, and only the ranks of the first
pipeline stage, which consume the vision features, register an
input-gradient callback.  The callback runs the vision backward of each
micro-batch as soon as the pipeline produces its feature gradient, inside
the pipeline schedule.  The other ranks never run the vision backward; the
reducer SUM over all ranks still yields the full vision gradient.
"""

from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from nnscaler.cli.scale_unit_parallelism import dp_scale_unit_param_config, dp_scale_unit_ranks
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import TrainerArgs
from nnscaler.runtime.executor import Executor
from tests.cli.test_nested_end2end import VLModel, _language_data
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_close


def vision_param_clss_fn(trainer_args: TrainerArgs, fqn: str) -> dict:
    # The ranks of a scale unit hold complementary vision gradients
    # (a slice of the batch in the baseline, all or nothing with callbacks);
    # keep the reducer SUM.
    if fqn.startswith('vision.'):
        return dp_scale_unit_param_config(trainer_args)
    return {}


def _resume_vision_backward(feature, grad):
    # The callback owns the propagation; Executor must not retain a copy as well.
    assert grad is not None and feature.grad is None
    torch.autograd.backward(feature, grad)


class FirstStageCallbackVLModel(VLModel):
    """The first-stage ranks resume the vision backward inside the pipeline schedule."""

    def train_step(self, samples, is_dummy_batch=None):
        first_stage_rank = dp_scale_unit_ranks(self.language.compute_config.plan_ngpus)[0]
        batches = []
        for sample in samples:
            feature = self.vision(sample['pixels'])
            # A registered callback must keep Executor from writing the retained grad.
            feature.retain_grad()
            # Only the first stage consumes ``feature``, so only its ranks register.
            if dist.get_rank() == first_stage_rank:
                Executor.register_input_grad_callback(
                    feature, partial(_resume_vision_backward, feature))
            batches.append((_language_data(sample), feature))
        # train_step raises if the schedule did not hand a registered feature back.
        return self.language.train_step(batches, is_dummy_batch)

    def infer_step(self, samples):
        return self.language.infer_step([
            (_language_data(sample), self.vision(sample['pixels'])) for sample in samples
        ])


_train_losses = []


def record_train_loss(trainer: Trainer, aggregated_outputs, loss):
    _train_losses.append(float(loss))


def input_grad_callback_worker(run_dir, callback: bool, use_fbw: bool = False, multi_stream: bool = False):
    _train_losses.clear()
    run_dir = Path(run_dir)
    module = 'tests.cli.test_nested_end2end_input_grad_callback'
    config_path = str(Path(__file__).with_name('trainer_args_nested_end2end.yaml').resolve())
    args = [
        '-f', config_path,
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(run_dir / 'gen'),
        '--checkpoint.save_dir', str(run_dir / 'ckpt'),
        '--hook.after_aggregate_train_step_outputs', f'{module}.record_train_loss',
        '--optimizer.param_clss_fn', f'{module}.vision_param_clss_fn',
        # replicated vision weights get a reducer; vision_param_clss_fn keeps its SUM
        '--model.parallel_modules.0.pas_policy', 'tests.cli.test_nested_end2end.replicate_policy',
        '--compute_config.reducer_replicated_params', 'true',
    ]
    if callback:
        args += [
            '--model.type', f'{module}.FirstStageCallbackVLModel',
            '--model.parallel_modules.1.compute_config.use_fbw', str(use_fbw),
        ]
        if multi_stream:
            args += ['--model.parallel_modules.1.compute_config.pas_config.pipeline_scheduler',
                     'tests.test_policies.sched_1f1b_multi_stream']
    else:
        # Mixed baseline: every scale-unit rank runs the vision tower on its slice of the batch.
        args += [
            '--model.type', 'tests.cli.test_nested_end2end.VLModel',
            '--model.args.vision_dp', 'true',
            '--model.parallel_modules.0.type', 'tests.cli.test_nested_end2end.VisionTowerDP',
            '--model.parallel_modules.1.pas_policy', 'tp',
            '--model.parallel_modules.1.compute_config.use_end2end', 'false',
        ]
    trainer = Trainer(args)
    trainer.run()
    dist.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((run_dir / 'ckpt' / 'last').glob('*.ckpt')),
            run_dir / 'merged.pt',
        )
    dist.barrier()
    return list(_train_losses)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_input_grad_callback_resumes_first_stage_backward(tmp_path):
    baseline_losses = launch_torchrun(4, input_grad_callback_worker, tmp_path / 'baseline', False)
    baseline = torch.load(tmp_path / 'baseline' / 'merged.pt', weights_only=False)

    # The callback runs inside backward or backward_input,
    # also on the named streams of a multi-stream schedule.
    for use_fbw, multi_stream in ((False, False), (True, False), (False, True)):
        run_dir = tmp_path / f'callback_fbw{int(use_fbw)}_ms{int(multi_stream)}'
        losses = launch_torchrun(4, input_grad_callback_worker, run_dir, True, use_fbw, multi_stream)
        # The dataset leaves a partial accumulation group,
        # so the second step pads with dummy micro-batches on both paths.
        assert len(losses[0]) == 2
        for rank in range(4):
            assert losses[rank] == pytest.approx(baseline_losses[rank], abs=1e-5)
        result = torch.load(run_dir / 'merged.pt', weights_only=False)
        assert_close(result['model'], baseline['model'], atol=1e-5, rtol=1e-5)
        assert_close(result['optimizer']['state'], baseline['optimizer']['state'], atol=1e-5, rtol=1e-5)
