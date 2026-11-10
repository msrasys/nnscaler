#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""An end-to-end pipeline module nested inside an eager wrapper.

A vision-language style layout: the wrapper owns a "vision" tower
parallelized as an ordinary (non end-to-end) module, either tensor parallel
or data parallel over the scale unit, and a "language" model compiled as an
end-to-end pipeline.  The wrapper's own
``train_step`` feeds vision features into the pipeline, reads back the
boundary gradient that ``Executor.backward`` retains on the first stage,
shares it with the other stage ranks and resumes the vision backward.
"""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch import nn

from nnscaler.cli.scale_unit_parallelism import (
    dp_scale_unit_all_gather,
    dp_scale_unit_chunk,
    dp_scale_unit_param_config,
    dp_scale_unit_ranks,
)
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import TrainerArgs
from nnscaler.policies import OpPlan, get_pas_ops
from nnscaler.runtime.device import DeviceGroup
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.common import assert_close
from tests.utils import init_random


class VLDataset(torch.utils.data.Dataset):
    def __init__(self, dim: int, size: int):
        generator = torch.Generator().manual_seed(0)
        self.tokens = torch.randn(size, dim, generator=generator)
        self.pixels = torch.randn(size, dim, generator=generator)
        self.target = torch.rand(size, dim, generator=generator)

    def __getitem__(self, idx: int):
        return {
            'tokens': self.tokens[idx],
            'pixels': self.pixels[idx],
            'target': self.target[idx],
        }

    def __len__(self):
        return len(self.tokens)


class VisionTower(nn.Module):
    def __init__(self, dim: int, nlayers: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(nlayers))

    def forward(self, pixels):
        x = pixels
        for layer in self.layers:
            x = layer(x)
        return x


class VisionTowerDP(VisionTower):
    """Replicated weights; every rank of the scale unit processes its own slice of the batch."""

    def __init__(self, dim: int, nlayers: int, plan_ngpus: int):
        super().__init__(dim, nlayers)
        self.plan_ngpus = plan_ngpus

    def forward(self, pixels):
        x = dp_scale_unit_chunk(pixels, self.plan_ngpus, 1, dim=0)
        x = super().forward(x)
        return dp_scale_unit_all_gather(x, self.plan_ngpus, 1, dim=0)


def replicate_policy(graph, compute_config):
    # the scale-unit DP helpers split the data; every op stays replicated
    for node in get_pas_ops(graph):
        yield OpPlan(node)


def init_scale_unit_groups(trainer: Trainer):
    # the DP collectives and the boundary-gradient broadcast use the scale-unit
    # groups; create them on every rank in the same order up front
    plan_ngpus = trainer.train_args.compute_config.plan_ngpus
    DeviceGroup().get_groups([
        list(dp_scale_unit_ranks(plan_ngpus, rank=rank))
        for rank in range(0, dist.get_world_size(), plan_ngpus)
    ])


def vision_param_clss_fn(trainer_args: TrainerArgs, fqn: str) -> dict:
    # DP vision ranks hold complementary gradients; keep the reducer SUM
    if fqn.startswith('vision.') and trainer_args.model.args['vision_dp']:
        return dp_scale_unit_param_config(trainer_args)
    return {}


class LanguageModel(nn.Module):
    def __init__(self, dim: int, nlayers: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(nlayers))
        self.loss_fn = nn.BCELoss()

    def forward(self, data, features):
        x = data['tokens'] + features
        for layer in self.layers:
            x = layer(x)
        return self.loss_fn(torch.sigmoid(x), data['target'])


def _dims(trainer_args: TrainerArgs):
    return trainer_args.micro_batch_size, trainer_args.model.args['dim']


def vision_forward_args_gen_fn(trainer_args: TrainerArgs):
    mbs, dim = _dims(trainer_args)
    return {'pixels': torch.randn(mbs, dim)}


def language_forward_args_gen_fn(trainer_args: TrainerArgs):
    mbs, dim = _dims(trainer_args)
    return {
        'data': {'tokens': torch.randn(mbs, dim), 'target': torch.rand(mbs, dim)},
        # the only pipeline input whose gradient must flow back to the wrapper
        'features': torch.randn(mbs, dim, requires_grad=True),
    }


def _language_data(sample):
    return {'tokens': sample['tokens'], 'target': sample['target']}


class VLModel(nn.Module):
    """Baseline: both submodules are ordinary parallel modules driven by autograd."""

    def __init__(self, dim: int, nlayers: int, plan_ngpus: int, vision_dp: bool = False):
        super().__init__()
        init_random()
        self.vision = VisionTowerDP(dim, nlayers, plan_ngpus) if vision_dp else VisionTower(dim, nlayers)
        self.language = LanguageModel(dim, nlayers)

    def forward(self, data):
        return self.language(_language_data(data), self.vision(data['pixels']))


class NestedEnd2EndVLModel(VLModel):
    """The language model is an end-to-end pipeline; this wrapper drives it."""

    def _pipeline_batches(self, samples):
        batches, features = [], []
        for sample in samples:
            feature = self.vision(sample['pixels'])
            if feature.requires_grad:
                # Executor.backward writes the pipeline input gradient back here.
                feature.retain_grad()
            batches.append((_language_data(sample), feature))
            features.append(feature)
        return batches, features

    def _boundary_grad(self, feature):
        """Only the first pipeline stage consumes ``feature``; share its gradient."""
        ranks = dp_scale_unit_ranks(self.language.compute_config.plan_ngpus)
        src, group = ranks[0], DeviceGroup().get_group(list(ranks))
        if dist.get_rank() == src:
            assert feature.grad is not None, 'pipeline did not retain the boundary gradient'
            grad = feature.grad
        else:
            grad = torch.zeros_like(feature)
        dist.broadcast(grad, src=src, group=group)
        return grad

    def train_step(self, samples, is_dummy_batch=None):
        batches, features = self._pipeline_batches(samples)
        outputs = self.language.train_step(batches, is_dummy_batch)
        for feature in features:
            grad = self._boundary_grad(feature)
            feature.grad = None
            torch.autograd.backward(feature, grad)
        return outputs

    def infer_step(self, samples):
        batches, _ = self._pipeline_batches(samples)
        return self.language.infer_step(batches)


_train_losses = []


def record_train_loss(trainer: Trainer, aggregated_outputs, loss):
    _train_losses.append(float(loss))


def nested_end2end_worker(run_dir, nested: bool, use_fbw: bool = False, vision_dp: bool = False):
    _train_losses.clear()
    run_dir = Path(run_dir)
    config_path = str(Path(__file__).with_name('trainer_args_nested_end2end.yaml').resolve())
    args = [
        '-f', config_path,
        '--enable_progress_bar', 'false',
        '--gen_savedir', str(run_dir / 'gen'),
        '--checkpoint.save_dir', str(run_dir / 'ckpt'),
        '--hook.after_aggregate_train_step_outputs',
        'tests.cli.test_nested_end2end.record_train_loss',
        '--model.args.vision_dp', str(vision_dp),
    ]
    if vision_dp:
        args += [
            '--model.parallel_modules.0.type', 'tests.cli.test_nested_end2end.VisionTowerDP',
            '--model.parallel_modules.0.pas_policy', 'tests.cli.test_nested_end2end.replicate_policy',
            # replicated weights get a reducer; vision_param_clss_fn keeps its SUM
            '--compute_config.reducer_replicated_params', 'true',
        ]
    if nested:
        args += ['--model.parallel_modules.1.compute_config.use_fbw', str(use_fbw)]
    else:
        args += [
            '--model.type', 'tests.cli.test_nested_end2end.VLModel',
            '--model.parallel_modules.1.pas_policy', 'tp',
            '--model.parallel_modules.1.compute_config.use_end2end', 'false',
        ]
    trainer = Trainer(args)
    trainer.run()
    if vision_dp:
        # Adam hides a wrong gradient scale, so check the reducer directly: the
        # replicated vision weights are summed over all ranks without division.
        buckets = [bucket for reducer in trainer.model.vision.reducers for bucket in reducer.buckets]
        assert buckets and all(bucket.nreplicas == 1 for bucket in buckets)
        assert all(tuple(reducer.ranks) == tuple(range(trainer.world_size)) for reducer in trainer.model.vision.reducers)
    dist.barrier()
    if trainer.rank == 0:
        Trainer.merge_checkpoint(
            list((run_dir / 'ckpt' / 'last').glob('*.ckpt')),
            run_dir / 'merged.pt',
        )
    dist.barrier()
    return list(_train_losses)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_nested_end2end_matches_mixed_baseline(tmp_path):
    baselines = {}
    for vision_dp in (False, True):
        layout = 'dp' if vision_dp else 'tp'
        baseline_dir = tmp_path / f'baseline_{layout}'
        baseline_losses = launch_torchrun(4, nested_end2end_worker, baseline_dir, False, False, vision_dp)
        baseline = torch.load(baseline_dir / 'merged.pt', weights_only=False)
        baselines[layout] = baseline

        for use_fbw in (False, True):
            run_dir = tmp_path / f'nested_{layout}_fbw{int(use_fbw)}'
            nested_losses = launch_torchrun(4, nested_end2end_worker, run_dir, True, use_fbw, vision_dp)

            # The dataset leaves a partial accumulation group, so the second step
            # pads with dummy micro-batches on both paths.
            assert len(nested_losses[0]) == 2
            for rank in range(4):
                assert nested_losses[rank] == pytest.approx(baseline_losses[rank], abs=1e-5)

            nested = torch.load(run_dir / 'merged.pt', weights_only=False)
            assert_close(nested['model'], baseline['model'], atol=1e-5, rtol=1e-5)
            assert_close(nested['optimizer']['state'], baseline['optimizer']['state'], atol=1e-5, rtol=1e-5)

    # The two vision layouts compute the same model; their baselines must agree too.
    assert_close(baselines['dp']['model'], baselines['tp']['model'], atol=1e-5, rtol=1e-5)
