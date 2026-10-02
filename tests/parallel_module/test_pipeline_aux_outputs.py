# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import nnscaler
from nnscaler import ComputeConfig, parallelize, build_optimizer
from nnscaler.flags import CompileFlag
from nnscaler.policies import get_pas_ops, OpPlan
from tests.launch_torchrun import launch_torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0


class AuxOutputMLP(torch.nn.Module):
    def __init__(self, early):
        super().__init__()
        self.early = early
        self.layers = torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False) for _ in range(4)])

    def forward(self, sample):
        first = self.layers[0](sample['x'])
        x = self.layers[1](first)
        x = self.layers[2](x)
        x = self.layers[3](x)
        aux = first.mean(dim=0) if self.early else x.mean(dim=0)
        return x.square().sum(), aux


def aux_policy(graph, cfg):
    for op in get_pas_ops(graph):
        stage = -1
        if torch.nn.Linear in op.module_class_chain:
            stage = int(op.get_module_fqn(torch.nn.Linear).split('.')[-1])
        yield OpPlan(op, stage_id=stage, partition=None)


def aux_worker(early, fbw):
    nnscaler.init()
    torch.manual_seed(7)
    source = AuxOutputMLP(early)
    eager = copy.deepcopy(source).cuda()
    eager_optimizer = torch.optim.SGD(eager.parameters(), lr=0.001)
    reference_parameters = dict(eager.named_parameters())
    cfg = ComputeConfig(2, 2, use_end2end=True, constant_folding=False,
                        pas_config={'pipeline_size': 2, 'pipeline_nmicros': 4,
                                    'pipeline_scheduler': '1f1b_interleaved'})
    directory = Path(tempfile.gettempdir()) / f'pipeline_aux_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir, patch.object(CompileFlag, 'use_fbw', fbw):
        model = parallelize(source, {'sample': {'x': torch.ones(8, 8)}},
                            aux_policy, cfg, gen_savedir=tempdir, reuse='override').cuda()
        optimizer = build_optimizer(model, torch.optim.SGD, lr=0.001)
        for step in range(2):
            samples = [{'x': torch.arange(64, device='cuda').reshape(8, 8).float() / 64 + micro + step}
                       for micro in range(4)]
            eager_optimizer.zero_grad()
            expected = [eager(sample) for sample in samples]
            torch.stack([output[0] for output in expected]).sum().backward()
            torch.testing.assert_close(model.train_step(samples), expected)
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(getattr(model, name).grad,
                                           reference_parameters[meta.orig_name].grad[meta.slicers])
            optimizer.step()
            eager_optimizer.step()
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(getattr(model, name),
                                           reference_parameters[meta.orig_name][meta.slicers])
            optimizer.zero_grad()
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires 2 GPUs')
@pytest.mark.parametrize('early', [False, True])
@pytest.mark.parametrize('fbw', [False, True])
def test_fn_pipeline_aux_outputs_match_eager(early, fbw):
    assert all(launch_torchrun(2, aux_worker, early, fbw).values())
