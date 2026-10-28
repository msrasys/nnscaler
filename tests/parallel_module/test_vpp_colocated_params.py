#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
import json
import tempfile
from pathlib import Path

import pytest
import torch

import nnscaler
from nnscaler import ComputeConfig, build_optimizer, parallelize
from nnscaler.policies import OpPartition, OpPlan, fn, get_pas_ops
from tests.launch_torchrun import launch_torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0


class ReusedBlocks(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 8, bias=False)
        self.b = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        x = self.a(x)
        x = self.b(x)
        x = self.a(x)
        x = self.b(x)
        return (x * x).sum()


def shared_vpp_policy(graph, cfg):
    def plans(graph, cfg):
        stage = -1
        for node in get_pas_ops(graph):
            if node.fn == torch.nn.functional.linear:
                stage += 1
                yield OpPlan(node, stage_id=stage, partition=OpPartition(0, 0))
            else:
                yield OpPlan(node, stage_id=stage, partition='auto')
    return fn(graph, cfg, plans)


def colocated_worker(async_reducer=False, pipeline_multiref_replicated_params=True):
    nnscaler.init()
    errors = {'loss_max_abs': 0., 'gradient_max_abs': 0., 'weight_max_abs': 0.}
    # Full and partitioned batches can select different TF32 kernels. Use
    # full FP32 precision to test placement and accumulation independently.
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(7)
    source = ReusedBlocks()
    reference = copy.deepcopy(source).cuda()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    cfg = ComputeConfig(4, 4, use_end2end=True, use_async_reducer=async_reducer,
                        pas_config={'pipeline_size': 2, 'pipeline_nmicros': 4,
                                    'pipeline_scheduler': '1f1b_interleaved',
                                    'pipeline_multiref_replicated_params': pipeline_multiref_replicated_params})
    directory = Path(tempfile.gettempdir()) / f'colocated_vpp_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        model = parallelize(source, {'x': torch.ones(8, 8)}, shared_vpp_policy,
                            cfg, gen_savedir=tempdir, reuse='override').cuda()
        optimizer = build_optimizer(model, torch.optim.SGD, lr=0.01)
        expected_name = 'a.weight' if torch.distributed.get_rank() < 2 else 'b.weight'
        assert {meta.orig_name for meta in model.fullmap.values()} == {expected_name}
        reference_params = dict(reference.named_parameters())
        for step in range(2):
            samples = [(torch.arange(64, device='cuda').reshape(8, 8).float() + micro + step) / 64
                       for micro in range(4)]
            reference_optimizer.zero_grad()
            expected_losses = [reference(sample) for sample in samples]
            torch.stack(expected_losses).sum().backward()
            outputs = model.train_step(samples)
            for actual, expected in zip(outputs, expected_losses):
                torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
                errors['loss_max_abs'] = max(errors['loss_max_abs'], (actual - expected).abs().max().item())
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(
                    getattr(model, name).grad, reference_params[meta.orig_name].grad[meta.slicers],
                    atol=1e-6, rtol=1e-5,
                )
                errors['gradient_max_abs'] = max(errors['gradient_max_abs'],
                    (getattr(model, name).grad - reference_params[meta.orig_name].grad[meta.slicers]).abs().max().item())
            optimizer.step()
            reference_optimizer.step()
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(
                    getattr(model, name), reference_params[meta.orig_name][meta.slicers],
                    atol=1e-6, rtol=1e-5,
                )
                errors['weight_max_abs'] = max(errors['weight_max_abs'],
                    (getattr(model, name) - reference_params[meta.orig_name][meta.slicers]).abs().max().item())
            optimizer.zero_grad()
    print('NNSCALER_TEST_METRICS ' + json.dumps({
        'case': 'colocated_vpp', 'rank': torch.distributed.get_rank(),
        'async_reducer': async_reducer, 'multiref': pipeline_multiref_replicated_params, **errors,
    }))
    return True


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires 4 GPUs')
@pytest.mark.parametrize('async_reducer', [False, True])
@pytest.mark.parametrize('pipeline_multiref_replicated_params', [False, True])
def test_colocated_vpp_matches_unpartitioned_optimizer_steps(async_reducer, pipeline_multiref_replicated_params):
    assert all(launch_torchrun(4, colocated_worker, async_reducer, pipeline_multiref_replicated_params).values())
