#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
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


def colocated_worker(async_reducer=False):
    nnscaler.init()
    # Full and partitioned batches can select different TF32 kernels. Use
    # full FP32 precision to test placement and accumulation independently.
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(7)
    source = ReusedBlocks()
    reference = copy.deepcopy(source).cuda()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    cfg = ComputeConfig(4, 4, use_end2end=True, use_async_reducer=async_reducer,
                        pas_config={'pipeline_size': 2, 'pipeline_nmicros': 4,
                                    'pipeline_scheduler': '1f1b_interleaved'})
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
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(
                    getattr(model, name).grad, reference_params[meta.orig_name].grad[meta.slicers],
                    atol=1e-6, rtol=1e-5,
                )
            optimizer.step()
            reference_optimizer.step()
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(
                    getattr(model, name), reference_params[meta.orig_name][meta.slicers],
                    atol=1e-6, rtol=1e-5,
                )
            optimizer.zero_grad()
    return True


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires 4 GPUs')
def test_colocated_vpp_matches_unpartitioned_optimizer_steps():
    assert all(launch_torchrun(4, colocated_worker).values())
