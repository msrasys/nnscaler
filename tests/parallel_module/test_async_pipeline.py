#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import copy
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import nnscaler
from nnscaler import ComputeConfig, parallelize, build_optimizer
from nnscaler.policies import get_pas_ops, OpPlan
from nnscaler.runtime.executor import AsyncCommHandler, Executor
from tests.launch_torchrun import torchrun
from tests.utils import clear_dir_on_rank0, PYTEST_RUN_ID


class AsyncPipelineMLP(torch.nn.Module):
    def __init__(self, nstages):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(8, 8, bias=False) for _ in range(nstages)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return (x * x).sum()


def async_pipeline_policy(graph, cfg):
    for op in get_pas_ops(graph):
        stage = -1
        if torch.nn.Linear in op.module_class_chain:
            stage = int(op.get_module_fqn(torch.nn.Linear).split('.')[-1])
        yield OpPlan(op, stage_id=stage, partition=None)


def worker_async_pipeline(scheduler, plan_ngpus, async_reducer):
    # Exercise the public initializer, including eager NCCL on recent PyTorch.
    nnscaler.init()
    torch.manual_seed(7)
    source = AsyncPipelineMLP(8 if scheduler == '1f1b_interleaved' else 4).double()
    eager = copy.deepcopy(source).cuda()
    params = dict(eager.named_parameters())
    reference_optimizer = torch.optim.SGD(eager.parameters(), lr=0.01)
    cfg = ComputeConfig(
        plan_ngpus, 4, use_end2end=True, use_async_comm=True,
        use_async_reducer=async_reducer,
        pas_config={
            'pipeline_size': plan_ngpus,
            'pipeline_nmicros': 8,
            'pipeline_scheduler': scheduler,
        },
    )
    directory = Path(tempfile.gettempdir()) / f'async_pipeline_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tmp, patch.object(
        torch.distributed, 'isend', wraps=torch.distributed.isend,
    ) as send, patch.object(
        torch.distributed, 'irecv', wraps=torch.distributed.irecv,
    ) as recv:
        model = parallelize(
            source, {'x': torch.ones(4, 8, dtype=torch.float64)}, async_pipeline_policy,
            cfg, gen_savedir=tmp, reuse='override',
        ).cuda()
        optimizer = build_optimizer(model, torch.optim.SGD, lr=0.01)
        for step in range(2):
            samples = [
                torch.arange(32, device='cuda', dtype=torch.float64).reshape(4, 8) / 32
                + micro * 0.1 + step * 0.01
                for micro in range(8)
            ]
            expected = [eager(sample) for sample in samples]
            (torch.stack(expected).sum() * (4 // plan_ngpus)).backward()
            torch.testing.assert_close(model.train_step(samples), expected)
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(getattr(model, name).grad, params[meta.orig_name].grad[meta.slicers])
            optimizer.step()
            reference_optimizer.step()
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(getattr(model, name), params[meta.orig_name][meta.slicers])
            optimizer.zero_grad()
            reference_optimizer.zero_grad()
            AsyncCommHandler().check_clear()
            Executor.check_clear()
        with torch.no_grad():
            model.eval()
            torch.testing.assert_close(model.infer_step(samples), [eager(sample) for sample in samples])
        assert send.call_count > 0 and recv.call_count > 0
        AsyncCommHandler().check_clear()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
@pytest.mark.parametrize('scheduler,plan_ngpus,async_reducer', [
    ('1f1b', 4, False),
    ('1f1b_interleaved', 4, False),
    ('1f1b_interleaved', 2, True),
])
def test_async_pipeline_matches_eager(scheduler, plan_ngpus, async_reducer):
    torchrun(4, worker_async_pipeline, scheduler, plan_ngpus, async_reducer)
