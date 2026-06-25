#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""End-to-end (multi-GPU) validation of pipeline operator rescheduling.

Runs a small PP2 x TP2 pipeline-parallel model for several training steps twice --
once with the operator reschedule disabled and once with it enabled on the pipeline
schedule (communication issued as early as legal) -- and asserts the merged weights
stay numerically identical. The reschedule only reorders communication, so the
training must produce the same result, and must not deadlock.

Requires 4 GPUs.
"""
import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer, merge_state_dicts

from .common import init_distributed, PASMegatron
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import init_random, clear_dir_on_rank0, PYTEST_RUN_ID

DIM = 16
NLAYERS = 8
MBS = 2
NMICROS = 4
NSTEPS = 4


class _MLP(nn.Module):
    def __init__(self, dim=DIM, nlayers=NLAYERS):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(nlayers))
        self.loss_fn = nn.BCELoss()

    def forward(self, data):
        x = data['data']
        for layer in self.layers:
            x = layer(x)
        return self.loss_fn(torch.sigmoid(x), data['target'])


def _make_data(nsteps, nmicros, seed=1234):
    g = torch.Generator().manual_seed(seed)
    steps = []
    for _ in range(nsteps):
        steps.append([
            {'data': torch.randn(MBS, DIM, generator=g, device='cpu'),
             'target': torch.rand(MBS, DIM, generator=g, device='cpu')}
            for _ in range(nmicros)
        ])
    return steps


def _pp_reschedule_worker(reschedule: bool):
    init_distributed()
    dev = torch.cuda.current_device()
    init_random()
    tag = 'on' if reschedule else 'off'
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'resched_e2e_{PYTEST_RUN_ID}_{tag}') as tempdir:
        init_random()
        model = parallelize(
            _MLP(),
            {'data': {'data': torch.randn(MBS, DIM, device=dev),
                      'target': torch.rand(MBS, DIM, device=dev)}},
            PASMegatron,
            ComputeConfig(4, 4, use_end2end=True,
                          pas_config=dict(pipeline_nstages=2, pipeline_nmicros=NMICROS,
                                          pipeline_scheduler='1f1b')),
            gen_savedir=tempdir,
            instance_name=f'resched_{tag}',
        )
        model.cuda()
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)

        data = _make_data(NSTEPS, NMICROS)
        states = []
        for step in range(NSTEPS):
            model.train()
            batch = [{k: v.to(dev) for k, v in mb.items()} for mb in data[step]]
            model.train_step(batch)
            optimizer.step()
            optimizer.zero_grad()
            states.append(clone_to_cpu_recursively(model.state_dict()))
        return states


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires 4 gpus')
def test_pipeline_reschedule_numeric_equivalence():
    saved = {k: os.environ.get(k) for k in
             ('ENABLE_OP_RESCHEDULE', 'OP_RESCHEDULE_SCOPE', 'OP_RESCHEDULE_PIPELINE')}
    try:
        # reschedule OFF (baseline)
        for k in saved:
            os.environ.pop(k, None)
        off = launch_torchrun(4, _pp_reschedule_worker, False)

        # reschedule ON, applied to the pipeline schedule (comm issued early)
        os.environ['ENABLE_OP_RESCHEDULE'] = '1'
        os.environ['OP_RESCHEDULE_SCOPE'] = 'both'
        os.environ['OP_RESCHEDULE_PIPELINE'] = '1'
        on = launch_torchrun(4, _pp_reschedule_worker, True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    assert off and on, 'workers returned no result'
    for step in range(NSTEPS):
        off_sd = merge_state_dicts([off[r][step] for r in range(4)])[0]
        on_sd = merge_state_dicts([on[r][step] for r in range(4)])[0]
        for k, a in off_sd.items():
            if not torch.is_tensor(a):
                continue
            b = on_sd[k]
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), \
                f'step {step} key {k} differs: max|diff|={(a - b).abs().max().item():.3e}'


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires 4 gpus')
def test_pipeline_reschedule_async_recv_numeric_equivalence():
    """Reschedule + ASYNC_RECV (cross-stage irecv issued early, waited lazily)
    must stay numerically identical to the synchronous baseline."""
    keys = ('ENABLE_OP_RESCHEDULE', 'OP_RESCHEDULE_SCOPE', 'OP_RESCHEDULE_PIPELINE', 'ASYNC_RECV')
    saved = {k: os.environ.get(k) for k in keys}
    try:
        # baseline (sync, no reschedule)
        for k in keys:
            os.environ.pop(k, None)
        off = launch_torchrun(4, _pp_reschedule_worker, False)

        # reschedule ON + async recv (irecv early + deferred wait)
        os.environ['ENABLE_OP_RESCHEDULE'] = '1'
        os.environ['OP_RESCHEDULE_SCOPE'] = 'both'
        os.environ['OP_RESCHEDULE_PIPELINE'] = '1'
        os.environ['ASYNC_RECV'] = '1'
        on = launch_torchrun(4, _pp_reschedule_worker, True)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    assert off and on, 'workers returned no result'
    for step in range(NSTEPS):
        off_sd = merge_state_dicts([off[r][step] for r in range(4)])[0]
        on_sd = merge_state_dicts([on[r][step] for r in range(4)])[0]
        for k, a in off_sd.items():
            if not torch.is_tensor(a):
                continue
            b = on_sd[k]
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), \
                f'step {step} key {k} differs: max|diff|={(a - b).abs().max().item():.3e}'
