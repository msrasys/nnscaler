#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Real, compiled, 4-GPU end-to-end proof of Step C at a genuine multi-rank
topology: PP2 x EP2 (2 physical pipeline stages, each with its own 2-rank
expert-parallel group -- ``ep_ranks_per_stage=[(0,1),(2,3)]``), matching
``tests/parallel_module/test_local_segments_multistage_e2e.py``'s/
``test_combined_1f1b_multistage_e2e.py``'s own multi-rank regression
conventions (repeated no-deadlock runs, numeric equivalence vs. serial
baseline).

This topology exercises real communication Step A/B/C all care about
simultaneously: cross-*stage* P2P activation/gradient adapters (stage 0's
devices 0/1 <-> stage 1's devices 2/3, Step A/B's own domain) *and*
intra-stage MoE dispatch/combine all-to-all (Step C's own domain, within
each stage's own EP group) -- a real, reasonable multi-rank topology per the
task's own requirement ("4GPU至少PP2×EP2或合理多rank拓扑").

Requires 4 GPUs.
"""
import tempfile
from pathlib import Path

import pytest
import torch

from nnscaler.parallel import parallelize, build_optimizer, ComputeConfig

from .common import init_distributed, assert_close
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import init_random, clear_dir_on_rank0, PYTEST_RUN_ID
from .phase_moe_common import MoEConfig, PhaseMoEModel, make_pas
from .test_phase_moe_e2e import _Alarm, _normalize_keys, _normalize_optimizer_state

DIM = 16
NHEADS = 2
SEQLEN = 4
FFN_HIDDEN = 32
NUM_STAGES = 2
LAYERS_PER_STAGE = 1
EP_RANKS_PER_STAGE = [(0, 1), (2, 3)]
NGPUS = 4
NMICROS = 4
NSTEPS = 2
T = 8


def _make_data(nsteps, nmicros, seed=4321):
    g = torch.Generator().manual_seed(seed)
    steps = []
    for _ in range(nsteps):
        steps.append([
            {'data': torch.randn(T, DIM, generator=g, device='cpu'),
             'target': torch.randn(T, DIM, generator=g, device='cpu')}
            for _ in range(nmicros)
        ])
    return steps


def _worker(use_phases: bool):
    init_distributed()
    dev = torch.cuda.current_device()
    init_random()
    tag = 'phase' if use_phases else 'serial'
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'phase_moe_multistage_{PYTEST_RUN_ID}_{tag}') as tempdir:
        init_random()
        cfg = MoEConfig(dim=DIM, n_heads=NHEADS, seq_len=SEQLEN, ffn_hidden=FFN_HIDDEN, capacity_factor=1.0)
        model = parallelize(
            PhaseMoEModel(cfg, NUM_STAGES, LAYERS_PER_STAGE, EP_RANKS_PER_STAGE, use_phases=use_phases),
            {'data': {'data': torch.randn(T, DIM, device=dev), 'target': torch.randn(T, DIM, device=dev)}},
            make_pas(NUM_STAGES, LAYERS_PER_STAGE, EP_RANKS_PER_STAGE, use_phases=use_phases),
            ComputeConfig(
                NGPUS, NGPUS, use_end2end=True,
                use_async_recv=True,
                pas_config=dict(pipeline_nmicros=NMICROS),
            ),
            gen_savedir=tempdir,
            instance_name=f'phase_moe_multistage_{tag}',
        )

        model.cuda()
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)

        data = _make_data(NSTEPS, NMICROS)
        states, opt_states = [], []
        for step in range(NSTEPS):
            model.train()
            batch = [{k: v.to(dev) for k, v in mb.items()} for mb in data[step]]
            model.train_step(batch)
            torch.cuda.synchronize()
            optimizer.step()
            optimizer.zero_grad()
            states.append(clone_to_cpu_recursively(model.state_dict()))
            opt_states.append(clone_to_cpu_recursively(optimizer.state_dict()))
        return states, opt_states


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires 4 gpus')
def test_phase_moe_multistage_no_deadlock_repeated():
    """PP2xEP2 must run to completion within a bounded wall-clock time,
    repeated 3x (a deadlock is the failure mode; one lucky pass proves
    little -- matches test_combined_1f1b_multistage_e2e.py's/
    test_local_segments_multistage_e2e.py's own convention)."""
    for attempt in range(3):
        with _Alarm(180, f'POSSIBLE DEADLOCK: PP2xEP2 phase-aware MoE pipeline '
                          f'did not complete within 180s (attempt {attempt})'):
            outputs = launch_torchrun(NGPUS, _worker, True)
        assert outputs is not None and len(outputs) == NGPUS
        for r in range(NGPUS):
            states, opt_states = outputs[r]
            assert len(states) == NSTEPS
            assert len(opt_states) == NSTEPS
    print('NO_DEADLOCK PASS: PP2xEP2 phase-aware MoE pipeline (3x repeated)')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires 4 gpus')
def test_phase_moe_multistage_numeric_equivalence():
    """PP2xEP2 phase-lowered + phase-aware-scheduled compile must match the
    plain/serial baseline compile's weights and optimizer state, per rank
    (see test_phase_moe_e2e.py's own numeric-equivalence test for why a
    per-rank, key-normalized comparison is used instead of cross-rank
    ``merge_state_dicts``)."""
    with _Alarm(180, 'possible deadlock: serial baseline (PP2xEP2) run did not finish in 180s'):
        off = launch_torchrun(NGPUS, _worker, False)
    with _Alarm(180, 'possible deadlock: phase-aware (PP2xEP2) run did not finish in 180s'):
        on = launch_torchrun(NGPUS, _worker, True)

    assert off and on, 'workers returned no result'
    for step in range(NSTEPS):
        for r in range(NGPUS):
            off_sd = _normalize_keys(off[r][0][step])
            on_sd = _normalize_keys(on[r][0][step])
            assert set(off_sd) == set(on_sd), (r, set(off_sd) ^ set(on_sd))
            for k, a in off_sd.items():
                if not torch.is_tensor(a):
                    continue
                b = on_sd[k]
                assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), \
                    f'step {step} rank {r} weight {k} differs: max|diff|={(a - b).abs().max().item():.3e}'
            off_opt_sd = _normalize_optimizer_state(off[r][1][step])
            on_opt_sd = _normalize_optimizer_state(on[r][1][step])
            assert_close(off_opt_sd, on_opt_sd, atol=1e-4, rtol=1e-4)
