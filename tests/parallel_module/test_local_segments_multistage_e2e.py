#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Real, compiled, 4-GPU (PP4) end-to-end regression for Step B (local
segments inside one physical pipeline stage,
``nnscaler.graph.schedule.local_segment``): local-segment splitting must
remain correct (numeric equivalence vs. an unsplit baseline) and
deadlock-free on a pipeline with more than 2 (and more than one independent
peer-pair of) physical stages -- NSTAGES=4 gives 3 independent peer-pairs
(0,1) (1,2) (2,3), exactly the topology
``test_combined_1f1b_multistage_e2e.py`` added for Step A's own multi-stage
regression. Every stage here is split into 2 local segments (only stage 0's
own B/F pair is actually interleaved by
``LocalSegmentSched.sched_1f1b_local_segments`` -- see that module's
docstring -- stages 1-3 place their local segments sequentially), so this
also exercises local segments correctly coexisting, on the same run, with
both an unconditionally-interleaved stage and non-interleaved stages.

Requires >= 4 GPUs.
"""
import signal
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer, merge_state_dicts
from nnscaler.ir.operator import IRFwOperation
import nnscaler.runtime.function as ncf
from nnscaler.graph.schedule.local_segment import AnchorBoundary, LocalSegmentSched, partition_stage_into_local_segments

from .common import init_distributed
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import init_random, clear_dir_on_rank0, PYTEST_RUN_ID

DIM = 16
NLAYERS = 12   # 3 layers/stage * 4 stages
MBS = 2
NSTAGES = 4    # 3 independent peer-pairs: (0,1) (1,2) (2,3)
NMICROS = 6    # > NSTAGES: genuine multi-microbatch in-flight overlap
NSTEPS = 2
LSEG_ANCHOR = 'lseg'


class _LSModel(nn.Module):
    """`NLAYERS` unbiased Linear layers; when `use_anchor`, an ``lseg``
    anchor is emitted before the *second* layer of every stage's 3 layers,
    so ``AnchorBoundary({'lseg'})`` splits each stage into 2 local segments
    (sizes 1 and 2)."""

    def __init__(self, dim=DIM, nlayers=NLAYERS, nstages=NSTAGES, use_anchor=False):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(nlayers))
        self.use_anchor = use_anchor
        per_stage = nlayers // nstages
        self._split_layer_idx = {sid * per_stage + 1 for sid in range(nstages)}

    def forward(self, data):
        x = data['data']
        for i, layer in enumerate(self.layers):
            if self.use_anchor and i in self._split_layer_idx:
                ncf.anchor(LSEG_ANCHOR)
            x = layer(x)
        return x.sum()


def _pas_local_segments(graph, config: ComputeConfig):
    num_stages = config.pas_config['pipeline_nstages']
    nmicros = config.pas_config['pipeline_nmicros']
    use_local_segments = config.pas_config['use_local_segments']

    all_fwd = [n for n in graph.nodes() if isinstance(n, IRFwOperation)]
    linear_positions = [i for i, n in enumerate(all_fwd) if n.name == 'linear']
    assert len(linear_positions) == NLAYERS
    per_stage = NLAYERS // num_stages
    stage_start_positions = [linear_positions[sid * per_stage] for sid in range(num_stages)]
    # See test_local_segments_e2e.py's _pas_local_segments for why this is
    # needed (the leading `getitem` op tracing `data['data']` must be
    # absorbed into the first stage, mirroring `IRGraph.blocking()`'s own
    # "adjust the start of the first stage" step).
    stage_start_positions[0] = 0

    all_segs = []
    for sid in range(num_stages):
        start = stage_start_positions[sid]
        end = stage_start_positions[sid + 1] if sid + 1 < num_stages else len(all_fwd)
        stage_nodes = all_fwd[start:end]
        boundary = AnchorBoundary({LSEG_ANCHOR}) if use_local_segments else None
        segs = partition_stage_into_local_segments(graph, stage_nodes, boundary)
        all_segs.append(segs)

    dataloader = graph.nodes()[0]
    sub_nodes = graph.replicate(dataloader, num_stages)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)
    for sid in range(num_stages):
        for seg in all_segs[sid]:
            for node in seg.nodes():
                graph.assign(node, sid)

    config.apply_pipeline_scheduler(graph, num_stages, nmicros, LocalSegmentSched.sched_1f1b_local_segments)
    return graph


def _make_data(nsteps, nmicros, seed=1234):
    g = torch.Generator().manual_seed(seed)
    steps = []
    for _ in range(nsteps):
        steps.append([
            {'data': torch.randn(MBS, DIM, generator=g, device='cpu')}
            for _ in range(nmicros)
        ])
    return steps


def _worker(use_local_segments: bool):
    init_distributed()
    dev = torch.cuda.current_device()
    init_random()
    tag = 'split' if use_local_segments else 'unsplit'
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'local_segments_multistage_e2e_{PYTEST_RUN_ID}_{tag}') as tempdir:
        init_random()
        model = parallelize(
            _LSModel(use_anchor=use_local_segments),
            {'data': {'data': torch.randn(MBS, DIM, device=dev)}},
            _pas_local_segments,
            ComputeConfig(
                NSTAGES, NSTAGES, use_end2end=True,
                use_async_recv=True,
                pas_config=dict(
                    pipeline_nstages=NSTAGES, pipeline_nmicros=NMICROS,
                    use_local_segments=use_local_segments,
                ),
            ),
            gen_savedir=tempdir,
            instance_name=f'local_segments_multistage_{tag}',
        )

        model.cuda()
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)

        data = _make_data(NSTEPS, NMICROS)
        states = []
        for step in range(NSTEPS):
            model.train()
            batch = [{k: v.to(dev) for k, v in mb.items()} for mb in data[step]]
            model.train_step(batch)
            torch.cuda.synchronize()
            optimizer.step()
            optimizer.zero_grad()
            states.append(clone_to_cpu_recursively(model.state_dict()))
        return states


class _Alarm:
    def __init__(self, seconds: int, message: str):
        self.seconds = seconds
        self.message = message
        self._supported = hasattr(signal, 'SIGALRM')

    def _handler(self, signum, frame):
        raise TimeoutError(self.message)

    def __enter__(self):
        if self._supported:
            signal.signal(signal.SIGALRM, self._handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._supported:
            signal.alarm(0)
        return False


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires >= 4 gpus')
def test_local_segments_multistage_no_deadlock():
    """PP4 (3 independent peer-pairs) with every stage split into 2 local
    segments must run to completion within a bounded wall-clock time,
    repeated (the failure mode is a deadlock -- one lucky pass proves
    little), matching ``test_combined_1f1b_multistage_e2e.py``'s own
    repeated-run convention."""
    for attempt in range(3):
        with _Alarm(180, f'POSSIBLE DEADLOCK (attempt {attempt}): PP4 local-segment pipeline '
                          'did not complete within 180s'):
            outputs = launch_torchrun(NSTAGES, _worker, True)
        assert outputs is not None and len(outputs) == NSTAGES
        for r in range(NSTAGES):
            assert len(outputs[r]) == NSTEPS
    print('NO_DEADLOCK PASS (PP4, x3): all ranks completed the local-segment pipeline run')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires >= 4 gpus')
def test_local_segments_multistage_numeric_equivalence():
    """PP4, every stage split into 2 local segments, must match the unsplit
    (single segment per stage) baseline -- both scheduled via the exact same
    ``LocalSegmentSched.sched_1f1b_local_segments``."""
    with _Alarm(180, 'possible deadlock: PP4 unsplit baseline run did not finish in 180s'):
        off = launch_torchrun(NSTAGES, _worker, False)
    with _Alarm(180, 'possible deadlock: PP4 local-segment-split run did not finish in 180s'):
        on = launch_torchrun(NSTAGES, _worker, True)

    assert off and on, 'workers returned no result'
    for step in range(NSTEPS):
        off_sd = merge_state_dicts([off[r][step] for r in range(NSTAGES)])[0]
        on_sd = merge_state_dicts([on[r][step] for r in range(NSTAGES)])[0]
        for k, a in off_sd.items():
            if not torch.is_tensor(a):
                continue
            b = on_sd[k]
            assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), \
                f'step {step} key {k} differs: max|diff|={(a - b).abs().max().item():.3e}'
