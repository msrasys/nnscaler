#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Real, compiled, end-to-end proof that ``GlobalCommSchedule`` is safe on a
genuinely multi-peer-pair pipeline (NSTAGES=4, giving 3 independent
peer-pairs: (0,1), (1,2), (2,3) -- ranks 1 and 2 are each members of TWO
distinct peer-pairs simultaneously), not just the single-peer-pair (NSTAGES=2)
topology ``test_combined_1f1b_pipeline_e2e.py`` exercises.

Why this file exists (self-audit background): an earlier version of
``GlobalCommSchedule`` (letting distinct peer-pairs reorder independently,
while all P2P shared the default process group) was found, via a dedicated
self-audit, to deadlock reproducibly on this exact topology on real 4-GPU
hardware -- confirmed via ``faulthandler``/``SIGUSR1`` stack dumps showing one
rank stuck in a plain synchronous ``send`` while the others were stuck inside
``irecv``'s enqueue call itself. Root cause and fix: see the module docstring
of ``nnscaler.execplan.planpass.global_schedule`` (dedicated per-peer-pair
process groups, established up front in a deterministic, all-ranks-consistent
order, with a fail-closed single-chain degradation when that cannot be
verified). This file is the real-hardware regression test for that fix, run
repeatedly (not just once) since the failure mode is a deadlock -- a single
lucky pass would not be reassuring.

Requires >= 4 GPUs.
"""
import os
import signal
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer
from nnscaler.graph.schedule.schedplan import StreamConfig, StreamContext

from .common import init_distributed, PASMegatron
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import init_random, clear_dir_on_rank0, PYTEST_RUN_ID

DIM = 16
NLAYERS = 12   # 3 layers/stage * 4 stages, cleanly divisible
MBS = 2
NSTAGES = 4    # 3 independent peer-pairs: (0,1) (1,2) (2,3); ranks 1,2 each in TWO pairs
NMICROS = 6    # > NSTAGES: genuine multi-microbatch in-flight overlap
NSTEPS = 2
COMM_STREAM = 'comm'
REPEATS = 3    # the failure mode is a deadlock -- repeat, don't trust one pass


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


def _pas_multi_stream(graph, config: ComputeConfig):
    """``PASMegatron`` plus routing inter-segment P2P onto a dedicated CUDA
    stream -- identical pattern to
    ``test_combined_1f1b_pipeline_e2e._pas_multi_stream``, just parameterized
    for NSTAGES=4 here."""
    graph = PASMegatron(graph, config)
    sched = graph.sched
    sched.stream_config = StreamConfig(
        inter_segment_move=StreamContext(stream=COMM_STREAM, wait_streams=['default']),
    )
    return graph


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


def _set_global_schedule_env(enabled: bool, max_outstanding: int = 6):
    keys = ('ENABLE_GLOBAL_P2P_RESCHEDULE', 'ASYNC_RECV_CHANNEL', 'ASYNC_RECV_MAX_OUTSTANDING')
    saved = {k: os.environ.get(k) for k in keys}
    if enabled:
        os.environ['ENABLE_GLOBAL_P2P_RESCHEDULE'] = '1'
        os.environ['ASYNC_RECV_CHANNEL'] = '1'
        os.environ['ASYNC_RECV_MAX_OUTSTANDING'] = str(max_outstanding)
    else:
        for k in keys:
            os.environ.pop(k, None)
    return saved


def _restore_env(saved):
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class _Alarm:
    """Hard wall-clock deadlock guard (SIGALRM). Only meaningful on Linux,
    which is what srgws-17 runs; skip gracefully elsewhere."""

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


def _worker(reschedule: bool, capture_per_file: bool = False):
    init_distributed()
    dev = torch.cuda.current_device()
    init_random()
    tag = 'on' if reschedule else 'off'
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'combined_1f1b_4stage_{PYTEST_RUN_ID}_{tag}') as tempdir:
        init_random()
        model = parallelize(
            _MLP(),
            {'data': {'data': torch.randn(MBS, DIM, device=dev),
                      'target': torch.rand(MBS, DIM, device=dev)}},
            _pas_multi_stream,
            ComputeConfig(NSTAGES, NSTAGES, use_end2end=True,
                          use_async_recv=True,
                          pas_config=dict(pipeline_nstages=NSTAGES, pipeline_nmicros=NMICROS,
                                          pipeline_scheduler='1f1b')),
            gen_savedir=tempdir,
            instance_name=f'combined_1f1b_4stage_{tag}',
        )

        per_file = {}
        if capture_per_file and torch.distributed.get_rank() == 0:
            per_file = {str(p.relative_to(tempdir)): p.read_text() for p in sorted(tempdir.rglob('*.py'))}

        model.cuda()
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)

        data = _make_data(NSTEPS, NMICROS)
        states = []
        for step in range(NSTEPS):
            model.train()
            batch = [{k: v.to(dev) for k, v in mb.items()} for mb in data[step]]
            model.train_step(batch)
            # see test_combined_1f1b_pipeline_e2e._worker for why this manual
            # sync is required with a custom inter-segment CUDA stream
            torch.cuda.synchronize()
            optimizer.step()
            optimizer.zero_grad()
            states.append(clone_to_cpu_recursively(model.state_dict()))
        return states, per_file


def _run(reschedule: bool, capture_per_file: bool = False, timeout: int = 180):
    saved = _set_global_schedule_env(reschedule, max_outstanding=6)
    try:
        with _Alarm(timeout, f'possible deadlock: 4-stage run (global_schedule='
                             f'{reschedule}) did not finish in {timeout}s'):
            outputs = launch_torchrun(NSTAGES, _worker, reschedule, capture_per_file)
    finally:
        _restore_env(saved)
    assert outputs and all(o is not None for o in outputs), 'a rank returned no result'
    return outputs[0]  # rank 0's (states, per_file)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires >= 4 gpus')
def test_combined_1f1b_multistage_no_deadlock_repeated():
    """The exact scenario that deadlocked pre-fix (see module docstring):
    4 stages, 3 independent peer-pairs, GlobalCommSchedule enabled. Repeated
    REPEATS times -- a deadlock is the failure mode being guarded against, so
    a single pass is not reassuring; each repetition uses a fresh process
    group (a new ``launch_torchrun`` call) to also exercise dedicated
    process-group creation repeatedly, not just once per test session."""
    for i in range(REPEATS):
        states, _ = _run(True, timeout=180)
        assert states and len(states) == NSTEPS, f'repeat {i}: incomplete result'


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires >= 4 gpus')
def test_combined_1f1b_multistage_numeric_equivalence():
    """GlobalCommSchedule (with dedicated per-peer-pair process groups) must
    not change the trained numerics vs the baseline, on a topology with 2+
    independent peer-pairs (never checked by the NSTAGES=2 e2e test)."""
    off_states, _ = _run(False)
    on_states, _ = _run(True)
    assert len(off_states) == len(on_states) == NSTEPS

    for step, (off_sd, on_sd) in enumerate(zip(off_states, on_states)):
        assert off_sd.keys() == on_sd.keys()
        for key in off_sd:
            off_val, on_val = off_sd[key], on_sd[key]
            if isinstance(off_val, torch.Tensor):
                max_diff = (off_val - on_val).abs().max().item()
                assert max_diff < 1e-5, (
                    f'step {step}, key {key!r}: 4-stage reschedule ON diverges from '
                    f'OFF beyond tolerance: max_diff={max_diff:.3e}'
                )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason='requires >= 4 gpus')
def test_combined_1f1b_multistage_gencode_structure():
    """Real generated code, on a 4-stage/3-peer-pair topology, must actually
    contain the channel/cap-tracked async-recv structure (not merely compile
    without error) -- i.e. the fix did not silently fall back to a
    no-op/never-taken code path on this topology."""
    _, per_file = _run(True, capture_per_file=True)
    assert per_file, 'rank 0 did not capture any generated code'
    all_code = '\n'.join(per_file.values())
    assert 'max_outstanding=6' in all_code, (
        'expected the channel-tracked async-recv launch (channel=<cid>, '
        'max_outstanding=6) literally in generated code on the 4-stage plan'
    )
    assert 'channel=' in all_code
