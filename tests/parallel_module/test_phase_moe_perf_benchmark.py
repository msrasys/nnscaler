#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Reproducible two-scale C-phase performance benchmark.

This is intentionally a measurement test, not a CI pass/fail gate.  It
compares the plain serial schedule with the optimized phase path and the
legacy dedicated-stream path as an ablation.  Every observation uses explicit
CUDA synchronizations, discards warmup, alternates the serial/default order
(AB/BA), reports raw timed samples, and summarizes per-round medians plus MAD.

Run manually on an otherwise idle 2-GPU host:

    CUDA_VISIBLE_DEVICES=0,1 pytest -s -q \
      tests/parallel_module/test_phase_moe_perf_benchmark.py

For CUPTI-backed timelines, run the accompanying profiler worker under
``torch.profiler``/Nsight Systems; its NVTX-friendly generated phase names
are retained in the generated source.  Do not enable line timers: they
synchronize CUDA and invalidate overlap measurements.
"""
from dataclasses import dataclass
import math
import statistics
import tempfile
import time
from pathlib import Path

import pytest
import torch

from nnscaler.parallel import parallelize, build_optimizer, ComputeConfig

from .common import init_distributed
from ..launch_torchrun import launch_torchrun
from ..utils import init_random, clear_dir_on_rank0, PYTEST_RUN_ID
from .phase_moe_common import MoEConfig, PhaseMoEModel, make_pas
from .test_phase_moe_e2e import _Alarm


@dataclass(frozen=True)
class TimingScale:
    name: str
    dim: int
    nheads: int
    seqlen: int
    ffn_hidden: int
    tokens: int
    nmicros: int


# The small case makes Python/context/allocator overhead visible; the large
# case gives real GEMM and all-to-all work a fair opportunity to hide latency.
SCALES = (
    TimingScale('small', dim=128, nheads=4, seqlen=8, ffn_hidden=512, tokens=128, nmicros=4),
    TimingScale('large', dim=512, nheads=8, seqlen=8, ffn_hidden=2048, tokens=512, nmicros=8),
)
NUM_STAGES = 1
LAYERS_PER_STAGE = 2
EP_RANKS_PER_STAGE = [(0, 1)]
NGPUS = 2
WARMUP_STEPS = 2
TIMED_STEPS = 5
N_ROUNDS = 3

SERIAL = 'serial'
PHASE_DEFAULT = 'phase-no-stream'
PHASE_DEDICATED_STREAM = 'phase-dedicated-stream'


def _mode_options(mode: str):
    if mode == SERIAL:
        return False, False
    if mode == PHASE_DEFAULT:
        return True, False
    if mode == PHASE_DEDICATED_STREAM:
        return True, True
    raise ValueError(f'unknown timing mode {mode!r}')


def _timing_worker(mode: str, scale: TimingScale, warmup_steps: int, timed_steps: int):
    init_distributed()
    dev = torch.cuda.current_device()
    init_random()
    use_phases, dedicated_moe_comm_stream = _mode_options(mode)
    cfg = MoEConfig(
        dim=scale.dim, n_heads=scale.nheads, seq_len=scale.seqlen,
        ffn_hidden=scale.ffn_hidden, capacity_factor=1.0,
    )
    tag = f'{scale.name}_{mode}'
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'phase_moe_timing_{PYTEST_RUN_ID}_{tag}') as tempdir:
        model = parallelize(
            PhaseMoEModel(cfg, NUM_STAGES, LAYERS_PER_STAGE, EP_RANKS_PER_STAGE, use_phases=use_phases),
            {'data': {'data': torch.randn(scale.tokens, scale.dim, device=dev),
                      'target': torch.randn(scale.tokens, scale.dim, device=dev)}},
            make_pas(
                NUM_STAGES, LAYERS_PER_STAGE, EP_RANKS_PER_STAGE, use_phases=use_phases,
                dedicated_moe_comm_stream=dedicated_moe_comm_stream,
            ),
            ComputeConfig(
                NGPUS, NGPUS, use_end2end=True, use_async_recv=True,
                pas_config=dict(pipeline_nmicros=scale.nmicros),
            ),
            gen_savedir=tempdir,
            instance_name=f'phase_moe_timing_{tag}',
        )
        model.cuda()
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)

        generator = torch.Generator().manual_seed(1234)
        total_steps = warmup_steps + timed_steps
        data = [
            [{'data': torch.randn(scale.tokens, scale.dim, generator=generator, device='cpu'),
              'target': torch.randn(scale.tokens, scale.dim, generator=generator, device='cpu')}
             for _ in range(scale.nmicros)]
            for _ in range(total_steps)
        ]
        samples = []
        for step in range(total_steps):
            model.train()
            batch = [{key: value.to(dev) for key, value in micro.items()} for micro in data[step]]
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.train_step(batch)
            torch.cuda.synchronize()
            t_train = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            t_total = time.perf_counter()
            if step >= warmup_steps:
                samples.append({
                    'train_seconds': t_train - t0,
                    'total_seconds': t_total - t0,
                    'optimizer_seconds': t_total - t_train,
                })
        return samples


def _median_and_mad(values):
    median = statistics.median(values)
    return median, statistics.median(abs(value - median) for value in values)


def _measure(mode: str, scale: TimingScale):
    with _Alarm(240, f'possible deadlock: C-phase timing {scale.name}/{mode}'):
        outputs = launch_torchrun(NGPUS, _timing_worker, mode, scale, WARMUP_STEPS, TIMED_STEPS)
    assert outputs and outputs[0] is not None
    samples = outputs[0]
    assert len(samples) == TIMED_STEPS
    train = [sample['train_seconds'] for sample in samples]
    total = [sample['total_seconds'] for sample in samples]
    optimizer = [sample['optimizer_seconds'] for sample in samples]
    return {
        'raw_train_ms': [value * 1e3 for value in train],
        'raw_total_ms': [value * 1e3 for value in total],
        'raw_optimizer_ms': [value * 1e3 for value in optimizer],
        'train_median': statistics.median(train),
        'total_median': statistics.median(total),
    }


def _round_order(round_idx: int):
    # Serial/default are AB then BA then AB; the stream ablation rotates to
    # first position once so allocator/thermal order cannot systematically
    # favor it either.
    orders = (
        (SERIAL, PHASE_DEFAULT, PHASE_DEDICATED_STREAM),
        (PHASE_DEFAULT, SERIAL, PHASE_DEDICATED_STREAM),
        (PHASE_DEDICATED_STREAM, SERIAL, PHASE_DEFAULT),
    )
    return orders[round_idx % len(orders)]


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 GPUs')
def test_phase_moe_overlap_benchmark():
    for scale in SCALES:
        by_mode = {mode: [] for mode in (SERIAL, PHASE_DEFAULT, PHASE_DEDICATED_STREAM)}
        for round_idx in range(N_ROUNDS):
            order = _round_order(round_idx)
            round_data = {}
            for mode in order:
                result = _measure(mode, scale)
                by_mode[mode].append(result)
                round_data[mode] = result
            print(
                f'[C perf {scale.name}] round={round_idx} order={order} '
                f'serial={round_data[SERIAL]["train_median"] * 1e3:.3f}ms '
                f'default={round_data[PHASE_DEFAULT]["train_median"] * 1e3:.3f}ms '
                f'dedicated_stream={round_data[PHASE_DEDICATED_STREAM]["train_median"] * 1e3:.3f}ms'
            )
            for mode in order:
                print(f'[C perf {scale.name}] {mode} raw train ms: '
                      f'{[f"{x:.3f}" for x in round_data[mode]["raw_train_ms"]]}')

        serial = [item['train_median'] for item in by_mode[SERIAL]]
        default = [item['train_median'] for item in by_mode[PHASE_DEFAULT]]
        dedicated_stream = [item['train_median'] for item in by_mode[PHASE_DEDICATED_STREAM]]
        default_diff = [base - optimized for base, optimized in zip(serial, default)]
        dedicated_stream_diff = [base - ablation for base, ablation in zip(serial, dedicated_stream)]
        serial_med, serial_mad = _median_and_mad(serial)
        default_med, default_mad = _median_and_mad(default)
        dedicated_stream_med, dedicated_stream_mad = _median_and_mad(dedicated_stream)
        diff_med, diff_mad = _median_and_mad(default_diff)
        dedicated_stream_diff_med, dedicated_stream_diff_mad = _median_and_mad(dedicated_stream_diff)
        default_wins = sum(value > 0 for value in default_diff)

        print(f'[C perf {scale.name}] serial medians ms={[f"{x * 1e3:.3f}" for x in serial]} '
              f'median={serial_med * 1e3:.3f} MAD={serial_mad * 1e3:.3f}')
        print(f'[C perf {scale.name}] default medians ms={[f"{x * 1e3:.3f}" for x in default]} '
              f'median={default_med * 1e3:.3f} MAD={default_mad * 1e3:.3f}')
        print(f'[C perf {scale.name}] dedicated-stream medians ms={[f"{x * 1e3:.3f}" for x in dedicated_stream]} '
              f'median={dedicated_stream_med * 1e3:.3f} MAD={dedicated_stream_mad * 1e3:.3f}')
        print(f'[C perf {scale.name}] paired serial-default ms={[f"{x * 1e3:+.3f}" for x in default_diff]} '
              f'median={diff_med * 1e3:+.3f} MAD={diff_mad * 1e3:.3f} wins={default_wins}/{N_ROUNDS}')
        print(f'[C perf {scale.name}] paired serial-dedicated-stream ms={[f"{x * 1e3:+.3f}" for x in dedicated_stream_diff]} '
              f'median={dedicated_stream_diff_med * 1e3:+.3f} MAD={dedicated_stream_diff_mad * 1e3:.3f}')

        # Report rather than assert: shared GPU hosts are noisy, and numerical
        # correctness/lifecycle tests are the hard regression gate.
        stable_gain = diff_med > 3 * max(diff_mad, 1e-9) and default_wins >= math.ceil(0.7 * N_ROUNDS)
        print(f'[C perf {scale.name}] RESULT=' + ('stable-default-gain' if stable_gain else 'measurement-only'))
