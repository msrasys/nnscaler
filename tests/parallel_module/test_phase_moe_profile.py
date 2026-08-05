#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Opt-in CPU/CUPTI/NVTX profile for the C-phase scheduling path.

This test intentionally writes no trace artifact into the repository.  It
prints a compact rank-0 table and cProfile sample when explicitly enabled:

    NN_SCALER_RUN_PHASE_PROFILE=1 CUDA_VISIBLE_DEVICES=0,1 \
      pytest -s -q tests/parallel_module/test_phase_moe_profile.py

The profiler wraps generated ``executor.fexecute``/``backward`` calls with
both ``torch.profiler.record_function`` and NVTX ranges, so Nsight Systems can
also consume the same labels if this worker is launched under ``nsys``.
"""
import cProfile
import io
import os
import pstats
import tempfile
from pathlib import Path

import pytest
import torch

from nnscaler.parallel import parallelize, build_optimizer, ComputeConfig

from .common import init_distributed
from ..launch_torchrun import launch_torchrun
from ..utils import init_random, clear_dir_on_rank0, PYTEST_RUN_ID
from .phase_moe_common import MoEConfig, PhaseMoEModel, make_pas
from .test_phase_moe_e2e import _Alarm


PROFILE_ENABLED = os.environ.get('NN_SCALER_RUN_PHASE_PROFILE') == '1'
MODES = ('serial', 'phase-dedicated-stream', 'phase-no-stream')


def _mode_options(mode):
    if mode == 'serial':
        return False, False
    if mode == 'phase-dedicated-stream':
        return True, True
    if mode == 'phase-no-stream':
        return True, False
    raise ValueError(mode)


def _profile_worker(mode):
    init_distributed()
    dev = torch.cuda.current_device()
    init_random()
    use_phases, dedicated = _mode_options(mode)
    cfg = MoEConfig(dim=128, n_heads=4, seq_len=8, ffn_hidden=512, capacity_factor=1.0)
    nmicros, tokens = 4, 128
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'phase_profile_{PYTEST_RUN_ID}_{mode}') as tempdir:
        model = parallelize(
            PhaseMoEModel(cfg, 1, 2, [(0, 1)], use_phases=use_phases),
            {'data': {'data': torch.randn(tokens, cfg.dim, device=dev),
                      'target': torch.randn(tokens, cfg.dim, device=dev)}},
            make_pas(1, 2, [(0, 1)], use_phases=use_phases,
                     dedicated_moe_comm_stream=dedicated),
            ComputeConfig(2, 2, use_end2end=True, use_async_recv=True,
                          pas_config=dict(pipeline_nmicros=nmicros)),
            gen_savedir=tempdir,
            instance_name=f'phase_profile_{mode}',
        )
        model.cuda()
        optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
        generator = torch.Generator(device=f'cuda:{dev}').manual_seed(901)

        def batch():
            return [
                {'data': torch.randn(tokens, cfg.dim, generator=generator, device=dev),
                 'target': torch.randn(tokens, cfg.dim, generator=generator, device=dev)}
                for _ in range(nmicros)
            ]

        # Compile, allocator and NCCL communicator warmup are deliberately
        # outside the measured profiler window.
        model.train_step(batch())
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()

        import nnscaler.runtime.executor as executor_module
        original_fexecute = executor_module.fexecute
        original_backward = executor_module.backward
        original_phase_forward = executor_module.PhaseExecutor.forward
        original_phase_backward = executor_module.PhaseExecutor.backward

        def _range(label, fn, *args, **kwargs):
            torch.cuda.nvtx.range_push(label)
            try:
                with torch.profiler.record_function(label):
                    return fn(*args, **kwargs)
            finally:
                torch.cuda.nvtx.range_pop()

        def profiled_fexecute(name, *args, **kwargs):
            return _range(f'nnscaler/fexecute/{name}', original_fexecute, name, *args, **kwargs)

        def profiled_backward(name, *args, **kwargs):
            return _range(f'nnscaler/backward/{name}', original_backward, name, *args, **kwargs)

        def profiled_phase_forward(phase_executor, slot, subgraph, *args, **kwargs):
            return _range(
                f'nnscaler/phase_fexecute/slot{slot}', original_phase_forward,
                phase_executor, slot, subgraph, *args, **kwargs,
            )

        def profiled_phase_backward(phase_executor, slot, outputs, grads):
            return _range(
                f'nnscaler/phase_backward/slot{slot}', original_phase_backward,
                phase_executor, slot, outputs, grads,
            )

        executor_module.fexecute = profiled_fexecute
        executor_module.backward = profiled_backward
        executor_module.PhaseExecutor.forward = profiled_phase_forward
        executor_module.PhaseExecutor.backward = profiled_phase_backward
        try:
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(activities=activities, profile_memory=True, record_shapes=False) as profiler:
                model.train_step(batch())
                torch.cuda.synchronize()
            cpu = cProfile.Profile()
            cpu.enable()
            model.train_step(batch())
            torch.cuda.synchronize()
            cpu.disable()
        finally:
            executor_module.fexecute = original_fexecute
            executor_module.backward = original_backward
            executor_module.PhaseExecutor.forward = original_phase_forward
            executor_module.PhaseExecutor.backward = original_phase_backward

        events = sorted(profiler.key_averages(), key=lambda event: event.self_cpu_time_total, reverse=True)
        selected_events = events[:30]
        selected_ids = {id(event) for event in selected_events}
        # Phase labels can be smaller than the top-30 CUDA/ATen entries; keep
        # them explicitly so the profile remains an API-count artifact.
        selected_events += [
            event for event in events
            if event.key.startswith('nnscaler/phase_') and id(event) not in selected_ids
        ]
        summary = []
        for event in selected_events:
            summary.append({
                'key': event.key,
                'count': event.count,
                'self_cpu_us': round(event.self_cpu_time_total, 1),
                'cpu_us': round(event.cpu_time_total, 1),
                'self_cuda_us': round(getattr(event, 'self_device_time_total', 0.0), 1),
                'cuda_us': round(getattr(event, 'device_time_total', 0.0), 1),
                'cpu_memory': getattr(event, 'cpu_memory_usage', 0),
            })
        stream = io.StringIO()
        pstats.Stats(cpu, stream=stream).strip_dirs().sort_stats('cumtime').print_stats(25)
        return {'mode': mode, 'events': summary, 'cprofile': stream.getvalue()}


@pytest.mark.skipif(not PROFILE_ENABLED, reason='set NN_SCALER_RUN_PHASE_PROFILE=1 to collect CUPTI/NVTX profile')
@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason='requires >= 2 GPUs')
def test_phase_moe_c_profile():
    for mode in MODES:
        with _Alarm(240, f'possible deadlock while profiling {mode}'):
            outputs = launch_torchrun(2, _profile_worker, mode)
        report = outputs[0]
        assert report['events']
        if mode != 'serial':
            assert any(item['key'].startswith('nnscaler/phase_fexecute/') for item in report['events'])
        print(f'\n[C profile] mode={mode}')
        for event in report['events']:
            print('[C profile]', event)
        print('[C profile] cProfile\n' + report['cprofile'])
