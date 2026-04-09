#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import os
from pathlib import Path

import pytest
import torch
import torch.distributed

from nnscaler.cli.trainer import Trainer
from tests.launch_torchrun import launch_torchrun


def trainer_profiling_worker(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args_profiling.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    trace_dir = save_dir / 'profiler_traces'
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_file_pattern = str(trace_dir / 'trace_rank{rank}_step{step_num}.json')
    stack_file_pattern = str(trace_dir / 'stack_rank{rank}_step{step_num}.txt')

    trainer = Trainer([
        '-f', config_path,
        '--gen_savedir', str(gen_savedir),
        '--debug.profile.with_stack', 'false',
        '--debug.profile.trace_handler.args.export_chrome_trace', trace_file_pattern,
        '--debug.profile.trace_handler.args.export_stacks', stack_file_pattern,
    ])
    trainer.run()

    torch.distributed.barrier()

    trace_files = list(trace_dir.glob('trace_rank*_step*.json'))
    assert len(trace_files) > 0, f"Expected chrome trace files in {trace_dir}, found none"
    for f in trace_files:
        assert f.stat().st_size > 0, f"Trace file {f} should not be empty"

    # with_stack is false, stack files should not be generated
    stack_files = list(trace_dir.glob('stack_rank*_step*.txt'))
    assert not len(stack_files)


def trainer_profiling_worker_tensorboard(save_dir):
    save_dir = Path(save_dir)
    config_path = str(Path(__file__).with_name('trainer_args_profiling.yaml').resolve())
    gen_savedir = save_dir / 'gen'
    trace_dir = save_dir / 'profiler_traces'
    trace_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer([
        '-f', config_path,
        '--gen_savedir', str(gen_savedir),
        '--debug.profile.trace_handler.name', 'tensorboard',
        '--debug.profile.trace_handler.args!',
        '--debug.profile.trace_handler.args.dir_name', str(trace_dir),
        '--debug.profile.trace_handler.args.worker_name', 'worker_{rank}',
        '--debug.profile.trace_handler.args.use_gzip', 'false',
    ])
    trainer.run()

    torch.distributed.barrier()

    trace_files = list(trace_dir.glob('worker_*.json'))
    assert len(trace_files) > 0
    for f in trace_files:
        assert f.stat().st_size > 0, f"Trace file {f} should not be empty"


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_trainer_profiling(tmp_path):
    launch_torchrun(2, trainer_profiling_worker, tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
def test_trainer_profiling_tensorboard(tmp_path):
    launch_torchrun(2, trainer_profiling_worker_tensorboard, tmp_path)
