"""Observe an unchanged NNScaler Trainer entrypoint with bounded local steps.

Example: torchrun ... profile_train.py --steps 24 --profile-step 8 --output OUT
         /path/to/nnscaler_train.py [original training arguments ...]

Only stop bounds are changed after setup; the original scheduler is constructed
with the original training horizon. Use --profile-step 0 for timing runs.
"""

import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import time

import torch
import torch.distributed as dist

from nnscaler.cli.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--profile-step', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('entrypoint', type=Path)
    parser.add_argument('training_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 <= args.profile_step <= args.steps or args.steps <= 0:
        parser.error('steps must be positive and profile-step must be in [0, steps]')
    args.output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get('RANK', 0))
    output = (args.output / f'rank{rank}.jsonl').open('x')

    def emit(**row):
        output.write(json.dumps(row, allow_nan=False) + '\n')
        output.flush()

    original_setup = Trainer._setup

    def setup(trainer, *a, **kw):
        result = original_setup(trainer, *a, **kw)
        emit(event='setup', configured_steps=trainer.max_train_steps,
             stop_steps=args.steps, profile_step=args.profile_step,
             module_streams=os.environ.get('NNSCALER_MODULE_STREAMS', '[]'),
             torch=torch.__version__, entrypoint=str(args.entrypoint.resolve()))
        trainer.max_train_steps = args.steps
        # Both the inner step loop and the outer epoch loop have stop bounds.
        trainer.train_args.max_train_steps = args.steps
        original_start = trainer.hook.on_step_start
        original_end = trainer.hook.on_step_end
        original_metrics = trainer.hook.before_log_train_metrics
        profiler = marker = None
        started = None

        def start(*a, **kw):
            nonlocal profiler, marker, started
            original_start(*a, **kw)
            step = trainer.train_status.finished_train_steps + 1
            torch.cuda.synchronize()
            dist.barrier()
            if step == args.profile_step:
                profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=False, profile_memory=False, with_stack=False)
                profiler.start()
                marker = torch.profiler.record_function(f'stream_overlap/step/{step}')
                marker.__enter__()
            started = time.perf_counter()

        def end(*a, **kw):
            nonlocal profiler, marker
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            step = trainer.train_status.finished_train_steps
            emit(event='step', step=step, elapsed_s=elapsed, profiled=profiler is not None,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                 peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if profiler is not None:
                marker.__exit__(None, None, None)
                profiler.stop()
                # PyTorch's exporter recognizes .gz and compresses the trace.
                profiler.export_chrome_trace(str(args.output / f'rank{rank}.trace.json.gz'))
                profiler = marker = None
            return original_end(*a, **kw)

        def metrics(_trainer, values, aggregated):
            original_metrics(_trainer, values, aggregated)
            keys = ('train_loss', 'gnorm', 'lr', 'train_wall', 'bsz', 'throughput',
                    'FLOPS', 'FLOPS_VISION', 'z_loss', 'moe_loss', 'mtp_loss')
            emit(event='metrics', step=trainer.train_status.finished_train_steps,
                 values={k: float(values[k]) for k in keys if k in values and values[k] is not None})

        trainer.hook.on_step_start = start
        trainer.hook.on_step_end = end
        trainer.hook.before_log_train_metrics = metrics
        return result

    Trainer._setup = setup
    entrypoint = args.entrypoint.resolve()
    sys.argv = [str(entrypoint), *args.training_args]
    sys.path.insert(0, str(entrypoint.parent))
    try:
        runpy.run_path(str(entrypoint), run_name='__main__')
    finally:
        Trainer._setup = original_setup
        output.close()


if __name__ == '__main__':
    main()
