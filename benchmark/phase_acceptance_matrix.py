#!/usr/bin/env python3
"""Balanced four-way phase-performance acceptance runner.

The driver deliberately starts a fresh 2-GPU torchrun process for every cell:
serial@new, C baseline, D-before-fastpath, and D-after-fastpath.  It measures
max latency across ranks, uses CUDA synchronization, balanced randomized Latin
orders, warmups, per-round medians/MAD, and writes small JSON/CSV artifacts.

Example (worktrees are intentionally detached):

  python benchmark/phase_acceptance_matrix.py \
    --c-worktree /tmp/nnscaler-c --d-old-worktree /tmp/nnscaler-d-old \
    --d-new-worktree /tmp/nnscaler-d-new \
    --output-json docs/phase_acceptance_raw.json \
    --output-csv docs/phase_acceptance_raw.csv

The script is self-hosting: worker subprocesses import nnscaler/tests from the
specified worktree, not from the driver's own checkout.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any


SCALES = {
    'small': dict(dim=128, nheads=4, seqlen=8, ffn_hidden=512, tokens=128, nmicros=4),
    'large': dict(dim=512, nheads=8, seqlen=8, ffn_hidden=2048, tokens=512, nmicros=8),
}
VARIANTS = ('serial', 'c-baseline', 'd-old', 'd-new')


def _load_runtime(worktree: str):
    worktree = str(Path(worktree).resolve())
    os.chdir(worktree)
    sys.path[:] = [worktree] + [path for path in sys.path if Path(path or '.').resolve() != Path(worktree)]
    import torch
    import torch.distributed as dist
    from nnscaler.parallel import ComputeConfig, build_optimizer, parallelize
    from tests.launch_torchrun import launch_torchrun
    from tests.parallel_module.common import init_distributed
    from tests.parallel_module.phase_moe_common import MoEConfig, PhaseMoEModel, make_pas
    from tests.utils import clear_dir_on_rank0, init_random, PYTEST_RUN_ID
    return dict(
        torch=torch, dist=dist, ComputeConfig=ComputeConfig, build_optimizer=build_optimizer,
        parallelize=parallelize, launch_torchrun=launch_torchrun,
        init_distributed=init_distributed, MoEConfig=MoEConfig,
        PhaseMoEModel=PhaseMoEModel, make_pas=make_pas,
        clear_dir_on_rank0=clear_dir_on_rank0, init_random=init_random,
        run_id=PYTEST_RUN_ID,
    )


def _worker(config: dict[str, Any]):
    rt = _load_runtime(config['worktree'])
    torch, dist = rt['torch'], rt['dist']
    rt['init_distributed']()
    rank = dist.get_rank()
    dev = torch.cuda.current_device()
    rt['init_random']()
    scale = config['scale']
    use_phases = config['variant'] != 'serial'
    cfg = rt['MoEConfig'](
        dim=scale['dim'], n_heads=scale['nheads'], seq_len=scale['seqlen'],
        ffn_hidden=scale['ffn_hidden'], capacity_factor=1.0,
    )
    artifact_dir = Path('/tmp') / f"phase_acceptance_{rt['run_id']}_{config['scale_name']}_{config['variant']}_{config['round']}"
    with rt['clear_dir_on_rank0'](artifact_dir) as tempdir:
        model = rt['PhaseMoEModel'](cfg, 1, 2, [(0, 1)], use_phases=use_phases)
        make_pas = rt['make_pas']
        pas_kwargs = {}
        parameters = inspect.signature(make_pas).parameters
        # Keep D variants on the measured no-extra-stream path where supported.
        if 'dedicated_moe_comm_stream' in parameters:
            pas_kwargs['dedicated_moe_comm_stream'] = False
        pas = make_pas(1, 2, [(0, 1)], use_phases=use_phases, **pas_kwargs)
        compiled = rt['parallelize'](
            model,
            {'data': {'data': torch.randn(scale['tokens'], scale['dim'], device=dev),
                      'target': torch.randn(scale['tokens'], scale['dim'], device=dev)}},
            pas,
            rt['ComputeConfig'](2, 2, use_end2end=True, use_async_recv=True,
                                pas_config=dict(pipeline_nmicros=scale['nmicros'])),
            gen_savedir=tempdir,
            instance_name=f"phase_acceptance_{config['variant']}",
        )
        compiled.cuda()
        optimizer = rt['build_optimizer'](compiled, torch.optim.Adam, lr=0.01)
        generator = torch.Generator().manual_seed(10_000 + config['round'])
        samples = []
        for step in range(config['warmup'] + config['timed']):
            batch = [
                {'data': torch.randn(scale['tokens'], scale['dim'], generator=generator, device='cpu').to(dev),
                 'target': torch.randn(scale['tokens'], scale['dim'], generator=generator, device='cpu').to(dev)}
                for _ in range(scale['nmicros'])
            ]
            dist.barrier()
            torch.cuda.synchronize()
            start = __import__('time').perf_counter()
            compiled.train_step(batch)
            torch.cuda.synchronize()
            elapsed = __import__('time').perf_counter() - start
            maximum = torch.tensor(elapsed, device=dev)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            if step >= config['warmup'] and rank == 0:
                samples.append(float(maximum.cpu()))
    return {'rank': rank, 'samples': samples}


def _run_worker(args, config: dict[str, Any]) -> dict[str, Any]:
    command = [
        sys.executable, str(Path(__file__).resolve()), '--worker',
        '--config', json.dumps(config, separators=(',', ':')),
    ]
    env = os.environ.copy()
    env['NNSCALER_BENCH_WORKTREE'] = config['worktree']
    completed = subprocess.run(command, cwd=config['worktree'], env=env,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               check=False)
    marker = 'PHASE_ACCEPTANCE_JSON='
    lines = [line for line in completed.stdout.splitlines() if line.startswith(marker)]
    if completed.returncode != 0 or not lines:
        raise RuntimeError(
            f"benchmark worker failed ({completed.returncode}) for {config}:\n"
            f"{completed.stdout[-8000:]}"
        )
    return json.loads(lines[-1][len(marker):])


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for scale_name in SCALES:
        result[scale_name] = {}
        for variant in VARIANTS:
            values = [row['round_median_s'] for row in rows
                      if row['scale'] == scale_name and row['variant'] == variant]
            median = statistics.median(values)
            mad = statistics.median(abs(value - median) for value in values)
            result[scale_name][variant] = dict(
                rounds=len(values), median_s=median, mad_s=mad,
                values_s=values,
            )
    return result


def _driver(args) -> None:
    worktrees = {
        'serial': str(Path(args.d_new_worktree).resolve()),
        'c-baseline': str(Path(args.c_worktree).resolve()),
        'd-old': str(Path(args.d_old_worktree).resolve()),
        'd-new': str(Path(args.d_new_worktree).resolve()),
    }
    for name, worktree in worktrees.items():
        if not (Path(worktree) / 'nnscaler').is_dir():
            raise ValueError(f'{name} worktree is not an nnscaler checkout: {worktree}')

    rng = random.Random(args.seed)
    rows = []
    for scale_name, scale in SCALES.items():
        base_order = list(VARIANTS)
        rng.shuffle(base_order)
        # Rotating one shuffled ordering gives every variant every ordinal
        # position exactly twice over eight rounds.
        for round_index in range(args.rounds):
            shift = round_index % len(VARIANTS)
            order = base_order[shift:] + base_order[:shift]
            for ordinal, variant in enumerate(order):
                config = dict(
                    worktree=worktrees[variant], variant=variant,
                    scale_name=scale_name, scale=scale, round=round_index,
                    warmup=args.warmup, timed=args.timed,
                )
                result = _run_worker(args, config)
                samples = result['samples']
                row = dict(
                    scale=scale_name, round=round_index, ordinal=ordinal,
                    variant=variant, worktree=worktrees[variant],
                    rank_max_samples_s=samples,
                    round_median_s=statistics.median(samples),
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True))

    payload = dict(
        seed=args.seed, rounds=args.rounds, warmup=args.warmup, timed=args.timed,
        rows=rows, summary=_summary(rows),
    )
    json_path = Path(args.output_json)
    csv_path = Path(args.output_csv)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2) + '\n')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'scale', 'round', 'ordinal', 'variant', 'worktree',
            'round_median_s', 'rank_max_samples_s',
        ))
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out['rank_max_samples_s'] = json.dumps(out['rank_max_samples_s'])
            writer.writerow(out)
    print(json.dumps(payload['summary'], indent=2))


def _worker_main(args) -> None:
    config = json.loads(args.config)
    rt = _load_runtime(config['worktree'])
    outputs = rt['launch_torchrun'](2, _worker, config)
    result = outputs[0]
    print('PHASE_ACCEPTANCE_JSON=' + json.dumps(result, separators=(',', ':')))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--config')
    parser.add_argument('--c-worktree')
    parser.add_argument('--d-old-worktree')
    parser.add_argument('--d-new-worktree')
    parser.add_argument('--output-json', default='docs/phase_acceptance_raw.json')
    parser.add_argument('--output-csv', default='docs/phase_acceptance_raw.csv')
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--timed', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260805)
    args = parser.parse_args()
    if args.worker:
        _worker_main(args)
    else:
        required = (args.c_worktree, args.d_old_worktree, args.d_new_worktree)
        if not all(required):
            parser.error('--c-worktree, --d-old-worktree, and --d-new-worktree are required')
        _driver(args)


if __name__ == '__main__':
    main()
