"""Bounded, model-independent activation transfer and forward-reuse benchmark.

Run with the host's CUDA environment initialized and an exclusively reserved GPU.
Source creation and correctness checks are outside the measured interval. Direct
copies are a reference implementation, not a theoretical bandwidth bound.
"""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from nnscaler.runtime.activation_config import CPUOffloadConfig
from nnscaler.runtime.activation_store import ActivationStore


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tensors', type=int, default=16)
    parser.add_argument('--tensor-mib', type=int, default=16)
    parser.add_argument('--batch-tensors', type=int, default=None,
                        help='tensors per transfer batch; default submits the whole payload together')
    parser.add_argument('--iterations', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--inflight-mib', type=int, default=512)
    parser.add_argument('--prefetch-mib', type=int, default=512)
    parser.add_argument('--cpu-budget-mib', type=int, default=1024)
    parser.add_argument('--compute-repeats', type=int, default=50)
    parser.add_argument('--compute-dim', type=int, default=2048)
    parser.add_argument('--allocate-on-compute-stream', action='store_true')
    args = parser.parse_args()
    if args.batch_tensors is None:
        args.batch_tensors = args.tensors
    if not 1 <= args.batch_tensors <= args.tensors:
        parser.error('batch-tensors must be within [1, tensors]')
    for name in ('tensors', 'tensor_mib', 'iterations', 'warmup', 'inflight_mib',
                 'prefetch_mib', 'cpu_budget_mib', 'compute_repeats', 'compute_dim'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if (args.tensors > 128 or args.tensors * args.tensor_mib > 1024
            or args.cpu_budget_mib > 2048 or args.compute_dim > 4096
            or args.compute_repeats > 1000 or args.iterations + args.warmup > 100):
        parser.error('Probe limits: 128 tensors, 1 GiB payload, 2 GiB CPU pool, '
                     '4096 compute dimension, 1000 repeats and 100 total iterations')
    if args.output.exists():
        parser.error('Output already exists; preserve the previous result')
    size = args.tensor_mib * 1024**2
    if args.cpu_budget_mib * 1024**2 < (1 << (size-1).bit_length()) * args.tensors:
        parser.error('CPU pool must fit rounded buffers for the fixed transfer volume')
    return args


def main():
    args = arguments()
    torch.cuda.set_device(0)
    free, _ = torch.cuda.mem_get_info()
    if free < 8 * 1024**3:
        raise RuntimeError('Probe requires 8 GiB GPU headroom')
    size = args.tensor_mib * 1024**2
    payload_bytes = args.tensors * size
    cache = ActivationStore(CPUOffloadConfig(
        max_cpu_bytes=args.cpu_budget_mib*1024**2, min_tensor_bytes=0,
        max_inflight_bytes=args.inflight_mib*1024**2,
        max_prefetch_bytes=args.prefetch_mib*1024**2, use_save_events=True,
        reload_on_compute_stream=args.allocate_on_compute_stream), device='cuda')
    direct_host = [torch.empty(size, dtype=torch.uint8, pin_memory=True) for _ in range(args.tensors)]
    d2h, h2d = torch.cuda.Stream(), torch.cuda.Stream()
    lhs = torch.randn(args.compute_dim, args.compute_dim, device='cuda', dtype=torch.bfloat16)
    rhs, scratch = torch.randn_like(lhs), torch.empty_like(lhs)

    def compute():
        for _ in range(args.compute_repeats):
            torch.mm(lhs, rhs, out=scratch)

    # Warm the independent compute kernel before comparing schedules.
    compute()
    torch.cuda.synchronize()

    def iteration(mode, observe=False):
        values = [torch.full((size,), (i+17) % 251, device='cuda', dtype=torch.uint8)
                  for i in range(args.tensors)]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        wall_start = time.perf_counter_ns()
        if mode == 'direct_copy':
            d2h.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(d2h):
                for source, host in zip(values, direct_host):
                    host.copy_(source, non_blocking=True)
                    source.record_stream(d2h)
                ready = torch.cuda.Event()
                ready.record(d2h)
            del source, values
            with torch.cuda.stream(h2d):
                h2d.wait_event(ready)
                restored = [host.to('cuda', non_blocking=True) for host in direct_host]
                done = torch.cuda.Event()
                done.record(h2d)
            torch.cuda.current_stream().wait_event(done)
            for value in restored:
                value.record_stream(torch.cuda.current_stream())
            del value
        else:
            handles = []
            for start_index in range(0, args.tensors, args.batch_tensors):
                with cache.batch() as batch:
                    for index in range(start_index, min(args.tensors, start_index + args.batch_tensors)):
                        handles.append(batch.save(values[index]))
                        values[index] = None  # Transfer retention responsibility to the handle.
            del values
            if mode == 'overlap_compute':
                compute()
            for handle in handles:
                handle.prefetch()
            if mode == 'overlap_compute':
                compute()
            restored = [handle.load() for handle in handles]
            if mode == 'serial_compute':
                compute()
                compute()
            for handle in handles:
                handle.release()
            del handle, handles
        end.record()
        submit_ms = (time.perf_counter_ns() - wall_start) / 1e6
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
        result = dict(submit_ms=submit_ms, wall_ms=wall_ms, gpu_span_ms=start.elapsed_time(end),
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if observe:
            # Validate entire copies outside the timed interval, not just pointers.
            for i, value in enumerate(restored):
                assert torch.all(value == (i+17) % 251).item()
        del restored
        return result

    results = {}
    for mode in ('direct_copy', 'store_roundtrip', 'serial_compute', 'overlap_compute'):
        for _ in range(args.warmup):
            iteration(mode)
        before = cache.snapshot()['pool']
        measured = [iteration(mode) for _ in range(args.iterations)]
        after = cache.snapshot()['pool']
        transferred = {key: after.get(key, 0) - before.get(key, 0)
                       for key in ('d2h_bytes', 'h2d_bytes', 'demand_loads', 'd2h_throttles',
                                   'budget_fallback_bytes', 'prefetch_budget_skips')}
        if mode != 'direct_copy':
            assert transferred['d2h_bytes'] == transferred['h2d_bytes'] == payload_bytes * args.iterations
            assert transferred['budget_fallback_bytes'] == 0
        assert after['live_cpu_bytes'] == after['prefetched_gpu_bytes'] == after['pending_cpu_bytes'] == 0
        with cache.trace(max_events=4096) as trace:
            iteration(mode, observe=True)
        results[mode] = dict(samples=measured, counters=transferred, pool=after,
                            median_wall_ms=statistics.median(v['wall_ms'] for v in measured),
                            median_submit_ms=statistics.median(v['submit_ms'] for v in measured),
                            diagnostic_trace=trace.to_dict())
    cache.close()
    del direct_host, lhs, rhs, scratch
    gc.collect()
    record = dict(configuration={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  torch_version=torch.__version__, gpu=torch.cuda.get_device_name(),
                  bytes_each_direction_per_iteration=payload_bytes, results=results,
                  notes=['Serial/overlap use the same two compute blocks and transfer bytes.',
                         'Source creation, warmup, validation and diagnostic trace are excluded from samples.',
                         'GPU spans include dependencies/host submission gaps, not just DMA kernel time.',
                         'Direct copy uses a whole-batch dependency; it is not a hardware lower bound.',
                         'Pool counters are requested/charged capacity; GPU peaks also include workspaces.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({name: {k: row[k] for k in ('median_wall_ms', 'median_submit_ms')}
                      for name, row in results.items()}, indent=2))


if __name__ == '__main__':
    main()
