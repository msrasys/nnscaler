"""Bounded replay of local virtual-stage F/B actions using the real offload adapter.

Each configuration executes the same actions, bytes and compute. Optional
compute-stream fences make action completion explicit; they do not wait for
independent copy streams. This is a scheduler probe, not model throughput.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from nnscaler.runtime.activation_config import CPUOffloadConfig
from nnscaler.runtime.activation_schedule import ActivationSchedulePlan
from nnscaler.runtime.cpu_offloading import ActivationOffloadManager, CPUOffloadContext


KEYS = (('stage0', 0), ('stage1', 0), ('stage0', 1), ('stage1', 1))
A, B, C, D = KEYS
ACTIONS = ((A, True, True), (B, True, True), (C, True, True), (A, False, True),
           (D, True, True), (B, False, True), (C, False, True), (D, False, True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lookahead', type=int, default=1)
    parser.add_argument('--unit', choices=('auto', 'scope', 'group'), default='auto')
    parser.add_argument('--forward-unit', choices=('auto', 'scope', 'group'), default='auto')
    parser.add_argument('--commit-mib', type=int, default=0)
    parser.add_argument('--copy-chunk-mib', type=int, default=0)
    parser.add_argument('--tensor-mib', type=int, default=64)
    parser.add_argument('--groups', type=int, default=2)
    parser.add_argument('--entry-groups', type=int, default=1,
                        help='backward entry groups when unit=group; scope mode requests the whole unit')
    parser.add_argument('--prefetch-mib', type=int, default=512)
    parser.add_argument('--inflight-mib', type=int, default=512)
    parser.add_argument('--compute-repeats', type=int, default=128)
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--sync-actions', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists')
    if (args.lookahead < 0 or args.lookahead > 64 or not 1 <= args.groups <= 16
            or not 1 <= args.tensor_mib or args.tensor_mib*args.groups*len(KEYS) > 1024
            or not 1 <= args.prefetch_mib <= 2048 or not 1 <= args.inflight_mib <= 2048
            or not 0 <= args.commit_mib <= 2048 or not 0 <= args.copy_chunk_mib <= 1024
            or not 1 <= args.iterations <= 64 or not 1 <= args.warmup <= 8
            or not 1 <= args.entry_groups <= 1024 or not args.groups <= args.compute_repeats <= 1024):
        parser.error('Invalid probe bounds: at most 1 GiB total saved data, 2 GiB windows and 64 iterations')
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 8*1024**3:
        raise RuntimeError('Probe requires 8 GiB GPU headroom')
    module = torch.nn.Module()
    manager = ActivationOffloadManager(module, CPUOffloadConfig(
        max_cpu_bytes=2*1024**3, min_tensor_bytes=0, use_save_events=True,
        max_prefetch_bytes=args.prefetch_mib*1024**2, max_inflight_bytes=args.inflight_mib*1024**2,
        reload_on_compute_stream=False, transfer_unit=args.unit, pipeline_prefetch_steps=args.lookahead,
        forward_commit_unit=args.forward_unit, commit_batch_bytes=args.commit_mib*1024**2,
        copy_chunk_bytes=args.copy_chunk_mib*1024**2,
        scope_prefetch_groups=args.entry_groups))
    module._get_activation_offload_manager = lambda: manager
    module.find_buffer_or_param_owner = lambda tensor: None
    plan = ActivationSchedulePlan.from_actions(ACTIONS)
    distances = {use.key: use.backward-use.forward for use in plan.uses}
    size = args.tensor_mib*1024**2
    lhs = torch.randn(2048,2048,device='cuda',dtype=torch.bfloat16)
    rhs, scratch = torch.randn_like(lhs), torch.empty_like(lhs)

    def compute(count):
        for _ in range(count):
            torch.mm(lhs, rhs, out=scratch)

    compute(args.compute_repeats)
    torch.cuda.synchronize()

    def iteration(diagnostic=False):
        saved, checks, expected, waits, boundaries = {}, [], [], [], []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter_ns()
        with manager.schedule(plan.entries()) as schedule:
            for index, (key, forward, _) in enumerate(ACTIONS):
                schedule.before(index)
                if forward:
                    pairs = []
                    with manager.scope(key, pipeline=True, backward_distance=distances[key]):
                        for group_index in range(args.groups):
                            value = 1 + KEYS.index(key)*args.groups + group_index
                            with CPUOffloadContext(module) as context:
                                tensor = torch.full((size,), value, dtype=torch.uint8, device='cuda')
                                packed = context._pack(tensor)
                                compute(args.compute_repeats//args.groups +
                                        int(group_index < args.compute_repeats % args.groups))
                            pairs.append((context, packed, value))
                            del tensor, packed, context
                    saved[key] = pairs
                    del pairs
                else:
                    before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before.record()
                    pairs = saved.pop(key)
                    for context, packed, value in reversed(pairs):
                        tensor = context._unpack(packed)
                        checks.append(tensor.sum())
                        expected.append(value*size)
                        del tensor
                    after.record()
                    waits.append((before, after))
                    compute(args.compute_repeats)
                    del pairs, context, packed
                    manager.finish(key)
                if args.sync_actions:
                    done = torch.cuda.Event()
                    done.record()
                    done.synchronize()  # Compute stream only; future transfers may remain in flight.
                schedule.after(index)
                if diagnostic:
                    boundaries.append(dict(action=index, key=key, forward=forward,
                                           allocated_bytes=torch.cuda.memory_allocated(),
                                           speculative_bytes=manager.pool.prefetched))
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter_ns()-started)/1e6
        result = dict(wall_ms=wall_ms,
                      backward_read_interval_ms=sum(a.elapsed_time(b) for a,b in waits),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        assert not saved and not manager.scopes
        torch.testing.assert_close(torch.stack(checks).cpu(), torch.tensor(expected), rtol=0, atol=0)
        stats = manager.pool.snapshot()
        assert all(stats[k] == 0 for k in ('live_cpu_bytes','pending_cpu_bytes','prefetched_gpu_bytes',
                                          'budget_fallback_bytes','headroom_fallback_bytes'))
        if diagnostic:
            result['boundaries'] = boundaries
        return result

    for _ in range(args.warmup):
        iteration()
    before = manager.pool.snapshot()
    samples = [iteration() for _ in range(args.iterations)]
    after = manager.pool.snapshot()
    transferred = {k: after.get(k,0)-before.get(k,0) for k in
                   ('d2h_bytes','h2d_bytes','demand_loads','prefetch_loads','prefetch_budget_skips',
                    'd2h_throttles','scope_commits','schedule_prefetch_calls','d2h_copy_chunks','h2d_copy_chunks')}
    expected_bytes = len(KEYS)*args.groups*size*args.iterations
    assert transferred['d2h_bytes'] == transferred['h2d_bytes'] == expected_bytes
    with manager.trace(max_events=10000, cuda_timing=True) as trace:
        diagnostic = iteration(True)
    output = dict(configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  plan=plan.describe(args.lookahead), samples=samples, counters=transferred,
                  diagnostic=diagnostic, trace=trace.to_dict(), pool=manager.pool.snapshot(),
                  median_ms=statistics.median(s['wall_ms'] for s in samples),
                  median_read_interval_ms=statistics.median(s['backward_read_interval_ms'] for s in samples),
                  notes=['Actions replay the same interleaved local F/B order; they are not real model stages.',
                         'Backward read intervals include reductions and CPU submission gaps, not just H2D wait.',
                         'Diagnostic cuda_waits bracket consumer dependencies; they include event overhead and may overlap.',
                         'DMA chunks retain whole-storage ownership; chunk size is not a GPU allocation cap.',
                         'Optional action fences wait only for the current compute stream.',
                         'Checks, trace collection and boundary samples are outside timing samples.'])
    manager.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('median_ms','median_read_interval_ms','counters')},indent=2))


if __name__ == '__main__':
    main()
