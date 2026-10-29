#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Unit tests of ``nnscaler.runtime.cpu_offload``, one section per module."""

import pytest
import torch

from nnscaler.runtime import cpu_offloading   # compatibility shim
from nnscaler.runtime.cpu_offload import (
    CPUOffloadContext, OffloadConfig, OffloadManager, comm_exclusive, configure,
    hint_forward, offload_report, offload_stats,
)
from nnscaler.runtime.cpu_offload.config import COMM_EXCLUSIVE_MODES, FAKE_MODES
from nnscaler.runtime.cpu_offload.hooks import ModuleTensorRef, OffloadedTensor, covered_bytes
from nnscaler.runtime.cpu_offload.host_memory import MemPolicy, PinnedPool, gpu_numa_info
from nnscaler.runtime.cpu_offload.stats import GroupProfile, OffloadStats, SizeHistogram, format_summary
from nnscaler.runtime.cpu_offload.transport import RANGE_ALIGN, REGION_ALIGN
from nnscaler.runtime.module import _ModuleTensorRegistry, ParallelModule

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')


class _Host(torch.nn.Module):
    """Stand-in for a ParallelModule: offloading only needs the owner lookup."""

    def __init__(self, width=256, depth=3):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(width, width) for _ in range(depth)])
        self._registry = _ModuleTensorRegistry(self)

    def find_buffer_or_param_owner(self, tensor):
        return self._registry.find(tensor)


@pytest.fixture(autouse=True)
def _fresh_manager(monkeypatch):
    monkeypatch.delenv('CPU_OFFLOADING_DISABLE', raising=False)
    OffloadManager.reset()
    yield
    OffloadManager.reset()


def _manager(**overrides):
    cfg = OffloadConfig(min_bytes=1024, numa_membind=False)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return configure(cfg)


def _segment_forward(host, x, name, manager):
    ctx = manager.begin_forward(name)
    h = x
    try:
        for i, layer in enumerate(host.layers):
            with CPUOffloadContext(host, group_id=i):
                h = torch.nn.functional.gelu(layer(h))
    finally:
        manager.end_forward(ctx)
    return h, ctx


def _segment_backward(loss, name, manager):
    ctx = manager.begin_backward(name)
    try:
        loss.backward()
    finally:
        manager.end_backward(ctx)
    return ctx


def _run_reference(host, x):
    h = x
    for layer in host.layers:
        h = torch.nn.functional.gelu(layer(h))
    return h


# --------------------------------------------------------------------------- config

def test_config_from_env_parses_every_knob():
    env = {
        'CPU_OFFLOADING_DISABLE': '1', 'CPU_OFFLOADING_MIN_BYTES': '4096',
        'CPU_OFFLOADING_MIN_BACKWARD_DISTANCE': '3', 'CPU_OFFLOADING_PREFETCH_GROUPS': '4',
        'CPU_OFFLOADING_PREFETCH_MB': '2', 'CPU_OFFLOADING_MAX_INFLIGHT_COPIES': '8',
        'CPU_OFFLOADING_COPY_CHUNK_MB': '32', 'CPU_OFFLOADING_COMM_EXCLUSIVE': '1',
        'CPU_OFFLOADING_NUMA_MEMBIND': '0', 'CPU_OFFLOADING_NUMA_AFFINITY': '1',
        'CPU_OFFLOADING_POOL_MAX_GB': '1.5', 'CPU_OFFLOADING_TIMING': '0',
        'CPU_OFFLOADING_DEBUG': '1', 'CPU_OFFLOADING_PROFILE_EVERY': '10',
        'CPU_OFFLOADING_FAKE_COPY': '2',
    }
    cfg = OffloadConfig.from_env(env)
    assert not cfg.enabled and cfg.min_bytes == 4096 and cfg.min_backward_distance == 3
    assert cfg.prefetch_groups == 4 and cfg.prefetch_bytes == 2 << 20
    assert cfg.max_inflight_copies == 8 and cfg.copy_chunk_bytes == 32 << 20
    assert cfg.comm_exclusive == 'strict' and cfg.fake_mode == 'd2h_only'
    assert not cfg.numa_membind and cfg.numa_affinity and cfg.pool_max_bytes == int(1.5 * 2**30)
    assert not cfg.timing and cfg.debug and cfg.profile_every == 10
    # names are accepted as well as the legacy integers
    assert OffloadConfig.from_env({'CPU_OFFLOADING_COMM_EXCLUSIVE': 'soft'}).comm_exclusive == 'soft'
    assert OffloadConfig.from_env({'CPU_OFFLOADING_FAKE_COPY': 'both_keep'}).fake_mode == 'both_keep'


def test_config_validation():
    with pytest.raises(ValueError, match='fake_mode'):
        OffloadConfig(fake_mode='nope').validate()
    with pytest.raises(ValueError, match='comm_exclusive'):
        OffloadConfig(comm_exclusive='maybe').validate()
    with pytest.raises(ValueError, match='non-negative'):
        OffloadConfig(prefetch_groups=-1).validate()
    with pytest.raises(ValueError, match='is not valid'):
        OffloadConfig.from_env({'CPU_OFFLOADING_MIN_BYTES': 'lots'})
    assert set(FAKE_MODES) == {'off', 'no_copy', 'd2h_only', 'both_keep'}
    assert set(COMM_EXCLUSIVE_MODES) == {'off', 'strict', 'soft'}


def test_configure_overrides_and_rejects_unknown_fields():
    manager = configure(OffloadConfig(numa_membind=False), prefetch_groups=5)
    assert manager.config.prefetch_groups == 5
    assert OffloadManager.get() is manager
    with pytest.raises(TypeError):
        configure(OffloadConfig(), not_a_field=1)


# --------------------------------------------------------------------------- host memory

def test_pool_slab_sizes_round_up_to_a_bounded_quantum():
    assert PinnedPool.slab_size(1) == PinnedPool.MIN_SLAB
    for nbytes in (3 << 20, 100 << 20, (1 << 30) + 12345):
        size = PinnedPool.slab_size(nbytes)
        assert size >= nbytes and size <= nbytes * (1 + 1 / 16) + PinnedPool.MIN_SLAB
    assert PinnedPool.slab_size((100 << 20) + 1000) == PinnedPool.slab_size((100 << 20) + 2000)


@cuda_only
def test_pool_best_fit_reuse_and_bound():
    pool = PinnedPool()
    stream = torch.cuda.Stream()
    small = pool.acquire(1 << 20, stream)
    large = pool.acquire(8 << 20, stream)
    assert pool.allocations == 2 and small.is_pinned() and large.numel() >= 8 << 20
    pool.release(large, None)
    pool.release(small, None)
    assert pool.acquire(1 << 20, stream) is small        # smallest fitting slab
    assert pool.acquire(3 << 20, stream) is large        # next one that fits
    assert pool.allocations == 2 and pool.free_bytes == 0

    # a busy slab is skipped while the pool may still grow ...
    done = torch.cuda.Event(); done.record(stream)
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000_000)                 # ~1 s of GPU time
    busy = torch.cuda.Event(); busy.record(stream)
    pool.release(small, busy)
    other = pool.acquire(1 << 20, stream)
    assert other is not small and pool.allocations == 3
    # ... and waited for at the bound
    bounded = PinnedPool(max_bytes=PinnedPool.MIN_SLAB)
    a = bounded.acquire(10, stream)
    bounded.release(a, done)
    assert bounded.acquire(10, stream) is a and bounded.waits == 0
    bounded.release(a, busy)
    assert bounded.acquire(10, stream) is a and bounded.waits == 1 and bounded.allocations == 1
    torch.cuda.synchronize()


def test_mempolicy_without_node_is_noop():
    with MemPolicy(None) as policy:
        assert not policy.failed


@cuda_only
def test_gpu_numa_info_shape():
    node, cpus = gpu_numa_info(torch.cuda.current_device())
    assert node is None or node >= 0
    assert isinstance(cpus, set)


# --------------------------------------------------------------------------- hooks / transport

def test_covered_bytes_handles_views_and_empty_tensors():
    base = torch.empty(8, 16)
    assert covered_bytes(base) == (0, 8 * 16 * 4)
    assert covered_bytes(base[:, :8]) == (0, (7 * 16 + 8) * 4)
    assert covered_bytes(base[2:4]) == (2 * 16 * 4, 4 * 16 * 4)
    assert covered_bytes(base[:0]) == (0, 0)


@cuda_only
def test_parameters_are_referenced_not_offloaded():
    host = _Host().cuda(); manager = _manager()
    ctx = manager.begin_forward('seg')
    context = CPUOffloadContext(host)
    with context:
        packed = context._pack(host.layers[0].weight)
        packed_act = context._pack(torch.randn(64, 256, device='cuda'))
    manager.end_forward(ctx)
    assert isinstance(packed, ModuleTensorRef) and isinstance(packed_act, OffloadedTensor)
    assert torch.equal(packed.unpack(), host.layers[0].weight)


@cuda_only
def test_small_tensors_stay_on_device_but_are_counted():
    host = _Host().cuda(); manager = _manager(min_bytes=1 << 20)
    ctx = manager.begin_forward('seg')
    context = CPUOffloadContext(host)
    with context:
        small = torch.randn(8, device='cuda')
        assert context._pack(small) is small
    manager.end_forward(ctx)
    assert offload_stats()['skipped_small_bytes'] == 32
    assert manager._profile_sizes.counts == {32 .bit_length(): 1}
    assert '*' in manager._profile_sizes.format(1, min_bytes=1 << 20)


@cuda_only
def test_overlapping_views_share_one_region_and_are_rebuilt_with_strides():
    host = _Host().cuda(); manager = _manager()
    base = torch.randn(8, 1024, device='cuda')
    other = torch.randn(64, 64, device='cuda')
    ctx = manager.begin_forward('seg')
    context = CPUOffloadContext(host)
    with context:
        a = context._pack(base[:, :512])
        b = context._pack(base[:, 512:])
        c = context._pack(base.t())
        c2 = context._pack(base.t())       # identical view
        d = context._pack(other[4:])       # different storage
        e = context._pack(base[:0])        # empty view: nothing to move
    manager.end_forward(ctx)
    transfer = context.batch.transfer
    # the three views of ``base`` overlap: one region; ``other`` gets its own
    assert len(transfer.regions) == 2 and a.region is b.region is c.region is c2.region
    assert d.region is not a.region and isinstance(e, torch.Tensor)
    assert a.region.lo % RANGE_ALIGN == 0 and d.region.offset % REGION_ALIGN == 0
    stats = offload_stats(reset=False)
    assert stats['offloaded_regions'] == 2 and stats['transfers'] == 1
    assert stats['offloaded_bytes'] == transfer.nbytes >= base.numel() * 4 + other[4:].numel() * 4
    # saved bytes minus copied bytes: the four views of ``base`` share one copy
    saved = sum(hi - lo for lo, hi in map(covered_bytes, (base[:, :512], base[:, 512:], base.t(), base.t())))
    assert stats['deduplicated_bytes'] == saved - a.region.nbytes
    manager.begin_backward('seg')
    torch.cuda.synchronize()
    assert torch.equal(context._unpack(a), base[:, :512])
    assert torch.equal(context._unpack(b), base[:, 512:])
    assert torch.equal(context._unpack(c), base.t())
    assert torch.equal(context._unpack(d), other[4:])
    assert context._unpack(e).shape == (0, 1024)
    manager.end_backward(manager.tracker._backward_stack[-1]) if manager.tracker._backward_stack else None
    assert transfer.device_buffer is None or transfer.unconsumed > 0


@cuda_only
def test_chunked_reload_waits_only_for_the_needed_chunk():
    torch.manual_seed(0)
    host = _Host(width=1024).cuda()
    manager = _manager(copy_chunk_bytes=256 << 10)
    x = torch.randn(256, 1024, device='cuda', requires_grad=True)
    h, ctx = _segment_forward(host, x, 'seg', manager)
    transfers = [b.transfer for b in ctx.batches]
    _segment_backward(h.float().sum(), 'seg', manager)
    assert all(len(t._chunk_events) > 1 for t in transfers)
    assert x.grad is not None


@cuda_only
def test_unpack_before_group_end_commits_on_demand():
    host = _Host().cuda(); manager = _manager()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    ctx = manager.begin_forward('seg')
    with CPUOffloadContext(host, group_id=0) as context:
        h = torch.nn.functional.gelu(host.layers[0](x))
        h.float().sum().backward()           # backward inside the offload block
    manager.end_forward(ctx)
    assert context.batch.transfer.committed and x.grad is not None


# --------------------------------------------------------------------------- tracker

@cuda_only
def test_contexts_are_consumed_fifo_per_segment():
    host = _Host().cuda(); manager = _manager(prefetch_groups=1)
    x0 = torch.randn(64, 256, device='cuda', requires_grad=True)
    x1 = torch.randn(64, 256, device='cuda', requires_grad=True)
    h0, ctx0 = _segment_forward(host, x0, 'seg', manager)
    h1, ctx1 = _segment_forward(host, x1, 'seg', manager)
    assert offload_stats(reset=False)['pending_contexts'] == 2
    popped = manager.begin_backward('seg')
    assert popped is ctx0
    assert ctx0.batches[-1].prefetched and ctx0.batches[-2].prefetched and not ctx0.batches[0].prefetched
    assert not ctx1.batches[-1].prefetched
    manager.prefetch_next_backward('seg')
    assert ctx1.batches[-1].prefetched
    h0.float().sum().backward(); manager.end_backward(popped)
    assert manager.begin_backward('seg') is ctx1
    h1.float().sum().backward(); manager.end_backward(ctx1)
    assert offload_stats()['pending_contexts'] == 0


@cuda_only
def test_prefetch_bytes_budget():
    host = _Host(width=512, depth=4).cuda()
    manager = _manager(prefetch_groups=0, prefetch_bytes=1)   # 1 byte: exactly one extra group
    x = torch.randn(64, 512, device='cuda', requires_grad=True)
    h, ctx = _segment_forward(host, x, 'seg', manager)
    manager.begin_backward('seg')
    flags = [b.prefetched for b in ctx.batches]
    assert flags == [False, False, True, True]
    h.float().sum().backward(); manager.end_backward(ctx)


@cuda_only
def test_hint_forward_disables_offload_for_immediate_backward():
    host = _Host().cuda(); manager = _manager()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    hint_forward('seg', 0)
    h, ctx = _segment_forward(host, x, 'seg', manager)
    assert not ctx.offload_enabled and not ctx.batches
    h.float().sum().backward()
    hint_forward('seg', 5)
    h, ctx = _segment_forward(host, x, 'seg', manager)
    assert ctx.offload_enabled and ctx.batches
    _segment_backward(h.float().sum(), 'seg', manager)


@cuda_only
def test_eager_chain_prefetch_without_executor():
    host = _Host().cuda(); _manager()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    h = x; contexts = []
    for i, layer in enumerate(host.layers):
        context = CPUOffloadContext(host, group_id=i)
        with context:
            h = torch.nn.functional.gelu(layer(h))
        contexts.append(context)
    assert all(c.batch.context is None for c in contexts)
    h.float().sum().backward()
    assert contexts[-2].batch.prefetched
    # nothing releases eager batches later: the slabs return to the pool once consumed
    assert all(c.batch.released for c in contexts)
    assert OffloadManager.get().pool.free_bytes == OffloadManager.get().pool.allocated_bytes


@cuda_only
def test_repeated_unpack_reloads_from_the_slab():
    host = _Host().cuda(); manager = _manager()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    h, ctx = _segment_forward(host, x, 'seg', manager)
    loss = h.float().sum()
    popped = manager.begin_backward('seg')
    loss.backward(retain_graph=True)
    first = x.grad.clone(); x.grad = None
    loss.backward()
    manager.end_backward(popped)
    assert torch.equal(first, x.grad)
    assert offload_stats()['prefetched_regions'] > offload_stats(reset=False)['offloaded_regions']


# --------------------------------------------------------------------------- transport policies

@cuda_only
@pytest.mark.parametrize('fake_mode', ['no_copy', 'd2h_only', 'both_keep'])
def test_fake_modes_keep_results_and_report_bytes(fake_mode):
    torch.manual_seed(0)
    host = _Host().cuda(); manager = _manager(fake_mode=fake_mode)
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    h, _ = _segment_forward(host, x, 'seg', manager)
    _segment_backward(h.float().square().mean(), 'seg', manager)
    stats = offload_stats()
    assert stats['offloaded_regions'] > 0
    if fake_mode == 'both_keep':
        assert stats['prefetched_regions'] == stats['offloaded_regions']
    if fake_mode == 'no_copy':
        # nothing moves: no copy events, no stalls, no pinned memory
        assert stats['stalls'] == 0 and stats['d2h_ms'] == 0 and stats['pool_allocated_bytes'] == 0
    assert x.grad is not None


@cuda_only
def test_inflight_window_defers_and_forces_copies():
    # large tensors so the copies are still running when the next one is scheduled
    host = _Host(width=2048, depth=6).cuda(); manager = _manager(max_inflight_copies=1, prefetch_groups=6)
    x = torch.randn(2048, 2048, device='cuda', requires_grad=True)
    h, ctx = _segment_forward(host, x, 'seg', manager)
    stats = offload_stats(reset=False)
    assert stats['deferred_d2h'] > 0
    _segment_backward(h.float().sum(), 'seg', manager)
    stats = offload_stats()
    assert stats['pending_contexts'] == 0 and stats['deferred_h2d'] + stats['forced_h2d'] > 0


@cuda_only
def test_comm_exclusive_window_counts_and_keeps_gradients():
    torch.manual_seed(0)
    host = _Host().cuda(); manager = _manager(comm_exclusive='strict')
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    ctx = manager.begin_forward('seg')
    h = x
    for i, layer in enumerate(host.layers):
        with CPUOffloadContext(host, group_id=i):
            h = torch.nn.functional.gelu(layer(h))
        with comm_exclusive():          # stands in for an all-to-all
            h = h * 1.0
    manager.end_forward(ctx)
    _segment_backward(h.float().sum(), 'seg', manager)
    assert offload_stats()['comm_windows'] == len(host.layers)
    assert x.grad is not None


# --------------------------------------------------------------------------- stats / manager

def test_stats_reset_keeps_state_fields():
    s = OffloadStats(offloaded_bytes=10, pool_allocated_bytes=7, pending_contexts=2)
    s.reset()
    assert s.offloaded_bytes == 0 and s.pool_allocated_bytes == 7 and s.pending_contexts == 2


def test_size_histogram_buckets_and_format():
    hist = SizeHistogram()
    for nbytes in (100, 1000, (1 << 20) - 1, 1 << 20):
        hist.add(nbytes)
    # buckets are [2**(b-1), 2**b): 1 MiB - 1 falls below 1M, 1 MiB itself below 2M
    assert hist.counts[20] == 1 and hist.counts[21] == 1
    text = hist.format(steps=1, min_bytes=1 << 20)
    assert '<128B*' in text and '<1K*' in text and '<1M*' in text and '<2M:' in text and 'below min_bytes' in text
    merged = SizeHistogram(); merged.merge(hist); merged.merge(hist)
    assert merged.counts == {k: 2 * v for k, v in hist.counts.items()}


def test_format_summary_is_readable():
    groups = {3: GroupProfile(executions=8, tensors=40, regions=24, bytes=8 * 400 << 20, d2h_ms=80.0)}
    total = OffloadStats(offloaded_bytes=8 * 400 << 20, offloaded_regions=24, transfers=8, packed_tensors=40,
                         d2h_ms=80.0, prefetched_bytes=8 * 400 << 20, h2d_ms=60.0)
    text = format_summary(groups, steps=2, stats_total=total, sizes=SizeHistogram(), min_bytes=1 << 20)
    assert 'GiB/step' in text and '4 transfers / 12 regions' in text
    assert text.splitlines()[-1].split()[0] == '3'


@cuda_only
def test_snapshot_and_report_roundtrip():
    torch.manual_seed(0)
    host = _Host().cuda(); manager = _manager(profile_every=0)
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    for _ in range(2):
        h, _ = _segment_forward(host, x, 'seg', manager)
        _segment_backward(h.float().sum(), 'seg', manager)
        offload_stats()
    text = offload_report()
    assert 'profile over 2 step(s)' in text and 'saved tensor sizes' in text
    rows = [line.split() for line in text.splitlines()[3:]]
    assert [r[0] for r in rows] == [str(gid) for gid in range(len(host.layers))]
    assert all(float(r[1]) == 1.0 for r in rows)        # executions per step


@cuda_only
def test_gradients_bitwise_identical_with_and_without_offload():
    torch.manual_seed(0)
    host = _Host().cuda()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)

    def run(offload):
        manager = _manager() if offload else None
        host.zero_grad(); x.grad = None
        if offload:
            h, _ = _segment_forward(host, x, 'seg', manager)
            loss = h.float().square().mean()
            _segment_backward(loss, 'seg', manager)
        else:
            loss = _run_reference(host, x).float().square().mean()
            loss.backward()
        return loss.detach().clone(), x.grad.clone(), [p.grad.clone() for p in host.parameters()]

    ref = run(False)
    for _ in range(2):
        out = run(True)
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
        assert all(torch.equal(a, b) for a, b in zip(out[2], ref[2]))


def test_disable_turns_contexts_into_noops():
    if not torch.cuda.is_available():
        pytest.skip('lack of gpu devices')
    host = _Host().cuda(); _manager(enabled=False)
    context = CPUOffloadContext(host)
    with context:
        assert context._hooks is None


def test_compatibility_shim_exposes_legacy_names():
    assert cpu_offloading.CPUOffloadContext is CPUOffloadContext
    assert cpu_offloading._OffloadedTensor is OffloadedTensor
    assert cpu_offloading._PREFETCH_STREAM_NAME == 'cpu_offload_h2d'


def test_parallel_module_cpu_offloading_hooks_accepts_group_id():
    class _Module(ParallelModule, skip_init=True):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self._module_tensor_registry = None

    context = _Module().cpu_offloading_hooks(group_id=7)
    assert isinstance(context, CPUOffloadContext) and context.group_id == 7


# --------------------------------------------------------------------------- tracker / executor contract

@cuda_only
def test_tracker_queue_mirrors_executor_when_a_forward_is_not_offloaded():
    host = _Host().cuda(); manager = _manager(min_backward_distance=2)
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    hint_forward('seg', 1)                       # too close: not offloaded, still queued
    h0, ctx0 = _segment_forward(host, x, 'seg', manager)
    hint_forward('seg', 2)
    h1, ctx1 = _segment_forward(host, x, 'seg', manager)
    assert not ctx0.offload_enabled and not ctx0.batches and ctx1.batches
    assert offload_stats(reset=False)['pending_contexts'] == 2
    assert manager.begin_backward('seg') is ctx0     # FIFO holds even for the empty one
    h0.float().sum().backward(); manager.end_backward(ctx0)
    assert manager.begin_backward('seg') is ctx1
    h1.float().sum().backward(); manager.end_backward(ctx1)
    assert offload_stats()['pending_contexts'] == 0 and x.grad is not None


@cuda_only
def test_skipped_forward_drops_its_hint_and_unbalanced_backward_raises():
    host = _Host().cuda(); manager = _manager()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    hint_forward('seg', 0)
    manager.forward_skipped('seg')               # Executor.fexecute(requires_grad=False)
    h, ctx = _segment_forward(host, x, 'seg', manager)
    assert ctx.offload_enabled and ctx.batches  # the stale hint did not disable it
    _segment_backward(h.float().sum(), 'seg', manager)
    with pytest.raises(RuntimeError, match='without a pending forward'):
        manager.begin_backward('seg')


@cuda_only
def test_announced_next_backward_is_prefetched_once_the_current_one_is_popped():
    host = _Host().cuda(); manager = _manager(prefetch_groups=1)
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    h0, ctx0 = _segment_forward(host, x, 'seg', manager)
    h1, ctx1 = _segment_forward(host, x, 'seg', manager)
    manager.prefetch_next_backward('seg')        # names the backward *after* the next one
    assert not ctx1.batches[-1].prefetched       # nothing happens before the current one is popped
    popped = manager.begin_backward('seg')
    assert popped is ctx0 and ctx0.batches[-1].prefetched and ctx1.batches[-1].prefetched
    assert not ctx1.batches[0].prefetched
    h0.float().sum().backward(); manager.end_backward(popped)
    ctx = manager.begin_backward('seg')
    h1.float().sum().backward(); manager.end_backward(ctx)


@cuda_only
def test_prefetch_depth_is_exactly_prefetch_groups_ahead():
    host = _Host(depth=5).cuda(); manager = _manager(prefetch_groups=1)
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    h, ctx = _segment_forward(host, x, 'seg', manager)
    manager.begin_backward('seg')
    flags = lambda: [b.prefetched for b in ctx.batches]
    assert flags() == [False, False, False, True, True]
    manager.tracker.on_unpack(ctx.batches[4].handles[0])    # demanded: group 4 -> group 3 already in flight
    assert flags() == [False, False, False, True, True]
    manager.tracker.on_unpack(ctx.batches[3].handles[0])    # demanded: group 3 -> group 2 starts
    assert flags() == [False, False, True, True, True]
    h.float().sum().backward(); manager.end_backward(ctx)


@cuda_only
def test_saves_after_an_early_commit_stay_on_device():
    host = _Host().cuda(); manager = _manager()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    ctx = manager.begin_forward('seg')
    with CPUOffloadContext(host, group_id=0) as context:
        y = torch.nn.functional.gelu(host.layers[0](x))
        torch.autograd.grad(y.float().sum(), x, retain_graph=True)   # unpack inside the group: commits
        z = torch.nn.functional.gelu(host.layers[1](y))              # saved after the commit
    manager.end_forward(ctx)
    stats = offload_stats(reset=False)
    assert context.batch.transfer.committed and stats['skipped_late_bytes'] > 0
    _segment_backward(z.float().sum(), 'seg', manager)
    assert x.grad is not None


@cuda_only
def test_soft_comm_window_holds_copies_until_it_ends():
    host = _Host().cuda(); manager = _manager(comm_exclusive='soft')
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    ctx = manager.begin_forward('seg')
    with comm_exclusive():
        with CPUOffloadContext(host, group_id=0) as context:
            h = torch.nn.functional.gelu(host.layers[0](x))
        assert not context.batch.transfer.d2h_issued and offload_stats(reset=False)['deferred_d2h'] == 1
    assert context.batch.transfer.d2h_issued        # issued when the window closed
    manager.end_forward(ctx)
    _segment_backward(h.float().sum(), 'seg', manager)
    assert x.grad is not None


@cuda_only
def test_timing_state_stays_bounded_without_polling():
    host = _Host().cuda(); manager = _manager()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)
    for _ in range(6):
        h, _ = _segment_forward(host, x, 'seg', manager)
        _segment_backward(h.float().sum(), 'seg', manager)
    torch.cuda.synchronize()
    manager.end_backward(None)                       # folds whatever finished
    assert len(manager.timing._d2h) + len(manager.timing._h2d) + len(manager.timing._stalls) == 0
    assert manager.stats.d2h_ms > 0
    text = offload_report()                          # no snapshot so far: counts as one step
    assert 'profile over 1 step(s)' in text and '18 transfers' in text


@cuda_only
def test_executor_backward_input_and_weight_release_the_context():
    from nnscaler.runtime.executor import Executor
    host = _Host().cuda(); manager = _manager()
    x = torch.randn(64, 256, device='cuda', requires_grad=True)

    def segment(inp):
        h = inp
        for i, layer in enumerate(host.layers):
            with CPUOffloadContext(host, group_id=i):
                h = torch.nn.functional.gelu(layer(h))
        return h

    out = Executor.fexecute('seg', segment, x)
    assert offload_stats(reset=False)['pending_contexts'] == 1
    Executor.backward_input('seg', [x], [out], [torch.ones_like(out)], host.parameters())
    assert offload_stats(reset=False)['pending_contexts'] == 0      # popped with the detached inputs
    assert all(p.grad is None for p in host.parameters())
    Executor.backward_weight('seg', host.parameters())
    assert all(p.grad is not None for p in host.parameters())
    assert manager.pool.free_bytes == manager.pool.allocated_bytes  # slabs released after the weight pass
