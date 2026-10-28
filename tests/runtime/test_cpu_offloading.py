# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
import gc
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nnscaler.parallel import ComputeConfig
from nnscaler.runtime.cpu_offloading import CPUOffloadConfig, mark_activation_resident
from nnscaler.runtime.module import ParallelModule, _ModuleTensorRegistry


class OffloadModule(ParallelModule, skip_init=True):
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.compute_config = SimpleNamespace(cpu_offload=CPUOffloadConfig(
            **dict(dict(max_cpu_bytes=32 * 1024**2, min_tensor_bytes=0,
                        host_reserve_bytes=0), **kwargs)))
        self.weight = torch.nn.Parameter(torch.randn(64, 64, device='cuda'))
        self.register_buffer('buffer', torch.randn(64, 64, device='cuda'))
        self._module_tensor_registry = _ModuleTensorRegistry(self)


@pytest.fixture(autouse=True)
def collect():
    from nnscaler.runtime.cpu_offloading import _pools
    gc.collect()
    _pools.clear()  # each test is an independent application configuration
    yield
    gc.collect()


@pytest.mark.parametrize('field,value', [('max_cpu_bytes', -1), ('prefetch_groups', True),
                                        ('max_prefetch_bytes', 1.5), ('min_offload_size_gap_bytes', -1),
                                        ('use_save_events', 1)])
def test_invalid_config(field, value):
    with pytest.raises(ValueError, match=field):
        CPUOffloadConfig(**{field: value})


@pytest.mark.parametrize('stats,expected', [
    ('inactive_file 50\nfile_dirty 0\nfile_writeback 0\n', 60),
    ('inactive_file 50\nfile_dirty 20\nfile_writeback 10\n', 30),
    ('inactive_file 10\nfile_dirty 20\nfile_writeback 10\n', 10),
    ('active_file 80\nanon 10\n', 10),
    (None, 10),
    ('inactive_file 1000\n', 100),  # Counter races cannot exceed the cgroup cap.
])
def test_cgroup_headroom_counts_only_clean_inactive_file_cache(tmp_path, stats, expected):
    from nnscaler.runtime.cpu_offloading import _cgroup_headroom
    (tmp_path / 'memory.max').write_text('100')
    (tmp_path / 'memory.current').write_text('90')
    if stats is not None:
        (tmp_path / 'memory.stat').write_text(stats)
    assert _cgroup_headroom(tmp_path) == expected


def test_unlimited_cgroup_does_not_override_host_or_ancestor_limits(tmp_path):
    from nnscaler.runtime.cpu_offloading import _cgroup_headroom
    (tmp_path / 'memory.max').write_text('max')
    assert _cgroup_headroom(tmp_path) is None


def test_compute_config_roundtrip(tmp_path):
    config = ComputeConfig(1, 1, cpu_offload=CPUOffloadConfig(max_cpu_bytes=12345))
    torch.save(asdict(config), tmp_path / 'config.pt')
    assert ComputeConfig.safe_load_from_file(tmp_path / 'config.pt') == config


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@cuda
@pytest.mark.skipif(torch.version.cuda is None, reason='CUDA sleep ordering probe requires NVIDIA')
@pytest.mark.parametrize('use_save_events', [False, True])
def test_saved_tensor_events_do_not_wait_for_later_unrelated_compute(use_save_events):
    module = OffloadModule(use_save_events=use_save_events)
    manager = module._get_activation_offload_manager()
    x = torch.randn(512, 1024, device='cuda', requires_grad=True)
    fn = lambda value: value.sin().cos().sum()
    expected = torch.autograd.grad(fn(x), x)[0]
    # Cold cudaHostAlloc may synchronize the device. Test the steady-state
    # dependency using a prewarmed pinned buffer, as normal training does.
    host = manager.pool.acquire(x.numel() * x.element_size())
    manager.pool.release(host, None)
    torch.cuda.synchronize()
    # A custom producer can use another stream, but must publish its result
    # to the saving stream before capture (the local DeepEP wrappers do this).
    payload = x.detach().clone()
    producer = torch.cuda.Stream()
    with torch.no_grad():
        x.zero_()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer), torch.no_grad():
        torch.cuda._sleep(100_000_000)
        x.copy_(payload)
    torch.cuda.current_stream().wait_stream(producer)
    later = torch.cuda.Event()
    with module.activation_offload_scope('ordering'):
        with module.cpu_offloading_hooks() as group:
            value = checkpoint(fn, x, use_reentrant=False)
            torch.cuda._sleep(1_000_000_000)
            later.record()
        group.watch(value)
    try:
        manager.d2h_stream.synchronize()
        overlapped = not later.query()
    finally:
        later.synchronize()
    actual = torch.autograd.grad(value, x)[0]
    module.finish_activation_offload('ordering')
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert overlapped == use_save_events
    assert bool(manager.pool.stats['saved_tensor_event_records']) == use_save_events
    assert manager.pool.snapshot()['live_cpu_bytes'] == 0


@cuda
def test_saved_tensor_events_cover_later_saves_and_multiple_producer_streams():
    module = OffloadModule(use_save_events=True)
    x = torch.randn(512, 512, device='cuda', requires_grad=True)
    def reference():
        a = x.sin()
        return a.cos().sum() + a.tanh().square().sum()
    expected = torch.autograd.grad(reference(), x)[0]
    first, second = torch.cuda.Stream(), torch.cuda.Stream()
    first.wait_stream(torch.cuda.current_stream())
    with module.activation_offload_scope('streams'):
        with module.cpu_offloading_hooks() as group:
            with torch.cuda.stream(first):
                a = x.sin()
                left = a.cos().sum()
            second.wait_stream(first)
            with torch.cuda.stream(second):
                right = a.tanh().square().sum()
                result = left + right
            del a
        group.watch(result)
    module.prefetch_activation('streams')
    actual = torch.autograd.grad(result, x)[0]
    module.finish_activation_offload('streams')
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(group.group.producer_events) == 2
    assert module.activation_offload_stats()['live_cpu_bytes'] == 0


@cuda
def test_size_and_schedule_distance_filter_uses_storage_and_preserves_views():
    from nnscaler.runtime.cpu_offloading import _OffloadGroup, _ResidentTensorRef, _SavedTensor
    mib = 1024**2
    module = OffloadModule(min_offload_size_gap_bytes=6*mib)
    manager = module._get_activation_offload_manager()
    large = torch.randn(1024, 1024, device='cuda')
    small = torch.randn(256, 256, device='cuda')
    with module.activation_offload_scope('short', backward_distance=2) as scope:
        group = _OffloadGroup(scope)
        a = manager.pack(large, group)
        b = manager.pack(large[:512], group)
        c = manager.pack(small, group)
        d = manager.pack(small[:128], group)
        assert isinstance(a, _SavedTensor) and isinstance(b, _SavedTensor)
        assert a.record is b.record
        assert isinstance(c, _ResidentTensorRef) and isinstance(d, _ResidentTensorRef)
        assert manager.pool.stats['size_gap_resident_storages'] == 1
        assert manager.pool.stats['size_gap_resident_bytes'] == small.untyped_storage().nbytes()
        group.commit()
    torch.testing.assert_close(a.unpack(), large, atol=0, rtol=0)
    torch.testing.assert_close(b.unpack(), large[:512], atol=0, rtol=0)
    small.add_(1)
    with pytest.raises(RuntimeError, match='modified by an inplace'):
        c.unpack()
    module.finish_activation_offload('short')
    with module.activation_offload_scope('unknown') as scope:
        group = _OffloadGroup(scope)
        assert isinstance(manager.pack(small, group), _SavedTensor)
    module.finish_activation_offload('unknown')


@cuda
@pytest.mark.parametrize('distance', [2, 64, None])
def test_size_gap_filter_preserves_gradients_for_resident_and_offloaded_inputs(distance):
    module = OffloadModule(min_offload_size_gap_bytes=2*1024**2)
    large = torch.randn(512, 512, device='cuda', requires_grad=True)
    small = torch.randn(128, 128, device='cuda', requires_grad=True)
    def compute():
        return large.sin().cos().sum() + small.sin().cos().sum()
    expected = torch.autograd.grad(compute(), (large, small))
    with module.activation_offload_scope('case', backward_distance=distance):
        with module.cpu_offloading_hooks() as group:
            value = compute()
        group.watch(value)
    module.prefetch_activation('case')
    actual = torch.autograd.grad(value, (large, small))
    module.finish_activation_offload('case')
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert module.activation_offload_stats()['live_cpu_bytes'] == 0


@cuda
def test_pinned_slab_budget_pending_events_and_coalescing():
    mib = 1024**2
    module = OffloadModule(max_cpu_bytes=8*mib, pool_slab_bytes=8*mib)
    pool = module._get_activation_offload_manager().pool
    a, b, c = (pool.acquire(n*mib) for n in (3, 3, 1))
    assert all(x is not None for x in (a,b,c))
    b.fill_(2);c.fill_(3)
    pointer = a.data_ptr()
    ready = [False]
    event = SimpleNamespace(query=lambda: ready[0])
    pool.release(a, event)
    assert pool.acquire(2*mib) is None  # in-flight slot cannot be overwritten
    ready[0] = True
    restored = pool.acquire(3*mib)
    assert restored.data_ptr() == pointer
    restored.fill_(9)
    assert (b == 2).all() and (c == 3).all()
    for buffer in (restored,b,c):pool.release(buffer, None)
    stats = pool.snapshot()
    assert stats['live_cpu_bytes'] == stats['live_pinned_bytes'] == 0
    assert stats['allocated_pinned_bytes'] == stats['free_slab_bytes'] == 8*mib
    whole = pool.acquire(8*mib)
    assert whole is not None and pool.snapshot()['pool_slabs'] == 1
    pool.release(whole, None)


@cuda
@pytest.mark.parametrize('reload_on_compute_stream', [False, True])
@pytest.mark.parametrize('use_save_events', [False, True])
def test_slab_backed_activation_restoration_is_exact(reload_on_compute_stream, use_save_events):
    mib = 1024**2
    module = OffloadModule(max_cpu_bytes=8*mib, pool_slab_bytes=8*mib,
                           reload_on_compute_stream=reload_on_compute_stream,
                           use_save_events=use_save_events)
    x = torch.randn(768,1024,device='cuda',requires_grad=True)
    z = torch.randn(256,1024,device='cuda',requires_grad=True)
    fn = lambda: x.sin().cos().sum() + z.sin().sum()
    reference = torch.autograd.grad(fn(), (x,z))
    with module.activation_offload_scope('slab'):
        with module.cpu_offloading_hooks() as group:
            y = fn()
        group.watch(y)
    module.prefetch_activation('slab')
    actual = torch.autograd.grad(y, (x,z))
    module.finish_activation_offload('slab')
    for a,b in zip(actual,reference):torch.testing.assert_close(a,b,atol=0,rtol=0)
    stats = module.activation_offload_stats()
    assert stats['budget_fallback_bytes'] == stats['live_cpu_bytes'] == 0
    assert stats['d2h_bytes'] == 7*mib
    assert stats['allocated_pinned_bytes'] == 8*mib


@cuda
@pytest.mark.parametrize('skip', [False, True])
def test_executor_retained_inputs_skip_only_unreclaimable_storage(skip):
    from nnscaler.runtime.executor import Executor
    module = OffloadModule(skip_retained_inputs=skip)
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    expected = torch.autograd.grad(x.sin().cos().sum(), x)[0]
    def forward(value):
        with module.cpu_offloading_hooks() as group:
            result = value.sin().cos()
        group.watch(result)
        return result
    key = ('retained-input-test', 0)
    with module.activation_offload_scope(key):
        y = Executor.fexecute('retained-input-test', forward, x)
    module.prefetch_activation(key)
    actual = Executor.backward('retained-input-test', [x], [y], [torch.ones_like(y)])
    module.finish_activation_offload(key)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stats = module.activation_offload_stats()
    assert stats['offloaded_storages'] == (1 if skip else 2)
    assert stats['d2h_bytes'] == x.numel() * x.element_size() * (1 if skip else 2)
    assert stats['live_cpu_bytes'] == 0
    assert stats.get('retained_input_bytes', 0) == (x.numel() * x.element_size() if skip else 0)
    assert not module._get_activation_offload_manager().scopes


@cuda
@pytest.mark.parametrize('distance,offloads', [(None, 2), (1, 0), (3, 2)])
def test_short_scope_policy_uses_scheduled_distance(distance, offloads):
    module = OffloadModule(min_offload_gap=2)
    x = torch.randn(64,64,device='cuda',requires_grad=True)
    expected, = torch.autograd.grad(x.sin().cos().sum(),x)
    with module.activation_offload_scope('gap', backward_distance=distance):
        with module.cpu_offloading_hooks() as group:
            y = x.sin().cos()
        group.watch(y)
    module.prefetch_activation('gap')
    actual, = torch.autograd.grad(y.sum(),x)
    module.finish_activation_offload('gap')
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    stats=module.activation_offload_stats()
    assert stats['offloaded_storages'] == offloads
    assert stats['live_cpu_bytes'] == 0


@cuda
def test_checkpoint_without_saved_intermediates_does_not_copy_dead_input():
    module = OffloadModule()
    x = torch.randn(64,64,device='cuda',requires_grad=True)
    with module.activation_offload_scope('no-replay'):
        with module.cpu_offloading_hooks() as group:
            y = checkpoint(lambda value: value + 1, x, use_reentrant=False)
        group.watch(y)
    actual, = torch.autograd.grad(y.sum(),x)
    module.finish_activation_offload('no-replay')
    torch.testing.assert_close(actual,torch.ones_like(x),atol=0,rtol=0)
    stats = module.activation_offload_stats()
    assert stats['d2h_bytes'] == 0
    assert stats['live_cpu_bytes'] == 0
    assert stats['allocated_pinned_bytes'] == 0


@cuda
@pytest.mark.parametrize('count,budget,loaded', [(1, 65536, 1), (3, 65536, 3), (3, 16384, 1)])
def test_scope_prefetch_depth_obeys_shared_byte_window(count, budget, loaded):
    module = OffloadModule(scope_prefetch_groups=count, max_prefetch_bytes=budget)
    ys = []
    with module.activation_offload_scope(('depth',0)):
        for _ in range(3):
            with module.cpu_offloading_hooks() as group:
                y = (torch.randn(64,64,device='cuda',requires_grad=True)+1).sin()
            group.watch(y)
            ys.append(y)
    module.prefetch_activation(('depth',0))
    scope = module._get_activation_offload_manager().scopes[('depth',0)]
    assert sum(group.records[0]().device_tensor is not None for group in scope.groups) == loaded
    sum(y.sum() for y in ys).backward()
    module.finish_activation_offload(('depth',0))
    assert module.activation_offload_stats()['live_cpu_bytes'] == 0


@cuda
@pytest.mark.parametrize('entry', ['scope', 'group'])
def test_prefetch_depth_skips_groups_whose_shared_storage_is_already_loaded(entry):
    module = OffloadModule(scope_prefetch_groups=2, prefetch_groups=1)
    a = torch.randn(64,64,device='cuda',requires_grad=True) + 1
    b = torch.randn(64,64,device='cuda',requires_grad=True) + 1
    ys = []
    with module.activation_offload_scope(('shared',0)):
        for value in (a,b,b,b):
            with module.cpu_offloading_hooks() as group:
                y = value.sin()
            group.watch(y)
            ys.append(y)
    del a,b,value
    scope = module._get_activation_offload_manager().scopes[('shared',0)]
    if entry == 'scope':
        module.prefetch_activation(('shared',0))
    else:
        scope.groups[-1].backward_entry()
    assert scope.groups[0].records[0]().device_tensor is not None
    assert scope.groups[1].records[0]().device_tensor is not None
    sum(y.sum() for y in ys).backward()
    module.finish_activation_offload(('shared',0))
    assert module.activation_offload_stats()['live_cpu_bytes'] == 0


@cuda
@pytest.mark.parametrize('recompute', [False, True])
def test_parity_and_checkpoint_boundary_only(recompute):
    module = OffloadModule()
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    expected = (x.sin().cos() @ module.weight).sum()
    reference = torch.autograd.grad(expected, (x, module.weight))
    with module.activation_offload_scope(('stage', 0)):
        with module.cpu_offloading_hooks() as group:
            fn = lambda z: z.sin().cos() @ module.weight
            y = checkpoint(fn, x, use_reentrant=False) if recompute else fn(x)
        group.watch(y)
    before = module.activation_offload_stats()
    module.prefetch_activation(('stage', 0))
    actual = torch.autograd.grad(y.sum(), (x, module.weight))
    module.finish_activation_offload(('stage', 0))
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    stats = module.activation_offload_stats()
    assert stats['offloaded_storages'] == (1 if recompute else 3)
    assert stats['d2h_bytes'] == before['d2h_bytes']  # no offload during replay
    assert stats.get('h2d_bytes', 0) + stats.get('resident_reused_bytes', 0) == stats['d2h_bytes']
    assert stats.get('prefetch_loads', 0) + stats.get('resident_reused_bytes', 0) > 0
    assert stats['live_cpu_bytes'] == 0
    assert not module._get_activation_offload_manager().scopes


@cuda
def test_views_shared_across_regions_and_repeated_backward():
    module = OffloadModule()
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with module.activation_offload_scope():
        with module.cpu_offloading_hooks() as first:
            a = x.sin()
        first.watch(a)
        with module.cpu_offloading_hooks() as second:
            b = x.t().cos() + x[:, ::2].sin().repeat_interleave(2, dim=1)
        second.watch(b)
    loss = a.sum() + b.sum()
    stats = module.activation_offload_stats()
    assert stats['offloaded_storages'] == 1
    assert stats['deduplicated_saves'] >= 2
    loss.backward(retain_graph=True)
    first_grad = x.grad.clone()
    assert module.activation_offload_stats()['live_cpu_bytes'] > 0
    x.grad = None
    loss.backward()
    torch.testing.assert_close(x.grad, first_grad, rtol=0, atol=0)
    stats = module.activation_offload_stats()
    assert stats['d2h_bytes'] == stats['resident_reused_bytes'] == x.numel() * x.element_size()
    assert stats.get('h2d_bytes', 0) == 0
    assert stats['live_cpu_bytes'] == 0


@cuda
def test_parameters_buffers_and_cached_weight_views_are_resident():
    module = OffloadModule()
    cached = mark_activation_resident(torch.randn(64, 64, device='cuda'))
    x = torch.randn_like(cached, requires_grad=True)
    baseline = x @ module.weight.t() + x @ module.buffer.t() + x @ cached.t()
    expected, = torch.autograd.grad(baseline.sum(), x)
    with module.cpu_offloading_hooks() as group:
        y = x @ module.weight.t() + x @ module.buffer.t() + x @ cached.t()
    group.watch(y)
    y.sum().backward()
    stats = module.activation_offload_stats()
    assert stats['offloaded_storages'] == 1  # x only
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)


@cuda
def test_budget_fallback_and_pool_reuse():
    module = OffloadModule(max_cpu_bytes=16384)
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    for _ in range(2):
        with module.cpu_offloading_hooks():
            y = x.sin().cos()
        y.sum().backward()
        torch.cuda.synchronize()
    stats = module.activation_offload_stats()
    assert stats['allocated_cpu_bytes'] == 16384
    assert stats['budget_fallback_bytes'] >= 32768
    assert stats['pool_reuses'] >= 1
    assert stats['live_cpu_bytes'] == 0


@cuda
def test_scopes_select_actual_microbatch_and_virtual_stage():
    module = OffloadModule()
    outputs = {}
    for key in [('chunk0', 0), ('chunk1', 1), ('chunk0', 1)]:
        with module.activation_offload_scope(key):
            with module.cpu_offloading_hooks() as group:
                y = (torch.randn(64, 64, device='cuda', requires_grad=True) + 1).sin()
            group.watch(y)
        outputs[key] = y
    module.prefetch_activation(('chunk0', 0))
    manager = module._get_activation_offload_manager()
    for key, scope in manager.scopes.items():
        record = scope.groups[0].records[0]()
        assert (record.device_tensor is not None) == (key == ('chunk0', 0))
    for key in [('chunk0', 0), ('chunk0', 1), ('chunk1', 1)]:
        outputs.pop(key).sum().backward()
        module.finish_activation_offload(key)
    del record
    assert module.activation_offload_stats()['live_cpu_bytes'] == 0


@cuda
def test_prefetch_is_bounded_and_demand_makes_progress():
    module = OffloadModule(max_prefetch_bytes=1)
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with module.activation_offload_scope('a'):
        with module.cpu_offloading_hooks():
            y = (x + 1).sin()
    module.prefetch_activation('a')
    assert module.activation_offload_stats().get('h2d_bytes', 0) == 0
    y.sum().backward()
    module.finish_activation_offload('a')
    stats = module.activation_offload_stats()
    assert stats.get('h2d_bytes', 0) + stats.get('resident_reused_bytes', 0) == stats['d2h_bytes']
    assert stats['prefetched_gpu_bytes'] == 0


@cuda
def test_pool_shared_by_parallel_modules():
    a, b = OffloadModule(), OffloadModule()
    assert a._get_activation_offload_manager().pool is b._get_activation_offload_manager().pool
    with pytest.raises(ValueError, match='same max_cpu_bytes'):
        OffloadModule(max_cpu_bytes=1)._get_activation_offload_manager()


@cuda
def test_nondefault_stream_and_allocator_reuse():
    module = OffloadModule(max_inflight_bytes=16384)
    stream = torch.cuda.Stream()
    for _ in range(10):
        with torch.cuda.stream(stream):
            x = torch.randn(64, 64, device='cuda', requires_grad=True)
            expected = x.cos()
            with module.cpu_offloading_hooks():
                y = x.sin()
            torch.empty_like(x).fill_(123)
            y.sum().backward()
            torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    stream.synchronize()
    assert module.activation_offload_stats()['live_cpu_bytes'] == 0


@cuda
def test_inplace_version_check():
    module = OffloadModule()
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with module.cpu_offloading_hooks():
        y = x.sin()
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match='inplace'):
        y.sum().backward()


@pytest.mark.parametrize('change', ['base', 'view', 'detach', 'custom_version'])
def test_resident_saved_tensor_contract_on_cpu(change):
    from nnscaler.runtime.cpu_offloading import _ResidentTensorRef
    x = torch.randn(8, 8, requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(_ResidentTensorRef, lambda saved: saved.unpack()):
        y = x.t().sin().sum()
    with torch.no_grad():
        if change == 'custom_version':
            # A raw-pointer kernel must explicitly report its mutation.
            torch.autograd.graph.increment_version(x)
        elif change == 'base':
            x.add_(1)
        elif change == 'view':
            x[:, :2].mul_(2)
        else:
            x.detach().zero_()
    with pytest.raises(RuntimeError, match='expected version.*found.*Custom operators'):
        y.backward()
    assert x.grad is None


def test_version_tracker_does_not_retain_activation_storage():
    import weakref
    from nnscaler.runtime.cpu_offloading import _version_tracker
    x = torch.ones(32)
    tracker = _version_tracker(x.t())
    storage = weakref.ref(x.untyped_storage())
    x.add_(1)
    assert tracker._version == x._version
    assert tracker.numel() == 0
    del x
    gc.collect()
    assert storage() is None


def test_custom_op_mutation_schema_reports_an_otherwise_untracked_write():
    from nnscaler.runtime.cpu_offloading import _ResidentTensorRef

    @torch.library.custom_op('offload_contract_test::raw_write', mutates_args=('x',))
    def raw_write(x: torch.Tensor) -> None:
        # NumPy bypasses torch's in-place operators, like a raw-pointer kernel.
        # The custom-op schema must report the write to the version counter.
        x.detach().numpy().fill(2)

    x = torch.ones(8)
    saved = _ResidentTensorRef(x)
    raw_write(x)
    with pytest.raises(RuntimeError, match='resident unpack'):
        saved.unpack()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_saved_storage_restores_views_with_shared_version_on_cpu(monkeypatch, dtype):
    from nnscaler.runtime.cpu_offloading import _StorageRecord, _SavedTensor
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: None)
    x = torch.arange(32).to(dtype).reshape(4, 8)
    record = _StorageRecord(x, SimpleNamespace(pool=None))
    saved = _SavedTensor(x[:, 1::2].t(), record)
    restored = saved.unpack()
    torch.testing.assert_close(restored, x[:, 1::2].t(), rtol=0, atol=0)
    assert restored.stride() == saved.stride
    assert restored.storage_offset() == saved.offset
    restored.add_(1)
    with pytest.raises(RuntimeError, match='resident storage unpack'):
        saved.unpack()


def test_saved_view_with_unused_trailing_storage_bytes(monkeypatch):
    from nnscaler.runtime.cpu_offloading import _StorageRecord, _SavedTensor
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: None)
    storage = torch.zeros(13, dtype=torch.uint8).untyped_storage()
    x = torch.empty(0, dtype=torch.float32).set_(storage, 1, (2,), (1,))
    x.fill_(3.0)
    record = _StorageRecord(x, SimpleNamespace(pool=None))
    restored = _SavedTensor(x, record).unpack()
    assert torch.equal(restored, x)
    assert restored.storage_offset() == 1


@cuda
def test_repeated_save_cannot_fork_a_mutated_live_storage():
    module = OffloadModule()
    x = torch.ones(64, 64, device='cuda')
    with module.cpu_offloading_hooks() as first:
        saved = first._pack(x)
    torch.cuda.synchronize()  # Isolate version detection from invalid DMA races.
    x.add_(1)
    with pytest.raises(RuntimeError, match='repeated save'):
        with module.cpu_offloading_hooks() as second:
            second._pack(x.t())
    assert len(module._get_activation_offload_manager().records) == 1
    del saved


@cuda
def test_resaved_restored_view_reuses_the_same_immutable_record():
    module = OffloadModule()
    x = torch.ones(64, 64, device='cuda')
    with module.cpu_offloading_hooks() as group:
        first = group._pack(x)
    restored = first.unpack()  # Original allocation is still owned by x.
    with module.cpu_offloading_hooks() as group:
        second = group._pack(restored.t())
    assert first.record is second.record
    restored.add_(1)
    with pytest.raises(RuntimeError, match='inplace'):
        first.unpack()
    with pytest.raises(RuntimeError, match='inplace'):
        second.unpack()


@cuda
def test_custom_mutation_version_is_checked_before_prefetch():
    module = OffloadModule()
    x = torch.ones(64, 64, device='cuda')
    with module.cpu_offloading_hooks() as group:
        saved = group._pack(x)
    torch.autograd.graph.increment_version(x)
    with pytest.raises(RuntimeError, match='prefetch'):
        saved.record.prefetch()
    del saved


@cuda
@pytest.mark.parametrize('reload_on_compute_stream', [False, True])
def test_restored_alias_mutation_cannot_silently_change_another_saved_view(reload_on_compute_stream):
    module = OffloadModule(reload_on_compute_stream=reload_on_compute_stream)
    x = torch.randn(64, 64, device='cuda')
    with module.cpu_offloading_hooks() as group:
        first = group._pack(x)
        second = group._pack(x.t())
    assert first.record is second.record
    del x
    gc.collect()
    restored = first.unpack()
    assert module.activation_offload_stats()['h2d_bytes'] == 64 * 64 * 4
    restored.mul_(2)
    with pytest.raises(RuntimeError, match='restored unpack'):
        second.unpack()


@cuda
def test_discarded_save_does_not_prohibit_later_inplace_and_save():
    module = OffloadModule()
    x = torch.ones(64, 64, device='cuda')
    with module.cpu_offloading_hooks() as group:
        old = group._pack(x)
        del old
        x.add_(1)
        current = group._pack(x)
    assert torch.equal(current.unpack(), x)
    assert module.activation_offload_stats()['offloaded_storages'] == 1


@cuda
def test_inplace_before_first_save_is_supported():
    class Double(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            x.mul_(2)
            ctx.mark_dirty(x)
            ctx.save_for_backward(x)
            return x

        @staticmethod
        def backward(ctx, grad):
            (saved,) = ctx.saved_tensors
            assert saved.shape == grad.shape
            return grad * 2

    module = OffloadModule()
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with module.cpu_offloading_hooks():
        loss = Double.apply(x + 0).square().sum()
    loss.backward()
    torch.testing.assert_close(x.grad, 8 * x, atol=0, rtol=0)


@cuda
def test_memory_is_released_after_forward_and_backward():
    module = OffloadModule()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    with module.cpu_offloading_hooks():
        x = torch.randn(1024, 1024, device='cuda', requires_grad=True)
        # Use non-leaf storage so leaf accumulation does not keep the data alive.
        a = x + 1
        y = a.sin()
        del a
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() - base
    assert used < 3 * x.numel() * x.element_size()
    y.sum().backward()
    assert module.activation_offload_stats()['live_cpu_bytes'] == 0


@cuda
def test_immediate_backward_keeps_only_tail_group_resident():
    module = OffloadModule()
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with module.activation_offload_scope('now', immediate_backward=True):
        with module.cpu_offloading_hooks() as first:
            y = (x + 1).sin()
        first.watch(y)
        with module.cpu_offloading_hooks(tail=True) as tail:
            z = y.cos()
        tail.watch(z)
    assert module.activation_offload_stats()['offloaded_storages'] == 1
    z.sum().backward()
    module.finish_activation_offload('now')
    assert module.activation_offload_stats()['live_cpu_bytes'] == 0
    torch.testing.assert_close(x.grad, -(x + 1).sin().sin() * (x + 1).cos())


@cuda
def test_inplace_check_survives_destruction_of_saved_view():
    module = OffloadModule()
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with module.cpu_offloading_hooks():
        view = x.t()
        y = view.sin()
        del view
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match='inplace'):
        y.sum().backward()


@cuda
@pytest.mark.parametrize('max_cpu_bytes', [0, 32768])
def test_resident_and_offloaded_paths_both_check_inplace(max_cpu_bytes):
    module = OffloadModule(max_cpu_bytes=max_cpu_bytes)
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with module.cpu_offloading_hooks():
        y = x.sin()
    with torch.no_grad():
        x.add_(1)
    with pytest.raises(RuntimeError, match='inplace'):
        y.sum().backward()


@cuda
def test_parameter_reinterpreted_dtype_view_stays_resident():
    from nnscaler.runtime.cpu_offloading import _ModuleTensorRef
    module = OffloadModule()
    view = module.weight.detach().view(torch.uint8)
    with module.cpu_offloading_hooks() as context:
        saved = context._pack(view)
    assert isinstance(saved, _ModuleTensorRef)
    torch.testing.assert_close(saved.unpack(), view)
    assert module.activation_offload_stats()['allocated_cpu_bytes'] == 0


@cuda
def test_host_headroom_falls_back_without_pinned_allocation(monkeypatch):
    import nnscaler.runtime.cpu_offloading as offload
    module = OffloadModule(host_reserve_bytes=1)
    monkeypatch.setattr(offload, 'available_host_bytes', lambda: 0)
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with module.cpu_offloading_hooks():
        y = x.sin()
    y.sum().backward()
    torch.testing.assert_close(x.grad, x.cos())
    stats = module.activation_offload_stats()
    assert stats['allocated_cpu_bytes'] == 0
    assert stats['headroom_fallback_bytes'] == x.numel() * x.element_size()


@cuda
def test_zero_prefetch_groups_is_really_demand_only():
    module = OffloadModule(prefetch_groups=0)
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with module.activation_offload_scope('demand'):
        with module.cpu_offloading_hooks() as group:
            y = (x + 1).sin().cos()
        group.watch(y)
    module.prefetch_activation('demand')
    y.sum().backward()
    module.finish_activation_offload('demand')
    stats = module.activation_offload_stats()
    assert stats['prefetch_loads'] == 0
    assert stats['demand_loads'] > 0
    assert stats['live_cpu_bytes'] == 0


@cuda
def test_process_budget_survives_sequential_module_instances():
    import weakref
    module = OffloadModule(max_cpu_bytes=16384)
    with module.cpu_offloading_hooks():
        x = torch.randn(64, 64, device='cuda', requires_grad=True)
        y = (x + 1).sin()
    y.sum().backward()
    pool_ref = weakref.ref(module._get_activation_offload_manager().pool)
    del module, x, y
    gc.collect()
    assert pool_ref() is not None
    torch.cuda.synchronize()
    second = OffloadModule(max_cpu_bytes=16384)
    with second.cpu_offloading_hooks():
        x = torch.randn(64, 64, device='cuda', requires_grad=True)
        y = (x + 1).sin()
    y.sum().backward()
    assert second._get_activation_offload_manager().pool is pool_ref()
    stats = second.activation_offload_stats()
    assert stats['allocated_cpu_bytes'] == 16384
    assert stats['pool_reuses'] == 1


def test_rounded_pinned_capacity_reuses_larger_requests_only_after_event(monkeypatch):
    import nnscaler.runtime.cpu_offloading as offload
    original_empty = torch.empty
    monkeypatch.setattr(offload, 'available_host_bytes', lambda: 1024**4)
    monkeypatch.setattr(offload.torch, 'empty',
                        lambda size, **kw: original_empty(size, dtype=torch.uint8))
    small = offload._OffloadPool(CPUOffloadConfig(max_cpu_bytes=200000, host_reserve_bytes=0))
    assert small.acquire(132000) is None  # tensor fits, its 262144-byte pinned block does not
    pool = offload._OffloadPool(CPUOffloadConfig(max_cpu_bytes=262144, host_reserve_bytes=0))
    buffer = pool.acquire(132000)
    assert buffer.numel() == 262144
    stats = pool.snapshot()
    assert stats['allocated_cpu_bytes'] == 262144
    assert stats['allocated_pinned_bytes'] == stats['live_pinned_bytes'] == 262144
    assert pool.acquire(1) is None
    ready = [False]
    pool.release(buffer, SimpleNamespace(query=lambda: ready[0]))
    assert pool.snapshot()['live_pinned_bytes'] == 0
    assert pool.acquire(140000) is None  # Pending DMA still owns the block.
    ready[0] = True
    assert pool.acquire(140000) is buffer  # Same charged class, larger request.
    assert pool.snapshot()['allocated_pinned_bytes'] == 262144
    pool.release(buffer, None)
    pool.close()
    assert pool.snapshot()['allocated_pinned_bytes'] == 0


@cuda
def test_pool_charge_matches_native_pinned_allocator():
    import nnscaler.runtime.cpu_offloading as offload
    read = getattr(torch.cuda.memory, 'host_memory_stats', None)
    if read is None:
        pytest.skip('native pinned allocator statistics unavailable')
    torch.cuda.init()
    torch.cuda.synchronize()
    # Reap finished stream dependencies from earlier cases before measuring.
    warm = torch.empty(1, pin_memory=True)
    del warm
    before = read()
    pool = offload._OffloadPool(CPUOffloadConfig(max_cpu_bytes=262144, host_reserve_bytes=0))
    buffer = pool.acquire(132000)
    after = read()
    charge = pool.snapshot()['allocated_pinned_bytes']
    assert charge == 262144
    # Physical owned capacity grows only when a new CUDA host block is created;
    # a cached block may already have existed before this test.
    created = after['num_host_alloc'] - before['num_host_alloc']
    assert created in (0, 1)
    assert after['allocated_bytes.current'] - before['allocated_bytes.current'] == created * charge
    pool.release(buffer, None)
    pool.close()


@cuda
def test_schedule_prefetch_skips_empty_trailing_regions():
    module = OffloadModule()
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with module.activation_offload_scope('with-empty-tail') as scope:
        with module.cpu_offloading_hooks():
            y = (x + 1).sin()
        with module.cpu_offloading_hooks():
            z = y + 1  # no tensors saved by addition
    assert not scope.groups[-1].records
    record = scope.groups[0].records[0]()
    assert record.device_tensor is None
    module.prefetch_activation('with-empty-tail')
    assert record.device_tensor is not None
    z.sum().backward()
    module.finish_activation_offload('with-empty-tail')


@cuda
def test_group_submission_does_not_wait_for_its_own_d2h():
    module = OffloadModule(max_inflight_bytes=16384)
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    # Warm kernels and two pinned buffers before delaying the copy stream.
    with module.cpu_offloading_hooks():
        warm = (x + 1).sin().cos()
    warm.sum().backward()
    del warm
    torch.cuda.synchronize()
    manager = module._get_activation_offload_manager()
    with torch.cuda.stream(manager.d2h_stream):
        torch.cuda._sleep(200_000_000)
    with module.cpu_offloading_hooks() as context:
        y = (x + 1).sin().cos()
    record = context.group.records[0]()
    # The complete group must be enqueued while its first copy is still pending,
    # even though total group bytes exceed the inflight threshold.
    assert record is not None and not record.d2h_event.query()
    y.sum().backward()


@cuda
def test_group_commit_waits_for_each_producer_stream():
    module = OffloadModule()
    left_stream, right_stream = torch.cuda.Stream(), torch.cuda.Stream()
    with module.cpu_offloading_hooks():
        with torch.cuda.stream(left_stream):
            torch.cuda._sleep(20_000_000)
            left = torch.full((64, 64), 0.3, device='cuda', requires_grad=True)
            left_output = (left + 1).sin()
        with torch.cuda.stream(right_stream):
            torch.cuda._sleep(20_000_000)
            right = torch.full((64, 64), 0.7, device='cuda', requires_grad=True)
            right_output = (right + 1).cos()
    main = torch.cuda.current_stream()
    main.wait_stream(left_stream)
    main.wait_stream(right_stream)
    (left_output.sum() + right_output.sum()).backward()
    main.wait_stream(left_stream)
    main.wait_stream(right_stream)
    torch.testing.assert_close(left.grad, (left + 1).cos())
    torch.testing.assert_close(right.grad, -(right + 1).sin())
