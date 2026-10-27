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
                                        ('max_prefetch_bytes', 1.5)])
def test_invalid_config(field, value):
    with pytest.raises(ValueError, match=field):
        CPUOffloadConfig(**{field: value})


def test_compute_config_roundtrip(tmp_path):
    config = ComputeConfig(1, 1, cpu_offload=CPUOffloadConfig(max_cpu_bytes=12345))
    torch.save(asdict(config), tmp_path / 'config.pt')
    assert ComputeConfig.safe_load_from_file(tmp_path / 'config.pt') == config


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


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


def test_budget_charges_rounded_pinned_blocks_without_resizing_tensor(monkeypatch):
    import nnscaler.runtime.cpu_offloading as offload
    original_empty = torch.empty
    monkeypatch.setattr(offload, 'available_host_bytes', lambda: 1024**4)
    monkeypatch.setattr(offload.torch, 'empty',
                        lambda size, **kw: original_empty(size, dtype=torch.uint8))
    small = offload._OffloadPool(CPUOffloadConfig(max_cpu_bytes=200000, host_reserve_bytes=0))
    assert small.acquire(132000) is None  # tensor fits, its 262144-byte pinned block does not
    pool = offload._OffloadPool(CPUOffloadConfig(max_cpu_bytes=262144, host_reserve_bytes=0))
    buffer = pool.acquire(132000)
    assert buffer.numel() == 132000
    stats = pool.snapshot()
    assert stats['allocated_cpu_bytes'] == 132000
    assert stats['allocated_pinned_bytes'] == stats['live_pinned_bytes'] == 262144
    assert pool.acquire(1) is None
    pool.release(buffer, None)
    assert pool.snapshot()['live_pinned_bytes'] == 0
    assert pool.acquire(130000) is buffer
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
