"""Pure scheduling rules and data ownership at pipeline F/B boundaries."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from nnscaler.parallel import ComputeConfig
from nnscaler.runtime.activation_config import CPUOffloadConfig
from nnscaler.runtime.activation_schedule import ActivationSchedulePlan, ActivationScheduleCursor
from nnscaler.runtime.cpu_offloading import ActivationOffloadManager, CPUOffloadContext

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def isolated_pool():
    import gc
    from nnscaler.runtime.activation_memory import _pools
    gc.collect()
    _pools.clear()
    yield
    gc.collect()
    _pools.clear()


A, B, C, D = ('stage0', 0), ('stage1', 0), ('stage0', 1), ('stage1', 1)
ACTIONS = [(A, True, True), (B, True, True), (C, True, True), (A, False, True),
           (D, True, True), (B, False, True), (C, False, True), (D, False, True)]


def test_pipeline_distance_counts_actual_local_actions_and_clamps_to_forward():
    plan = ActivationSchedulePlan.from_actions(ACTIONS)
    assert plan.entries() == ((A, 0, 3), (B, 1, 5), (C, 2, 6), (D, 4, 7))
    assert [row['first_prefetch'] for row in plan.describe(1)] == [
        ('before', 2), ('before', 4), ('before', 5), ('before', 6)]
    assert [row['first_prefetch'] for row in plan.describe(4)] == [
        ('after', 0), ('after', 1), ('after', 2), ('after', 4)]
    assert all(row['first_prefetch'] is None for row in plan.describe(0))
    assert plan.candidates(0, 4) == ()
    assert [use.key for use in plan.candidates(4, 4)] == [B, C]
    assert [use.key for use in plan.candidates(4, 4, after=True)] == [B, C, D]


def test_nonoffloaded_actions_still_contribute_to_distance():
    actions = [(A, True, True), ('other', True, False), ('other', False, False), (A, False, True)]
    plan = ActivationSchedulePlan.from_actions(actions)
    assert plan.entries() == ((A, 0, 3),)
    assert plan.describe(2)[0]['first_prefetch'] == ('before', 1)


def test_cursor_retries_nearest_blocked_reader_without_stealing_its_budget():
    plan = ActivationSchedulePlan.from_actions(ACTIONS)
    requested = []
    blocked = True
    def request(key, lead):
        requested.append((key, lead))
        return not (blocked and key == B)
    cursor = ActivationScheduleCursor(plan, 4, request)
    cursor.before(4)
    assert requested == [(B, 1)]
    requested.clear()
    blocked = False
    cursor.after(4)
    assert requested == [(B, 0), (C, 1), (D, 2)]
    requested.clear()
    cursor.before(5)
    assert requested == []  # Completed requests are not rescanned every action.


def test_same_plan_uses_new_lookahead_configuration_on_the_next_iteration(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    manager = ActivationOffloadManager(torch.nn.Module(), CPUOffloadConfig(pipeline_prefetch_steps=0))
    requests = []
    manager.prefetch = lambda key: requests.append(key) or True
    entries = ActivationSchedulePlan.from_actions(ACTIONS).entries()
    with manager.schedule(entries) as cursor:
        cursor.before(2)
    assert requests == []
    manager.config = replace(manager.config, pipeline_prefetch_steps=4)
    with manager.schedule(entries) as cursor:
        cursor.before(2)
    assert requests == [A, B]
    requests.clear()
    manager.config = replace(manager.config, prefetch_groups=0)
    with manager.schedule(entries) as cursor:
        cursor.before(2)
    assert requests == []


@pytest.mark.parametrize('entries', [[(A, 2, 1)], [(A, 0, 0)], [(A, 0, 1), (A, 2, 3)]])
def test_invalid_scope_schedules_fail_before_execution(entries):
    with pytest.raises(ValueError):
        ActivationSchedulePlan(entries)


def test_legacy_codegen_config_does_not_silently_reuse_one_action_hints(tmp_path):
    current = ComputeConfig(1, 1, cpu_offload=CPUOffloadConfig())
    old = asdict(current)
    del old['cpu_offload']['_schedule_version']
    path = tmp_path/'old.pt'
    torch.save(old, path)
    restored = ComputeConfig.safe_load_from_file(path, return_none_on_error=False)
    assert restored.cpu_offload._schedule_version == 0
    assert not ComputeConfig.safe_equals(restored, current)
    assert restored.graph_config == current.graph_config  # Raw trace remains reusable.


def test_submission_options_roundtrip_without_changing_schedule_abi(tmp_path):
    current = ComputeConfig(1,1,cpu_offload=CPUOffloadConfig(
        transfer_unit='scope', forward_commit_unit='group', commit_batch_bytes=1024, copy_chunk_bytes=64))
    path = tmp_path/'current.pt'
    torch.save(asdict(current), path)
    restored = ComputeConfig.safe_load_from_file(path, return_none_on_error=False)
    assert ComputeConfig.safe_equals(restored,current)
    assert restored.cpu_offload._schedule_version == 1
    old = asdict(current)
    for key in ('forward_commit_unit','commit_batch_bytes','copy_chunk_bytes'):
        old['cpu_offload'].pop(key)
    torch.save(old,path)
    restored = ComputeConfig.safe_load_from_file(path, return_none_on_error=False)
    assert restored.cpu_offload.commit_unit(pipeline=True) == 'scope'
    assert restored.cpu_offload.commit_batch_bytes == restored.cpu_offload.copy_chunk_bytes == 0
    assert restored.cpu_offload._schedule_version == 1


def test_schedule_exception_drops_scope_metadata_with_gc_disabled(monkeypatch):
    import gc
    import weakref
    from nnscaler.runtime.activation_memory import _pools
    _pools.clear()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    module = torch.nn.Module()
    manager = ActivationOffloadManager(module, CPUOffloadConfig())
    module._get_activation_offload_manager = lambda: manager
    enabled = gc.isenabled()
    gc.disable()
    try:
        with pytest.raises(RuntimeError, match='abort'):
            with manager.schedule(((A, 0, 2),)):
                with manager.scope(A, pipeline=True) as scope:
                    with CPUOffloadContext(module):
                        x = torch.ones(8, requires_grad=True)
                        y = x.sin()
                observed = weakref.ref(scope)
                del scope
                raise RuntimeError('abort')
        del x, y
        assert manager.scopes == {} and observed() is None
    finally:
        if enabled:
            gc.enable()
        _pools.clear()


def module_and_manager(**settings):
    module = torch.nn.Module()
    config = CPUOffloadConfig(**dict(dict(max_cpu_bytes=64*1024**2, min_tensor_bytes=0,
                                         host_reserve_bytes=0), **settings))
    manager = ActivationOffloadManager(module, config)
    module._get_activation_offload_manager = lambda: manager
    module.find_buffer_or_param_owner = lambda tensor: None
    return module, manager


@cuda
@pytest.mark.parametrize('unit,pipeline,scope_batch', [
    ('auto', True, True), ('auto', False, False), ('scope', False, True), ('group', True, False)])
def test_transfer_unit_controls_commit_boundary_and_preserves_gradients(unit, pipeline, scope_batch):
    module, manager = module_and_manager(transfer_unit=unit, reload_on_compute_stream=False)
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    expected, = torch.autograd.grad(((x+1).sin()+2).sin().sum(), x)
    with manager.scope(A, pipeline=pipeline, backward_distance=4):
        with CPUOffloadContext(module) as first:
            a = (x+1).sin()
        first.watch(a)
        assert bool(manager.pool.snapshot()['d2h_bytes']) == (not scope_batch)
        with CPUOffloadContext(module) as second:
            b = (a+2).sin()
        second.watch(b)
    assert manager.pool.snapshot()['d2h_committed_groups'] == (1 if scope_batch else 2)
    manager.prefetch(A)
    b.sum().backward()
    manager.finish(A)
    torch.testing.assert_close(x.grad, expected)
    torch.cuda.synchronize()
    assert manager.pool.snapshot()['live_cpu_bytes'] == 0


@cuda
@pytest.mark.parametrize('restore', ['scope', 'group'])
@pytest.mark.parametrize('forward', ['scope', 'group'])
@pytest.mark.parametrize('chunk', [0, 512])
def test_independent_submission_and_chunking_preserve_real_gradients(restore, forward, chunk):
    module, manager = module_and_manager(transfer_unit=restore, forward_commit_unit=forward,
        commit_batch_bytes=4096, copy_chunk_bytes=chunk, reload_on_compute_stream=False,
        use_save_events=True, scope_prefetch_groups=2)
    x = torch.randn(64,64,device='cuda',requires_grad=True)
    reference = ((x+1).sin().t()+2).sin().sum()
    expected, = torch.autograd.grad(reference, x)
    with manager.trace(cuda_timing=True) as trace:
        with manager.scope(A, pipeline=True, backward_distance=4):
            with CPUOffloadContext(module) as first:
                a = (x+1).sin()
            first.watch(a)
            assert manager.pool.stats['d2h_bytes'] > 0  # Before F closes, including scope restore.
            with CPUOffloadContext(module) as second:
                b = (a.t()+2).sin()
            second.watch(b)
            del a
        manager.prefetch(A)
        b.sum().backward()
        manager.finish(A)
        torch.cuda.synchronize()
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    stats = manager.pool.snapshot()
    assert stats['d2h_bytes'] > 0 and stats['h2d_bytes'] > 0
    assert stats['peak_d2h_batch_bytes'] <= x.numel()*x.element_size()
    assert all(stats[k]==0 for k in ('live_cpu_bytes','pending_cpu_bytes','prefetched_gpu_bytes'))
    waits = trace.to_dict()['cuda_waits']
    assert waits and all(wait['complete'] and wait['elapsed_ms'] >= 0 for wait in waits)


@cuda
def test_scope_prefetch_retries_after_demand_frees_byte_capacity():
    size = 64*64*4
    module, manager = module_and_manager(max_prefetch_bytes=2*size, reload_on_compute_stream=False)
    tokens = []
    with manager.scope(A, pipeline=True, backward_distance=4):
        for index in range(4):
            with CPUOffloadContext(module) as group:
                tokens.append(group._pack(torch.full((64,64), float(index), device='cuda')))
    assert manager.prefetch(A) is False
    assert manager.pool.snapshot()['h2d_bytes'] == 2*size
    torch.testing.assert_close(tokens[-1].unpack(), torch.full((64,64),3.0,device='cuda'))
    assert manager.prefetch(A) is False
    assert manager.pool.snapshot()['h2d_bytes'] == 3*size
    tokens.clear()
    manager.finish(A)
    torch.cuda.synchronize()
    assert manager.pool.snapshot()['live_cpu_bytes'] == 0


@cuda
def test_group_entry_window_does_not_expand_on_repeated_schedule_hints():
    size = 64*64*4
    module, manager = module_and_manager(transfer_unit='group', scope_prefetch_groups=1)
    tokens = []
    with manager.scope(A, pipeline=True, backward_distance=4):
        for index in range(3):
            with CPUOffloadContext(module) as group:
                tokens.append(group._pack(torch.full((64,64), float(index), device='cuda')))
    assert manager.prefetch(A) is True
    assert manager.prefetch(A) is True
    assert manager.pool.snapshot()['h2d_bytes'] == size
    tokens.clear()
    manager.finish(A)
    torch.cuda.synchronize()
    assert manager.pool.snapshot()['live_cpu_bytes'] == 0


@cuda
def test_oversize_scope_value_does_not_starve_a_later_fitting_scope():
    size = 64*64*4
    module, manager = module_and_manager(max_prefetch_bytes=size, reload_on_compute_stream=False,
                                         pipeline_prefetch_steps=4)
    tokens = []
    for key, shape in ((A,(128,128)), (B,(64,64))):
        with manager.scope(key, pipeline=True, backward_distance=4):
            with CPUOffloadContext(module) as group:
                tokens.append(group._pack(torch.ones(shape,device='cuda')))
    with manager.schedule(((A,0,4),(B,1,5))) as cursor:
        cursor.before(2)
        assert tokens[0].record.device_tensor is None
        assert tokens[1].record.device_tensor is not None
        assert manager.pool.snapshot()['scope_prefetch_oversize_bytes'] == 4*size
        torch.testing.assert_close(tokens[0].unpack(), torch.ones(128,128,device='cuda'))
        tokens.clear()
    torch.cuda.synchronize()
    assert manager.scopes == {} and manager.pool.snapshot()['live_cpu_bytes'] == 0


@cuda
def test_scope_commit_joins_all_producer_streams():
    module, manager = module_and_manager(use_save_events=True)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    values = [torch.empty(64,64,device='cuda') for _ in streams]
    buffers = [manager.pool.acquire(64*64*4) for _ in streams]
    for buffer in buffers:
        manager.pool.release(buffer, None)
    del buffers, buffer
    torch.cuda.synchronize()
    tokens = []
    with manager.scope(A, pipeline=True, backward_distance=4) as scope:
        for index, (stream, value) in enumerate(zip(streams, values)):
            with torch.cuda.stream(stream):
                torch.cuda._sleep(20_000_000)
                value.fill_(index+2)
                with CPUOffloadContext(module) as group:
                    tokens.append(group._pack(value))
        assert len(scope.commits.batch.producer_events) == 2
    manager.d2h_stream.synchronize()
    for index, token in enumerate(tokens):
        host = token.record.host[:token.record.size].view(torch.float32).reshape(64,64)
        torch.testing.assert_close(host, torch.full((64,64), float(index+2)))
    del token, host
    tokens.clear()
    manager.finish(A)
    assert manager.pool.snapshot()['live_cpu_bytes'] == 0


@cuda
@pytest.mark.skipif(torch.version.cuda is None, reason='CUDA producer-delay probe')
def test_scope_batch_does_not_gate_ready_copies_on_later_producers():
    module, manager = module_and_manager(use_save_events=True)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    tensors = [torch.ones(1024,1024,device='cuda') for _ in streams]
    buffers = [manager.pool.acquire(4*1024**2) for _ in streams]
    for buffer in buffers:
        manager.pool.release(buffer, None)
    del buffers, buffer
    torch.cuda.synchronize()
    tokens = []
    with manager.scope(A, pipeline=True, backward_distance=4):
        with torch.cuda.stream(streams[0]):
            with CPUOffloadContext(module) as group:
                tokens.append(group._pack(tensors[0]))
        with torch.cuda.stream(streams[1]):
            torch.cuda._sleep(200_000_000)
            tensors[1].fill_(2)
            with CPUOffloadContext(module) as group:
                tokens.append(group._pack(tensors[1]))
                late_ready = tokens[1].record.producer_ready_event
    tokens[0].record.d2h_event.synchronize()
    assert not late_ready.query(), 'Earlier ready data was blocked by the later producer'
    manager.d2h_stream.synchronize()
    tokens.clear()
    manager.finish(A)
    assert manager.pool.snapshot()['live_cpu_bytes'] == 0


@cuda
def test_immediate_pipeline_backward_keeps_the_whole_scope_resident():
    module, manager = module_and_manager()
    with manager.scope(A, pipeline=True, immediate_backward=True, backward_distance=1):
        with CPUOffloadContext(module):
            y = torch.randn(64,64,device='cuda',requires_grad=True).sin()
    y.sum().backward()
    manager.finish(A)
    assert manager.pool.snapshot()['d2h_bytes'] == 0
