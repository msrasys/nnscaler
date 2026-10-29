"""Ownership, forward reuse and transport contracts of the public store."""
import gc
from dataclasses import replace
import json
import weakref
from types import SimpleNamespace

import pytest
import torch

from nnscaler.runtime.activation_config import CPUOffloadConfig
from nnscaler.runtime.activation_memory import _pools
from nnscaler.runtime.activation_store import ActivationStore, _GradientAnchor, _RestoreGradient

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture(autouse=True)
def isolated_pool():
    gc.collect()
    _pools.clear()
    yield
    gc.collect()


def store(device='cpu', **kwargs):
    return ActivationStore(CPUOffloadConfig(**dict(
        dict(max_cpu_bytes=64*1024**2, min_tensor_bytes=0, host_reserve_bytes=0), **kwargs)), device=device)


@pytest.mark.parametrize('layout', ['dense', 'slice', 'expand'])
def test_gradient_anchor_does_not_retain_nonleaf_payload_and_preserves_higher_gradients(layout):
    def transform(value):
        if layout == 'slice':
            return value.t()[::2]
        if layout == 'expand':
            return value[:, :1].expand(-1, 5)
        return value

    x = torch.randn(4, 5, requires_grad=True)
    value = transform(x.sin())
    payload = value.detach().clone()
    original_storage = weakref.ref(value.untyped_storage())
    anchor = _GradientAnchor.apply(value)
    del value
    assert original_storage() is None
    assert anchor.untyped_storage().nbytes() == anchor.element_size()
    a = _RestoreGradient.apply(anchor, payload)
    b = _RestoreGradient.apply(anchor, payload.clone())
    grad, = torch.autograd.grad(a.square().sum() + b.sum(), x, create_graph=True)
    reference = transform(x.sin())
    expected, = torch.autograd.grad(reference.square().sum() + reference.sum(), x, create_graph=True)
    torch.testing.assert_close(grad, expected)
    second, = torch.autograd.grad(grad.sum(), x)
    torch.testing.assert_close(second, torch.autograd.grad(expected.sum(), x)[0])


def test_resident_forward_handles_and_close_ownership():
    cache = store()
    x = torch.randn(3, 5, requires_grad=True)
    with cache.batch() as batch:
        handle = batch.save(x.sin())
        assert handle.status()['reason'] == 'host_value'
    with pytest.raises(RuntimeError, match='handles'):
        cache.close()
    a, b = handle.load(), handle.load()
    handle.release()
    handle.release()
    cache.close()
    (a.square().sum() + b.sum()).backward()
    torch.testing.assert_close(x.grad, (2*x.sin()+1)*x.cos())
    for action in (handle.load, handle.prefetch, handle.evict):
        with pytest.raises(RuntimeError, match='released'):
            action()
    with pytest.raises(RuntimeError, match='closed'):
        cache.save(x)


def test_store_discovery_initializes_only_when_explicitly_requested():
    from nnscaler.runtime.cpu_offloading import activation_offload_stores

    class LazyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.compute_config = SimpleNamespace(cpu_offload=CPUOffloadConfig())
            self.calls = 0

        def _get_activation_offload_manager(self):
            self.calls += 1
            self._activation_offload_manager = store()
            return self._activation_offload_manager

    module = LazyModule()
    assert activation_offload_stores(module) == () and module.calls == 0
    initialized = activation_offload_stores(module, initialize=True)
    assert initialized == activation_offload_stores(module, initialize=True)
    assert module.calls == 1
    initialized[0].close()


def test_exception_and_manual_commit_do_not_leave_active_batches():
    cache = store()
    with pytest.raises(ValueError, match='abort'):
        with cache.batch() as failed:
            handle = failed.save(torch.ones(4))
            raise ValueError('abort')
    torch.testing.assert_close(handle.load(), torch.ones(4))
    handle.release()
    with cache.batch() as batch:
        batch.commit()
        with pytest.raises(RuntimeError, match='committed'):
            cache.save(torch.ones(4))
    with cache.batch():
        with pytest.raises(RuntimeError, match='Nested'):
            with cache.batch():
                pass
    cache.close()


def test_batch_readiness_policy_is_fixed_for_the_whole_batch():
    cache = store(use_save_events=True)
    with cache.batch() as batch:
        cache.config = replace(cache.config, use_save_events=False)
        assert batch.capture_save_events
    with cache.batch() as batch:
        assert not batch.capture_save_events
    cache.close()


def test_trace_is_bounded_serializable_and_does_not_retain_store_or_tensor():
    cache = store()
    value = torch.randn(8)
    with cache.trace(max_events=2) as trace:
        with cache.batch():
            handles = [cache.save(value) for _ in range(5)]
        for handle in handles:
            handle.release()
    assert len(trace.events) == 2 and trace.dropped_events > 0
    json.dumps(trace.to_dict())
    reference, storage = weakref.ref(cache), weakref.ref(value.untyped_storage())
    cache.close()
    del cache, value, handles, handle
    assert reference() is None and storage() is None


def test_resident_mutations_are_not_silently_accepted():
    cache = store()
    value = torch.ones(8)
    handle = cache.save(value)
    value.add_(1)
    with pytest.raises(RuntimeError, match='inplace'):
        handle.load()
    handle.release()
    cache.close()


def test_saved_tensor_adapter_shares_explicit_batch_and_preserves_gradients():
    cache = store()
    x = torch.randn(4, 5, requires_grad=True)
    weight = torch.nn.Parameter(torch.randn(5, 3))
    with cache.batch() as batch:
        with cache.saved_tensors(exclude=[weight]) as saved:
            assert saved.group is batch
            handle = cache.save((x @ weight).sigmoid())
    with cache.saved_tensors(exclude=[weight]):
        loss = handle.load().square().sum()
    handle.release()
    gradients = torch.autograd.grad(loss, (x, weight), create_graph=True)
    reference = (x @ weight).sigmoid().square().sum()
    expected = torch.autograd.grad(reference, (x, weight), create_graph=True)
    torch.testing.assert_close(gradients, expected)
    torch.testing.assert_close(torch.autograd.grad(gradients[0].sum(), x),
                               torch.autograd.grad(expected[0].sum(), x))
    cache.close()


@cuda
def test_saved_tensor_adapter_offloads_producer_saves_and_reconnects_forward_cache():
    cache = store('cuda')
    x = torch.randn(256, 256, device='cuda', requires_grad=True)
    with cache.saved_tensors():
        activation = x.sigmoid()  # Backward normally retains this output.
        original_storage = weakref.ref(activation.untyped_storage())
        handle = cache.save(activation)
    del activation
    torch.cuda.synchronize()
    assert original_storage() is None
    with cache.saved_tensors():
        loss = handle.load().square().sum()
    handle.release()
    loss.backward()
    torch.testing.assert_close(x.grad, 2*x.sigmoid().square()*(1-x.sigmoid()))
    assert cache.snapshot()['pool']['live_cpu_bytes'] == 0
    cache.close()


@cuda
@pytest.mark.parametrize('copy_stream', [False, True])
def test_forward_cache_evict_restore_preserves_gradients_and_frees_original(copy_stream):
    cache = store('cuda', reload_on_compute_stream=not copy_stream)
    leaf = torch.randn(256, 256, device='cuda', requires_grad=True)
    source = leaf.sin()
    storage = weakref.ref(source.untyped_storage())
    with cache.batch() as batch:
        handle = batch.save(source)
    del source
    torch.cuda.synchronize()
    assert storage() is None
    total = 0
    for _ in range(3):
        assert handle.prefetch()
        value = handle.load()
        total = total + value.sum()  # Backward needs no saved payload.
        borrowed_storage = weakref.ref(value.untyped_storage())
        assert handle.evict()
        assert borrowed_storage() is not None  # The consumer still owns it.
        del value
        torch.cuda.synchronize()
        assert borrowed_storage() is None
    handle.release()
    total.backward()
    torch.testing.assert_close(leaf.grad, 3*leaf.cos())
    stats = cache.snapshot()['pool']
    assert stats['d2h_bytes'] == leaf.numel()*leaf.element_size()
    assert stats['h2d_bytes'] == 3*stats['d2h_bytes']
    assert stats['live_cpu_bytes'] == stats['prefetched_gpu_bytes'] == 0
    cache.close()


@cuda
def test_evicted_borrow_mutation_and_resaved_views_share_one_record():
    cache = store('cuda')
    source = torch.arange(64*64, device='cuda', dtype=torch.float32).reshape(64, 64)
    expected = source.clone()
    handle = cache.save(source)
    del source
    value = handle.load()
    alias = cache.save(value[:, 1::2].t())
    assert alias.status()['storage_id'] == handle.status()['storage_id']
    torch.testing.assert_close(alias.load(), expected[:, 1::2].t())
    assert handle.evict()
    value.add_(1)
    with pytest.raises(RuntimeError, match='evicted view'):
        alias.load()
    del value
    handle.release()
    alias.release()
    cache.close()


@cuda
def test_evict_reuses_a_live_borrow_without_forking_counters_or_recopying():
    cache = store('cuda')
    handle = cache.save(torch.ones(64, 64, device='cuda'))
    borrowed = handle.load()
    h2d_bytes = cache.snapshot()['pool']['h2d_bytes']
    for _ in range(12):
        handle.evict()
        restored = handle.load()
        assert restored.untyped_storage() is borrowed.untyped_storage()
    assert cache.snapshot()['pool']['h2d_bytes'] == h2d_bytes
    assert handle.status()['observed_gpu_copies'] == 1
    borrowed.add_(1)
    with pytest.raises(RuntimeError, match='inplace'):
        handle.load()
    handle.release()
    cache.close()


@cuda
def test_storage_lookup_is_not_the_owner_registry():
    cache = store('cuda')
    x = torch.randn(64, 64, device='cuda', requires_grad=True)
    with cache.saved_tensors():
        loss = x.sin().square().sum()
    # Freed GPU storage identities may be reused by unrelated allocations.
    # Removing lookup entries must not hide live saved values from close().
    cache.records.clear()
    with pytest.raises(RuntimeError, match='autograd'):
        cache.close()
    loss.backward()
    torch.testing.assert_close(x.grad, 2*x.sin()*x.cos())
    cache.close()


@cuda
@pytest.mark.skipif(torch.version.cuda is None, reason='CUDA sleep dependency probe')
@pytest.mark.parametrize('mode', ['uncommitted', 'budget', 'small'])
def test_resident_loads_preserve_readiness_and_consumer_lifetime_across_streams(mode):
    cache = store('cuda', max_cpu_bytes=0 if mode == 'budget' else 64*1024**2,
                  min_tensor_bytes=1024**2 if mode == 'small' else 0,
                  use_save_events=True)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    source = torch.zeros(64, 64, device='cuda')
    output = torch.empty_like(source)
    torch.cuda.synchronize()

    def read(handle):
        with torch.cuda.stream(consumer):
            value = handle.load()
            torch.cuda._sleep(100_000_000)
            output.copy_(value)

    with cache.batch():
        with torch.cuda.stream(producer):
            torch.cuda._sleep(20_000_000)
            source.fill_(3)
            handle = cache.save(source)
        if mode == 'uncommitted':
            read(handle)
            handle.release()
    if mode != 'uncommitted':
        read(handle)
        handle.release()
    del source
    producer.synchronize()
    # The original allocation must not be recycled on its allocation stream
    # while the other consumer stream still reads it.
    replacement = torch.empty_like(output).fill_(9)
    consumer.synchronize()
    torch.testing.assert_close(output, torch.full_like(output, 3))
    del replacement
    cache.close()


@cuda
def test_host_budget_fallback_and_store_close_keep_other_store_valid():
    a, b = store('cuda', max_cpu_bytes=0), store('cuda', max_cpu_bytes=0)
    x = torch.randn(128, device='cuda')
    ha, hb = a.save(x), b.save(x + 1)
    assert ha.status()['state'] == 'resident'
    torch.testing.assert_close(ha.load(), x)
    ha.release()
    a.close()
    torch.testing.assert_close(hb.load(), x + 1)
    hb.release()
    b.close()
