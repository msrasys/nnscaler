"""D2H opportunities must preserve source ownership, deadlines and budgets."""
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import gc
import weakref

import pytest
import torch

from nnscaler.runtime.activation_batch import ActivationBatch, ForwardCommitQueue, DeferredCommitQueue, ScopeRestoreQueue
from nnscaler.runtime.activation_config import CPUOffloadConfig
from nnscaler.runtime.activation_memory import _pools
from tests.runtime.test_cpu_offloading import OffloadModule


class Record:
    def __init__(self, uid, size=3):
        self.uid, self.size = uid, size
        self.live_saves, self.deferred_offload = 1, False
        self.d2h_event = self.device_tensor = None

    def prefetch(self, **kwargs):
        self.device_tensor = True
        return True


class Transport:
    inflight_bytes = 0

    def __init__(self):
        self.waits = []

    def poll(self):
        pass

    def throttle(self, size=0):
        if self.inflight_bytes:
            self.waits.append(size)
            self.inflight_bytes = 0

    def prefetch_batch(self):
        return nullcontext()


class Store:
    def __init__(self, limit=32):
        self.config = CPUOffloadConfig(forward_commit_unit='scope', d2h_launch_phase='backward',
                                      max_pending_bytes=limit, max_inflight_bytes=limit//2,
                                      backpressure_mode='stream')
        self.pool = SimpleNamespace(stats=defaultdict(int))
        self.transport = Transport()
        self.copies = []
        self.fail = False

    def offload_batch(self, records, events=None):
        if self.fail:
            raise ValueError('copy failure')
        self.copies.append(tuple(r.uid for r in records))
        for record in records:
            record.d2h_event = True
            self.transport.inflight_bytes += record.size


def add(store, queue, key, *records):
    group = ActivationBatch(store)
    group.pending.extend(records)
    group.records.extend(weakref.ref(r) for r in records)
    commits = ForwardCommitQueue(store, unit='scope', target_bytes=1, deferred=queue)
    commits.add(group)
    commits.finish()
    queue.enqueue(key, commits)
    return commits, group


def test_deferred_source_is_not_mistaken_for_resident_during_prefetch():
    store = Store(); queue = DeferredCommitQueue(store); record = Record(1)
    _, group = add(store, queue, 'a', record)
    restore = ScopeRestoreQueue(store); restore.seal([group])
    assert not store.copies and record.deferred_offload
    assert restore.prefetch() is False and restore.pending
    queue.launch()
    assert store.copies == [(1,)] and not record.deferred_offload
    assert restore.prefetch() is True and not restore.pending
    assert queue.queued_bytes == 0


def test_backward_quota_skips_current_consumer_aliases_and_keeps_fifo():
    store = Store(); queue = DeferredCommitQueue(store)
    records = [Record(i) for i in range(3)]
    for record in records:
        add(store, queue, record.uid, record)
    queue.launch(protected={0})
    assert store.copies == [(1,)]
    queue.launch()
    assert store.copies == [(1,), (0,)]
    queue.cancel(2)
    assert queue.queued_bytes == 0 and not records[2].deferred_offload


def test_source_budget_charges_deferred_and_unfinished_submitted_bytes():
    store = Store(limit=4); queue = DeferredCommitQueue(store)
    add(store, queue, 'a', Record(1))
    add(store, queue, 'b', Record(2))
    assert store.copies == [(1,)] and store.transport.waits == [3]
    assert queue.queued_bytes + store.transport.inflight_bytes <= 4
    assert store.pool.stats['d2h_deferred_budget_flushes'] == 1
    queue.abort()


def test_indivisible_source_can_cross_budget_but_next_admission_waits():
    store = Store(limit=4); queue = DeferredCommitQueue(store)
    add(store, queue, 'large', Record(1, 12))
    assert store.copies == [(1,)] and store.transport.inflight_bytes == 12
    add(store, queue, 'small', Record(2, 1))
    assert store.transport.waits == [1]
    assert queue.queued_bytes == 1 and store.transport.inflight_bytes == 0
    queue.abort()


def test_abort_and_submission_failure_drop_owned_sources_without_gc():
    enabled = gc.isenabled(); gc.disable()
    try:
        store = Store(); queue = DeferredCommitQueue(store); record = Record(1)
        ref = weakref.ref(record)
        commits, _ = add(store, queue, 'a', record)
        del record
        commits.abort()  # Charge must survive an external scope abort until removal.
        queue.abort()
        assert ref() is None and queue.queued_bytes == 0
        record = Record(2); commits, _ = add(store, queue, 'b', record)
        store.fail = True
        with pytest.raises(ValueError, match='copy failure'):
            queue.launch()
        assert not record.deferred_offload and not commits.batch.pending
        assert queue.queued_bytes == 0
    finally:
        if enabled: gc.enable()


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.fixture
def module():
    gc.collect(); _pools.clear()
    result = OffloadModule(forward_commit_unit='scope', d2h_launch_phase='backward',
                           max_pending_bytes=128*1024, max_inflight_bytes=64*1024,
                           backpressure_mode='stream', pipeline_prefetch_steps=2,
                           reload_on_compute_stream=False, use_save_events=True)
    yield result
    gc.collect()


@cuda
@pytest.mark.parametrize('checkpoint', [False, True])
@pytest.mark.parametrize('restore', ['scope', 'group'])
def test_backward_window_submits_future_scope_and_keeps_current_input(module, checkpoint, restore):
    manager=module._get_activation_offload_manager()
    manager.config=replace(manager.config,transfer_unit=restore)
    leaves = [torch.randn(64,64,device='cuda',requires_grad=True) for _ in range(2)]
    outputs = []
    size = leaves[0].numel()*leaves[0].element_size()
    with module.activation_offload_schedule([('a',0,5),('b',1,3)]) as cursor:
        for position,key in enumerate(('a','b')):
            cursor.before(position)
            with module.activation_offload_scope(key,pipeline=True):
                with module.cpu_offloading_hooks() as saved:
                    value=leaves[position]+1
                    y=saved.checkpoint(torch.sin,value) if checkpoint else value.sin()
                saved.watch(y)
                outputs.append(y)
                del value,y
            cursor.after(position)
        assert module.activation_offload_stats()['d2h_bytes'] == 0
        cursor.before(3); cursor.before(3)  # A retried hint is not another compute window.
        assert module.activation_offload_stats()['d2h_bytes'] == size
        actual, = torch.autograd.grad(outputs[1].sum(),leaves[1])
        torch.testing.assert_close(actual,(leaves[1]+1).cos())
        module.finish_activation_offload('b'); cursor.after(3)
        cursor.before(5)
        actual, = torch.autograd.grad(outputs[0].sum(),leaves[0])
        torch.testing.assert_close(actual,(leaves[0]+1).cos())
        module.finish_activation_offload('a'); cursor.after(5)
    assert module.activation_offload_stats()['live_cpu_bytes'] == 0
    assert module._get_activation_offload_manager().deferred_commits is None


@cuda
def test_alias_needed_now_does_not_force_a_late_round_trip(module):
    leaf=torch.randn(64,64,device='cuda',requires_grad=True); value=leaf+1
    outputs=[]
    with module.activation_offload_schedule([('a',0,5),('b',1,3)]) as cursor:
        for position,key in enumerate(('a','b')):
            with module.activation_offload_scope(key,pipeline=True):
                with module.cpu_offloading_hooks() as saved:y=value.sin()
                saved.watch(y); outputs.append(y)
            cursor.after(position)
        del value,y
        for position,key,index in ((3,'b',1),(5,'a',0)):
            cursor.before(position)
            actual,=torch.autograd.grad(outputs[index].sum(),leaf,retain_graph=index==1)
            torch.testing.assert_close(actual,(leaf+1).cos())
            module.finish_activation_offload(key);cursor.after(position)
        assert module.activation_offload_stats()['d2h_bytes'] == 0


@cuda
def test_mutation_while_waiting_for_a_backward_window_is_rejected(module):
    leaf=torch.randn(64,64,device='cuda',requires_grad=True); value=leaf+1
    with pytest.raises(RuntimeError,match='inplace'):
        with module.activation_offload_schedule([('a',0,5),('b',1,3)]) as cursor:
            with module.activation_offload_scope('a',pipeline=True):
                with module.cpu_offloading_hooks() as saved:y=value.sin()
                saved.watch(y)
            value.add_(1)
            cursor.before(3)
    manager=module._get_activation_offload_manager()
    assert not manager.scopes and manager.deferred_commits is None


@cuda
def test_region_larger_than_deferred_budget_submits_before_next_action(module):
    leaf=torch.randn(512,512,device='cuda',requires_grad=True)
    with module.activation_offload_schedule([('a',0,3)]) as cursor:
        with module.activation_offload_scope('a',pipeline=True):
            with module.cpu_offloading_hooks() as saved:y=(leaf+1).sin()
            saved.watch(y)
        assert module.activation_offload_stats()['d2h_deferred_budget_flushes'] == 1
        assert module.activation_offload_stats()['d2h_bytes'] == leaf.numel()*leaf.element_size()
        cursor.before(3)
        actual,=torch.autograd.grad(y.sum(),leaf)
        torch.testing.assert_close(actual,(leaf+1).cos())
        module.finish_activation_offload('a')
    assert module.activation_offload_stats()['live_cpu_bytes'] == 0


@cuda
def test_schedule_failure_releases_deferred_records_with_gc_disabled(module):
    enabled=gc.isenabled();gc.disable()
    try:
        leaf=torch.randn(64,64,device='cuda',requires_grad=True)
        with pytest.raises(ValueError,match='aborted forward'):
            with module.activation_offload_schedule([('a',0,3)]):
                with module.activation_offload_scope('a',pipeline=True):
                    with module.cpu_offloading_hooks() as saved:y=(leaf+1).sin()
                    saved.watch(y)
                record_ref=weakref.ref(saved.group.records[0]())
                raise ValueError('aborted forward')
        del y
        assert record_ref() is None
        manager=module._get_activation_offload_manager()
        assert not manager.scopes and manager.deferred_commits is None
    finally:
        if enabled:gc.enable()
