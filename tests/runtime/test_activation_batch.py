"""Submission/restore contracts using CPU-controlled transfer completions."""
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace
import weakref

import pytest
import torch

from nnscaler.runtime.activation_batch import ActivationBatch, ScopeRestoreQueue, record_batches
from nnscaler.runtime.activation_config import CPUOffloadConfig
from nnscaler.runtime.activation_memory import _pools
from nnscaler.runtime.cpu_offloading import ActivationOffloadManager, CPUOffloadContext


class Record:
    def __init__(self, uid, size, *, live=1, admitted=True):
        self.uid, self.size, self.live_saves = uid, size, live
        self.admitted = admitted
        self.d2h_event = self.device_tensor = None
        self.prefetch_calls = []

    def prepare_offload(self):
        return self.admitted and self.d2h_event is None

    def prefetch(self, *, from_schedule=False):
        self.prefetch_calls.append(from_schedule)
        if self.admitted:
            self.device_tensor = object()
        return self.admitted


@pytest.fixture(autouse=True)
def cpu_pool(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    _pools.clear()
    yield
    _pools.clear()


def setup_manager(monkeypatch, **options):
    module = torch.nn.Module()
    config = CPUOffloadConfig(**options)
    manager = ActivationOffloadManager(module, config)
    module._get_activation_offload_manager = lambda: manager
    submitted = []

    def submit(records, events=None):
        submitted.append(tuple(record.uid for record in records))
        for record in records:
            record.d2h_event = object()

    monkeypatch.setattr(manager.transport, 'submit_batch', submit)
    return module, manager, submitted


def save_group(module, records):
    with CPUOffloadContext(module) as context:
        context.group.pending.extend(records)
        context.group.records.extend(weakref.ref(record) for record in records)
    return context.group


@pytest.mark.parametrize('restore,submit', [('scope', 'group'), ('group', 'scope')])
def test_forward_submission_is_independent_of_backward_coverage(monkeypatch, restore, submit):
    module, manager, batches = setup_manager(monkeypatch, transfer_unit=restore, forward_commit_unit=submit,
                                              scope_prefetch_groups=1)
    records = [Record(i, 8) for i in range(3)]
    with manager.scope('unit', pipeline=True) as scope:
        for index, record in enumerate(records):
            save_group(module, [record])
            assert len(batches) == (index+1 if submit == 'group' else 0)
        if restore == 'scope':
            assert scope.transfer.prefetch() is False  # Producer scope is still open.
    assert batches == ([(0,), (1,), (2,)] if submit == 'group' else [(0, 1, 2)])
    assert manager.prefetch('unit') is True
    assert [r.device_tensor is not None for r in records] == ([True]*3 if restore == 'scope' else [False,False,True])
    manager.finish('unit')


def test_scope_batch_target_flushes_before_forward_end_and_bounds_submission(monkeypatch):
    module, manager, batches = setup_manager(monkeypatch, transfer_unit='scope', commit_batch_bytes=8)
    records = [Record(i, size) for i, size in enumerate((3, 3, 3, 11, 2))]
    with manager.scope('unit', pipeline=True):
        save_group(module, records[:2])
        assert batches == []
        save_group(module, records[2:3])
        assert batches == [(0, 1), (2,)]
        save_group(module, records[3:])
        assert batches == [(0, 1), (2,), (3,), (4,)]  # One oversized storage makes progress.
    assert manager.prefetch('unit') is True
    manager.finish('unit')


def test_dead_and_budget_declined_values_do_not_create_empty_transfer_waits(monkeypatch):
    module, manager, batches = setup_manager(monkeypatch, transfer_unit='scope', commit_batch_bytes=8)
    def unexpected_wait(*args):
        pytest.fail('A batch with no admitted records must not wait for other copies')
    monkeypatch.setattr(manager, 'throttle', unexpected_wait)
    records = [Record(0, 9, live=0), Record(1, 11, admitted=False)]
    with manager.scope('unit', pipeline=True):
        save_group(module, records)
    assert not batches and manager.pool.stats['discarded_save_bytes'] == 9
    manager.finish('unit')


def test_scope_abort_releases_pending_sources_without_cyclic_gc(monkeypatch):
    import gc
    module, manager, batches = setup_manager(monkeypatch, transfer_unit='scope')
    record = Record(0, 8)
    observed = weakref.ref(record)
    enabled = gc.isenabled()
    gc.disable()
    try:
        with pytest.raises(RuntimeError, match='abort'):
            with manager.scope('unit', pipeline=True) as scope:
                group = save_group(module, [record])
                del record
                assert observed() is not None
                raise RuntimeError('abort')
        assert observed() is None
        assert scope.commits.batch.pending == [] and not scope.transfer.pending
        assert manager.scopes == {} and not batches
        assert group.pending == []
    finally:
        if enabled:
            gc.enable()


def test_restore_queue_deduplicates_views_retries_budget_and_skips_dead_values():
    stats = Counter()
    store = SimpleNamespace(config=CPUOffloadConfig(max_prefetch_bytes=8),
                            pool=SimpleNamespace(stats=stats),
                            transport=SimpleNamespace(prefetch_batch=nullcontext))
    a, b, oversized, dead = [Record(i, size) for i, size in enumerate((4, 4, 9, 4))]
    for r in (a, b, oversized, dead):
        r.d2h_event = object()
    oversized.admitted = b.admitted = False
    groups = [SimpleNamespace(records=[weakref.ref(r) for r in (a,b)]),
              SimpleNamespace(records=[weakref.ref(r) for r in (b,oversized,dead)])]
    del r, dead
    queue = ScopeRestoreQueue(store)
    queue.seal(groups)
    assert queue.prefetch(from_schedule=True) is False
    assert len(oversized.prefetch_calls) == len(b.prefetch_calls) == 1 and not a.prefetch_calls
    b.admitted = True
    assert queue.prefetch(from_schedule=True) is True
    assert b.prefetch_calls == [True,True] and a.prefetch_calls == [True]
    assert stats['scope_prefetch_oversize_bytes'] == 9


def test_batch_flush_renews_producer_event_before_later_saves(monkeypatch):
    events = []
    class Event:
        def __init__(self):
            events.append(self)
        def record(self, stream):
            pass
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    store = SimpleNamespace(config=CPUOffloadConfig(use_save_events=True), pool=SimpleNamespace(stats=Counter()),
                            offload_batch=lambda *args: None)
    batch = ActivationBatch(store)
    stream = SimpleNamespace(cuda_stream=7)
    first = batch.capture_producer_event(stream)
    assert batch.capture_producer_event(stream) is first
    batch.flush()
    second = batch.capture_producer_event(stream)
    assert second is not first and not batch.closed
    batch.commit()
    assert batch.closed and batch.producer_events is None


@pytest.mark.parametrize('target', [0, 4, 8, 16])
def test_physical_batching_preserves_order_and_storage_atomicity(target):
    records = [Record(i, size) for i,size in enumerate((3, 4, 17, 1, 2, 8))]
    batches = list(record_batches(records, target))
    assert [r for batch in batches for r in batch] == records
    if target:
        assert all(sum(r.size for r in batch) <= target or len(batch)==1 for batch in batches)


def test_legacy_config_defaults_preserve_coupled_transfer_behavior():
    for unit in ('auto', 'scope', 'group'):
        fields = asdict(CPUOffloadConfig(transfer_unit=unit))
        for key in ('forward_commit_unit','commit_batch_bytes','copy_chunk_bytes'):
            fields.pop(key)
        config = CPUOffloadConfig(**fields)
        assert config.commit_batch_bytes == config.copy_chunk_bytes == 0
        for pipeline in (False, True):
            assert config.commit_unit(pipeline=pipeline) == config.restore_unit(pipeline=pipeline)
