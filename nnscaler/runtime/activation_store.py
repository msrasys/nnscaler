# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Model-independent immutable activation values and explicit ownership handles."""
from contextlib import contextmanager
from contextvars import ContextVar
from itertools import count
import threading
import time
import weakref
import torch
from .activation_config import CPUOffloadConfig
from .activation_batch import ActivationBatch, record_batches
from .activation_autograd import _GradientAnchor, _RestoreGradient
from .activation_memory import _shared_pool
from .activation_transfer import ActivationTransferEngine
from .activation_observer import _Observation

_record_ids = count()


class _ResidentTensorRef:
    def __init__(self, tensor, reason='resident'):
        self.tensor = tensor.detach()
        self.reason = reason
        self.version = None if torch.is_inference(tensor) else tensor._version
        self.producer_stream = (torch.cuda.current_stream(tensor.device)
                                if tensor.device.type == 'cuda' and tensor.layout == torch.strided else None)
        if self.producer_stream is not None:
            tensor.record_stream(self.producer_stream)

    def unpack(self):
        # saved_tensors_hooks bypasses PyTorch's normal SavedVariable check,
        # including on the resident/budget-fallback path. Restore that check.
        if self.version is not None and self.tensor._version != self.version:
            _raise_activation_mutation('resident unpack', self.version, self.tensor._version)
        if self.producer_stream is not None:
            consumer = torch.cuda.current_stream(self.tensor.device)
            if consumer.cuda_stream != self.producer_stream.cuda_stream:
                consumer.wait_stream(self.producer_stream)
                self.tensor.record_stream(consumer)
        return self.tensor


def _raise_activation_mutation(phase, expected, actual):
    raise RuntimeError(
        f'CPU activation offload: an activation retained by the store was modified by an inplace operation '
        f'({phase}: expected version {expected}, found {actual}). '
        'Saved tensors and their storage aliases must remain immutable until all saved values and handles are released. '
        'Custom operators must correctly update PyTorch version counters; writes bypassing those counters '
        'and independently versioned storage aliases are unsupported.'
    )


def _version_tracker(tensor):
    # Keep the shared version counter, without retaining activation storage.
    # Only this private detached tracker changes its data pointer. The original
    # tensor and every user-visible view remain untouched.
    tracker = tensor.detach()
    tracker.data = torch.empty(0, dtype=tensor.dtype, device=tensor.device)
    return tracker


def _byte_view(tensor):
    """View the entire backing allocation while sharing the tensor's counter.

    A zero-element strided view makes dtype reinterpretation valid even for a
    noncontiguous slice. The final byte view includes unused trailing bytes.
    These are ordinary views; no user or private payload pointer is replaced.
    """
    empty = torch.as_strided(tensor.detach(), (0,), (1,), 0).view(torch.uint8)
    return torch.as_strided(empty, (tensor.untyped_storage().nbytes(),), (1,), 0)


class _GPUCopy:
    """Weak allocation/view identities and one counter for a physical copy."""
    def __init__(self, tensor, ready=None, tracker=None):
        self.storage = weakref.ref(tensor.untyped_storage())
        self.key = (tensor.device, tensor.untyped_storage()._cdata)
        self.tracker = tracker
        self.version = tensor._version
        self.ready = ready
        self.views = weakref.WeakValueDictionary()
        self.remember(tensor)

    def remember(self, tensor):
        # Normal views retain their root. Detached aliases remain safe but may
        # not provide a surviving known tensor from which to reuse the counter.
        base = tensor._base if tensor._base is not None else tensor
        self.views[id(base)] = base

    def reuse(self):
        for tensor in tuple(self.views.values()):
            if tensor.untyped_storage() is self.storage():
                return _byte_view(tensor)
        return None

    def freeze_counter(self, tensor):
        # While resident, the record already owns the tensor and checks its
        # counter directly. Allocate a storage-free tracker only upon eviction.
        if self.tracker is None:
            self.tracker = _version_tracker(tensor)


class _StorageRecord:
    # Compatibility for older diagnostic implementations; storage owns no model.
    manager = property(lambda self: self.store)

    def __init__(self, tensor, store, host=None):
        self.uid = next(_record_ids)
        self.live_saves = 0
        self.store, self.pool = store, store.pool
        self.device = tensor.device
        self.producer_stream = torch.cuda.current_stream(tensor.device)
        self.producer_ready_event = None
        self.size = tensor.untyped_storage().nbytes()
        self.host = host
        self.source = _byte_view(tensor)
        self.source_version = self.source._version
        self.version_tracker = _version_tracker(tensor)
        self.version = tensor._version
        self.storage_ref = weakref.ref(tensor.untyped_storage())
        self.original_copy = _GPUCopy(tensor, tracker=self.version_tracker)
        self.original_copy.remember(self.source)
        self.copies = [self.original_copy]
        self.device_copy = None
        self.device_tensor = None
        self.restored_version = None
        self.d2h_event = self.h2d_event = None
        self.speculative = False
        self.lock = threading.RLock()

    def check_version(self, phase='unpack'):
        if self.source is not None and self.source._version != self.source_version:
            _raise_activation_mutation('resident storage ' + phase, self.source_version, self.source._version)
        if self.restored_version is not None and self.device_tensor._version != self.restored_version:
            _raise_activation_mutation('restored ' + phase, self.restored_version, self.device_tensor._version)
        if self.version_tracker._version != self.version:
            _raise_activation_mutation(phase, self.version, self.version_tracker._version)
        live = []
        for copy in self.copies:
            if copy.tracker is not None and copy.tracker._version != copy.version:
                _raise_activation_mutation('evicted view ' + phase, copy.version, copy.tracker._version)
            if copy.storage() is not None:
                live.append(copy)
            elif self.store.records.get(copy.key) is self:
                self.store.records.pop(copy.key, None)
        self.copies = live

    def owns_storage(self, storage):
        if self.storage_ref() is storage:
            return True
        return any(copy.storage() is storage for copy in self.copies)

    def _register_restored_storage(self):
        # A later forward can save a restored view again through the same
        # store. Preserve its immutable record instead of creating a new copy.
        storage = self.device_tensor.untyped_storage()
        self.store.records[(self.device, storage._cdata)] = self

    def evict(self):
        """Drop only this record's reload reference; returned tensors stay valid."""
        with self.lock:
            self.check_version('evict')
            if self.d2h_event is None or self.device_tensor is None:
                return False
            self.device_copy.freeze_counter(self.device_tensor)
            self.device_tensor = None
            self.device_copy = None
            self.restored_version = None
            if self.speculative:
                with self.pool.lock:
                    self.pool.prefetched -= self.size
                self.speculative = False
            self.pool.stats['evicted_gpu_bytes'] += self.size
            if self.store.observation.enabled:
                self.store.observation.emit('evict', storage_id=self.uid, nbytes=self.size)
            return True

    def prepare_offload(self):
        with self.lock:
            if self.d2h_event is not None:
                return False
            self.check_version('D2H preparation')
            if self.host is None:
                self.host = self.pool.acquire(self.size)
                if self.host is None and self.store.observation.enabled:
                    self.store.observation.emit('resident', storage_id=self.uid, nbytes=self.size,
                                                  reason='host_allocation_declined')
            return self.host is not None

    def offload(self):
        with self.lock:
            if self.d2h_event is not None:
                return False
            if self.host is None and not self.prepare_offload():
                return False
            self.check_version('D2H submission')
            self.d2h_event = self.store.transport.to_host(self.source, self.host[:self.size])
            self.original_copy.ready = self.d2h_event
            self.producer_ready_event = None
            self.pool.stats['d2h_bytes'] += self.size
            self.pool.stats['offloaded_storages'] += 1
            self.source = None
            if self.store.observation.enabled:
                self.store.observation.emit('d2h_submit', storage_id=self.uid, nbytes=self.size)
            return True

    def prefetch(self, *, demanded=False, from_schedule=False):
        with self.lock:
            self.check_version('unpack' if demanded else 'prefetch')
            if self.device_tensor is not None:
                if demanded and self.speculative:
                    with self.pool.lock:
                        self.pool.prefetched -= self.size
                    self.speculative = False
                return True
            if self.d2h_event is None:
                return False
            # Pipeline inputs, outputs and YOCO caches can still own the original
            # allocation. Reuse it instead of creating a second GPU copy. A weak
            # storage reference cannot keep an otherwise dead allocation alive.
            for copy in self.copies:
                reused = copy.reuse()
                if reused is not None:
                    self.device_tensor, self.device_copy = reused, copy
                    self.restored_version = reused._version
                    self.h2d_event = copy.ready
                    self.pool.stats['resident_reused_bytes'] += self.size
                    self._register_restored_storage()
                    if self.store.observation.enabled:
                        self.store.observation.emit('reuse_gpu', storage_id=self.uid, nbytes=self.size)
                    return True
            with self.pool.lock:
                if not demanded:
                    if self.pool.prefetched + self.size > self.store.config.max_prefetch_bytes:
                        self.pool.stats['prefetch_budget_skips'] += 1
                        if self.store.observation.enabled:
                            self.store.observation.emit('prefetch_skip', storage_id=self.uid,
                                                          nbytes=self.size, reason='prefetch_budget')
                        return False
                    self.pool.prefetched += self.size
                    self.pool.stats['peak_prefetched_gpu_bytes'] = max(
                        self.pool.stats['peak_prefetched_gpu_bytes'], self.pool.prefetched)
                    self.speculative = True
                try:
                    self.device_tensor, self.h2d_event = self.store.transport.to_device(
                        self.host[:self.size], self.d2h_event)
                except BaseException:
                    if self.speculative:
                        self.pool.prefetched -= self.size
                        self.speculative = False
                    raise
                self.restored_version = self.device_tensor._version
                self.device_copy = _GPUCopy(self.device_tensor, self.h2d_event)
                self.copies.append(self.device_copy)
                self._register_restored_storage()
                self.pool.stats['h2d_bytes'] += self.size
                self.pool.stats['demand_loads' if demanded else 'prefetch_loads'] += 1
                if from_schedule:
                    self.pool.stats['schedule_prefetch_loads'] += 1
                if self.store.observation.enabled:
                    self.store.observation.emit('h2d_submit', storage_id=self.uid, nbytes=self.size,
                                                  reason='demand' if demanded else 'prefetch')
            return True

    def unpack(self, value):
        observation = getattr(self.store, 'observation', None)
        if observation is not None and observation.enabled:
            with observation.range('load'):
                return self._unpack_value(value)
        return self._unpack_value(value)

    def _unpack_value(self, value):
        if self.d2h_event is None:
            self.check_version()
            raw = self.source
            if self.device.type == 'cuda':
                consumer = torch.cuda.current_stream(self.device)
                if consumer.cuda_stream != self.producer_stream.cuda_stream:
                    if self.producer_ready_event is None:
                        consumer.wait_stream(self.producer_stream)
                    else:
                        consumer.wait_event(self.producer_ready_event)
                    raw.record_stream(consumer)
        else:
            self.prefetch(demanded=True)
            raw = self.device_tensor
            stream = torch.cuda.current_stream(self.device)
            self.store.transport.wait_for_restore(stream, self.h2d_event, storage_id=self.uid, nbytes=self.size)
            raw.record_stream(stream)
        observation = getattr(self.store, 'observation', None)
        if observation is not None and observation.enabled:
            observation.emit('load', storage_id=self.uid, nbytes=self.size)
        # set_(storage) creates an independent version counter for every view.
        # Reinterpret and as_strided instead so writes through any restored
        # view are visible to the shared record on the next access. A storage
        # may have unused trailing bytes smaller than one typed element.
        usable = raw.numel() // value.element_size * value.element_size
        typed = (raw if usable == raw.numel() else raw[:usable]).view(value.dtype)
        result = torch.as_strided(typed, value.shape, value.stride, value.offset)
        (self.device_copy or self.original_copy).remember(result)
        return result

    def __del__(self):
        # Saved references own records; groups and dedup tables only hold weakrefs.
        # CUDA records every consumer stream before a restored storage is released.
        if getattr(self, 'host', None) is not None:
            self.pool.release(self.host, self.h2d_event or self.d2h_event)
            if self.speculative:
                with self.pool.lock:
                    self.pool.prefetched -= self.size


class _SavedTensor:
    def __init__(self, tensor, record):
        self.record = record
        record.live_saves += 1
        self.shape, self.stride = tensor.shape, tensor.stride()
        self.offset, self.dtype = tensor.storage_offset(), tensor.dtype
        self.element_size = tensor.element_size()
        # Supported source aliases share the record's original counter; views
        # produced by unpack share its source/reload counter. Track those once
        # per storage instead of allocating an empty tracker for every save.

    def unpack(self):
        return self.record.unpack(self)

    def __del__(self):
        self.record.live_saves -= 1
        observation = getattr(self.record.store, 'observation', None)
        if observation is not None and observation.enabled:
            observation.emit('release_save', storage_id=self.record.uid, nbytes=self.record.size)



class ActivationHandle:
    """Owned immutable activation view, optionally carrying its autograd edge.

    Keep this handle instead of the original tensor during an idle interval.
    Returned tensors are ordinary tensor owners and must also be dropped to
    reclaim GPU memory. Explicit release is idempotent and only closes this
    handle; tensors already returned to consumers and their gradients survive.
    """
    def __init__(self, value, anchor=None):
        self._value, self._anchor = value, anchor

    @property
    def released(self):
        return self._value is None

    def _checked_value(self):
        if self.released:
            raise RuntimeError('Activation handle has been released')
        return self._value

    def prefetch(self):
        value = self._checked_value()
        return value.record.prefetch() if isinstance(value, _SavedTensor) else False

    def load(self):
        value = self._checked_value().unpack()
        if self._anchor is not None and torch.is_grad_enabled():
            value = _RestoreGradient.apply(self._anchor, value)
        return value

    def evict(self):
        value = self._checked_value()
        return value.record.evict() if isinstance(value, _SavedTensor) else False

    def status(self):
        if self.released:
            return dict(released=True)
        value = self._value
        if isinstance(value, _ResidentTensorRef):
            tensor = value.tensor
            return dict(released=False, state='resident', reason=value.reason,
                        tensor_bytes=tensor.numel() * tensor.element_size())
        record = value.record
        with record.lock:
            return dict(released=False, state=('resident' if record.d2h_event is None else
                        'restored' if record.device_tensor is not None else 'host_copy_submitted'),
                        storage_id=record.uid, storage_bytes=record.size,
                        original_storage_alive=record.storage_ref() is not None,
                        observed_gpu_copies=sum(copy.storage() is not None for copy in record.copies))

    def release(self):
        self._value = self._anchor = None

    def __enter__(self):
        self._checked_value()
        return self

    def __exit__(self, *exc):
        self.release()


class ActivationStore:
    """Activation values independent of modules, autograd scheduling and recipes.

    CUDA transfers use a process-wide host pool and this store's device streams.
    CPU and unsupported-layout values stay resident. The caller must report all
    mutations through PyTorch version counters and release external GPU owners.
    """
    def __init__(self, config=None, *, device=None):
        config = CPUOffloadConfig() if config is None else config
        if not isinstance(config, CPUOffloadConfig):
            raise TypeError('config must be CPUOffloadConfig')
        self.pool = _shared_pool(config)
        self.records = weakref.WeakValueDictionary()
        # Ownership is independent of the transient storage-identity index.
        self._live_records = weakref.WeakSet()
        self.device = torch.device(device) if device is not None else (
            torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu'))
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('ActivationStore supports CPU-resident or CUDA values')
        self.observation = _Observation()
        self.transport = ActivationTransferEngine(config, self.device, self.pool.stats, self.observation)
        self._active_batch = ContextVar('activation_store_batch', default=None)
        self._handles = weakref.WeakSet()
        self._lock = threading.RLock()
        self.closed = False

    @property
    def config(self):
        return self.transport.config

    @config.setter
    def config(self, config):
        if not isinstance(config, CPUOffloadConfig):
            raise TypeError('config must be CPUOffloadConfig')
        if (config.max_cpu_bytes != self.pool.config.max_cpu_bytes
                or config.pool_slab_bytes != self.pool.config.pool_slab_bytes):
            raise ValueError('Cannot change a live process pool budget or allocator')
        self.transport.config = config

    # Compatibility accessors for existing runtime diagnostic integrations.
    # The transport engine is the sole owner of streams and queue accounting.
    d2h_stream = property(lambda self: self.transport.d2h_stream)
    h2d_stream = property(lambda self: self.transport.h2d_stream)
    inflight = property(lambda self: self.transport.inflight)
    inflight_bytes = property(lambda self: self.transport.inflight_bytes,
                              lambda self, value: setattr(self.transport, 'inflight_bytes', value))

    def throttle(self, size=0):
        self.transport.throttle(size)

    def offload_batch(self, records, producer_events=None):
        with self.observation.range('batch_submit'):
            return self._submit_records(records, producer_events)

    def _submit_records(self, records, producer_events=None):
        # Prepare before admission so allocation never gaps a submitted batch.
        started = time.perf_counter_ns() if self.observation.enabled else None
        count = nbytes = 0
        for batch in record_batches(records, self.config.commit_batch_bytes):
            batch = [record for record in batch if record.prepare_offload()]
            if batch:
                self.throttle(0)
                self.transport.submit_batch(batch, producer_events)
                if started is not None:
                    count += len(batch)
                    nbytes += sum(record.size for record in batch)
        if started is not None:
            self.observation.emit('batch_submit', count=count, nbytes=nbytes,
                                  cpu_duration_ns=time.perf_counter_ns() - started)

    def trace(self, *, max_events=10000, cuda_timing=False):
        """Observe without retaining tensors; optional timing keeps bounded CUDA events."""
        return self.observation.trace(max_events, cuda_timing=cuda_timing)

    def saved_tensors(self, *, exclude=()):
        """Store autograd saves in this store; exclude parameters/buffers by storage."""
        from .activation_autograd import SavedActivationContext
        return SavedActivationContext(self, exclude=exclude)

    def prefetch_many(self, handles):
        """Prefetch distinct records in one submission batch.

        Returns the number of admitted/already-restored records, not completed
        GPU copies. Per-record producer events and the shared byte budget still
        apply. Validate ownership before submitting any work.
        """
        if self.closed:
            raise RuntimeError('Activation store has been closed')
        records = []
        for handle in handles:
            if not isinstance(handle, ActivationHandle):
                raise TypeError('prefetch_many expects activation handles')
            value = handle._checked_value()
            if isinstance(value, _SavedTensor):
                if value.record.store is not self:
                    raise ValueError('Prefetch handles must belong to this activation store')
                records.append(value.record)
        return self._prefetch_records(records)

    def _prefetch_records(self, records, *, from_schedule=False):
        unique = {id(record): record for record in records if record is not None}
        if not unique:
            return 0
        with self.transport.prefetch_batch():
            return sum(record.prefetch(from_schedule=from_schedule) for record in unique.values())

    @contextmanager
    def batch(self):
        if self.closed:
            raise RuntimeError('Activation store has been closed')
        if self._active_batch.get() is not None:
            raise RuntimeError('Nested batches on the same activation store are not supported')
        batch = ActivationBatch(self)
        token = self._active_batch.set(batch)
        try:
            yield batch
            batch.commit()
        finally:
            batch.abort()
            self._active_batch.reset(token)

    def save(self, tensor):
        if self.closed:
            raise RuntimeError('Activation store has been closed')
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('ActivationStore.save expects a tensor')
        if tensor.device.type == 'cuda' and tensor.layout != torch.strided:
            raise ValueError('Explicit CUDA activation handles require strided tensors')
        batch = self._active_batch.get()
        if batch is None:
            with self.batch() as batch:
                return batch.save(tensor)
        if batch.closed:
            raise RuntimeError('Activation batch has already been committed')
        value = self._capture(tensor, batch)
        anchor = None
        if torch.is_grad_enabled() and tensor.requires_grad:
            if isinstance(value, _ResidentTensorRef):
                # Resident values already own their payload. Preserve their
                # normal edge directly, including unsupported tensor layouts.
                value.tensor = tensor
            else:
                anchor = _GradientAnchor.apply(tensor)
        handle = ActivationHandle(value, anchor)
        self._handles.add(handle)
        return handle

    def residency_reason(self, tensor):
        """Explain static admission without allocating or transferring data.

        None means eligible, subject to runtime host-budget admission and any
        adapter-specific ownership/distance rules. It is not a copy guarantee.
        """
        if tensor.device.type != 'cuda':
            return 'host_value'
        if tensor.layout != torch.strided:
            return 'unsupported_layout'
        if tensor.device != self.device:
            raise ValueError(f'Activation store expects device {self.device}, got {tensor.device}')
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        tensor_size = tensor.numel() * tensor.element_size()
        if getattr(storage, '_nnscaler_activation_resident', False):
            return 'marked_resident'
        if tensor.is_conj() or tensor.is_neg() or torch.is_inference(tensor):
            return 'unsupported_view_or_version'
        if not size or tensor_size < self.config.min_tensor_bytes:
            return 'below_minimum_size'
        if self.config.max_storage_to_tensor_ratio and size > self.config.max_storage_to_tensor_ratio * tensor_size:
            return 'disproportionate_storage'
        return None

    def _capture(self, tensor, batch, *, admitted=False):
        reason = None if admitted else self.residency_reason(tensor)
        if reason is not None:
            return self._resident(tensor, reason)
        storage = tensor.untyped_storage()
        tensor.record_stream(torch.cuda.current_stream(tensor.device))
        with self._lock:
            key = (tensor.device, storage._cdata)
            record = self.records.get(key)
            if record is not None and record.live_saves and record.owns_storage(storage):
                with record.lock:
                    record.check_version('repeated save')
                self.pool.stats['deduplicated_saves'] += 1
            else:
                record = _StorageRecord(tensor, self)
                self._live_records.add(record)
                if batch.capture_save_events:
                    record.producer_ready_event = batch.capture_producer_event(record.producer_stream)
                self.records[key] = record
                batch.pending.append(record)
            batch.records.append(weakref.ref(record))
            if self.observation.enabled:
                self.observation.emit('save', storage_id=record.uid, nbytes=record.size)
            return _SavedTensor(tensor, record)

    def _resident(self, tensor, reason):
        if self.observation.enabled:
            self.observation.emit('resident', nbytes=tensor.numel() * tensor.element_size(), reason=reason)
        return _ResidentTensorRef(tensor, reason)

    def snapshot(self):
        self.transport.poll()
        records = list(self._live_records)
        storages = {}
        for record in records:
            candidates = [copy.storage() for copy in record.copies]
            for storage in candidates:
                if storage is not None:
                    storages[(record.device, storage._cdata)] = storage.nbytes()
        return dict(closed=self.closed, live_handles=sum(not h.released for h in self._handles),
                    managed_storages=sum(record.live_saves > 0 for record in records),
                    managed_gpu_storage_bytes=sum(storages.values()),
                    queued_d2h_bytes=self.transport.inflight_bytes, pool=self.pool.snapshot(),
                    residency_note='Observed backing storages of managed CUDA records, deduplicated; '
                                   'excludes untracked resident saves, pending frees and allocator cache.')

    def close(self):
        if self.closed:
            return
        if self._active_batch.get() is not None or any(not h.released for h in self._handles):
            raise RuntimeError('Cannot close activation store with an active batch or handles')
        if any(record.live_saves for record in self._live_records):
            raise RuntimeError('Cannot close activation store with saved autograd values')
        for stream in (self.d2h_stream, self.h2d_stream):
            if stream is not None:
                stream.synchronize()
        self.transport.inflight.clear()
        self.transport.inflight_bytes = 0
        self.pool.snapshot()  # Reap completions without closing another store's pool.
        self.closed = True
