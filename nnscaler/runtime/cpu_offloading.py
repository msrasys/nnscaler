#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Saved activation storage with explicit forward scopes and bounded transfers.

The graph selects regions. A scheduler may prefetch a scope by its execution
key; ordinary autograd uses group output hooks. Neither storage identity nor
backward order is inferred from a process-global stack of forward calls.

Saved activations and their storage aliases must remain immutable until their
backward consumers release them. All custom writes must maintain PyTorch's
shared version counter. Raw-pointer/.data writes that bypass that counter and
independently versioned storage aliases are outside this contract. Checks detect
violations at save/transfer/unpack boundaries; they do not intercept writes or
make an invalid program's asynchronous accesses race-free.
"""
from collections import Counter, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import weakref

import torch
from torch.autograd.graph import saved_tensors_hooks


@dataclass(frozen=True)
class CPUOffloadConfig:
    # Rounded pinned block capacity, shared across this process's modules.
    max_cpu_bytes: int = 8 * 1024**3
    min_tensor_bytes: int = 1024**2
    max_inflight_bytes: int = 512 * 1024**2
    max_prefetch_bytes: int = 512 * 1024**2
    prefetch_groups: int = 1
    scope_prefetch_groups: int = 1
    reload_on_compute_stream: bool = True
    skip_retained_inputs: bool = True
    pool_slab_bytes: int = 0
    min_offload_gap: int = 0
    # Minimum backing-storage bytes * scheduled local F/B action distance.
    # Zero disables this size-aware lifetime filter; unknown distances retain
    # the existing behavior (ordinary non-pipeline autograd has no schedule).
    min_offload_size_gap_bytes: int = 0
    # Leave headroom for the dataloader, optimizer, compilation and other ranks.
    host_reserve_bytes: int = 32 * 1024**3

    def __post_init__(self):
        for name, value in vars(self).items():
            if name in ('reload_on_compute_stream', 'skip_retained_inputs'):
                if not isinstance(value, bool):
                    raise ValueError(f'{name} must be a bool')
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{name} must be a non-negative integer, got {value!r}')


def _cgroup_headroom(path):
    """Estimate allocatable bytes without treating clean inactive cache as live.

    Only inactive file pages are considered reclaimable, conservatively minus
    all dirty/writeback file pages. Active cache, anonymous and pinned memory
    remain charged. No cache eviction is requested by this function.
    """
    limit = (path / 'memory.max').read_text().strip()
    if limit == 'max':
        return None
    used = int((path / 'memory.current').read_text())
    reclaimable = 0
    try:
        stats = {key: int(value) for key, value in
                 (line.split() for line in (path / 'memory.stat').read_text().splitlines())}
        reclaimable = min(used, max(0, stats.get('inactive_file', 0)
                                   - stats.get('file_dirty', 0) - stats.get('file_writeback', 0)))
    except (OSError, ValueError):
        pass  # Older/unavailable stats retain the conservative raw-usage check.
    return max(0, int(limit) - used + reclaimable)


def available_host_bytes():
    """Host availability constrained by every visible cgroup v2 ancestor."""
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemAvailable', 'MemTotal'):
            values[key] = int(value.split()[0]) * 1024
    available = values['MemAvailable']
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if not line.startswith('0::'):
            continue
        root = Path('/sys/fs/cgroup')
        path = root / line.split('::', 1)[1].lstrip('/')
        # Namespaced mounts expose the current cgroup at the mount root.
        if not path.exists():
            path = root
        while True:
            try:
                headroom = _cgroup_headroom(path)
                if headroom is not None:
                    available = min(available, headroom)
            except (OSError, ValueError):
                pass
            if path == root:
                break
            path = path.parent
    return available


def mark_activation_resident(tensor: torch.Tensor) -> torch.Tensor:
    """Exclude a tensor and all its storage views (e.g. cached MX weights)."""
    tensor.untyped_storage()._nnscaler_activation_resident = True
    return tensor


class _OffloadPool:
    """Reusable tensors sized to the host allocator's charged block capacity."""
    def __init__(self, config):
        self.config = config
        self.lock = threading.RLock()
        self.free = []
        self.pending = []
        self.allocated = 0
        self.allocated_pinned = 0
        self.live = 0
        self.live_pinned = 0
        self.prefetched = 0
        self.stats = Counter()

    @staticmethod
    def allocation_charge(size):
        # ATen CachingHostAllocator rounds to PowerOf2Ceil before CUDA host
        # allocation (also reported by torch.cuda.memory.host_memory_stats).
        # Make the whole charged capacity available for subsequent leases.
        return 1 << (size - 1).bit_length() if size else 0

    def _reap(self):
        pending = []
        for buffer, event in self.pending:
            if event is None or event.query():
                self.free.append(buffer)
            else:
                pending.append((buffer, event))
        self.pending = pending

    def acquire(self, size):
        with self.lock:
            self._reap()
            candidates = [(b.numel(), i) for i, b in enumerate(self.free) if b.numel() >= size]
            if candidates:
                _, index = min(candidates)
                buffer = self.free.pop(index)
                self.stats['pool_reuses'] += 1
            else:
                charge = self.allocation_charge(size)
                if self.allocated_pinned + charge > self.config.max_cpu_bytes:
                    self.stats['budget_fallback_bytes'] += size
                    return None
                # This check is intentionally only on allocations, not every save.
                if available_host_bytes() < charge + self.config.host_reserve_bytes:
                    self.stats['headroom_fallback_bytes'] += size
                    return None
                # ATen already reserves this rounded capacity. Exposing it in
                # the tensor lets a later, slightly larger request reuse the
                # block instead of paying another cudaHostAlloc in the same
                # size class. Transfers still use only record.size bytes.
                buffer = torch.empty(charge, dtype=torch.uint8, device='cpu', pin_memory=True)
                self.allocated += charge
                self.allocated_pinned += charge
            self.live += buffer.numel()
            self.live_pinned += self.allocation_charge(buffer.numel())
            self.stats['peak_live_cpu_bytes'] = max(self.stats['peak_live_cpu_bytes'], self.live)
            self.stats['peak_live_pinned_bytes'] = max(self.stats['peak_live_pinned_bytes'], self.live_pinned)
            return buffer

    def release(self, buffer, event):
        with self.lock:
            self.live -= buffer.numel()
            self.live_pinned -= self.allocation_charge(buffer.numel())
            self.pending.append((buffer, event))

    def snapshot(self):
        with self.lock:
            self._reap()
            counters = dict.fromkeys(('d2h_bytes', 'h2d_bytes', 'resident_reused_bytes',
                                      'offloaded_storages', 'prefetch_loads', 'demand_loads',
                                      'schedule_prefetch_calls', 'schedule_prefetch_loads', 'budget_fallback_bytes',
                                      'headroom_fallback_bytes'), 0)
            counters.update(self.stats)
            return dict(counters, allocated_cpu_bytes=self.allocated,
                        allocated_pinned_bytes=self.allocated_pinned,
                        live_pinned_bytes=self.live_pinned,
                        live_cpu_bytes=self.live, prefetched_gpu_bytes=self.prefetched,
                        pending_cpu_bytes=sum(b.numel() for b, _ in self.pending))

    def close(self):
        """Release idle buffers; never invalidate an active saved tensor."""
        with self.lock:
            if self.live:
                raise RuntimeError('Cannot close activation offload pool with live saved tensors')
            for _, event in self.pending:
                if event is not None:
                    event.synchronize()
            self.pending.clear()
            self.free.clear()
            self.allocated = 0
            self.allocated_pinned = 0


# Keep the process pool alive between module instances. Otherwise destroying a
# module could drop our accounting while PyTorch still caches its pinned pages,
# and the next module would receive a fresh, uncharged allocation allowance.
_pools = {}
_pools_lock = threading.Lock()


def _shared_pool(config):
    with _pools_lock:
        pool = _pools.get(os.getpid())
        if pool is None:
            if config.pool_slab_bytes:
                from nnscaler.runtime.activation_pool import SlabOffloadPool
                pool = SlabOffloadPool(config)
            else:
                pool = _OffloadPool(config)
            _pools[os.getpid()] = pool
        elif pool.config.max_cpu_bytes != config.max_cpu_bytes:
            raise ValueError('All CPU-offloading modules in a process must use the same max_cpu_bytes')
        elif pool.config.pool_slab_bytes != config.pool_slab_bytes:
            raise ValueError('All CPU-offloading modules in a process must use the same pool_slab_bytes')
        return pool


class _ModuleTensorRef:
    """Reconstruct parameter views after ZeRO-3 has gathered them again."""
    def __init__(self, owner, tensor):
        self.owner = owner
        self.shape, self.stride = tensor.shape, tensor.stride()
        self.offset = tensor.storage_offset()
        self.dtype = tensor.dtype
        self.version = owner._version

    def unpack(self):
        if self.owner._version != self.version:
            raise RuntimeError('A parameter saved for backward was modified by an inplace operation')
        if self.owner.dtype != self.dtype:
            return torch.empty(0, dtype=self.dtype, device=self.owner.device).set_(
                self.owner.untyped_storage(), self.offset, self.shape, self.stride)
        return torch.as_strided(self.owner, self.shape, self.stride, self.offset)


class _ResidentTensorRef:
    def __init__(self, tensor):
        self.tensor = tensor.detach()
        self.version = None if torch.is_inference(tensor) else tensor._version

    def unpack(self):
        # saved_tensors_hooks bypasses PyTorch's normal SavedVariable check,
        # including on the resident/budget-fallback path. Restore that check.
        if self.version is not None and self.tensor._version != self.version:
            _raise_activation_mutation('resident unpack', self.version, self.tensor._version)
        return self.tensor


def _raise_activation_mutation(phase, expected, actual):
    raise RuntimeError(
        f'CPU activation offload: an activation saved for backward was modified by an inplace operation '
        f'({phase}: expected version {expected}, found {actual}). '
        'Saved tensors and their storage aliases must remain immutable until backward releases them. '
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


class _StorageRecord:
    def __init__(self, tensor, manager, host=None):
        self.live_saves = 0
        self.manager, self.pool = manager, manager.pool
        self.device = tensor.device
        self.producer_stream = torch.cuda.current_stream(tensor.device)
        self.size = tensor.untyped_storage().nbytes()
        self.host = host
        self.source = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
            tensor.untyped_storage(), 0, (self.size,), (1,))
        self.source_version = self.source._version
        self.version_tracker = _version_tracker(tensor)
        self.version = tensor._version
        self.storage_ref = weakref.ref(tensor.untyped_storage())
        self.device_tensor = None
        self.restored_version = None
        self.d2h_event = self.h2d_event = None
        self.speculative = False
        self.lock = threading.RLock()

    def check_version(self, phase='unpack'):
        if self.version_tracker._version != self.version:
            _raise_activation_mutation(phase, self.version, self.version_tracker._version)
        if self.source is not None and self.source._version != self.source_version:
            _raise_activation_mutation('resident storage ' + phase, self.source_version, self.source._version)
        if self.restored_version is not None and self.device_tensor._version != self.restored_version:
            _raise_activation_mutation('restored ' + phase, self.restored_version, self.device_tensor._version)

    def prepare_offload(self):
        with self.lock:
            if self.d2h_event is not None:
                return False
            self.check_version('D2H preparation')
            if self.host is None:
                self.host = self.pool.acquire(self.size)
            return self.host is not None

    def offload(self):
        with self.lock:
            if self.d2h_event is not None:
                return False
            if self.host is None and not self.prepare_offload():
                return False
            self.check_version('D2H submission')
            manager = self.manager
            stream = manager.d2h_stream
            with torch.cuda.stream(stream):
                self.host[:self.size].copy_(self.source, non_blocking=True)
                self.source.record_stream(stream)
                self.d2h_event = torch.cuda.Event()
                self.d2h_event.record(stream)
            manager.inflight.append((self.size, self.d2h_event))
            manager.inflight_bytes += self.size
            self.pool.stats['d2h_bytes'] += self.size
            self.pool.stats['offloaded_storages'] += 1
            self.source = None
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
            storage = self.storage_ref()
            if storage is not None and storage.nbytes() == self.size:
                self.device_tensor = torch.empty(0, dtype=torch.uint8, device=self.device).set_(
                    storage, 0, (self.size,), (1,))
                self.restored_version = self.device_tensor._version
                self.h2d_event = self.d2h_event
                self.pool.stats['resident_reused_bytes'] += self.size
                return True
            with self.pool.lock:
                if not demanded:
                    if self.pool.prefetched + self.size > self.manager.config.max_prefetch_bytes:
                        self.pool.stats['prefetch_budget_skips'] += 1
                        return False
                    self.pool.prefetched += self.size
                    self.speculative = True
                stream = self.manager.h2d_stream
                if self.manager.config.reload_on_compute_stream:
                    # Reuse the compute stream's allocator arena. A block may
                    # still be in use by work already enqueued there, so the
                    # copy stream must join that prefix before overwriting it.
                    consumer = torch.cuda.current_stream(self.device)
                    self.device_tensor = torch.empty(self.size, dtype=torch.uint8, device=self.device)
                    stream.wait_stream(consumer)
                with torch.cuda.stream(stream):
                    stream.wait_event(self.d2h_event)
                    if self.device_tensor is None:
                        self.device_tensor = self.host[:self.size].to(self.device, non_blocking=True)
                    else:
                        self.device_tensor.copy_(self.host[:self.size], non_blocking=True)
                        self.device_tensor.record_stream(stream)
                    self.h2d_event = torch.cuda.Event()
                    self.h2d_event.record(stream)
                self.restored_version = self.device_tensor._version
                self.pool.stats['h2d_bytes'] += self.size
                self.pool.stats['demand_loads' if demanded else 'prefetch_loads'] += 1
                if from_schedule:
                    self.pool.stats['schedule_prefetch_loads'] += 1
            return True

    def unpack(self, value):
        if self.d2h_event is None:
            self.check_version()
            raw = self.source
        else:
            self.prefetch(demanded=True)
            raw = self.device_tensor
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(self.h2d_event)
            raw.record_stream(stream)
        # set_(storage) creates an independent version counter for every view.
        # Reinterpret and as_strided instead so writes through any restored
        # view are visible to the shared record on the next access. A storage
        # may have unused trailing bytes smaller than one typed element.
        usable = raw.numel() // value.element_size * value.element_size
        typed = (raw if usable == raw.numel() else raw[:usable]).view(value.dtype)
        return torch.as_strided(typed, value.shape, value.stride, value.offset)

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


class _OffloadGroup:
    def __init__(self, scope):
        self.scope = scope
        self.records = []
        self.pending = []
        self.previous = scope.groups[-1] if scope.groups else None
        scope.groups.append(self)

    def commit(self):
        # Admit before a region starts, then enqueue its whole batch without
        # host waits between copies. Wait once for each producing stream.
        manager = self.scope.manager
        pending = []
        for record in self.pending:
            if record.live_saves:
                pending.append(record)
            else:
                # A checkpoint with no saved intermediates can discard its
                # input saver before leaving the region. No backward will ask
                # for this record, so its staging reference must not cause DMA.
                manager.pool.stats['discarded_save_bytes'] += record.size
                manager.pool.stats['discarded_save_storages'] += 1
        # Finish host allocation before starting the batch of DMA submissions.
        # Cold pinned allocations must not create gaps between its copies.
        pending = [record for record in pending if record.prepare_offload()]
        producers = {record.producer_stream.cuda_stream: record.producer_stream
                     for record in pending}
        for stream in producers.values():
            manager.d2h_stream.wait_stream(stream)
        copied = sum(bool(record.offload()) for record in pending)
        if copied:
            manager.pool.stats['d2h_committed_groups'] += 1
            manager.pool.stats['peak_queued_d2h_bytes'] = max(
                manager.pool.stats['peak_queued_d2h_bytes'], manager.inflight_bytes)
        self.pending.clear()

    def has_offloaded_records(self):
        return any((record := ref()) is not None and record.d2h_event is not None
                   for ref in self.records)

    def has_pending_loads(self):
        return any((record := ref()) is not None and record.d2h_event is not None
                   and record.device_tensor is None for ref in self.records)

    def prefetch(self, *, from_schedule=False):
        for ref in reversed(self.records):
            if (record := ref()) is not None:
                record.prefetch(from_schedule=from_schedule)

    def backward_entry(self, *unused):
        if not self.scope.manager.config.prefetch_groups:
            return
        self.prefetch()
        group = self.previous
        remaining = self.scope.manager.config.prefetch_groups
        while group is not None and remaining:
            if group.has_pending_loads():
                group.prefetch()
                remaining -= 1
            group = group.previous

    def watch(self, outputs):
        # Node prehooks execute before the group's backward compute, providing
        # overlap for the previous group without modifying outputs or aliasing.
        seen = set()
        def visit(value):
            if torch.is_tensor(value) and value.grad_fn is not None:
                node = value.grad_fn
                if node not in seen:
                    seen.add(node)
                    node.register_prehook(self.backward_entry)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
        visit(outputs)


_scope = ContextVar('nnscaler_activation_offload_scope', default=None)


def mark_retained_scope_tensors(values):
    """Exclude storage already retained by the executor until backward.

    Identities belong to this forward scope, not the tensor or process. The
    executor's saved inputs provide the strong references; this set adds none.
    """
    scope = _scope.get()
    if scope is None or not scope.manager.config.skip_retained_inputs:
        return
    def visit(value):
        if torch.is_tensor(value):
            if value.device.type == 'cuda' and value.layout == torch.strided:
                scope.retained_storages.add((value.device, value.untyped_storage()._cdata))
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
    visit(values)


class _ForwardScope:
    def __init__(self, manager, immediate_backward=False, backward_distance=None):
        self.manager = manager
        self.backward_distance = backward_distance
        self.immediate_backward = immediate_backward
        self.keep_resident = (backward_distance is not None
                              and backward_distance < manager.config.min_offload_gap)
        self.groups = []
        self.retained_storages = set()
        self.retained_saved = set()

    def prefetch(self):
        # Shape-only ops, filtered tensors and checkpoint tails can create
        # empty regions. They must not hide the actual backward entry group.
        remaining = self.manager.config.scope_prefetch_groups
        for group in reversed(self.groups):
            if not remaining:
                break
            if group.has_pending_loads():
                group.prefetch(from_schedule=True)
                remaining -= 1


class ActivationOffloadManager:
    def __init__(self, module, config):
        self.module = weakref.ref(module)
        self.config = config
        self.pool = _shared_pool(config)
        self.records = weakref.WeakValueDictionary()
        self.scopes = {}
        self.device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else None
        self.d2h_stream = torch.cuda.Stream(device=self.device) if self.device else None
        self.h2d_stream = torch.cuda.Stream(device=self.device) if self.device else None
        self.inflight = deque()
        self.inflight_bytes = 0

    def throttle(self, size):
        # Called at region admission. A single group may exceed the threshold;
        # the next group cannot start until older queued bytes fall below it.
        while self.inflight and self.inflight[0][1].query():
            nbytes, _ = self.inflight.popleft()
            self.inflight_bytes -= nbytes
        while self.inflight and self.inflight_bytes + size > self.config.max_inflight_bytes:
            nbytes, event = self.inflight.popleft()
            event.synchronize()
            self.inflight_bytes -= nbytes
            self.pool.stats['d2h_throttles'] += 1

    @contextmanager
    def scope(self, key=None, immediate_backward=False, backward_distance=None):
        current = _scope.get()
        if key is None and current is not None and current.manager is self:
            yield current
            return
        scope = _ForwardScope(self, immediate_backward, backward_distance)
        if key is not None:
            if key in self.scopes:
                raise RuntimeError(f'Duplicate activation forward scope: {key}')
            self.scopes[key] = scope
        token = _scope.set(scope)
        try:
            yield scope
        except BaseException:
            self.scopes.pop(key, None)
            raise
        finally:
            _scope.reset(token)

    def prefetch(self, key):
        scope = self.scopes.get(key)
        if scope is not None and self.config.prefetch_groups:
            self.pool.stats['schedule_prefetch_calls'] += 1
            scope.prefetch()

    def finish(self, key):
        self.scopes.pop(key, None)

    def pack(self, tensor, group):
        if tensor.device.type != 'cuda' or tensor.layout != torch.strided:
            return _ResidentTensorRef(tensor)
        if tensor.device != self.device:
            raise ValueError(f'CPU offloading only supports one CUDA device per module: {self.device}')
        module = self.module()
        owner = module.find_buffer_or_param_owner(tensor)
        if owner is not None:
            return _ModuleTensorRef(owner, tensor)
        storage = tensor.untyped_storage()
        identity = (tensor.device, storage._cdata)
        if identity in group.scope.retained_storages:
            key = (*identity, tensor._version)
            if key not in group.scope.retained_saved:
                group.scope.retained_saved.add(key)
                self.pool.stats['retained_input_bytes'] += storage.nbytes()
                self.pool.stats['retained_input_storages'] += 1
            return _ResidentTensorRef(tensor)
        if (getattr(storage, '_nnscaler_activation_resident', False)
                or tensor.is_conj() or tensor.is_neg() or torch.is_inference(tensor)):
            return _ResidentTensorRef(tensor)
        if tensor.numel() * tensor.element_size() < self.config.min_tensor_bytes:
            return _ResidentTensorRef(tensor)
        size = storage.nbytes()
        # Avoid copying a huge backing allocation for a small slice.
        if size == 0 or size > 8 * tensor.numel() * tensor.element_size():
            return _ResidentTensorRef(tensor)
        if (self.config.min_offload_size_gap_bytes
                and (distance := group.scope.backward_distance) is not None
                and size * distance < self.config.min_offload_size_gap_bytes):
            # Use storage size so ordinary views within one scope make the
            # same decision. Keep SavedVariable's version check on this path.
            seen = getattr(group.scope, '_size_gap_saved', None)
            if seen is None:
                seen = group.scope._size_gap_saved = set()
            saved_key = (*identity, tensor._version)
            if saved_key not in seen:
                seen.add(saved_key)
                self.pool.stats['size_gap_resident_bytes'] += size
                self.pool.stats['size_gap_resident_storages'] += 1
            return _ResidentTensorRef(tensor)
        key = identity
        tensor.record_stream(torch.cuda.current_stream(tensor.device))
        record = self.records.get(key)
        if record is not None and record.live_saves and record.storage_ref() is storage:
            # There is exactly one immutable version for a live saved storage.
            # Never fork a second record to tolerate mutation of a live save.
            record.check_version('repeated save')
            # Compare each counter to its own captured value, not another
            # tensor's numeric version. A restored view that reuses resident
            # storage has the reload buffer's counter, also checked above.
            self.pool.stats['deduplicated_saves'] += 1
        else:
            # Reserve pinned memory only when the completed group still has
            # a backward consumer. Budget failures retain record.source on GPU.
            record = _StorageRecord(tensor, self)
            self.records[key] = record
            group.pending.append(record)
        group.records.append(weakref.ref(record))
        return _SavedTensor(tensor, record)


class CPUOffloadContext:
    """Collect saved tensors and submit D2H together at the region boundary.

    With non-reentrant checkpointing this context wraps the checkpoint *call*,
    so only checkpoint inputs are saved here. Replayed intermediates remain on
    the GPU. Call ``watch(outputs)`` to install backward group lookahead hooks.
    """
    def __init__(self, module, *, tail=False):
        self.manager = module._get_activation_offload_manager()
        self._scope_context = None
        self.tail = tail

    def __enter__(self):
        self._scope_context = self.manager.scope()
        scope = self._scope_context.__enter__()
        try:
            if (torch.is_grad_enabled() and not scope.keep_resident
                    and not (self.tail and scope.immediate_backward)):
                self.manager.throttle(0)
            self.group = _OffloadGroup(scope)
            self._hooks = saved_tensors_hooks(self._pack, self._unpack)
            self._hooks.__enter__()
        except BaseException as error:
            self._scope_context.__exit__(type(error), error, error.__traceback__)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._hooks.__exit__(exc_type, exc_value, traceback)
            if exc_type is None:
                self.group.commit()
            else:
                self.group.pending.clear()
        finally:
            self._scope_context.__exit__(exc_type, exc_value, traceback)

    def _pack(self, tensor):
        if self.group.scope.keep_resident or (self.tail and self.group.scope.immediate_backward):
            owner = self.manager.module().find_buffer_or_param_owner(tensor) if tensor.layout == torch.strided else None
            if owner is not None:
                return _ModuleTensorRef(owner, tensor)
            label = ('short_scope_resident_bytes' if self.group.scope.keep_resident
                     else 'immediate_backward_resident_bytes')
            self.manager.pool.stats[label] += tensor.numel() * tensor.element_size()
            return _ResidentTensorRef(tensor)
        return self.manager.pack(tensor, self.group)

    def _unpack(self, value):
        if isinstance(value, _SavedTensor):
            # Correctness fallback for disconnected outputs or branched graphs.
            # The explicit scheduler and group hooks normally prefetch earlier.
            tensor = value.unpack()
            self.group.backward_entry()
            return tensor
        if isinstance(value, (_ModuleTensorRef, _ResidentTensorRef)):
            return value.unpack()
        return value

    def watch(self, outputs):
        if not self.group.scope.keep_resident:
            self.group.watch(outputs)
