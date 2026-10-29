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
from contextlib import contextmanager
from contextvars import ContextVar
import weakref

import torch
from .activation_config import CPUOffloadConfig
from .activation_autograd import SavedActivationContext, _ModuleTensorRef
from .activation_memory import _OffloadPool, _shared_pool, _pools, available_host_bytes, _cgroup_headroom
from .activation_store import (ActivationStore, ActivationBatch, _ResidentTensorRef, _StorageRecord, _SavedTensor)


def mark_activation_resident(tensor: torch.Tensor) -> torch.Tensor:
    """Exclude a tensor and all its storage views (e.g. cached MX weights)."""
    tensor.untyped_storage()._nnscaler_activation_resident = True
    return tensor


def activation_offload_stores(module: torch.nn.Module, *, initialize=False) -> tuple[ActivationStore, ...]:
    """Discover module stores; optional initialization is explicit for profiling.

    The default is read-only and never creates CUDA streams or a memory pool.
    A profiler may request initialization before its diagnostic step begins.
    """
    stores = {}
    for child in module.modules():
        manager = getattr(child, '_activation_offload_manager', None)
        config = getattr(getattr(child, 'compute_config', None), 'cpu_offload', None)
        factory = getattr(child, '_get_activation_offload_manager', None)
        if manager is None and initialize and config is not None and callable(factory):
            manager = factory()
        if manager is not None:
            stores[id(manager)] = manager
    return tuple(stores.values())


def activation_offload_pool_stats(module: torch.nn.Module) -> list[dict]:
    """Snapshot each initialized offload pool used by a module tree once.

    Pools are shared across compiled modules. Summing per-module snapshots
    would count that same memory repeatedly. This query never creates a manager
    or a pool; returned dictionaries neither own nor expose runtime resources.
    A shared pool's counters include all its users, even outside this tree.
    """
    pools = {}
    for manager in activation_offload_stores(module):
        pools[id(manager.pool)] = manager.pool
    return [pool.snapshot() for pool in pools.values()]


def _iter_tensors(values):
    """Walk nested values without a recursive closure retaining autograd nodes."""
    pending = [values]
    containers = set()
    while pending:
        value = pending.pop()
        if torch.is_tensor(value):
            yield value
        elif isinstance(value, (tuple, list, dict)):
            if id(value) in containers:
                continue
            containers.add(id(value))
            items = tuple(value.values()) if isinstance(value, dict) else value
            pending.extend(reversed(items))


class _OffloadGroup(ActivationBatch):
    def __init__(self, scope):
        # The scope owns its groups for schedule lookahead. A strong reverse
        # link would leak every group's metadata when cyclic GC is disabled.
        # Backward only needs the manager and previous-group chain; the scope
        # itself is needed only while packing tensors in the active forward.
        super().__init__(scope.manager)
        self._scope = weakref.ref(scope)
        self.manager = scope.manager
        self.keep_resident = scope.keep_resident
        self.previous = scope.groups[-1] if scope.groups else None
        scope.groups.append(self)

    @property
    def scope(self):
        return self._scope()

    def backward_entry(self, *unused):
        if not self.manager.config.prefetch_groups:
            return
        self.prefetch()
        group = self.previous
        remaining = self.manager.config.prefetch_groups
        while group is not None and remaining:
            if group.has_pending_loads():
                group.prefetch()
                remaining -= 1
            group = group.previous

    def watch(self, outputs):
        # Node prehooks execute before the group's backward compute, providing
        # overlap for the previous group without modifying outputs or aliasing.
        seen = set()
        for value in _iter_tensors(outputs):
            if value.grad_fn is not None:
                node = value.grad_fn
                if node not in seen:
                    seen.add(node)
                    node.register_prehook(self.backward_entry)


_scope = ContextVar('nnscaler_activation_offload_scope', default=None)


def mark_retained_scope_tensors(values):
    """Exclude storage already retained by the executor until backward.

    Identities belong to this forward scope, not the tensor or process. The
    executor's saved inputs provide the strong references; this set adds none.
    """
    scope = _scope.get()
    if scope is None or not scope.manager.config.skip_retained_inputs:
        return
    for value in _iter_tensors(values):
        if value.device.type == 'cuda' and value.layout == torch.strided:
            scope.retained_storages.add((value.device, value.untyped_storage()._cdata))


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


class ActivationOffloadManager(ActivationStore):
    def __init__(self, module, config):
        super().__init__(config)
        self.module = weakref.ref(module)
        self.scopes = {}

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
            return self._resident(tensor, 'host_or_unsupported_layout')
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
            return self._resident(tensor, 'executor_retained_input')
        reason = self.residency_reason(tensor)
        if reason is not None:
            return self._resident(tensor, reason)
        size = storage.nbytes()
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
            return self._resident(tensor, 'below_size_gap')
        return self._capture(tensor, group, admitted=True)


class CPUOffloadContext(SavedActivationContext):
    """Collect saved tensors and submit D2H together at the region boundary.

    With non-reentrant checkpointing this context wraps the checkpoint *call*,
    so only checkpoint inputs are saved here. Replayed intermediates remain on
    the GPU. Call ``watch(outputs)`` to install backward group lookahead hooks.
    """
    def __init__(self, module, *, tail=False):
        self.manager = module._get_activation_offload_manager()
        super().__init__(self.manager)
        self.tail = tail

    def _forward_context(self):
        return self.manager.scope()

    def _make_group(self, scope):
        return _OffloadGroup(scope)

    def _finish_forward(self, exc_type):
        if exc_type is None:
            self.group.commit()
        else:
            self.group.pending.clear()

    def _pack(self, tensor):
        if self.group.scope.keep_resident or (self.tail and self.group.scope.immediate_backward):
            owner = self.manager.module().find_buffer_or_param_owner(tensor) if tensor.layout == torch.strided else None
            if owner is not None:
                return _ModuleTensorRef(owner, tensor)
            label = ('short_scope_resident_bytes' if self.group.scope.keep_resident
                     else 'immediate_backward_resident_bytes')
            self.manager.pool.stats[label] += tensor.numel() * tensor.element_size()
            return self.manager._resident(tensor, label)
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
        if not self.group.keep_resident:
            self.group.watch(outputs)
