#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""CPU offloading of tensors saved for backward.

Planning marks consecutive forward operators with an ``offload_id`` (see
``OpPlan`` in ``nnscaler.policies``).  Code generation wraps every such group in
``with self.cpu_offloading_hooks(group_id):`` which installs
``saved_tensors_hooks``.  This module implements the runtime side:

* pack: every strided CUDA tensor saved by autograd inside the group is copied
  to pinned host memory on a dedicated D2H stream and the device storage is
  released as soon as the copy has finished.  Parameters / buffers are not
  offloaded (a cheap reference is kept instead).  Tensors sharing the same
  storage region are copied once.
* unpack: the region is copied back on a dedicated H2D stream; the consumer
  stream waits on the copy event.

Prefetch ordering is derived from execution contexts instead of guessing:

* ``Executor.fexecute(name, ...)`` opens an :class:`_ExecContext` for the
  segment ``name``; the offload groups executed inside it are recorded in
  forward order.  Finished contexts are queued per segment name (FIFO).
* ``Executor.backward(name, ...)`` pops the oldest pending context of ``name``.
  Within one context, groups are consumed in reverse forward order, so when a
  group is demanded the preceding groups are prefetched (``lookahead`` groups).
  This is exact for 1F1B, interleaved 1F1B and gradient accumulation because
  the backward of a given segment always consumes micro-batches in the order
  their forwards ran.
* The scheduler code generator additionally emits
  ``prefetch_next_backward(name)`` before every backward call with the segment
  name of the *following* backward, so its first group is already in flight
  while the current backward computes.

Everything is configurable through environment variables (read once):

``CPU_OFFLOADING_MIN_BYTES``       smallest tensor to offload (default 1 MiB)
``CPU_OFFLOADING_PREFETCH_GROUPS`` groups prefetched ahead of the demanded one
                                   inside a context (default 1)
``CPU_OFFLOADING_DISABLE``         set to 1 to turn offloading into a no-op
``CPU_OFFLOADING_PREFETCH_LEVEL``  legacy knob from the first prototype; it is
                                   still accepted but ignored.
"""

from __future__ import annotations

import collections
import logging
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch
from torch.autograd.graph import saved_tensors_hooks

from nnscaler.runtime.device import DeviceGroup
from nnscaler.runtime.utils import load_int_from_env

if TYPE_CHECKING:
    from nnscaler.runtime.module import ParallelModule


_logger = logging.getLogger(__name__)

_D2H_STREAM_NAME = 'cpu_offload_d2h'
_H2D_STREAM_NAME = 'cpu_offload_h2d'
# kept for backward compatibility with the first prototype / its tests
_PREFETCH_STREAM_NAME = _H2D_STREAM_NAME


def _get_prefetch_stream() -> torch.cuda.Stream:
    return DeviceGroup().get_stream(_H2D_STREAM_NAME)


def _get_d2h_stream() -> torch.cuda.Stream:
    return DeviceGroup().get_stream(_D2H_STREAM_NAME)


# ---------------------------------------------------------------------------
# pinned host memory pool
# ---------------------------------------------------------------------------

class _PinnedPool:
    """Free-list of pinned ``uint8`` buffers organised in geometric size classes.

    ``cudaHostAlloc`` costs ~0.3 ms/MiB, so buffers are recycled across
    training steps.  Sizes are rounded up to the next class (ratio 1.25, at
    most 25% slack) so that tensors whose shapes change every step (MoE expert
    inputs depend on routing) still find a buffer to reuse.  A buffer returned
    to the pool may still be the source of an in-flight H2D copy; the recorded
    event is waited for before reuse.  ``CPU_OFFLOADING_POOL_MAX_GB`` bounds the
    free list; surplus buffers are released to the OS instead of being cached.
    """

    _MIN_CLASS = 64 * 1024
    _RATIO = 1.25

    def __init__(self) -> None:
        self._free: Dict[int, List[Tuple[torch.Tensor, Optional[torch.cuda.Event]]]] = collections.defaultdict(list)
        self._lock = threading.Lock()
        self.allocated_bytes = 0   # bytes currently owned by the pool (free + in use)
        self.free_bytes = 0
        self.allocations = 0
        self.max_free_bytes = load_int_from_env('CPU_OFFLOADING_POOL_MAX_GB', 0) << 30

    @classmethod
    def size_class(cls, nbytes: int) -> int:
        size = cls._MIN_CLASS
        while size < nbytes:
            size = int(size * cls._RATIO) // 256 * 256
        return size

    def acquire(self, nbytes: int, stream: torch.cuda.Stream) -> torch.Tensor:
        size = self.size_class(nbytes)
        with self._lock:
            bucket = self._free.get(size)
            if bucket:
                buf, event = bucket.pop()
                self.free_bytes -= size
                if event is not None:
                    stream.wait_event(event)
                return buf
        buf = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        with self._lock:
            self.allocated_bytes += size
            self.allocations += 1
        return buf

    def release(self, buf: torch.Tensor, event: Optional[torch.cuda.Event]) -> None:
        size = buf.numel()
        with self._lock:
            if self.max_free_bytes and self.free_bytes + size > self.max_free_bytes:
                # drop it: the pinned memory is returned to the OS when the
                # tensor dies (after the event, which the caller already waited
                # for through ``record_stream`` on the device side)
                self.allocated_bytes -= size
                return
            self._free[size].append((buf, event))
            self.free_bytes += size

    def clear(self) -> None:
        with self._lock:
            self._free.clear()
            self.allocated_bytes = 0
            self.free_bytes = 0
            self.allocations = 0


# ---------------------------------------------------------------------------
# offloaded storage regions and views
# ---------------------------------------------------------------------------

def _covered_bytes(tensor: torch.Tensor) -> Tuple[int, int]:
    """Byte range ``[lo, hi)`` of ``tensor``'s storage that the view touches."""
    elem = tensor.element_size()
    extent = 1
    for size, stride in zip(tensor.shape, tensor.stride()):
        if size == 0:
            return 0, 0
        extent += (size - 1) * stride
    lo = tensor.storage_offset() * elem
    return lo, lo + extent * elem


class _HostRegion:
    """One contiguous byte range of a device storage living in pinned memory."""

    __slots__ = (
        'manager', 'batch', 'device', 'nbytes', 'host', 'd2h_event',
        'device_tensor', 'h2d_event', 'lock', 'pending_views', 'released',
    )

    def __init__(self, manager: 'OffloadManager', batch: '_OffloadBatch',
                 storage_view: torch.Tensor) -> None:
        self.manager = manager
        self.batch = batch
        self.device = storage_view.device
        self.nbytes = storage_view.numel()
        self.lock = threading.Lock()
        self.device_tensor: Optional[torch.Tensor] = None
        self.h2d_event: Optional[torch.cuda.Event] = None
        self.pending_views = 0
        self.released = False

        d2h = manager.d2h_stream
        current = torch.cuda.current_stream(self.device)
        self.host = manager.pool.acquire(self.nbytes, d2h)
        # the copy must observe every kernel that produced the tensor
        d2h.wait_stream(current)
        with torch.cuda.stream(d2h):
            self.host[:self.nbytes].copy_(storage_view, non_blocking=True)
            # the caching allocator must not hand the device memory to anybody
            # until the D2H copy has finished
            storage_view.record_stream(d2h)
            self.d2h_event = torch.cuda.Event()
            self.d2h_event.record(d2h)
        manager.stats.offloaded_bytes += self.nbytes
        manager.stats.offloaded_regions += 1

    def prefetch(self) -> None:
        with self.lock:
            if self.device_tensor is not None or self.released:
                return
            h2d = self.manager.h2d_stream
            with torch.cuda.stream(h2d):
                h2d.wait_event(self.d2h_event)
                self.device_tensor = torch.empty(self.nbytes, dtype=torch.uint8, device=self.device)
                self.device_tensor.copy_(self.host[:self.nbytes], non_blocking=True)
                self.h2d_event = torch.cuda.Event()
                self.h2d_event.record(h2d)
            self.manager.stats.prefetched_bytes += self.nbytes
            self.manager.stats.prefetched_regions += 1

    def fetch(self) -> torch.Tensor:
        """Return the device copy, blocking the current stream on the H2D copy."""
        self.prefetch()
        assert self.device_tensor is not None and self.h2d_event is not None
        if not self.h2d_event.query():
            self.manager.stats.stalls += 1
        current = torch.cuda.current_stream(self.device)
        current.wait_event(self.h2d_event)
        self.device_tensor.record_stream(current)
        return self.device_tensor

    def view_consumed(self) -> None:
        with self.lock:
            self.pending_views -= 1
            if self.pending_views == 0:
                # autograd keeps its own reference to the unpacked views; the
                # region does not need to pin the device copy anymore
                self.device_tensor = None

    def release(self) -> None:
        """Return the pinned buffer to the pool; called when the batch is done."""
        with self.lock:
            if self.released:
                return
            self.released = True
            self.device_tensor = None
            self.manager.pool.release(self.host, self.h2d_event)
            self.host = None  # type: ignore[assignment]


@dataclass(eq=False)
class _OffloadedTensor:
    """What autograd stores instead of the saved tensor."""

    region: _HostRegion
    dtype: torch.dtype
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    offset_bytes: int  # relative to the region start
    consumed: bool = False

    def unpack(self) -> torch.Tensor:
        base = self.region.fetch()
        elem = torch.empty(0, dtype=self.dtype).element_size()
        typed = base.view(self.dtype)
        tensor = torch.as_strided(typed, self.shape, self.stride, self.offset_bytes // elem)
        if not self.consumed:
            self.consumed = True
            self.region.view_consumed()
        return tensor


class _ModuleTensorRef:
    """Reference to a parameter/buffer view; these are never offloaded."""

    def __init__(self, owner: torch.Tensor, tensor: torch.Tensor) -> None:
        self.owner = owner
        self.shape = tensor.shape
        self.stride = tensor.stride()
        self.storage_offset = tensor.storage_offset()
        self.version = tensor._version

    def unpack(self) -> torch.Tensor:
        if self.owner._version != self.version:
            raise RuntimeError(
                'one of the variables needed for gradient computation has been modified by an inplace operation: '
                f'tensor is at version {self.owner._version}; expected version {self.version} instead.'
            )
        return torch.as_strided(self.owner, self.shape, self.stride, self.storage_offset)


# ---------------------------------------------------------------------------
# batches (one offload group execution) and execution contexts (one segment call)
# ---------------------------------------------------------------------------

class _OffloadBatch:
    """Saved tensors of one execution of one offload group."""

    last_batch: Optional[weakref.ReferenceType['_OffloadBatch']] = None

    def __init__(self, manager: 'OffloadManager', group_id: Optional[int],
                 context: Optional['_ExecContext']) -> None:
        self.manager = manager
        self.group_id = group_id
        self.context = context
        self.index = -1  # position inside the context
        self.regions: Dict[Tuple[int, int, int], _HostRegion] = {}
        self.handles: List[_OffloadedTensor] = []
        self.prefetched = False
        self.released = False
        # Strong references to the device storages packed in this batch.  They
        # are dropped in ``done()``: until then no address can be recycled by
        # the caching allocator, which keeps the (data_ptr, lo, hi) dedup key
        # unambiguous.  The D2H copies themselves are protected by
        # ``record_stream``.
        self._live_storages: List[torch.UntypedStorage] = []
        # chain used when no execution context is available (plain eager use)
        self.previous: Optional[weakref.ReferenceType['_OffloadBatch']] = (
            _OffloadBatch.last_batch if context is None else None)

    @property
    def nbytes(self) -> int:
        return sum(region.nbytes for region in self.regions.values())

    def append(self, tensor: torch.Tensor) -> _OffloadedTensor:
        lo, hi = _covered_bytes(tensor)
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), lo, hi)
        region = self.regions.get(key)
        if region is None:
            storage_view = torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(storage)[lo:hi]
            region = _HostRegion(self.manager, self, storage_view)
            self.regions[key] = region
            self._live_storages.append(storage)
        else:
            self.manager.stats.deduplicated_bytes += hi - lo
        self.manager.stats.packed_tensors += 1
        region.pending_views += 1
        handle = _OffloadedTensor(
            region=region,
            dtype=tensor.dtype,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            offset_bytes=tensor.storage_offset() * tensor.element_size() - lo,
        )
        self.handles.append(handle)
        return handle

    def done(self) -> None:
        self._live_storages = []
        if self.regions and self.context is None:
            _OffloadBatch.last_batch = weakref.ref(self)

    def prefetch(self) -> None:
        if self.prefetched or self.released:
            return
        self.prefetched = True
        # reverse pack order: the last saved tensors are needed first
        for region in reversed(list(self.regions.values())):
            region.prefetch()

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        for region in self.regions.values():
            region.release()
        self.regions = {}
        self.handles = []


class _ExecContext:
    """Offload batches produced by one forward execution of a segment."""

    def __init__(self, name: str, seq: int) -> None:
        self.name = name
        self.seq = seq
        self.batches: List[_OffloadBatch] = []
        self.backward_started = False
        self.offload_enabled = True

    def add(self, batch: _OffloadBatch) -> None:
        batch.index = len(self.batches)
        self.batches.append(batch)

    @property
    def nbytes(self) -> int:
        return sum(batch.nbytes for batch in self.batches)

    def prefetch_from(self, index: int, count: int) -> None:
        """Prefetch batches ``index, index-1, ...`` (``count`` of them)."""
        for i in range(index, max(-1, index - count), -1):
            self.batches[i].prefetch()

    def release(self) -> None:
        for batch in self.batches:
            batch.release()


@dataclass
class OffloadStats:
    offloaded_bytes: int = 0
    offloaded_regions: int = 0
    deduplicated_bytes: int = 0
    prefetched_bytes: int = 0
    prefetched_regions: int = 0
    stalls: int = 0  # unpacks that had to wait for an in-flight H2D copy
    skipped_small_bytes: int = 0
    packed_tensors: int = 0
    pack_seconds: float = 0.0
    unpack_seconds: float = 0.0
    pool_allocated_bytes: int = 0
    pool_allocations: int = 0

    def reset(self) -> None:
        for name in ('offloaded_bytes', 'offloaded_regions', 'deduplicated_bytes',
                     'prefetched_bytes', 'prefetched_regions', 'stalls', 'skipped_small_bytes',
                     'packed_tensors'):
            setattr(self, name, 0)
        self.pack_seconds = 0.0
        self.unpack_seconds = 0.0


# ---------------------------------------------------------------------------
# the manager
# ---------------------------------------------------------------------------

class OffloadManager:
    """Process-wide bookkeeping shared by all offload contexts."""

    _instance: Optional['OffloadManager'] = None

    def __init__(self) -> None:
        self.min_bytes = load_int_from_env('CPU_OFFLOADING_MIN_BYTES', 1 << 20)
        self.lookahead_groups = load_int_from_env('CPU_OFFLOADING_PREFETCH_GROUPS', 1)
        self.enabled = load_int_from_env('CPU_OFFLOADING_DISABLE', 0) == 0
        # a forward whose own backward is scheduled within this many segment
        # executions gains nothing from offloading (the tensors would come back
        # right away); the generated schedule reports the distance via
        # ``hint_forward``.  0 disables the heuristic.
        self.min_backward_distance = load_int_from_env('CPU_OFFLOADING_MIN_BACKWARD_DISTANCE', 1)
        self._next_forward_distance: Optional[int] = None
        self.pool = _PinnedPool()
        self.stats = OffloadStats()
        self.lock = threading.Lock()
        self._d2h_stream: Optional[torch.cuda.Stream] = None
        self._h2d_stream: Optional[torch.cuda.Stream] = None
        # forward side
        self._forward_stack: List[_ExecContext] = []
        self._forward_seq: Dict[str, int] = collections.defaultdict(int)
        # finished forward contexts waiting for their backward, per segment name
        self._pending: Dict[str, Deque[_ExecContext]] = collections.defaultdict(collections.deque)
        # backward side
        self._backward_stack: List[_ExecContext] = []

    @classmethod
    def get(cls) -> 'OffloadManager':
        if cls._instance is None:
            cls._instance = OffloadManager()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None
        _OffloadBatch.last_batch = None

    # -- streams --------------------------------------------------------
    @property
    def d2h_stream(self) -> torch.cuda.Stream:
        if self._d2h_stream is None:
            self._d2h_stream = _get_d2h_stream()
        return self._d2h_stream

    @property
    def h2d_stream(self) -> torch.cuda.Stream:
        if self._h2d_stream is None:
            self._h2d_stream = _get_prefetch_stream()
        return self._h2d_stream

    # -- execution contexts (driven by Executor) --------------------------
    def hint_forward(self, name: str, backward_distance: int) -> None:
        """Scheduler hint: the next ``fexecute`` of ``name`` has its backward
        ``backward_distance`` segment executions later."""
        self._next_forward_distance = backward_distance

    def begin_forward(self, name: str) -> _ExecContext:
        with self.lock:
            seq = self._forward_seq[name]
            self._forward_seq[name] += 1
            ctx = _ExecContext(name, seq)
            distance = self._next_forward_distance
            self._next_forward_distance = None
            if distance is not None and distance < self.min_backward_distance:
                ctx.offload_enabled = False
            self._forward_stack.append(ctx)
        return ctx

    def end_forward(self, ctx: _ExecContext) -> None:
        with self.lock:
            assert self._forward_stack and self._forward_stack[-1] is ctx, 'unbalanced begin/end_forward'
            self._forward_stack.pop()
            if ctx.batches:
                self._pending[ctx.name].append(ctx)

    def begin_backward(self, name: str) -> Optional[_ExecContext]:
        with self.lock:
            queue = self._pending.get(name)
            if not queue:
                return None
            ctx = queue.popleft()
            ctx.backward_started = True
            self._backward_stack.append(ctx)
        # the last forward group is the first one needed by backward
        ctx.prefetch_from(len(ctx.batches) - 1, 1 + self.lookahead_groups)
        return ctx

    def end_backward(self, ctx: Optional[_ExecContext]) -> None:
        if ctx is None:
            return
        with self.lock:
            if self._backward_stack and self._backward_stack[-1] is ctx:
                self._backward_stack.pop()
            else:
                self._backward_stack = [c for c in self._backward_stack if c is not ctx]
        ctx.release()

    def prefetch_next_backward(self, name: str) -> None:
        """Hint emitted by the scheduler: the next backward will be ``name``."""
        with self.lock:
            queue = self._pending.get(name)
            ctx = queue[0] if queue else None
        if ctx is not None and ctx.batches:
            ctx.prefetch_from(len(ctx.batches) - 1, 1 + self.lookahead_groups)

    # -- hooks used by the saved_tensors_hooks context -----------------------
    def new_batch(self, group_id: Optional[int]) -> Optional[_OffloadBatch]:
        with self.lock:
            ctx = self._forward_stack[-1] if self._forward_stack else None
        if ctx is not None and not ctx.offload_enabled:
            return None
        batch = _OffloadBatch(self, group_id, ctx)
        if ctx is not None:
            ctx.add(batch)
        return batch

    def on_unpack(self, handle: _OffloadedTensor) -> None:
        batch = handle.region.batch
        batch.prefetch()
        ctx = batch.context
        if ctx is not None:
            if batch.index > 0:
                ctx.prefetch_from(batch.index - 1, self.lookahead_groups)
        else:
            # eager / non-scheduled execution: follow the creation chain backwards
            previous = batch.previous() if batch.previous is not None else None
            count = self.lookahead_groups
            while previous is not None and count > 0:
                previous.prefetch()
                count -= 1
                previous = previous.previous() if previous.previous is not None else None

    # -- diagnostics ------------------------------------------------------
    def snapshot(self, reset: bool = True) -> Dict[str, Any]:
        self.stats.pool_allocated_bytes = self.pool.allocated_bytes
        self.stats.pool_allocations = self.pool.allocations
        data = dict(self.stats.__dict__)
        with self.lock:
            data['pending_contexts'] = sum(len(q) for q in self._pending.values())
            data['pending_bytes'] = sum(c.nbytes for q in self._pending.values() for c in q)
        if reset:
            self.stats.reset()
        return data


# ---------------------------------------------------------------------------
# the context installed by generated code
# ---------------------------------------------------------------------------

class CPUOffloadContext:
    """Offload tensors saved for backward inside the ``with`` block.

    Contexts must not be nested.  ``prefetch_level`` is accepted for backward
    compatibility with the first prototype and is ignored; prefetch behaviour is
    controlled by :class:`OffloadManager`.
    """

    def __init__(self, module: 'ParallelModule', prefetch_level: int = 0,
                 group_id: Optional[int] = None) -> None:
        if isinstance(prefetch_level, bool) or not isinstance(prefetch_level, int):
            raise ValueError(f'prefetch_level must be an integer, got {prefetch_level}')
        self.module = module
        self.group_id = group_id
        self.prefetch_level = prefetch_level
        self.manager = OffloadManager.get()
        self.batch: Optional[_OffloadBatch] = None
        self._hooks: Optional[saved_tensors_hooks] = None
        self.device: Optional[torch.device] = None

    def __enter__(self) -> 'CPUOffloadContext':
        if not self.manager.enabled or not torch.cuda.is_available():
            return self
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.batch = self.manager.new_batch(self.group_id)
        if self.batch is None:
            # the scheduler told us the backward follows immediately
            return self
        self._hooks = saved_tensors_hooks(self._pack, self._unpack)
        self._hooks.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Optional[bool]:
        if self._hooks is None:
            return None
        hooks = self._hooks
        self._hooks = None
        result = hooks.__exit__(exc_type, exc_value, traceback)
        if exc_type is None and self.batch is not None:
            self.batch.done()
        return result

    def _pack(self, tensor: torch.Tensor) -> Any:
        start = time.perf_counter()
        try:
            return self._pack_impl(tensor)
        finally:
            self.manager.stats.pack_seconds += time.perf_counter() - start

    def _pack_impl(self, tensor: torch.Tensor) -> Any:
        if tensor.device.type != 'cuda':
            return tensor
        if tensor.device != self.device:
            raise ValueError(
                f'CPU offloading only supports one CUDA device, expected {self.device}, got {tensor.device}'
            )
        if tensor.layout != torch.strided:
            return tensor

        owner = self.module.find_buffer_or_param_owner(tensor)
        if owner is not None:
            return _ModuleTensorRef(owner, tensor)

        nbytes = tensor.numel() * tensor.element_size()
        if nbytes < self.manager.min_bytes:
            self.manager.stats.skipped_small_bytes += nbytes
            return tensor

        assert self.batch is not None
        return self.batch.append(tensor)

    def _unpack(self, value: Any) -> torch.Tensor:
        if isinstance(value, _ModuleTensorRef):
            return value.unpack()
        if not isinstance(value, _OffloadedTensor):
            return value
        start = time.perf_counter()
        self.manager.on_unpack(value)
        result = value.unpack()
        self.manager.stats.unpack_seconds += time.perf_counter() - start
        return result


def prefetch_next_backward(name: str) -> None:
    """Module-level entry used by generated schedule code."""
    manager = OffloadManager._instance
    if manager is not None and manager.enabled:
        manager.prefetch_next_backward(name)


def hint_forward(name: str, backward_distance: int) -> None:
    """Module-level entry used by generated schedule code."""
    manager = OffloadManager.get()
    if manager.enabled:
        manager.hint_forward(name, backward_distance)


def offload_stats(reset: bool = True) -> Optional[Dict[str, Any]]:
    manager = OffloadManager._instance
    return None if manager is None else manager.snapshot(reset=reset)
