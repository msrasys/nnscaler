#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Moving bytes between device and pinned host memory.

The unit of transport is a :class:`Transfer`: everything one execution of one
offload group saved.  Saved views are collected while the group runs and
*committed* when it ends: overlapping views of the same storage are merged
into regions, the regions are laid out in one pinned slab, one D2H copy per
region is issued behind a single event that orders them after the producing
stream.  The reload is one device buffer and one H2D copy per transfer (or a
few chunks, each with its own event), so the per-tensor cost is a dictionary
lookup at pack time and one event query at unpack time.

:class:`CopyEngine` owns the two copy streams and every policy about *how*
bytes move (chunking, bounded in-flight windows, communication-exclusive
windows, the fake A/B modes).  Neither class knows anything about autograd or
pipeline schedules.
"""

from __future__ import annotations

import bisect
import collections
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

import torch

from nnscaler.runtime.device import DeviceGroup
from .config import OffloadConfig
from .hooks import OffloadedTensor, covered_bytes
from .host_memory import PinnedPool
from .stats import GroupProfile, OffloadStats, TimingCollector

D2H_STREAM_NAME = 'cpu_offload_d2h'
H2D_STREAM_NAME = 'cpu_offload_h2d'

REGION_ALIGN = 512     # slab offset alignment of a region
RANGE_ALIGN = 16       # region start alignment inside its storage (largest torch element size)


class Region:
    """A contiguous byte range ``[lo, hi)`` of one device storage, placed at ``offset`` in the slab."""

    __slots__ = ('lo', 'hi', 'offset', 'src')

    def __init__(self, lo: int, hi: int, offset: int) -> None:
        self.lo = lo
        self.hi = hi
        self.offset = offset
        self.src: Optional[torch.Tensor] = None     # uint8 view of the device bytes

    @property
    def nbytes(self) -> int:
        return self.hi - self.lo


class _SavedStorage:
    """Handles saved from one device storage; keeps the storage alive until commit."""

    __slots__ = ('storage', 'handles')

    def __init__(self, storage: torch.UntypedStorage) -> None:
        self.storage = storage
        self.handles: List[OffloadedTensor] = []


class Transfer:
    """The saved tensors of one offload group execution, moved as one unit.

    Life cycle: ``add`` while the group runs -> ``commit`` at group end (regions
    laid out, D2H scheduled) -> device memory released -> ``prefetch`` /
    ``issue_h2d`` -> ``fetch`` per saved tensor (consumer waits for the chunk
    holding it) -> ``release`` with the owning batch (slab back to the pool).
    """

    __slots__ = (
        'engine', 'owner', 'group_id', 'device', 'handles', 'regions', 'nbytes', '_storages',
        'host', 'device_buffer', 'keep_src', 'consumer_stream', 'lock',
        'ready_event', 'd2h_start', 'd2h_event', 'h2d_start', 'h2d_event', '_chunk_ends', '_chunk_events',
        '_chunk_state',
        'committed', 'd2h_issued', 'h2d_issued', 'released', 'unconsumed', '_recorded_stream', '_h2d_tracked',
    )

    def __init__(self, engine: 'CopyEngine', group_id: Optional[int], device: torch.device) -> None:
        self.engine = engine
        self.owner = None            # the tracker's batch, for prefetch decisions at unpack
        self.group_id = group_id if group_id is not None else -1
        self.device = device
        self.handles: List[OffloadedTensor] = []
        self.regions: List[Region] = []
        self.nbytes = 0
        self._storages: Dict[int, _SavedStorage] = {}
        self.host: Optional[torch.Tensor] = None
        self.device_buffer: Optional[torch.Tensor] = None
        self.keep_src = engine.config.fake_mode != 'off'
        self.consumer_stream: Optional[torch.cuda.Stream] = None
        self.lock = threading.Lock()
        self.ready_event: Optional[torch.cuda.Event] = None
        self.d2h_start: Optional[torch.cuda.Event] = None
        self.d2h_event: Optional[torch.cuda.Event] = None
        self.h2d_start: Optional[torch.cuda.Event] = None
        self.h2d_event: Optional[torch.cuda.Event] = None
        self._chunk_ends: List[int] = []
        self._chunk_events: List[torch.cuda.Event] = []
        # per chunk: None = unknown, True = observed complete, or the stream already waiting on it
        self._chunk_state: List[Any] = []
        self.committed = False
        self.d2h_issued = False
        self.h2d_issued = False
        self.released = False
        self.unconsumed = 0
        self._recorded_stream: Optional[torch.cuda.Stream] = None
        self._h2d_tracked = False

    # -- collection -----------------------------------------------------------
    def add(self, tensor: torch.Tensor) -> OffloadedTensor:
        """Register a saved view; nothing moves until :meth:`commit`."""
        if self.committed:
            raise RuntimeError('cannot add a saved tensor to a committed transfer')
        lo, hi = covered_bytes(tensor)
        storage = tensor.untyped_storage()
        key = storage.data_ptr()
        saved = self._storages.get(key)
        if saved is None:
            saved = self._storages[key] = _SavedStorage(storage)
        handle = OffloadedTensor(self, tensor.dtype, tuple(tensor.shape), tuple(tensor.stride()), lo, hi)
        saved.handles.append(handle)
        self.handles.append(handle)
        if hi > lo:
            self.unconsumed += 1      # empty views are rebuilt without a fetch
        return handle

    def commit(self) -> None:
        """Lay the saved views out in one slab and schedule the D2H copies."""
        if self.committed:
            return
        self.committed = True
        offset = 0
        saved_bytes = 0
        for saved in self._storages.values():
            storage_regions: List[Region] = []
            region: Optional[Region] = None
            for handle in sorted(saved.handles, key=lambda h: h.lo):
                saved_bytes += handle.hi - handle.lo
                if handle.hi == handle.lo:
                    continue
                if region is not None and handle.lo <= region.hi:
                    region.hi = max(region.hi, handle.hi)     # overlapping or adjacent view
                else:
                    region = Region(handle.lo // RANGE_ALIGN * RANGE_ALIGN, handle.hi, 0)
                    storage_regions.append(region)
                handle.region = region
            if storage_regions:
                storage_view = torch.empty(0, dtype=torch.uint8, device=self.device).set_(saved.storage)
                for region in storage_regions:
                    region.src = storage_view[region.lo:region.hi]
                    offset = self._place(region, offset)
        self._storages = {}
        self.nbytes = offset
        regions = self.regions
        stats = self.engine.stats
        stats.deduplicated_bytes += max(saved_bytes - sum(r.nbytes for r in regions), 0)
        if not regions:
            return
        stats.offloaded_bytes += self.nbytes
        stats.offloaded_regions += len(regions)
        stats.transfers += 1
        profile = self.engine.groups[self.group_id]
        profile.bytes += self.nbytes
        profile.regions += len(regions)
        if self.engine.config.fake_mode == 'no_copy':
            self.d2h_issued = self.h2d_issued = True
            self.d2h_event = torch.cuda.Event()
            self.d2h_event.record(torch.cuda.current_stream())
            return
        # one event orders every copy after the kernels that produced the views
        self.ready_event = torch.cuda.Event()
        self.ready_event.record(torch.cuda.current_stream())
        self.engine.schedule_d2h(self)

    def _place(self, region: Region, offset: int) -> int:
        region.offset = offset
        self.regions.append(region)
        return (offset + region.nbytes + REGION_ALIGN - 1) // REGION_ALIGN * REGION_ALIGN

    # -- D2H -------------------------------------------------------------------
    def issue_d2h(self) -> None:
        if self.d2h_issued:
            return
        self.d2h_issued = True
        engine, stats = self.engine, self.engine.stats
        t0 = time.perf_counter()
        d2h = engine.d2h_stream
        self.host = engine.pool.acquire(self.nbytes, d2h)
        t1 = time.perf_counter()
        stats.acquire_seconds += t1 - t0
        host = self.host
        with torch.cuda.stream(d2h):
            d2h.wait_event(self.ready_event)
            engine.wait_comm_barrier(d2h)
            if engine.timed:
                self.d2h_start = torch.cuda.Event(enable_timing=True)
                self.d2h_start.record(d2h)
            for region in self.regions:
                engine.chunked_copy(host[region.offset:region.offset + region.nbytes], region.src)
                # the caching allocator must not reuse the device memory before the copy finished
                region.src.record_stream(d2h)
                if not self.keep_src:
                    region.src = None
            self.d2h_event = torch.cuda.Event(enable_timing=engine.timed)
            self.d2h_event.record(d2h)
        stats.d2h_issue_seconds += time.perf_counter() - t1
        if engine.timed:
            engine.timing.track_d2h(self, self.group_id)

    # -- H2D -------------------------------------------------------------------
    def prefetch(self) -> None:
        """Request the reload; issued now or when the in-flight window has room."""
        if not self.regions or self.h2d_issued or self.released:
            return
        with self.lock:
            if self.consumer_stream is None:
                self.consumer_stream = torch.cuda.current_stream()
        self.engine.schedule_h2d(self)

    def issue_h2d(self) -> None:
        if self.h2d_issued or self.released:
            return
        self.h2d_issued = True
        if not self.d2h_issued:
            self.issue_d2h()
        engine, stats = self.engine, self.engine.stats
        if engine.config.fake_mode == 'd2h_only':
            self.h2d_event = self.d2h_event
            self._chunk_ends, self._chunk_events, self._chunk_state = [self.nbytes], [self.d2h_event], [None]
            return
        t0 = time.perf_counter()
        h2d = engine.h2d_stream
        consumer = self.consumer_stream or torch.cuda.current_stream()
        # Allocate on the consumer stream: the caching allocator keeps a pool per
        # stream; allocating on the copy stream would strand the memory there.
        with torch.cuda.stream(consumer):
            buffer = torch.empty(self.nbytes, dtype=torch.uint8, device=self.device)
        host = self.host[:self.nbytes]
        chunk = engine.config.copy_chunk_bytes or self.nbytes
        ends: List[int] = []
        events: List[torch.cuda.Event] = []
        with torch.cuda.stream(h2d):
            h2d.wait_event(self.d2h_event)
            h2d.wait_stream(consumer)
            engine.wait_comm_barrier(h2d)
            if engine.timed:
                self.h2d_start = torch.cuda.Event(enable_timing=True)
                self.h2d_start.record(h2d)
            for start in range(0, self.nbytes, chunk):
                end = min(start + chunk, self.nbytes)
                buffer[start:end].copy_(host[start:end], non_blocking=True)
                event = torch.cuda.Event(enable_timing=engine.timed and end == self.nbytes)
                event.record(h2d)
                ends.append(end)
                events.append(event)
            buffer.record_stream(h2d)
        self._chunk_ends, self._chunk_events, self._chunk_state = ends, events, [None] * len(events)
        self.h2d_event = events[-1]
        self.device_buffer = buffer
        self._recorded_stream = None
        stats.h2d_issue_seconds += time.perf_counter() - t0
        stats.prefetched_bytes += self.nbytes
        stats.prefetched_regions += len(self.regions)
        if engine.timed and not self._h2d_tracked:
            self._h2d_tracked = True
            engine.timing.track_h2d(self, self.group_id)

    def fetch(self, handle: OffloadedTensor) -> torch.Tensor:
        """Device bytes ``[handle.lo, handle.hi)``; the current stream waits for their reload."""
        if not self.committed:
            self.commit()     # unpacked before the group ended (backward inside forward)
        region = handle.region
        if region is None:
            raise RuntimeError('offloaded tensor has no bytes (empty view)')
        engine = self.engine
        if self.device_buffer is None and not self.keep_src:
            if self.h2d_issued:
                # every view was unpacked once and the reload buffer handed to
                # autograd; a second unpack (retain_graph) reloads from the slab
                if self.host is None:
                    raise RuntimeError('offloaded tensor unpacked again after its transfer was released')
                self.h2d_issued = False
            engine.force_h2d(self)
        current = torch.cuda.current_stream()     # one CUDA device per process (checked at pack time)
        if self.h2d_event is not None and engine.config.fake_mode != 'no_copy':
            end = region.offset + (handle.hi - region.lo)
            index = bisect.bisect_left(self._chunk_ends, end)
            state = self._chunk_state[index]
            # a chunk observed complete stays complete, and a stream that already
            # waits on it needs no second wait: most fetches cost no CUDA call
            if state is not True and state is not current:
                event = self._chunk_events[index]
                if event.query():
                    self._chunk_state[index] = True
                else:
                    engine.stats.stalls += 1
                    if engine.timed:
                        # GPU-side measurement: nothing else runs on ``current`` between the events
                        before = torch.cuda.Event(enable_timing=True)
                        before.record(current)
                        current.wait_event(event)
                        after = torch.cuda.Event(enable_timing=True)
                        after.record(current)
                        engine.timing.track_stall(before, after, self.group_id)
                    else:
                        current.wait_event(event)
                    self._chunk_state[index] = current
        if self.keep_src:
            base, start = region.src, 0
        else:
            base, start = self.device_buffer, region.offset
            if self._recorded_stream != current:
                base.record_stream(current)
                self._recorded_stream = current
        start += handle.lo - region.lo
        return base[start:start + handle.hi - handle.lo]

    def view_consumed(self) -> None:
        with self.lock:
            self.unconsumed -= 1
            done = self.unconsumed == 0
            if done and not self.keep_src:
                # autograd holds the unpacked views; drop our reference early
                self.device_buffer = None
        if done and self.owner is not None and self.owner.context is None:
            # eager use (no executor): nothing releases the batch later
            self.owner.release()

    def release(self) -> None:
        with self.lock:
            if self.released:
                return
            self.released = True
            self.device_buffer = None
            for region in self.regions:
                region.src = None
            self._storages = {}
            if self.host is not None:
                # the slab's last reader is the H2D copy, or the D2H writer if none happened
                self.engine.pool.release(self.host, self.h2d_event or self.d2h_event)
                self.host = None


class CopyEngine:
    """Streams and copy policies shared by all transfers of a device."""

    def __init__(self, config: OffloadConfig, pool: PinnedPool, stats: OffloadStats,
                 timing: TimingCollector, groups: Dict[int, GroupProfile]) -> None:
        self.config = config
        self.pool = pool
        self.stats = stats
        self.timing = timing
        self.groups = groups
        self.timed = config.timing
        self._d2h_stream: Optional[torch.cuda.Stream] = None
        self._h2d_stream: Optional[torch.cuda.Stream] = None
        self._lock = threading.Lock()
        self._inflight_d2h: Deque[Transfer] = collections.deque()
        self._pending_d2h: Deque[Transfer] = collections.deque()
        self._inflight_h2d: Deque[Transfer] = collections.deque()
        self._pending_h2d: Deque[Transfer] = collections.deque()
        self._comm_barrier: Optional[torch.cuda.Event] = None
        self._comm_depth = 0

    # -- streams ------------------------------------------------------------------
    @property
    def d2h_stream(self) -> torch.cuda.Stream:
        if self._d2h_stream is None:
            self._d2h_stream = DeviceGroup().get_stream(D2H_STREAM_NAME)
        return self._d2h_stream

    @property
    def h2d_stream(self) -> torch.cuda.Stream:
        if self._h2d_stream is None:
            self._h2d_stream = DeviceGroup().get_stream(H2D_STREAM_NAME)
        return self._h2d_stream

    def new_transfer(self, group_id: Optional[int], device: torch.device) -> Transfer:
        return Transfer(self, group_id, device)

    # -- copy helpers ---------------------------------------------------------------
    def chunked_copy(self, dst: torch.Tensor, src: torch.Tensor) -> None:
        chunk = self.config.copy_chunk_bytes
        n = dst.numel()
        if not chunk or n <= chunk:
            dst.copy_(src, non_blocking=True)
            return
        for off in range(0, n, chunk):
            dst[off:off + chunk].copy_(src[off:off + chunk], non_blocking=True)

    # -- communication windows ---------------------------------------------------
    def wait_comm_barrier(self, stream: torch.cuda.Stream) -> None:
        event = self._comm_barrier
        if event is not None:
            stream.wait_event(event)

    def comm_begin(self) -> None:
        """A communication-heavy region starts on the current stream.

        ``strict``: the compute stream first waits for every copy issued so far
        and copies requested until ``comm_end`` are issued after the region.
        ``soft``: only the second half (copies requested inside the window are
        held back until it ends; the window never waits for copies).
        """
        mode = self.config.comm_exclusive
        if mode == 'off' or self.config.fake_mode == 'no_copy':
            return
        self._comm_depth += 1
        if self._comm_depth > 1:
            return
        current = torch.cuda.current_stream()
        with self._lock:
            if mode == 'strict':
                for stream in (self.d2h_stream, self.h2d_stream):
                    marker = torch.cuda.Event()
                    marker.record(stream)
                    current.wait_event(marker)
            barrier = torch.cuda.Event()
            barrier.record(current)
            self._comm_barrier = barrier
        self.stats.comm_windows += 1

    def comm_end(self) -> None:
        if self.config.comm_exclusive == 'off' or self.config.fake_mode == 'no_copy':
            return
        self._comm_depth -= 1
        if self._comm_depth > 0:
            return
        barrier = torch.cuda.Event()
        barrier.record(torch.cuda.current_stream())
        with self._lock:
            self._comm_barrier = barrier
        self.pump()      # issue the transfers held back during the window

    # -- bounded in-flight windows -------------------------------------------------
    @staticmethod
    def _drain(inflight: Deque[Transfer], direction: str) -> None:
        while inflight:
            transfer = inflight[0]
            event = transfer.d2h_event if direction == 'd2h' else transfer.h2d_event
            if event is None or not event.query():
                break
            inflight.popleft()

    def _has_room(self, inflight: Deque[Transfer]) -> bool:
        if self._comm_depth > 0:
            return False          # inside a communication window: hold copies back
        limit = self.config.max_inflight_copies
        return not limit or len(inflight) < limit

    def pump(self) -> None:
        """Issue deferred transfers whose window has room; cheap, called at group boundaries."""
        if not (self._pending_d2h or self._pending_h2d) or self._comm_depth > 0:
            return
        with self._lock:
            self._drain(self._inflight_d2h, 'd2h')
            while self._pending_d2h and self._has_room(self._inflight_d2h):
                transfer = self._pending_d2h.popleft()
                if not (transfer.released or transfer.d2h_issued):
                    transfer.issue_d2h()
                    self._inflight_d2h.append(transfer)
            self._drain(self._inflight_h2d, 'h2d')
            while self._pending_h2d and self._has_room(self._inflight_h2d):
                transfer = self._pending_h2d.popleft()
                if not (transfer.released or transfer.h2d_issued):
                    transfer.issue_h2d()
                    self._inflight_h2d.append(transfer)

    def schedule_d2h(self, transfer: Transfer) -> None:
        with self._lock:
            self._drain(self._inflight_d2h, 'd2h')
            if self._has_room(self._inflight_d2h):
                transfer.issue_d2h()
                self._inflight_d2h.append(transfer)
            else:
                self._pending_d2h.append(transfer)
                self.stats.deferred_d2h += 1

    def schedule_h2d(self, transfer: Transfer) -> None:
        with self._lock:
            self._drain(self._inflight_h2d, 'h2d')
            if self._has_room(self._inflight_h2d):
                transfer.issue_h2d()
                self._inflight_h2d.append(transfer)
            else:
                self._pending_h2d.append(transfer)
                self.stats.deferred_h2d += 1

    def force_h2d(self, transfer: Transfer) -> None:
        """The bytes are needed now: bypass the window."""
        with self._lock:
            if transfer.device_buffer is None and not transfer.h2d_issued:
                if transfer.consumer_stream is None:
                    transfer.consumer_stream = torch.cuda.current_stream()
                transfer.issue_h2d()
                self._inflight_h2d.append(transfer)
                self.stats.forced_h2d += 1

    def pending_d2h_bytes(self) -> int:
        with self._lock:
            return sum(t.nbytes for t in self._pending_d2h)
