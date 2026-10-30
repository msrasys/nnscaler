#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Moving bytes between device and pinned host memory.

The unit of transport is a :class:`Transfer`: what one execution of one offload
group saved (or a slice of it when ``commit_bytes`` flushes early).  Saved
views are collected while the group runs and *committed* when it ends:
overlapping views of the same storage are merged into regions, the regions are
laid out in one pinned slab and grouped into chunks, one D2H copy per region is
issued behind a single event that orders them after the producing stream.  The
reload is one device buffer and one H2D copy per chunk, each with its own
event, so the first tensors of a group can be consumed before the rest arrived
and a chunk's device memory is released as soon as its views were consumed.
Per saved tensor this costs a dictionary lookup at pack time and one event
query at unpack time.

:class:`CopyEngine` owns the two copy streams and every policy about *how*
bytes move (chunking, bounded in-flight windows, communication-exclusive
windows, the reload budget, the fake A/B modes).  Neither class knows anything
about autograd or pipeline schedules.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Deque, Dict, List, Optional, TYPE_CHECKING

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

    __slots__ = ('storage_ptr', 'lo', 'hi', 'offset', 'src', 'chunk')

    def __init__(self, storage_ptr: int, lo: int, hi: int) -> None:
        self.storage_ptr = storage_ptr
        self.lo = lo
        self.hi = hi
        self.offset = 0
        self.src: Optional[torch.Tensor] = None     # uint8 view of the device bytes
        self.chunk: Optional[Chunk] = None

    @property
    def nbytes(self) -> int:
        return self.hi - self.lo

    def covers(self, lo: int, hi: int) -> bool:
        return self.lo <= lo and hi <= self.hi


class Chunk:
    """A slab range reloaded as one copy into one device buffer."""

    __slots__ = ('start', 'end', 'unconsumed', 'buffer', 'event', 'state')

    def __init__(self, start: int) -> None:
        self.start = start
        self.end = start
        self.unconsumed = 0                 # saved views not yet rebuilt from this chunk
        self.buffer: Optional[torch.Tensor] = None
        self.event: Optional[torch.cuda.Event] = None
        # None = unknown, True = observed complete, or the stream already waiting on it
        self.state: Any = None

    @property
    def nbytes(self) -> int:
        return self.end - self.start


class _SavedStorage:
    """Handles saved from one device storage; keeps the storage alive until commit."""

    __slots__ = ('storage', 'handles')

    def __init__(self, storage: torch.UntypedStorage) -> None:
        self.storage = storage
        self.handles: List[OffloadedTensor] = []


class Transfer:
    """Saved tensors of one offload group execution, moved as one unit.

    Life cycle: ``add`` while the group runs -> ``commit`` (regions and chunks
    laid out, D2H scheduled) -> device memory released -> ``prefetch`` /
    ``issue_h2d`` -> ``fetch`` per saved tensor (consumer waits for the chunk
    holding it) -> ``release`` with the owning batch (slab back to the pool).
    """

    __slots__ = (
        'engine', 'owner', 'group_id', 'device', 'handles', 'regions', 'chunks', 'nbytes', 'pending_bytes',
        '_storages', 'host', 'keep_src', 'hold_src', 'consumer_stream', 'lock',
        'ready_event', 'd2h_start', 'd2h_event', 'h2d_start', 'h2d_event',
        'committed', 'd2h_issued', 'h2d_issued', 'released', 'unconsumed', '_h2d_tracked', '_recorded',
    )

    def __init__(self, engine: 'CopyEngine', group_id: Optional[int], device: torch.device) -> None:
        self.engine = engine
        self.owner = None            # the tracker's batch, for prefetch decisions at unpack
        self.group_id = group_id if group_id is not None else -1
        self.device = device
        self.handles: List[OffloadedTensor] = []
        self.regions: List[Region] = []
        self.chunks: List[Chunk] = []
        self.nbytes = 0
        self.pending_bytes = 0       # bytes of saved views registered so far (before commit)
        self._storages: Dict[int, _SavedStorage] = {}
        self.host: Optional[torch.Tensor] = None
        self.keep_src = engine.config.fake_mode != 'off'
        self.hold_src = False        # keep sources after the D2H copy (cross-group dedup)
        self.consumer_stream: Optional[torch.cuda.Stream] = None
        self.lock = threading.Lock()
        self.ready_event: Optional[torch.cuda.Event] = None
        self.d2h_start: Optional[torch.cuda.Event] = None
        self.d2h_event: Optional[torch.cuda.Event] = None
        self.h2d_start: Optional[torch.cuda.Event] = None
        self.h2d_event: Optional[torch.cuda.Event] = None
        self.committed = False
        self.d2h_issued = False
        self.h2d_issued = False
        self.released = False
        self.unconsumed = 0
        self._h2d_tracked = False
        self._recorded: Dict[int, torch.cuda.Stream] = {}   # chunk index -> stream recorded on

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
        self.pending_bytes += hi - lo
        if hi > lo:
            self.unconsumed += 1      # empty views are rebuilt without a fetch
        return handle

    def find_region(self, storage_ptr: int, lo: int, hi: int) -> Optional[Region]:
        """A committed region still holding its source that covers ``[lo, hi)`` of the storage."""
        if not self.committed or self.released:
            return None
        for region in self.regions:
            if region.storage_ptr == storage_ptr and region.src is not None and region.covers(lo, hi):
                return region
        return None

    def share(self, handle: OffloadedTensor, region: Region) -> None:
        """Serve ``handle`` (saved by a later group) from this transfer's ``region``."""
        handle.transfer = self
        handle.region = region
        with self.lock:
            self.unconsumed += 1
            region.chunk.unconsumed += 1

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
                    region = Region(saved.storage.data_ptr(), handle.lo // RANGE_ALIGN * RANGE_ALIGN, handle.hi)
                    storage_regions.append(region)
                handle.region = region
            if storage_regions:
                storage_view = torch.empty(0, dtype=torch.uint8, device=self.device).set_(saved.storage)
                for region in storage_regions:
                    region.src = storage_view[region.lo:region.hi]
                    offset = self._place(region, offset)
        self._storages = {}
        self.nbytes = offset
        if self.chunks:
            self.chunks[-1].end = offset
        regions = self.regions
        for handle in self.handles:
            if handle.region is not None:
                handle.region.chunk.unconsumed += 1
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
        chunk_bytes = self.engine.config.copy_chunk_bytes
        chunk = self.chunks[-1] if self.chunks else None
        if chunk is None or (chunk_bytes and chunk.end > chunk.start and chunk.end - chunk.start + region.nbytes > chunk_bytes):
            if chunk is not None:
                chunk.end = offset
            chunk = Chunk(offset)
            self.chunks.append(chunk)
        region.offset = offset
        region.chunk = chunk
        self.regions.append(region)
        end = (offset + region.nbytes + REGION_ALIGN - 1) // REGION_ALIGN * REGION_ALIGN
        chunk.end = end
        return end

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
            self.d2h_event = torch.cuda.Event(enable_timing=engine.timed)
            self.d2h_event.record(d2h)
        if not (self.keep_src or self.hold_src):
            self.drop_sources()
        stats.d2h_issue_seconds += time.perf_counter() - t1
        if engine.timed:
            engine.timing.track_d2h(self, self.group_id)

    def drop_sources(self) -> None:
        """Release the device memory of the saved views (the copies hold it via ``record_stream``)."""
        self.hold_src = False
        if self.keep_src or not self.d2h_issued:
            return
        for region in self.regions:
            region.src = None

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
        for chunk in self.chunks:
            chunk.state = None
        self._recorded = {}
        if engine.config.fake_mode == 'd2h_only':
            for chunk in self.chunks:
                chunk.event = self.d2h_event
            self.h2d_event = self.d2h_event
            return
        t0 = time.perf_counter()
        h2d = engine.h2d_stream
        consumer = self.consumer_stream or torch.cuda.current_stream()
        host = self.host
        on_copy_stream = engine.config.reload_on_copy_stream
        with torch.cuda.stream(h2d):
            h2d.wait_event(self.d2h_event)
            if not on_copy_stream:
                # buffers come from the consumer's allocator pool: their memory may
                # still be read by work queued there
                h2d.wait_stream(consumer)
            engine.wait_comm_barrier(h2d)
            if engine.timed:
                self.h2d_start = torch.cuda.Event(enable_timing=True)
                self.h2d_start.record(h2d)
            for chunk in self.chunks:
                if on_copy_stream:
                    buffer = torch.empty(chunk.nbytes, dtype=torch.uint8, device=self.device)
                else:
                    with torch.cuda.stream(consumer):
                        buffer = torch.empty(chunk.nbytes, dtype=torch.uint8, device=self.device)
                    buffer.record_stream(h2d)
                buffer.copy_(host[chunk.start:chunk.end], non_blocking=True)
                chunk.buffer = buffer
                chunk.event = torch.cuda.Event(enable_timing=engine.timed and chunk is self.chunks[-1])
                chunk.event.record(h2d)
        self.h2d_event = self.chunks[-1].event
        engine.reload_issued(self.nbytes)
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
        chunk = region.chunk
        engine = self.engine
        if not self.keep_src and (chunk.buffer is None or not self.h2d_issued):
            if self.h2d_issued:
                # every view of the chunk was unpacked once and its buffer handed
                # to autograd; a second unpack (retain_graph) reloads from the slab
                if self.host is None:
                    raise RuntimeError('offloaded tensor unpacked again after its transfer was released')
                self.h2d_issued = False
            engine.force_h2d(self)
        current = torch.cuda.current_stream()     # one CUDA device per process (checked at pack time)
        if chunk.event is not None and engine.config.fake_mode != 'no_copy':
            state = chunk.state
            # a chunk observed complete stays complete, and a stream that already
            # waits on it needs no second wait: most fetches cost no CUDA call.
            # (torch.cuda.Stream only defines __eq__; never compare it with None via !=)
            if state is not True and (state is None or state != current):
                if chunk.event.query():
                    chunk.state = True
                else:
                    engine.stats.stalls += 1
                    if engine.timed:
                        # GPU-side measurement: nothing else runs on ``current`` between the events
                        before = torch.cuda.Event(enable_timing=True)
                        before.record(current)
                        current.wait_event(chunk.event)
                        after = torch.cuda.Event(enable_timing=True)
                        after.record(current)
                        engine.timing.track_stall(before, after, self.group_id)
                    else:
                        current.wait_event(chunk.event)
                    chunk.state = current
        if self.keep_src:
            base, start = region.src, 0
        else:
            base, start = chunk.buffer, region.offset - chunk.start
            index = id(chunk)
            recorded = self._recorded.get(index)
            if recorded is None or recorded != current:
                base.record_stream(current)
                self._recorded[index] = current
        start += handle.lo - region.lo
        return base[start:start + handle.hi - handle.lo]

    def view_consumed(self, handle: OffloadedTensor) -> None:
        chunk = handle.region.chunk if handle.region is not None else None
        with self.lock:
            self.unconsumed -= 1
            done = self.unconsumed == 0
            if chunk is not None:
                chunk.unconsumed -= 1
                if chunk.unconsumed == 0 and not self.keep_src and chunk.buffer is not None:
                    # autograd holds the unpacked views; drop our reference early
                    chunk.buffer = None
                    self.engine.reload_consumed(chunk.nbytes)
        if done and self.owner is not None and self.owner.context is None:
            # eager use (no executor): nothing releases the batch later
            self.owner.release()

    def release(self) -> None:
        with self.lock:
            if self.released:
                return
            self.released = True
            remaining = 0
            for chunk in self.chunks:
                if chunk.buffer is not None:
                    remaining += chunk.nbytes
                chunk.buffer = None
            for region in self.regions:
                region.src = None
            self._storages = {}
            if remaining and self.h2d_issued:
                self.engine.reload_consumed(remaining)
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
        # the budget gauge has its own lock: it is updated from issue_h2d, which
        # runs under ``_lock`` when a window or the budget pump issues the copy
        self._budget_lock = threading.Lock()
        self.reloaded_bytes = 0      # reloaded and not yet consumed (the prefetch budget's gauge)

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

    # -- reload budget -----------------------------------------------------------
    def reload_issued(self, nbytes: int) -> None:
        with self._budget_lock:
            self.reloaded_bytes += nbytes

    def reload_consumed(self, nbytes: int) -> None:
        with self._budget_lock:
            self.reloaded_bytes = max(self.reloaded_bytes - nbytes, 0)

    def within_reload_budget(self) -> bool:
        """Whether an *ahead-of-demand* reload may be issued now."""
        budget = self.config.prefetch_bytes
        return not budget or self.reloaded_bytes < budget

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
        """The bytes are needed now: bypass the window and the budget."""
        with self._lock:
            if not transfer.h2d_issued:
                if transfer.consumer_stream is None:
                    transfer.consumer_stream = torch.cuda.current_stream()
                transfer.issue_h2d()
                self._inflight_h2d.append(transfer)
                self.stats.forced_h2d += 1

    def pending_d2h_bytes(self) -> int:
        with self._lock:
            return sum(t.nbytes for t in self._pending_d2h)
