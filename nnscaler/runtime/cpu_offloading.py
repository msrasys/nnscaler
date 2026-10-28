from __future__ import annotations

import threading
import weakref
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch.autograd.graph import saved_tensors_hooks
from torch.multiprocessing.reductions import StorageWeakRef

from nnscaler.runtime.device import DeviceGroup

if TYPE_CHECKING:
    from nnscaler.runtime.module import ParallelModule


_PREFETCH_STREAM_NAME = 'cpu_offload_prefetch'
_OFFLOAD_STREAM_NAME = 'cpu_offload_d2h'


def _get_prefetch_stream() -> torch.cuda.Stream:
    return DeviceGroup().get_stream(_PREFETCH_STREAM_NAME)


def _get_offload_stream() -> torch.cuda.Stream:
    return DeviceGroup().get_stream(_OFFLOAD_STREAM_NAME)


class _OffloadBatch:
    last_batch: Optional[weakref.ReferenceType['_OffloadBatch']] = None
    def __init__(self) -> None:
        self.handles: list['_OffloadedStorage'] = []
        self.previous: Optional[weakref.ReferenceType['_OffloadBatch']] = _OffloadBatch.last_batch
        self.triggered = False

    def append(self, handle: '_OffloadedStorage') -> None:
        self.handles.append(handle)

    def done(self) -> None:
        # don't put empty batches into the last_batch chain.
        if self.handles:
            _OffloadBatch.last_batch = weakref.ref(self)

    def prefetch(self, *, prefetch_level: int = 1) -> None:
        if not self.triggered:
            self.triggered = True
            for handle in reversed(self.handles):
                handle.prefetch()

        if prefetch_level > 1:
            previous = self.previous() if self.previous is not None else None
            if previous is not None:
                previous.prefetch(prefetch_level=prefetch_level - 1)

    def prefetch_tensors(self, handle: '_OffloadedStorage', count: int) -> None:
        handle_idx = self.handles.index(handle)
        handles = self.handles[:handle_idx]
        batch: Optional[_OffloadBatch] = self
        while count > 0:
            for handle in reversed(handles):
                handle.prefetch()
                count -= 1
                if count == 0:
                    return
            batch = batch.previous() if batch.previous is not None else None
            if batch is None:
                return
            handles = batch.handles


class _VersionCounter:
    """Track a saved tensor's version counter without keeping its storage alive.

    PyTorch skips its own saved-tensor version check when pack hooks are
    installed, so inplace mutations of offloaded tensors must be detected here.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self.version = tensor._version
        self.tensor = tensor.detach()
        # detach() makes self.tensor share tensor's version counter.
        # Replacing self.tensor.data releases its reference to the source
        # storage, but preserves that shared counter without incrementing it.
        # Later inplace mutations of tensor or its normal views therefore also
        # change self.tensor._version, allowing changed() to detect them.
        # Do not use set_(): it would increment the shared counter here.
        self.tensor.data = torch.empty(0, device=tensor.device)

    def changed(self) -> bool:
        return self.tensor._version != self.version

    def check(self) -> None:
        if self.changed():
            raise RuntimeError(
                'one of the variables needed for gradient computation has been modified by an inplace operation: '
                f'tensor is at version {self.tensor._version}; expected version {self.version} instead.'
            )


class _ModuleTensorRef:
    def __init__(self, owner: torch.Tensor, tensor: torch.Tensor) -> None:
        self.owner = owner
        self.shape = tensor.shape
        self.stride = tensor.stride()
        self.storage_offset = tensor.storage_offset()
        self.version = _VersionCounter(tensor)

    def unpack(self) -> torch.Tensor:
        self.version.check()
        return torch.as_strided(
            self.owner,
            self.shape,
            self.stride,
            self.storage_offset,
        )


class _OffloadedStorage:
    def __init__(
        self,
        tensor: torch.Tensor,
        offload_stream: torch.cuda.Stream,
        prefetch_stream: torch.cuda.Stream,
    ) -> None:
        self.device = tensor.device
        self.prefetch_stream = prefetch_stream
        # Keep the snapshot's expected version while sharing the source's live
        # counter. A source mutation then requires a fresh copy for later saves,
        # while unpacking an earlier save must still report the mutation.
        self.version = _VersionCounter(tensor)
        # autograd (and load activations) may be multi-threaded,
        # so we must lock access to the handle's state
        self.lock = threading.Lock()
        self.device_tensor: Optional[torch.Tensor] = None
        self.h2d_event: Optional[torch.cuda.Event] = None

        # Copy the whole storage: later saved views may cover regions outside
        # the first saved view.
        source = torch.empty(0, dtype=torch.uint8, device=self.device).set_(
            tensor.untyped_storage(),
        )
        self.host_tensor = torch.empty_like(source, device='cpu', pin_memory=True)
        offload_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(offload_stream):
            self.host_tensor.copy_(source, non_blocking=True)
            self.d2h_event = offload_stream.record_event()
        source.record_stream(offload_stream)

    def prefetch(self) -> None:
        with self.lock:
            if self.device_tensor is not None:
                return
            stream = self.prefetch_stream
            with torch.cuda.stream(stream):
                stream.wait_event(self.d2h_event)
                self.device_tensor = self.host_tensor.to(self.device, non_blocking=True)
                self.h2d_event = stream.record_event()

    def unpack(self) -> torch.Tensor:
        self.version.check()
        self.prefetch()
        assert self.device_tensor is not None and self.h2d_event is not None
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.h2d_event)
        self.device_tensor.record_stream(stream)
        return self.device_tensor


class _OffloadedTensor:
    def __init__(self, tensor: torch.Tensor, storage: _OffloadedStorage) -> None:
        self.storage = storage
        # Aliases created with set_ can have independent version counters.
        self.version = _VersionCounter(tensor)
        self.shape = tensor.shape
        self.stride = tensor.stride()
        self.storage_offset = tensor.storage_offset()
        self.dtype = tensor.dtype
        self.is_conj = tensor.is_conj()
        self.is_neg = tensor.is_neg()

    def unpack(self) -> torch.Tensor:
        self.version.check()
        data = self.storage.unpack()
        tensor = torch.empty(0, dtype=self.dtype, device=data.device).set_(
            data.untyped_storage(), self.storage_offset, self.shape, self.stride,
        )
        if self.is_conj:
            tensor = tensor.conj()
        if self.is_neg:
            tensor = torch._neg_view(tensor)
        return tensor


class CPUOffloadContext:
    """Offload saved tensors and prefetch them in reverse forward order.

    Contexts must be entered and exited sequentially on one thread. Nested or
    concurrent use is unsupported because batch ordering is process-global.

    Within each context, views share one CPU copy and one prefetched GPU copy
    per valid activation storage snapshot. Offload uses a dedicated CUDA stream.
    Views retain their dtype, shape, strides and storage offset. The whole
    storage is copied, even when only a small view is saved.

    Backward rejects a saved tensor whose version counter has changed,
    including changes through normal views sharing that counter. Saving the
    storage again after such a change takes a fresh snapshot; earlier saves
    remain invalid. Multiple snapshots of one storage can coexist in a batch.

    Writes through .data or storage aliases with independent version counters,
    such as those created by set_(), are unsupported: they can bypass snapshot
    invalidation and silently produce incorrect gradients.
    """

    def __init__(self, module: 'ParallelModule', prefetch_level: int = 2) -> None:
        """
        Initialize the CPU offload context.

        Args:
            module (ParallelModule): The module whose tensors will be offloaded.
            prefetch_level (int): How far to prefetch after loading the demanded tensor.
                0: load only the demanded snapshot, without lookahead.
                Positive values set a snapshot lookahead in reverse pack order:
                1 prefetches the next snapshot, 2 prefetches the next two, and
                so on, continuing into previous batches when necessary.
                Different versions of one storage count as separate snapshots.
                Negative values count batches: -1 prefetches the current batch,
                -2 prefetches the current and previous batch, and so on.
                ...
            Batch ordering:
                Each successfully completed, non-empty context becomes the
                latest batch. A new context captures that latest live batch as
                its predecessor, forming a process-local chain in forward
                execution order.

                This order is valid for non-pipeline execution and within one
                pipeline stage for single forward pass.
                Across pipeline stages, however, process-local
                forward order does not describe which batch will be adjacent
                during backward. The chain therefore doesn't work well for
                prefetching across stage boundaries.

                When multiple forward passes occur before backward(i.e. accumulation of gradients),
                the chain may not accurately reflect the true backward order,
                potentially affecting prefetching efficiency.
        """
        if isinstance(prefetch_level, bool) or not isinstance(prefetch_level, int):
            raise ValueError(f'prefetch_level must be an integer, got {prefetch_level}')

        self.batch = _OffloadBatch()
        self.module = module
        self._hooks: Optional[saved_tensors_hooks] = None
        self.prefetch_stream: Optional[torch.cuda.Stream] = None
        self.offload_stream: Optional[torch.cuda.Stream] = None
        self._storages: dict[StorageWeakRef, _OffloadedStorage] = {}
        self.device: Optional[torch.device] = None
        self.prefetch_level = prefetch_level

    def __enter__(self) -> CPUOffloadContext:
        if torch.cuda.is_available():
            self.prefetch_stream = _get_prefetch_stream()
            self.offload_stream = _get_offload_stream()
            self.device = self.prefetch_stream.device

        self._hooks = saved_tensors_hooks(self._pack, self._unpack)
        self._hooks.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Optional[bool]:
        assert self._hooks is not None
        hooks = self._hooks
        self._hooks = None
        result = hooks.__exit__(exc_type, exc_value, traceback)
        if exc_type is None:
            self.batch.done()
        return result

    def _pack(self, tensor: torch.Tensor) -> Any:
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

        assert self.prefetch_stream is not None and self.offload_stream is not None
        key = StorageWeakRef(tensor.untyped_storage())
        storage = self._storages.get(key)
        # Normal views share the source's counter, so checking storage.version
        # detects writes through any of them and prevents stale snapshot reuse.
        # Independent-counter aliases (e.g. set_) do not propagate their version
        # changes here; such writes are unsupported.
        if storage is None or storage.version.changed():
            storage = _OffloadedStorage(tensor, self.offload_stream, self.prefetch_stream)
            self._storages[key] = storage
            # Keep both snapshots in pack order for prefetching. Existing saves
            # still reference the old snapshot, whose guard shares the source's
            # live counter but retains the old expected version; unpacking those
            # saves therefore reports the mutation instead of using the new data.
            self.batch.append(storage)
        return _OffloadedTensor(tensor, storage)

    def _unpack(self, value: Any) -> torch.Tensor:
        if isinstance(value, _ModuleTensorRef):
            return value.unpack()
        if not isinstance(value, _OffloadedTensor):
            return value
        tensor = value.unpack()
        if self.prefetch_level > 0:
            self.batch.prefetch_tensors(value.storage, self.prefetch_level)
        elif self.prefetch_level < 0:
            self.batch.prefetch(prefetch_level=-self.prefetch_level)
        return tensor
