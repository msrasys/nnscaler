from __future__ import annotations

import threading
import weakref
from typing import Any, Optional, TYPE_CHECKING

import torch
from torch.autograd.graph import saved_tensors_hooks

from nnscaler.runtime.device import DeviceGroup

if TYPE_CHECKING:
    from nnscaler.runtime.module import ParallelModule


_PREFETCH_STREAM_NAME = 'cpu_offload_prefetch'


def _get_prefetch_stream() -> torch.cuda.Stream:
    return DeviceGroup().get_stream(_PREFETCH_STREAM_NAME)


class _OffloadBatch:
    last_batch: Optional[weakref.ReferenceType['_OffloadBatch']] = None
    def __init__(self) -> None:
        self.handles: list['_OffloadedTensor'] = []
        self.previous: Optional[weakref.ReferenceType['_OffloadBatch']] = _OffloadBatch.last_batch
        self.triggered = False

    def append(self, handle: '_OffloadedTensor') -> None:
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

    def prefetch_tensors(self, handle: '_OffloadedTensor', count: int) -> None:
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


class _ModuleTensorRef:
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
        return torch.as_strided(
            self.owner,
            self.shape,
            self.stride,
            self.storage_offset,
        )


class _OffloadedTensor:
    def __init__(self, tensor: torch.Tensor, prefetch_stream: torch.cuda.Stream) -> None:
        self.device = tensor.device
        self.prefetch_stream = prefetch_stream
        # autograd (and load activations) may be multi-threaded,
        # so we must lock access to the handle's state
        self.lock = threading.Lock()
        self.device_tensor: Optional[torch.Tensor] = None
        self.h2d_event: Optional[torch.cuda.Event] = None

        self.host_tensor = torch.empty_like(
            tensor,
            device='cpu',
            pin_memory=True,
            memory_format=torch.preserve_format,
        )
        # TODO: Use a dedicated D2H stream to overlap offload with forward
        # compute; it must wait for the producer stream and retain the source
        # storage until the copy completes.
        stream = torch.cuda.current_stream(self.device)
        self.host_tensor.copy_(tensor, non_blocking=True)
        tensor.record_stream(stream) # required as we have multi-stream support
        self.d2h_event = torch.cuda.Event()
        self.d2h_event.record(stream)

    def prefetch(self) -> None:
        with self.lock:
            if self.device_tensor is not None:
                return
            stream = self.prefetch_stream
            with torch.cuda.stream(stream):
                # cross-stream wait_event is safe.
                stream.wait_event(self.d2h_event)
                self.device_tensor = self.host_tensor.to(self.device, non_blocking=True)
                self.h2d_event = torch.cuda.Event()
                self.h2d_event.record(stream)

    def unpack(self) -> torch.Tensor:
        self.prefetch()
        assert self.device_tensor is not None and self.h2d_event is not None
        stream = torch.cuda.current_stream(self.device)
        stream.wait_event(self.h2d_event)
        self.device_tensor.record_stream(stream)
        # unpack can return a different tensor object
        return self.device_tensor


class CPUOffloadContext:
    """Offload saved tensors and prefetch them in reverse forward order.

    Contexts must be entered and exited sequentially on one thread. Nested or
    concurrent use is unsupported because batch ordering is process-global.
    """

    def __init__(self, module: 'ParallelModule', prefetch_level: int = 2) -> None:
        """
        Initialize the CPU offload context.

        Args:
            module (ParallelModule): The module whose tensors will be offloaded.
            prefetch_level (int): How far to prefetch after loading the demanded tensor.
                0: no prefetching, only offload on demand.
                Positive values set a tensor lookahead in reverse pack order:
                1 prefetches the next tensor, 2 prefetches the next two, and so
                on, continuing into previous batches when necessary.
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
        self.device: Optional[torch.device] = None
        self.prefetch_level = prefetch_level

    def __enter__(self) -> CPUOffloadContext:
        if torch.cuda.is_available():
            self.prefetch_stream = _get_prefetch_stream()
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

        assert self.prefetch_stream is not None
        handle = _OffloadedTensor(tensor, self.prefetch_stream)
        self.batch.append(handle)
        return handle

    def _unpack(self, value: Any) -> torch.Tensor:
        if isinstance(value, _ModuleTensorRef):
            return value.unpack()
        if not isinstance(value, _OffloadedTensor):
            return value
        # Submit the demanded tensor first, then prefetch the configured batch
        # or tensor window in reverse pack order.
        value.prefetch()
        if self.prefetch_level > 0:
            self.batch.prefetch_tensors(value, self.prefetch_level)
        elif self.prefetch_level < 0:
            self.batch.prefetch(prefetch_level=-self.prefetch_level)
        return value.unpack()
