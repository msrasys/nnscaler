#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""The ``saved_tensors_hooks`` context installed by generated code."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import torch
from torch.autograd.graph import saved_tensors_hooks

if TYPE_CHECKING:
    from .manager import OffloadManager
    from .tracker import OffloadBatch
    from .transport import Region, Transfer

_logger = logging.getLogger(__name__)

_ELEMENT_SIZE: Dict[torch.dtype, int] = {}


def element_size(dtype: torch.dtype) -> int:
    size = _ELEMENT_SIZE.get(dtype)
    if size is None:
        size = _ELEMENT_SIZE[dtype] = torch.empty((), dtype=dtype).element_size()
    return size


def covered_bytes(tensor: torch.Tensor) -> Tuple[int, int]:
    """Byte range ``[lo, hi)`` of ``tensor``'s storage that the view touches."""
    elem = tensor.element_size()
    extent = 1
    for size, stride in zip(tensor.shape, tensor.stride()):
        if size == 0:
            return 0, 0
        extent += (size - 1) * stride
    lo = tensor.storage_offset() * elem
    return lo, lo + extent * elem


class OffloadedTensor:
    """What autograd stores instead of the saved tensor.

    ``lo``/``hi`` are the view's byte range in its original storage; ``region``
    is assigned when the transfer commits (overlapping views share a region).
    """

    __slots__ = ('transfer', 'dtype', 'shape', 'stride', 'lo', 'hi', 'region', 'consumed')

    def __init__(self, transfer: 'Transfer', dtype: torch.dtype, shape: Tuple[int, ...],
                 stride: Tuple[int, ...], lo: int, hi: int) -> None:
        self.transfer = transfer
        self.dtype = dtype
        self.shape = shape
        self.stride = stride
        self.lo = lo
        self.hi = hi
        self.region: Optional['Region'] = None
        self.consumed = False

    @property
    def nbytes(self) -> int:
        return self.hi - self.lo

    def unpack(self) -> torch.Tensor:
        if self.hi == self.lo:
            return torch.empty(self.shape, dtype=self.dtype, device=self.transfer.device)
        bytes_ = self.transfer.fetch(self)
        tensor = bytes_.view(self.dtype).as_strided(self.shape, self.stride)
        if not self.consumed:
            self.consumed = True
            self.transfer.view_consumed()
        return tensor


class ModuleTensorRef:
    """Reference to a parameter/buffer view; these are never offloaded.

    Keeps ZeRO-3 semantics: a gathered parameter may be replaced by its shard
    after forward, the view is rebuilt from the parameter object at unpack.
    """

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
                f'tensor is at version {self.owner._version}; expected version {self.version} instead.')
        return torch.as_strided(self.owner, self.shape, self.stride, self.storage_offset)


class CPUOffloadContext:
    """Offload the tensors autograd saves inside the ``with`` block.

    Generated code calls ``module.cpu_offloading_hooks(group_id)``; the context
    must not be nested.  Tensors owned by the module (parameters, buffers) are
    referenced; tensors below ``OffloadConfig.min_bytes`` stay on the device.
    """

    def __init__(self, module, group_id: Optional[int] = None,
                 manager: Optional['OffloadManager'] = None) -> None:
        from .manager import OffloadManager
        self.module = module
        self.group_id = group_id
        self.manager = manager or OffloadManager.get()
        self.batch: Optional['OffloadBatch'] = None
        self.device: Optional[torch.device] = None
        self._hooks: Optional[saved_tensors_hooks] = None

    def __enter__(self) -> 'CPUOffloadContext':
        if not self.manager.config.enabled or not torch.cuda.is_available():
            return self
        self.device = torch.device('cuda', torch.cuda.current_device())
        self.batch = self.manager.tracker.new_batch(self.group_id, self.device)
        if self.batch is None:      # the schedule says the backward follows right away
            return self
        self._hooks = saved_tensors_hooks(self._pack, self._unpack)
        self._hooks.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> Optional[bool]:
        if self._hooks is None:
            return None
        hooks, self._hooks = self._hooks, None
        result = hooks.__exit__(exc_type, exc_value, traceback)
        if exc_type is None and self.batch is not None:
            stats = self.manager.stats
            start = time.perf_counter()
            issue_before = stats.acquire_seconds + stats.d2h_issue_seconds
            self.batch.done()
            # bookkeeping only: pinned allocation and copy issue are reported separately
            stats.commit_seconds += (time.perf_counter() - start) - (
                stats.acquire_seconds + stats.d2h_issue_seconds - issue_before)
            self.manager.debug_group_done(self.group_id, self.batch)
        return result

    # -- pack / unpack ----------------------------------------------------------
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
                f'CPU offloading only supports one CUDA device, expected {self.device}, got {tensor.device}')
        if tensor.layout != torch.strided:
            return tensor
        owner = self.module.find_buffer_or_param_owner(tensor)
        if owner is not None:
            return ModuleTensorRef(owner, tensor)
        nbytes = tensor.numel() * tensor.element_size()
        self.manager.sizes.add(nbytes)
        if nbytes < self.manager.config.min_bytes:
            self.manager.stats.skipped_small_bytes += nbytes
            return tensor
        assert self.batch is not None
        if self.batch.transfer.committed:
            # a backward inside the group already shipped the transfer; later
            # saves stay on the device rather than joining a closed transfer
            self.manager.stats.skipped_late_bytes += nbytes
            return tensor
        handle = self.batch.append(tensor)
        self.manager.debug_tensor(self.group_id, tensor, nbytes)
        return handle

    def _unpack(self, value: Any) -> torch.Tensor:
        if isinstance(value, ModuleTensorRef):
            return value.unpack()
        if not isinstance(value, OffloadedTensor):
            return value
        stats = self.manager.stats
        start = time.perf_counter()
        issue_before = stats.h2d_issue_seconds
        self.manager.tracker.on_unpack(value)
        result = value.unpack()
        # bookkeeping only: reloads issued from here are reported as h2d_issue_seconds
        stats.unpack_seconds += (time.perf_counter() - start) - (stats.h2d_issue_seconds - issue_before)
        return result
