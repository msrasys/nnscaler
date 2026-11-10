#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Bounded-memory, synchronous checkpoint merging to standard PyTorch files."""

import ctypes
import mmap
import os
import pickle
import sys
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from nnscaler.utils import check_recursively


@dataclass
class MergeStats:
    """Reconstruction memory only; excludes inputs, metadata and writer buffers."""

    live_bytes: int = 0
    peak_bytes: int = 0
    storages: int = 0


@dataclass(frozen=True)
class _Layout:
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    offset: int

    @classmethod
    def of(cls, tensor: torch.Tensor) -> '_Layout':
        return cls(tuple(tensor.shape), tuple(tensor.stride()), tensor.storage_offset())

    def view(self, flat: torch.Tensor) -> torch.Tensor:
        return flat.as_strided(self.shape, self.stride, self.offset)


@dataclass(frozen=True)
class _Source:
    layout: _Layout
    version: Optional['_Version']
    tensor: Optional[torch.Tensor]


@dataclass(frozen=True)
class _Version:
    elements: int
    dtype: torch.dtype
    writes: Tuple[Tuple[_Layout, _Source], ...] = ()


class _DeferredMerge(TorchDispatchMode):
    # These operations only change tensor metadata, never read placeholder pages.
    _views = {
        'aten.slice.Tensor', 'aten.select.int', 'aten.view.default',
        'aten._unsafe_view.default', 'aten.as_strided.default', 'aten.detach.default',
        'aten.alias.default', 'aten.transpose.int', 'aten.t.default',
        'aten.permute.default', 'aten.expand.default', 'aten.squeeze.default',
        'aten.squeeze.dim', 'aten.unsqueeze.default',
    }

    def __init__(self, budget: int, stats: MergeStats, release: Callable[[torch.UntypedStorage], None]):
        super().__init__()
        self.budget = budget
        self.stats = stats
        self.release = release
        self.versions: dict[int, _Version] = {}

    def _allocate(self, shape, dtype, stride=None):
        layout = (torch.empty(shape, dtype=dtype, device='meta') if stride is None
                  else torch.empty_strided(shape, stride, dtype=dtype, device='meta'))
        nbytes = layout.untyped_storage().nbytes()
        if not nbytes:
            return torch.empty_strided(shape, layout.stride(), dtype=dtype, device='cpu')
        # Untouched anonymous pages reserve addresses, not resident tensor memory.
        mapping = mmap.mmap(-1, nbytes)
        tensor = torch.frombuffer(mapping, dtype=dtype).as_strided(shape, layout.stride())
        key = tensor.untyped_storage()._cdata
        self.versions[key] = _Version(nbytes // layout.element_size(), dtype)
        weakref.finalize(mapping, self.versions.pop, key, None)
        return tensor

    def _source(self, tensor: torch.Tensor) -> _Source:
        version = self.versions.get(tensor.untyped_storage()._cdata)
        return _Source(_Layout.of(tensor), version, tensor if version is None else None)

    def _copy(self, destination: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        if source.device.type != 'cpu':
            raise ValueError('Tensor merge requires CPU inputs')
        source = source.expand(destination.shape)
        if not destination.numel():
            return destination
        key = destination.untyped_storage()._cdata
        if key not in self.versions:
            raise RuntimeError('Tensor merge cannot modify an input tensor')
        if destination.dtype != source.dtype:
            raise ValueError('Tensor merge requires same-dtype copies')
        if source.untyped_storage()._cdata == key and _Layout.of(source) == _Layout.of(destination):
            return destination
        previous = self.versions[key]
        # Snapshot the source version: subsequent writes must not change a clone.
        self.versions[key] = _Version(
            previous.elements, previous.dtype,
            previous.writes + ((_Layout.of(destination), self._source(source)),),
        )
        return destination

    def _materialize(self, version: _Version) -> torch.Tensor:
        element_size = torch.empty((), dtype=version.dtype, device='meta').element_size()
        nbytes = version.elements * element_size
        if self.budget and self.stats.live_bytes + nbytes > self.budget:
            raise MemoryError(
                f'Tensor merge needs another {nbytes} bytes with {self.stats.live_bytes} bytes live; '
                f'increase ram_budget_bytes (currently {self.budget})'
            )
        mapping = mmap.mmap(-1, nbytes)
        self.stats.live_bytes += nbytes
        self.stats.peak_bytes = max(self.stats.peak_bytes, self.stats.live_bytes)

        def released():
            self.stats.live_bytes -= nbytes

        weakref.finalize(mapping, released)
        result = torch.frombuffer(mapping, dtype=version.dtype)
        del mapping
        try:
            for layout, source in version.writes:
                value = self._value(source)
                try:
                    layout.view(result).copy_(value)
                finally:
                    try:
                        self.release(value.untyped_storage())
                    finally:
                        del value
            return result
        finally:
            # Also release allocations when an exception traceback retains this frame.
            del result

    def _value(self, source: _Source) -> torch.Tensor:
        if source.version is None:
            assert source.tensor is not None
            return source.tensor
        return source.layout.view(self._materialize(source.version))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if name in ('aten.empty.memory_format', 'aten.zeros.default', 'aten.empty_strided.default'):
            if (torch.device(kwargs.get('device') or 'cpu').type != 'cpu'
                    or kwargs.get('layout', torch.strided) != torch.strided
                    or kwargs.get('pin_memory', False)
                    or kwargs.get('memory_format') not in (None, torch.contiguous_format)):
                raise ValueError(f'Unsupported tensor merge allocation: {name}, {kwargs}')
            return self._allocate(args[0], kwargs.get('dtype') or torch.get_default_dtype(),
                                  args[1] if name == 'aten.empty_strided.default' else None)
        if name == 'aten.clone.default':
            source = args[0]
            layout = torch.empty_like(source, device='meta', **kwargs)
            return self._copy(self._allocate(source.shape, source.dtype, layout.stride()), source)
        if name == 'aten.copy_.default':
            return self._copy(args[0], args[1])
        if name == 'aten.equal.default':
            if args[0].shape != args[1].shape:
                return False
            left = right = None
            try:
                left = self._value(self._source(args[0]))
                right = self._value(self._source(args[1]))
                return torch.equal(left, right)
            finally:
                try:
                    if left is not None:
                        self.release(left.untyped_storage())
                    if right is not None:
                        self.release(right.untyped_storage())
                finally:
                    left = right = None
        if name == 'aten._local_scalar_dense.default':
            value = self._value(self._source(args[0]))
            try:
                return func(value, **kwargs)
            finally:
                try:
                    self.release(value.untyped_storage())
                finally:
                    del value
        if name in self._views:
            return func(*args, **kwargs)
        raise RuntimeError(f'Unsupported operation in tensor merge: {name}')


class _StorageWriter:
    """Adapt PyTorch's ZIP writer without replacing its pickle/storage protocol."""

    def __init__(self, writer, merge: _DeferredMerge):
        self.writer = writer
        self.merge = merge

    def __getattr__(self, name):
        return getattr(self.writer, name)

    def write_record(self, name, data, size):
        if not isinstance(data, torch.UntypedStorage):
            return self.writer.write_record(name, data, size)
        version = self.merge.versions.get(data._cdata)
        if version is None:
            try:
                return self.writer.write_record(name, data, size)
            finally:
                self.merge.release(data)
        tensor = self.merge._materialize(version)
        try:
            self.writer.write_record(name, tensor.untyped_storage(), size)
            self.merge.stats.storages += 1
        finally:
            del tensor


class CheckpointMerger:
    """Merge and serialize one reconstructed storage (plus dependencies) at a time.

    ``load`` uses read-only CPU mmap inputs (PyTorch >= 2.1, ZIP checkpoints).
    ``merge`` runs the existing nnscaler merge algorithm, records its copies, and
    synchronously writes a normal ``torch.load``-compatible file. It never returns
    deferred tensors. Only CPU strided tensors and merge operations are supported;
    unsupported operations fail explicitly instead of falling back to eager mode.

    ``ram_budget_bytes=0`` means no reconstruction limit. The limit does NOT include
    mapped input pages, Python metadata, or serialization buffers. One output
    storage and its temporary dependencies must fit, including during validation.
    Statistics accumulate over calls to ``merge`` on this instance.

    Custom checkpoint layouts can load each payload with ``load`` and supply their
    own zero-argument merge callable. Application metadata/layout remains the
    caller's responsibility. Inputs must not be modified during merging.
    """

    def __init__(self, *, ram_budget_bytes: int = 0):
        if not isinstance(ram_budget_bytes, int) or ram_budget_bytes < 0:
            raise ValueError('ram_budget_bytes must be a nonnegative integer')
        if tuple(int(v) for v in torch.__version__.split('.')[:2]) < (2, 1):
            raise RuntimeError('Tensor checkpoint merging requires PyTorch >= 2.1')
        self.ram_budget_bytes = ram_budget_bytes
        self.stats = MergeStats()
        self._inputs: weakref.WeakValueDictionary[int, torch.UntypedStorage] = weakref.WeakValueDictionary()
        self._madvise = None
        if sys.platform == 'linux':
            self._madvise = ctypes.CDLL(None, use_errno=True).madvise
            self._madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            self._madvise.restype = ctypes.c_int

    def load(self, filename: str | Path) -> Any:
        """Load a trusted PyTorch ZIP checkpoint without eagerly reading tensors."""
        state = torch.load(filename, map_location='cpu', weights_only=False, mmap=True)
        regions = []
        if self._madvise is not None:
            file_stat = os.stat(filename)
            for line in Path('/proc/self/maps').read_text().splitlines():
                address, _, _, device, inode, *_ = line.split(maxsplit=5)
                major, minor = (int(part, 16) for part in device.split(':'))
                if int(inode) == file_stat.st_ino and os.makedev(major, minor) == file_stat.st_dev:
                    regions.append(tuple(int(part, 16) for part in address.split('-')))

        def register(value):
            if isinstance(value, torch.Tensor):
                if value.layout != torch.strided or value.device.type != 'cpu':
                    raise ValueError('Tensor merge requires CPU strided tensors')
                storage = value.untyped_storage()
                # Custom pickle reducers can create heap tensors even with mmap=True.
                # Advising their pages would corrupt the allocator or unrelated data.
                if any(start <= storage.data_ptr() and storage.data_ptr() + storage.nbytes() <= end
                       for start, end in regions):
                    self._inputs[storage._cdata] = storage
            elif isinstance(value, dict):
                for item in value.values():
                    register(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    register(item)

        register(state)
        return state

    def _release_input(self, storage: torch.UntypedStorage) -> None:
        if self._madvise is None or storage._cdata not in self._inputs or not storage.nbytes():
            return
        # Only advise known, unmodified file mappings, never ordinary heap tensors.
        page = mmap.PAGESIZE
        start = storage.data_ptr() // page * page
        end = (storage.data_ptr() + storage.nbytes() + page - 1) // page * page
        if self._madvise(start, end - start, mmap.MADV_DONTNEED):
            error = ctypes.get_errno()
            raise OSError(error, f'Cannot release checkpoint input pages: {os.strerror(error)}')

    def merge(self, merge_fn: Callable[[], Any], output_file: str | Path) -> MergeStats:
        """Atomically publish ``merge_fn()`` as a standard PyTorch checkpoint.

        The callback must only assemble metadata and call nnscaler merge APIs;
        loading, training, or arbitrary tensor computation belongs outside it.
        Existing output is unchanged on merge/write/fsync failure. No intermediate
        full checkpoint or tensor spill files are created.
        """
        output_file = Path(output_file)
        mode = _DeferredMerge(self.ram_budget_bytes, self.stats, self._release_input)
        try:
            with torch.no_grad(), mode:
                result = merge_fn()
            if check_recursively(
                result,
                lambda value: isinstance(value, torch.Tensor)
                and (value.device.type != 'cpu' or value.layout != torch.strided),
            ):
                raise ValueError('Tensor merge requires CPU strided tensors')
            with tempfile.NamedTemporaryFile(
                prefix=f'.{output_file.name}.', suffix='.tmp', dir=output_file.parent, delete=False,
            ) as output:
                temporary = Path(output.name)
                try:
                    with torch.serialization._open_zipfile_writer(output) as writer:
                        torch.serialization._save(
                            result, _StorageWriter(writer, mode), pickle,
                            torch.serialization.DEFAULT_PROTOCOL, False,
                        )
                    output.flush()
                    os.fsync(output.fileno())
                    os.replace(temporary, output_file)
                finally:
                    temporary.unlink(missing_ok=True)
            return self.stats
        finally:
            mode.versions.clear()
