#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from typing import Any, TypedDict
import pickle
import base64
import copy

import torch
from safetensors.torch import save_file
from safetensors import safe_open

from nnscaler.utils import transform_recursively, check_recursively
from nnscaler.version import __version__


class MetadataDict(TypedDict):
    obj: str
    nnscaler: str


class _Index:
    def __init__(self, index: int):
        self.index = index

    def __repr__(self):
        return f"_Index({self.index})"


def save(obj: Any, f, *, format="safetensors") -> None:
    """
    Saves an object containing tensors into a safetensors file.
    Args:
        obj (`Any`):
            The object you want to save. It can be a nested structure containing
            tensors, lists, tuples, and dictionaries.
        f:
            The file-like object or filename where to save the safetensors file.
        format (`str`, *optional*, defaults to `"safetensors"`):
            The format to save the object. Currently `"safetensors"` and `"pt"` is supported.
    """
    if format == 'pt':
        torch.save(obj, f)
        return

    if format != 'safetensors':
        raise ValueError(f"Unsupported format: {format}")

    index = 0

    # all tensors to be saved
    tensors = {}
    # detect shared tensors
    # because safetensors does not support shared tensors, we need to
    # save shared tensors only once and replace other occurrences
    # TODO: Currently we only detect shared tensors that are exactly the same
    # (i.e., share the same data_ptr and shape and stride).
    # We may improve it in the future if needed.
    # key: (tensor.data_ptr(), tensor.shape, tensor.stride()), value: _Index
    tensor_ids: dict[tuple[int, tuple[int, ...], tuple[int, ...]], _Index] = {}
    def transform_fn(o: Any) -> Any:
        nonlocal index
        if isinstance(o, torch.Tensor):
            key = (o.data_ptr(), o.shape, o.stride())
            if key in tensor_ids:
                idx = tensor_ids[key]
            else:
                idx = _Index(index)
                tensor_ids[key] = idx
                tensors[f'{index}'] = o
                index += 1
            return idx
        return o
    metadata = transform_recursively(obj, transform_fn, target_types=(torch.Tensor,))
    save_file(tensors, f, metadata={
        'obj': base64.b64encode(pickle.dumps(metadata)).decode('utf-8'),
        'nnscaler': __version__
    })


class _LazyContainer:
    """
    Mock class for dictionary, list, and tuple that loads tensors lazily from safetensors file.
    """
    def __init__(self, data: dict | tuple | list, tensors: safe_open):
        self.data = data
        self.tensors = tensors

    def __getitem__(self, key):
        return self._v(self.data[key])

    def __setitem__(self, key, value):
        raise NotImplementedError("Lazy containers are read-only.")

    def __delitem__(self, key):
        raise NotImplementedError("Lazy containers are read-only.")

    def pop(self, key, default=None):
        raise NotImplementedError("Lazy containers are read-only.")

    def __len__(self):
        return len(self.data)

    def __contains__(self, item):
        return self.data.__contains__(item)

    def get(self, key, default=None):
        return self._v(self.data.get(key, default))

    def keys(self):
        return self.data.keys()

    def values(self):
        return map(self._v, self.data.values())

    def items(self):
        return ((k, self._v(v)) for k, v in self.data.items())

    def _v(self, v):
        return _wrap_value(v, self.tensors)

    def load_all(self):
        def _load(v):
            if isinstance(v, _Index):
                return self.tensors.get_tensor(f'{v.index}')
            return v
        return transform_recursively(self.data, _load, target_types=(_Index,))

    def __copy__(self):
        return copy.copy(self.load_all())

    def __deepcopy__(self, memo):
        return copy.deepcopy(self.load_all(), memo)

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.data)})"


class _LazyList(_LazyContainer, list):
    pass


class _LazyDict(_LazyContainer, dict):
    pass


class _LazyTuple(_LazyContainer, tuple):
    # tuple is immutable, so we need to override __new__
    def __new__(cls, *args, **kwargs):
        return tuple.__new__(cls, ())


def _wrap_value(v: Any, tensors: safe_open) -> Any:
    if isinstance(v, _Index):
        return tensors.get_tensor(f'{v.index}')
    if not check_recursively(v, lambda k: isinstance(k, _Index)):
        return v
    if isinstance(v, dict):
        return _LazyDict(v, tensors)
    if isinstance(v, list):
        return _LazyList(v, tensors)
    if isinstance(v, tuple):
        return _LazyTuple(v, tensors)
    # should not reach here
    return v


class LazyLoader:
    def __init__(self, filename, device="cpu"):
        self.filename = filename
        self.device = device
        self.tensor_loader = safe_open(self.filename, framework="pt", device=self.device)
        self.tensors = None
        self.data = None

    def __enter__(self):
        self.tensors = self.tensor_loader.__enter__()
        metadata: MetadataDict = self.tensors.metadata()
        metadata_obj_b64 = metadata['obj']
        self.data = pickle.loads(base64.b64decode(metadata_obj_b64.encode('utf-8')))
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.tensor_loader.__exit__(_exc_type, _exc_value, _traceback)

    def get_lazy_data(self) -> _LazyContainer | Any:
        if self.tensors is None:
            raise RuntimeError("LazyLoader context is not entered.")
        return _wrap_value(self.data, self.tensors)


def load(f, *, device="cpu", format="safetensors", lazy=False) -> LazyLoader | Any:
    """
    Loads an object containing tensors from a safetensors file lazily.
    Args:
        f: The file-like object or filename from which to load the safetensors file.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device where the tensors will be loaded.
        lazy (`bool`, *optional*, defaults to `False`):
            If set to `False`, loads all tensors into memory eagerly.
    Returns:
        (`LazyLoader` | `Any`):
            The lazy loader object that can be used to access the data.
            If `lazy` is set to `False`, returns the loaded object with all tensors
            loaded into memory.
    """
    if format == 'pt':
        return torch.load(f, map_location=device, weights_only=False)
    if format != 'safetensors':
        raise ValueError(f"Unsupported format: {format}")

    if not lazy:
        with LazyLoader(f, device=device) as loader:
            data = loader.get_lazy_data()
            assert isinstance(data, _LazyContainer)
            return data.load_all()
    return LazyLoader(f, device=device)


def convert(src: str, dst: str, *, src_format="safetensors", dst_format="pt", device="cpu") -> None:
    """
    Converts a serialized file from one format to another.
    Args:
        src (`str`):
            The source filename.
        dst (`str`):
            The destination filename.
        src_format (`str`, *optional*, defaults to `"safetensors"`):
            The format of the source file. Currently `"safetensors"` and `"pt"` is supported.
        dst_format (`str`, *optional*, defaults to `"pt"`):
            The format of the destination file. Currently `"safetensors"` and `"pt"` is supported.
        device (`str`, *optional*, defaults to `"cpu"`):
            The device where the tensors will be loaded.

    Returns:
        (`None`):
            This function does not return anything.
    """
    if src_format == dst_format:
        raise ValueError("Source and destination formats are the same.")

    save(
        load(src, device=device, format=src_format, lazy=False),
        dst,
        format=dst_format
    )
