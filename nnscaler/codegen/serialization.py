#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Helpers for serializing code-generation state."""

from contextlib import contextmanager
import importlib
import io
import logging
import pickle
import sys
import types
from typing import Any, BinaryIO, Optional

import dill
import torch


# Partitioned training graphs can be substantially deeper than Python's
# default recursion limit. dill follows those object links recursively while
# writing and reading the multi-process codegen payload.
CODEGEN_PICKLE_RECURSION_LIMIT = 10_000
_logger = logging.getLogger(__name__)


@contextmanager
def codegen_pickle_recursion_limit():
    """Temporarily allow pickle/dill to traverse a deeply partitioned graph."""
    previous_limit = sys.getrecursionlimit()
    if previous_limit < CODEGEN_PICKLE_RECURSION_LIMIT:
        sys.setrecursionlimit(CODEGEN_PICKLE_RECURSION_LIMIT)
    try:
        yield
    finally:
        if previous_limit < CODEGEN_PICKLE_RECURSION_LIMIT:
            sys.setrecursionlimit(previous_limit)


def _importable_module(obj: Any) -> Optional[str]:
    """Return the module when ordinary pickle can import this exact object."""
    module_name = getattr(obj, '__module__', None)
    qualname = getattr(obj, '__qualname__', '')
    module = sys.modules.get(module_name)
    if (module_name in (None, '__main__', '__mp_main__')
            or module is None or getattr(module, '__spec__', None) is None
            or not qualname or '<locals>' in qualname):
        return None
    value = module
    try:
        for part in qualname.split('.'):
            value = getattr(value, part)
    except AttributeError:
        return None
    if value is obj or (isinstance(obj, types.MethodType) and value == obj):
        return module_name
    return None


def _registered_op_module(runtime_fn: Any) -> Optional[str]:
    # Function.apply is inherited from torch.autograd.Function: import the
    # registered subclass's module, not torch.autograd.function.
    if isinstance(runtime_fn, types.MethodType) and isinstance(runtime_fn.__self__, type):
        return _importable_module(runtime_fn.__self__)
    return _importable_module(runtime_fn)


def _load_registered_factory(module_name: str, signature: str, fallback: bytes):
    from nnscaler.graph.parser.register import CustomizedOps

    importlib.import_module(module_name)
    # A module may expose its runtime function globally but register the op
    # only from a setup function in the parent process.
    if signature in CustomizedOps.kOpMap:
        return CustomizedOps.kOpMap[signature]
    return _CodegenUnpickler(io.BytesIO(fallback)).load()


def _load_dill_callable(data: bytes):
    return _CodegenUnpickler(io.BytesIO(data)).load()


def _load_codegen_storage(data: bytes):
    # Codegen inspects tensor values but never executes the traced graph.
    # Restoring CUDA storages in every worker duplicates them on the original
    # device (usually cuda:0), exhausting GPU memory even on small payloads.
    return torch.load(io.BytesIO(data), map_location='cpu', weights_only=False)


class _CodegenUnpickler(dill.Unpickler):
    def find_class(self, module: str, name: str):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return _load_codegen_storage
        return super().find_class(module, name)


def _can_copy_callable_state(value: Any) -> bool:
    """Whether a separate dill memo cannot split mutable captured state."""
    if type(value) in (type(None), bool, int, float, complex, str, bytes):
        return True
    if type(value) in (tuple, frozenset):
        return all(_can_copy_callable_state(item) for item in value)
    if isinstance(value, types.ModuleType):
        return value.__name__ not in ('__main__', '__mp_main__') and sys.modules.get(value.__name__) is value
    if isinstance(value, (types.FunctionType, type)):
        return _importable_module(value) is not None
    return False


def _can_copy_closure_cell(function, name: str, value: Any) -> bool:
    if _can_copy_callable_state(value):
        return True
    # Reshape partition rules capture private dimension lookup tables. They
    # are lists in existing graph caches, but the modifier only reads them;
    # neither table escapes _reshape_anno except through this closure.
    return (
        function.__module__ == 'nnscaler.graph.function.function'
        and function.__qualname__ == '_reshape_anno.<locals>.modifier'
        and name in ('ifirst', 'ofirst')
        and type(value) is list
        and all(item is None or type(item) is str for item in value)
    )


_EMPTY_CELL = object()


def _exact_key(value: Any) -> Any:
    """A hashable key equal only for values dill serializes identically.

    Tags every value with its type (1, 1.0 and True compare equal) and spells
    floats with repr (-0.0 == 0.0). Only state accepted by the copy checks
    reaches this: immutable scalars, tuples/frozensets of them, read-only lists
    of strings, modules, importable functions and types (by identity).
    """
    kind = type(value)
    if kind in (list, tuple):
        return kind, tuple(_exact_key(item) for item in value)
    if kind is frozenset:
        return kind, frozenset(_exact_key(item) for item in value)
    if kind in (float, complex):
        return kind, repr(value)
    if kind in (type(None), bool, int, str, bytes):
        return kind, value
    if isinstance(value, (types.ModuleType, types.FunctionType, type)):
        return kind, id(value)
    raise TypeError(f'no exact key for {kind.__name__}')


def _callable_key(obj: types.FunctionType) -> Optional[tuple]:
    """Key of everything dill writes for a function, or None if not exact."""
    try:
        cells = []
        for cell in obj.__closure__ or ():
            try:
                cells.append(_exact_key(cell.cell_contents))
            except ValueError:  # empty cell
                cells.append(_EMPTY_CELL)
        return (
            id(obj.__code__), id(obj.__globals__), obj.__module__, obj.__name__, obj.__qualname__,
            _exact_key(obj.__doc__), tuple(cells),
            _exact_key(tuple(obj.__defaults__ or ())),
            _exact_key(tuple((obj.__kwdefaults__ or {}).items())),
            _exact_key(tuple(obj.__dict__.items())),
            _exact_key(tuple(getattr(obj, '__annotations__', {}).items())),
        )
    except TypeError:
        return None


class _CodegenPickler(pickle.Pickler):
    """Use C pickle for the graph and dill only for non-importable functions."""

    def __init__(self, stream: BinaryIO, payload: dict[str, Any], graph: Any = None):
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        from nnscaler.graph.parser.register import CustomizedOps

        references = {}
        self._factories = {}
        self._factory_fallbacks = {}
        self._registered_factories = {}
        self._closure_owners = {}
        # Many graph nodes carry distinct closures of one local function with
        # equal state (e.g. 6k `_reshape_anno` modifiers, 5 distinct dumps).
        # Reuse their dill bytes; the entry keeps the code and globals alive
        # so their ids in the key stay unique.
        self._callable_dumps = {}
        for signature, factory in CustomizedOps.kOpMap.items():
            module_name = _registered_op_module(CustomizedOps.kOpRuntime.get(signature))
            if module_name is not None:
                references[signature] = (module_name, signature)
                self._factories[id(factory)] = references[signature]
                self._registered_factories[signature] = factory

        # graph.ckp recreates local factory functions through dill. Their
        # identities differ from kOpMap even though their signatures match.
        if graph is None:
            module_codegen = payload.get('module_codegen')
            graph = getattr(getattr(module_codegen, 'execplan', None), 'graph', None)
        if graph is not None:
            for node in graph.nodes(flatten=True):
                factory = getattr(node, '_create_fn', ())
                reference = references.get(getattr(node, 'signature', None))
                if factory and reference is not None:
                    self._factories[id(factory[0])] = reference

    def reducer_override(self, obj):
        if isinstance(obj, types.FunctionType):
            reference = self._factories.get(id(obj))
            if reference is not None:
                signature = reference[1]
                if signature not in self._factory_fallbacks:
                    self._factory_fallbacks[signature] = dill.dumps(
                        self._registered_factories[signature], protocol=pickle.HIGHEST_PROTOCOL)
                return _load_registered_factory, (*reference, self._factory_fallbacks[signature])
            if _importable_module(obj) is None:
                # Separately dumping functions that share closure cells would
                # split their state. Preserve the legacy whole-payload memo
                # in that case instead.
                for name, cell in zip(obj.__code__.co_freevars, obj.__closure__ or ()):
                    owner = self._closure_owners.setdefault(id(cell), obj)
                    if owner is not obj:
                        raise pickle.PicklingError('codegen callables share closure state')
                    try:
                        value = cell.cell_contents
                    except ValueError:  # An empty closure cell has no state to share.
                        continue
                    if not _can_copy_closure_cell(obj, name, value):
                        raise pickle.PicklingError(
                            f'codegen callable {obj.__qualname__} captures {type(value).__name__} state')
                state = (*(obj.__defaults__ or ()), *(obj.__kwdefaults__ or {}).values(), *obj.__dict__.values())
                if not all(_can_copy_callable_state(value) for value in state):
                    raise pickle.PicklingError('codegen callable has mutable defaults or attributes')
                key = _callable_key(obj)
                if key is None:
                    return _load_dill_callable, (dill.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL),)
                entry = self._callable_dumps.get(key)
                if entry is None:
                    entry = self._callable_dumps[key] = (
                        dill.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL), obj.__code__, obj.__globals__)
                return _load_dill_callable, (entry[0],)
        return NotImplemented


def try_dumps_graph(obj: Any, graph: Any) -> Optional[bytes]:
    """Pickle `obj` (which contains `graph`) with the C pickler, or return None.

    The result loads with `dill.load`: graph nodes and tensors pickle as with dill,
    and only non-importable functions are embedded as dill bytes. Callers fall
    back to `dill.dumps` when this returns None.
    """
    stream = io.BytesIO()
    with codegen_pickle_recursion_limit():
        try:
            _CodegenPickler(stream, {}, graph=graph).dump(obj)
        except (pickle.PicklingError, AttributeError, TypeError, RecursionError) as exc:
            _logger.info('Fast graph serialization unavailable; falling back to dill: %s', exc)
            return None
    return stream.getvalue()


def dump_codegen_payload(payload: dict[str, Any], stream: BinaryIO) -> str:
    """Write a worker payload to a seekable stream, returning the serializer.

    Registered factories are restored by importing their defining modules.
    Unsupported objects (e.g. local classes) retain the legacy dill path.
    Rewind any partial pickle before fallback; workers never see a mixed file.
    """
    offset = stream.tell()
    with codegen_pickle_recursion_limit():
        try:
            _CodegenPickler(stream, payload).dump(payload)
        except (pickle.PicklingError, AttributeError, TypeError, RecursionError) as exc:
            _logger.info('Fast codegen serialization unavailable; falling back to dill: %s', exc)
            stream.seek(offset)
            stream.truncate()
            dill.dump(payload, stream)
            return 'dill'
    return 'pickle'


def load_codegen_payload(stream: BinaryIO) -> dict[str, Any]:
    """Read fast and legacy payloads with tensor storage restored on CPU.

    Device metadata in the IR is preserved. This mapping is limited to worker
    payloads; ordinary model and checkpoint loading retain their device rules.
    """
    with codegen_pickle_recursion_limit():
        return _CodegenUnpickler(stream).load()
