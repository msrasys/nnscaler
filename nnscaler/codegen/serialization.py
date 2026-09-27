#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Helpers for serializing code-generation state."""

from contextlib import contextmanager
import importlib
import logging
import pickle
import sys
import types
from typing import Any, BinaryIO, Optional

import dill


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
    return dill.loads(fallback)


def _load_dill_callable(data: bytes):
    return dill.loads(data)


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


class _CodegenPickler(pickle.Pickler):
    """Use C pickle for the graph and dill only for non-importable functions."""

    def __init__(self, stream: BinaryIO, payload: dict[str, Any]):
        super().__init__(stream, protocol=pickle.HIGHEST_PROTOCOL)
        from nnscaler.graph.parser.register import CustomizedOps

        references = {}
        self._factories = {}
        self._factory_fallbacks = {}
        self._registered_factories = {}
        self._closure_owners = {}
        for signature, factory in CustomizedOps.kOpMap.items():
            module_name = _registered_op_module(CustomizedOps.kOpRuntime.get(signature))
            if module_name is not None:
                references[signature] = (module_name, signature)
                self._factories[id(factory)] = references[signature]
                self._registered_factories[signature] = factory

        # graph.ckp recreates local factory functions through dill. Their
        # identities differ from kOpMap even though their signatures match.
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
                return _load_dill_callable, (dill.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL),)
        return NotImplemented


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
    """Read both fast pickle payloads and legacy/fallback dill payloads."""
    with codegen_pickle_recursion_limit():
        return dill.load(stream)
