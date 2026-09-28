#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Helpers for serializing code-generation state."""

from contextlib import contextmanager
import dis
import logging
import pickle
import sys
import types
from typing import Any, BinaryIO

import cloudpickle
import dill


# Partitioned training graphs can be substantially deeper than Python's
# default recursion limit. Picklers follow those object links recursively while
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


class _CodegenPickler(cloudpickle.CloudPickler):
    def reducer_override(self, obj):
        # Cloudpickle preserves captured objects, but recreates closure cells
        # separately for each function. Rebinding a shared nonlocal would then
        # silently change behavior. Keep dill's shared-cell memo in that case.
        if isinstance(obj, types.FunctionType) and obj.__closure__:
            freevars = obj.__code__.co_freevars
            if any(inst.opname in ('STORE_DEREF', 'DELETE_DEREF') and inst.argval in freevars
                   for inst in dis.get_instructions(obj)):
                raise pickle.PicklingError('codegen function rebinds captured state')
        return super().reducer_override(obj)


def dump_codegen_payload(payload: dict[str, Any], stream: BinaryIO) -> str:
    """Serialize one shared object graph, falling back to legacy dill if needed.

    Keep one pickler memo for functions and graph objects so captured mutable
    state retains its aliases. I/O errors propagate without retrying writes.
    """
    offset = stream.tell()
    with codegen_pickle_recursion_limit():
        try:
            _CodegenPickler(stream, protocol=pickle.HIGHEST_PROTOCOL).dump(payload)
        except (pickle.PicklingError, AttributeError, TypeError, RecursionError) as exc:
            _logger.info('Cloudpickle unavailable for codegen payload; falling back to dill: %s', exc)
            stream.seek(offset)
            stream.truncate()
            dill.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            return 'dill'
    return 'cloudpickle'


def load_codegen_payload(stream: BinaryIO) -> dict[str, Any]:
    """Read cloudpickle payloads and legacy/fallback dill payloads."""
    with codegen_pickle_recursion_limit():
        return dill.load(stream)
