#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Helpers for serializing code-generation state."""

from contextlib import contextmanager
import logging
import pickle
import sys
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


def dump_codegen_payload(payload: dict[str, Any], stream: BinaryIO) -> str:
    """Serialize one shared object graph, falling back to legacy dill if needed.

    Keep one pickler memo for functions and graph objects so captured mutable
    state retains its aliases. I/O errors propagate without retrying writes.
    """
    offset = stream.tell()
    with codegen_pickle_recursion_limit():
        try:
            cloudpickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
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
