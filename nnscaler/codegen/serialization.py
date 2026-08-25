#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Helpers for serializing code-generation state."""

from contextlib import contextmanager
import sys


# Partitioned training graphs can be substantially deeper than Python's
# default recursion limit. dill follows those object links recursively while
# writing and reading the multi-process codegen payload.
CODEGEN_PICKLE_RECURSION_LIMIT = 10_000


@contextmanager
def codegen_pickle_recursion_limit():
    """Temporarily allow dill to traverse a deeply partitioned graph."""
    previous_limit = sys.getrecursionlimit()
    if previous_limit < CODEGEN_PICKLE_RECURSION_LIMIT:
        sys.setrecursionlimit(CODEGEN_PICKLE_RECURSION_LIMIT)
    try:
        yield
    finally:
        if previous_limit < CODEGEN_PICKLE_RECURSION_LIMIT:
            sys.setrecursionlimit(previous_limit)
