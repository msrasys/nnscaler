# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Separate an operator's asynchronous submission from its autograd binding.

The submit callback returns an AsyncResult. Its wait
callback joins the submitted work to the consuming stream. The complete callback
binds the original inputs to the prepared outputs, including their backward.
Backends must protect storage used on their private streams until that join.
"""

from dataclasses import dataclass
from typing import Any, Callable

from .stream import input_stream_lifetime

_operators = set()


@dataclass
class AsyncResult:
    value: Any
    wait: Callable[[], None]
    # Opt in only when the backend already protects every input/output storage
    # on both its private streams and the caller, including opaque owners.
    manages_lifetime: bool = False


def register_async_op(operator, *, submit, complete):
    """Register an imported Python operator's submit/complete implementation.

    ``complete(prepared_value, *inputs, **kwargs)`` must preserve the operator's
    values and gradients. It must use the supplied inputs for autograd, since
    the compiler can add backward ordering identities after submission.
    """
    if not all(callable(fn) for fn in (operator, submit, complete)):
        raise TypeError('Async operator implementations must be callable')
    signature = f'{operator.__module__}.{operator.__name__}'
    implementation = (submit, complete)
    operator._nnscaler_async_op = implementation
    _operators.add(signature)


def registered_async_ops():
    return tuple(_operators)


class PendingOp:
    def __init__(self, operator, args, kwargs):
        try:
            submit, self._complete = operator._nnscaler_async_op
        except AttributeError as error:
            raise ValueError(f'{operator} has no async implementation') from error
        self._inputs = (args, kwargs)
        self._result = None
        self._waited = False
        self._closed = False
        result = submit(*args, **kwargs)
        if not isinstance(result, AsyncResult) or not callable(result.wait):
            raise TypeError('Async submission must return AsyncResult(value, wait)')
        if type(result.manages_lifetime) is not bool:
            raise TypeError('manages_lifetime must be a boolean')
        self._result = result

    def _wait(self):
        if not self._waited:
            self._result.wait()
            self._waited = True

    def complete(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError('An async operator can only complete once')
        try:
            self._wait()
            return self._complete(self._result.value, *args, **kwargs)
        finally:
            self.close()

    def close(self):
        """Keep submitted inputs/payload safe even if a side branch raises."""
        if self._closed:
            return
        self._wait()
        if not self._result.manages_lifetime:
            with input_stream_lifetime((self._inputs, self._result.value), True):
                pass
        self._closed = True
        self._inputs = self._result = self._complete = None
