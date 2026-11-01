# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Publish one immutable CUDA completion for a batch of activation transfers."""
import threading

import torch


class TransferFence:
    """A batch event cannot be consumed before its record has been submitted.

    CUDA treats a wait on an unrecorded event as already complete. Saved values
    may be inspected by another autograd thread while their batch is still
    being submitted, so publication is a separate, CPU-side state transition.
    Waiting for publication never waits for DMA completion.
    """

    def __init__(self, event=None):
        self.event = torch.cuda.Event() if event is None else event
        self.used = False
        self._owner = threading.get_ident()
        self._published = threading.Event()
        self._failure = None
        self._complete = False

    def publish(self, stream):
        if self._published.is_set():
            raise RuntimeError('Activation transfer fences cannot be re-recorded')
        try:
            self.event.record(stream)
        except BaseException as error:
            # Do not retain traceback frames containing activation tensors.
            self._failure = str(error)
            raise
        finally:
            self._published.set()

    def cuda_event(self):
        if not self._published.is_set():
            if threading.get_ident() == self._owner:
                raise RuntimeError('Cannot consume an activation transfer before its batch closes')
            self._published.wait()
        if self._failure is not None:
            raise RuntimeError('Activation transfer event publication failed: ' + self._failure)
        return self.event

    def query(self):
        # Polling an unpublished or failed transfer must never recycle memory.
        if not self._complete and self._published.is_set() and self._failure is None:
            if self.event.query():
                # A slow concurrent query may return an older False result
                # after another reader observed completion. Never regress.
                self._complete = True
        return self._complete

    def synchronize(self):
        self.cuda_event().synchronize()
        self._complete = True


def cuda_event(dependency):
    """Accept native events as well as published activation batch fences."""
    return dependency.cuda_event() if isinstance(dependency, TransferFence) else dependency
