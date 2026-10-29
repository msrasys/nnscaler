# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Saved-tensor adapters; activation storage and transport live below this layer."""
from contextlib import nullcontext
import sys

import torch
from torch.autograd.graph import saved_tensors_hooks


class _GradientAnchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        # One scalar carries shape/dtype and the incoming gradient edge. It
        # does not retain a view of the activation or save its payload.
        return torch.empty((), dtype=tensor.dtype, device=tensor.device).expand(tensor.shape)

    @staticmethod
    def backward(ctx, grad):
        return grad


class _RestoreGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, anchor, payload):
        return payload

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class _ModuleTensorRef:
    """Reconstruct excluded parameter/buffer views, including after ZeRO gather."""
    def __init__(self, owner, tensor):
        self.owner = owner
        self.shape, self.stride = tensor.shape, tensor.stride()
        self.offset = tensor.storage_offset()
        self.dtype = tensor.dtype
        self.version = owner._version

    def unpack(self):
        if self.owner._version != self.version:
            raise RuntimeError('A parameter saved for backward was modified by an inplace operation')
        if self.owner.dtype != self.dtype:
            return torch.empty(0, dtype=self.dtype, device=self.owner.device).set_(
                self.owner.untyped_storage(), self.offset, self.shape, self.stride)
        return torch.as_strided(self.owner, self.shape, self.stride, self.offset)


class SavedActivationContext:
    """Use a store for autograd saves, without storing gradient edges twice.

    ``exclude`` may contain parameters, buffers or other tensor owners that
    should remain resident. Ordinary views of those storages are also excluded.
    The NNScaler adapter supplies its dynamic parameter registry instead.
    """
    def __init__(self, store, *, exclude=()):
        self.store = store
        self._excluded = {}
        for tensor in exclude:
            if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
                raise TypeError('exclude must contain strided tensors')
            self._excluded[(tensor.device, tensor.untyped_storage()._cdata)] = tensor
        self._scope_context = self._hooks = None

    def _forward_context(self):
        batch = self.store._active_batch.get()
        return nullcontext(batch) if batch is not None else self.store.batch()

    def _make_group(self, scope):
        if scope.closed:
            raise RuntimeError('Activation batch has already been committed')
        return scope

    def __enter__(self):
        if self._scope_context is not None:
            raise RuntimeError('Saved activation context is already active')
        self._scope_context = self._forward_context()
        try:
            scope = self._scope_context.__enter__()
        except BaseException:
            self._scope_context = None
            raise
        try:
            self.group = self._make_group(scope)
            self._hooks = saved_tensors_hooks(self._pack, self._unpack)
            self._hooks.__enter__()
        except BaseException:
            self._scope_context.__exit__(*sys.exc_info())
            self._scope_context = self._hooks = None
            raise
        return self

    def _finish_forward(self, exc_type):
        pass  # The store's batch context commits or aborts on exit.

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            try:
                self._hooks.__exit__(exc_type, exc_value, traceback)
                self._finish_forward(exc_type)
            except BaseException:
                self._scope_context.__exit__(*sys.exc_info())
                raise
            return self._scope_context.__exit__(exc_type, exc_value, traceback)
        finally:
            # Hooks retain bound callbacks. Break that cycle even when commit
            # fails and training disables Python's cyclic garbage collector.
            self._hooks = self._scope_context = None

    def _pack(self, tensor):
        if tensor.layout == torch.strided:
            owner = self._excluded.get((tensor.device, tensor.untyped_storage()._cdata))
            if owner is not None:
                return _ModuleTensorRef(owner, tensor)
        return self.store._capture(tensor, self.group)

    @staticmethod
    def _unpack(value):
        return value.unpack()

    def prefetch(self):
        self.group.prefetch()
