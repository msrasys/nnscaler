# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Saved-tensor adapters; activation storage and transport live below this layer."""
from contextlib import contextmanager, nullcontext
import sys
import threading
import weakref

import torch
from torch.autograd.graph import saved_tensors_hooks


class CheckpointSubmissionPolicy:
    """Avoid early copies for checkpoints observed to retain no saved inputs.

    This changes submission timing only. A checkpoint that starts saving inputs
    again falls back to ordinary group commit on that invocation. Code keys are
    weak and never retain functions, closures, modules or activation tensors.
    """
    def __init__(self):
        self._retains_inputs = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()

    def early(self, code):
        if code is None:
            return True
        with self._lock:
            return self._retains_inputs.get(code, True)

    def observe(self, code, saved_inputs):
        # No admitted CUDA inputs provides no evidence about replay. This can
        # happen on a resident pipeline tail or with a smaller input shape.
        if code is not None and saved_inputs:
            # A different checkpoint or handle can still own the same storage.
            # Only this invocation's saved-value tokens establish replay needs.
            retained = any(ref() is not None for ref in saved_inputs)
            with self._lock:
                self._retains_inputs[code] = retained


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

    def checkpoint(self, function, *args, **kwargs):
        """Checkpoint under this saved-tensor context, optionally submitting early.

        PyTorch saves the boundary inputs before entering the forward context.
        In checkpoint submission mode, send that batch before the region's
        compute is launched, so compute can overlap the immutable D2H reads.
        Replay uses an empty context and never offloads its intermediates.
        Other submission modes preserve their ordinary region/scope boundary.

        A first early submission can copy an input even if the function needs
        no replay. Later calls of that code defer to group commit until saved
        inputs are observed again. Dynamic branches remain correct either way.
        """
        from torch.utils.checkpoint import checkpoint

        if self._hooks is None:
            raise RuntimeError('Checkpoint requires an active saved activation context')
        kwargs.setdefault('use_reentrant', False)
        policy = None
        if self.store.config.forward_commit_unit == 'checkpoint':
            if kwargs['use_reentrant'] or 'context_fn' in kwargs:
                raise ValueError('Early checkpoint offload requires non-reentrant checkpoint '
                                 'without a custom context_fn')
            policy = self.store.checkpoint_submission
            code = getattr(function, '__code__', None)
            if policy.early(code):
                kwargs['context_fn'] = self._checkpoint_contexts
        # Preserve PyTorch's default context_fn (and debug support) unless
        # early submission was explicitly selected.
        if policy is None:
            return checkpoint(function, *args, **kwargs)
        with self._checkpoint_input_tokens() as saved_inputs:
            result = checkpoint(function, *args, **kwargs)
        policy.observe(code, saved_inputs)
        return result

    @contextmanager
    def _checkpoint_input_tokens(self):
        from .activation_store import _SavedTensor

        saved_inputs = []

        def pack(tensor):
            value = self._pack(tensor)
            if isinstance(value, _SavedTensor):
                saved_inputs.append(weakref.ref(value))
            return value

        # The checkpoint's internal hooks still own replay intermediates. This
        # outer adapter only observes eligible boundary saves, without retaining
        # their tokens, storages or autograd graph. Exclusions remain unchanged.
        with saved_tensors_hooks(pack, self._unpack):
            yield saved_inputs

    def _checkpoint_contexts(self):
        return self._submit_checkpoint_inputs(), nullcontext()

    @contextmanager
    def _submit_checkpoint_inputs(self):
        self.group.flush()
        yield
