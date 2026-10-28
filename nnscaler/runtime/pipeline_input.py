# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Input identities and gradient targets that do not own activation storage."""
import weakref

import torch


class InputReference:
    """An input's identity after its final scheduled data read.

    A weak reference distinguishes object-id reuse and preserves retain_grad
    behavior when the caller still owns the original tensor. It must never be
    passed to a forward/data consumer in place of a tensor.
    """
    def __init__(self, tensor):
        self.tensor_id = id(tensor)
        self.original = weakref.ref(tensor)


class InputGradSlot:
    """Backward writes dInput here without retaining a detached input leaf."""
    requires_grad = True

    def __init__(self, tensor):
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.grad = None

    def retain_grad(self):
        pass  # The boundary's backward explicitly captures the gradient.


class _InputBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, anchor, slot):
        ctx.slot = slot
        return value

    @staticmethod
    def backward(ctx, grad):
        slot = ctx.slot
        slot.grad = grad if slot.grad is None else slot.grad + grad
        return None, None, None


def make_releasable_input(tensor):
    # Record the forward consumer even if no operation saves this input.
    # Releasing the caller's reference must not race a different allocator
    # stream while the forward kernels still read its storage.
    if tensor.device.type == 'cuda':
        tensor.record_stream(torch.cuda.current_stream(tensor.device))
    slot = InputGradSlot(tensor)
    anchor = torch.empty(0, device=tensor.device, dtype=tensor.dtype, requires_grad=True)
    value = _InputBoundary.apply(tensor.detach(), anchor, slot)
    return value, slot


def release_input_tensor(tensor):
    """Drop a schedule-local data reference, never resize or mutate its value."""
    if isinstance(tensor, InputReference):
        return tensor
    if not torch.is_tensor(tensor) or not tensor.requires_grad:
        return tensor
    return InputReference(tensor)


def input_id(value):
    return value.tensor_id if isinstance(value, InputReference) else id(value)


def saved_input_id(key):
    return key.tensor_id if isinstance(key, InputReference) else key


def is_grad_target(value):
    return isinstance(value, (torch.Tensor, InputGradSlot)) and value.requires_grad


def supports_input_slots(hook):
    return hook is None or getattr(hook, '_nnscaler_supports_input_grad_slots', False) is True
