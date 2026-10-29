# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Input identities and gradient targets that do not own activation storage."""
import weakref

import torch


class _TensorIdentity:
    def __init__(self, tensor):
        self.original = weakref.ref(tensor)


# Only live references own identities. The registry never owns a tensor or
# annotates caller tensors, and a recycled Python id is checked against its weak
# referent before reuse. Weak values remove entries when the last handle dies.
_tensor_identities = weakref.WeakValueDictionary()


class InputReference:
    """An input's identity after its final scheduled data read.

    A weak reference distinguishes object-id reuse and preserves retain_grad
    behavior when the caller still owns the original tensor. It must never be
    passed to a forward/data consumer in place of a tensor.
    """
    def __init__(self, tensor):
        self.tensor_id = id(tensor)
        identity = _tensor_identities.get(self.tensor_id)
        if identity is None or identity.original() is not tensor:
            identity = _TensorIdentity(tensor)
            _tensor_identities[self.tensor_id] = identity
        self.identity = identity
        self.original = identity.original

    def record_stream(self, stream):
        """Match tensor stream bookkeeping; this identity owns no GPU storage."""
        pass


class OutputReference(InputReference):
    """A backward root after its final data read, without owning its payload.

    This is also an input identity if another local segment consumed the same
    tensor. Autograd continues to own any values its backward actually needs.
    """
    def __init__(self, tensor, edge):
        super().__init__(tensor)
        self.edge = edge

    @property
    def backward_identity(self):
        # Preserve Tensor deduplication while the object is alive, and avoid
        # confusing its recycled Python id with a later tensor after release.
        return self.tensor_id if self.original() is not None else self.identity


class InputGradSlot:
    """Backward writes dInput here without retaining a detached input leaf."""
    requires_grad = True

    def __init__(self, tensor):
        self.shape = tensor.shape
        self.strides = tensor.stride()
        self.dtype = tensor.dtype
        self.device = tensor.device
        self.grad = None
        # Match AccumulateGrad's layout contract without keeping the input.
        # Dense permutations preserve their strides; sliced/expanded inputs
        # receive a contiguous gradient. Unit dimensions do not affect density.
        self.dense = True
        expected = 1
        for stride, size in sorted((stride, size) for size, stride in zip(self.shape, self.strides) if size > 1):
            if stride != expected:
                self.dense = False
                break
            expected *= size

    def accumulate(self, grad):
        if self.grad is None:
            if self.dense:
                self.grad = torch.empty_strided(self.shape, self.strides, dtype=self.dtype, device=self.device)
            else:
                self.grad = torch.empty(self.shape, dtype=self.dtype, device=self.device)
            # Own the gradient, including when SumBackward supplies an expanded
            # view of the incoming cotangent. Forwarding that borrowed view
            # would expose overlapping strides and allow later writes to alter
            # another gradient consumer's input.
            self.grad.copy_(grad)
        else:
            self.grad.add_(grad)

    def retain_grad(self):
        pass  # The boundary's backward explicitly captures the gradient.


class _InputBoundary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, anchor, slot):
        ctx.slot = slot
        return value

    @staticmethod
    def backward(ctx, grad):
        ctx.slot.accumulate(grad)
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


def supports_output_references(hook):
    return hook is None or getattr(hook, '_nnscaler_supports_output_references', False) is True
