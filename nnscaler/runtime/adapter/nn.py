#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
This module offers autograd functions for communication
primitives. This is typically used in the training with tensor 
parallelism scenario.
"""

from typing import List, Tuple
import weakref
import torch

from nnscaler.flags import RuntimeFlag
from nnscaler.profiler.timer import CudaTimer
from nnscaler.runtime.device import DeviceGroup
from .collectives import (
    all_reduce,
    all_gather,
    reduce_scatter,
    all_to_all,
    all_to_all_single,
    chunk
)


_DEFERRED_IDENTITY_ALLREDUCE_GRADS = {}
_DEFERRED_IDENTITY_ALLREDUCE_EXPECTED_COUNTS = {}


def _launch_deferred_identity_allreduce(entry) -> None:
    if entry[3]:
        return
    param, ranks = entry[:2]
    all_reduce(param.grad, ranks, async_op=True, clone=False)
    entry[3] = True


def defer_identity_allreduce_grad(
    param: torch.nn.Parameter,
    ranks: Tuple[int, ...],
    *,
    ready: bool = True,
) -> None:
    """Defer a replicated leaf-gradient reduction until the FBW step ends.

    Split-W kernels can accumulate every microbatch directly into the final
    leaf ``.grad``. Reducing that storage after each contribution would reduce
    earlier contributions repeatedly, while reducing a separate contribution
    requires a full-size temporary. Delaying the linear reduction lets us use
    one in-place collective on the accumulated final gradient, matching the
    main-grad lifecycle used by Megatron gradient-accumulation fusion.
    """
    if not (
        isinstance(param, torch.nn.Parameter)
        and param.is_leaf
        and param.requires_grad
    ):
        raise TypeError("deferred identity all-reduce requires a trainable leaf parameter")
    ranks = tuple(ranks)
    key = id(param)
    entry = _DEFERRED_IDENTITY_ALLREDUCE_GRADS.get(key)
    if entry is not None:
        existing_param, existing_ranks = entry[:2]
        if existing_param is not param or existing_ranks != ranks:
            raise RuntimeError(
                "Conflicting deferred identity all-reduce registration for parameter"
            )
        if ready and entry[3]:
            raise RuntimeError(
                "Deferred identity all-reduce received a gradient after communication started"
            )
    else:
        entry = [param, ranks, 0, False]
        _DEFERRED_IDENTITY_ALLREDUCE_GRADS[key] = entry

    if not ready:
        return
    entry[2] += 1

    expected = _DEFERRED_IDENTITY_ALLREDUCE_EXPECTED_COUNTS.get(key)
    if expected is None:
        return
    expected_param_ref, expected_ranks, expected_count = expected
    if expected_param_ref() is not param or expected_ranks != ranks:
        _DEFERRED_IDENTITY_ALLREDUCE_EXPECTED_COUNTS.pop(key, None)
        return
    if entry[2] == expected_count:
        _launch_deferred_identity_allreduce(entry)
    elif entry[2] > expected_count:
        raise RuntimeError(
            "Deferred identity all-reduce contribution count exceeded the previous step: "
            f"expected={expected_count}, actual={entry[2]}"
        )


def mark_deferred_identity_allreduce_grad_ready(param: torch.nn.Parameter) -> bool:
    """Mark one registered leaf-gradient contribution as fully accumulated."""
    entry = _DEFERRED_IDENTITY_ALLREDUCE_GRADS.get(id(param))
    if entry is None or entry[0] is not param:
        return False
    defer_identity_allreduce_grad(param, entry[1], ready=True)
    return True


@torch.no_grad()
def flush_deferred_identity_allreduce_grads() -> None:
    """Launch any remaining reductions and learn each leaf's ready count."""
    pending = tuple(_DEFERRED_IDENTITY_ALLREDUCE_GRADS.items())
    for key, entry in pending:
        param, ranks, contribution_count, launched = entry
        if param.grad is None:
            raise RuntimeError(
                "Deferred identity all-reduce parameter has no accumulated gradient"
            )
        expected = _DEFERRED_IDENTITY_ALLREDUCE_EXPECTED_COUNTS.get(key)
        if expected is not None:
            expected_param_ref, expected_ranks, expected_count = expected
            if (
                expected_param_ref() is param
                and expected_ranks == ranks
                and launched
                and contribution_count != expected_count
            ):
                raise RuntimeError(
                    "Deferred identity all-reduce contribution count changed after "
                    "communication started: "
                    f"expected={expected_count}, actual={contribution_count}"
                )
        if not launched:
            _launch_deferred_identity_allreduce(entry)
        _DEFERRED_IDENTITY_ALLREDUCE_EXPECTED_COUNTS[key] = (
            weakref.ref(param),
            ranks,
            contribution_count,
        )
    _DEFERRED_IDENTITY_ALLREDUCE_GRADS.clear()


def clear_deferred_identity_allreduce_grads() -> None:
    """Discard registrations left by an interrupted FBW step."""
    _DEFERRED_IDENTITY_ALLREDUCE_GRADS.clear()


class AllReduceIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, ranks: Tuple[int]):
        return all_reduce(itensor, ranks)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def allreduce_identity(tensor: torch.Tensor, ranks: List[int]):
    return AllReduceIdentity.apply(tensor, ranks)


class IdentityAllreduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._leaf_param = (
            itensor
            if isinstance(itensor, torch.nn.Parameter)
            and itensor.is_leaf
            and itensor.requires_grad
            else None
        )
        return itensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        if (
            RuntimeFlag.fbw_phase in ("weight", "native_weight")
            and ctx._leaf_param is not None
        ):
            # The leaf accumulation happens after this Function returns. The
            # split-W runtime marks the contribution ready only after writing
            # the final leaf/reducer storage.
            defer_identity_allreduce_grad(ctx._leaf_param, ranks, ready=False)
            return grad, None
        # A split-W contribution is freshly produced for this traversal and is
        # discarded after it is mapped back to the stage parameter. This holds
        # for both phase-aware custom tasks (``weight``) and the remaining
        # native autograd traversal (``native_weight``).
        # Reuse that storage for the reduction instead of cloning a potentially
        # multi-GiB dWeight. Ordinary autograd keeps the non-mutating behavior.
        grad = all_reduce(
            grad,
            ranks,
            clone=RuntimeFlag.fbw_phase not in ("weight", "native_weight"),
        )
        return grad, None


def identity_allreduce(tensor: torch.Tensor, ranks: Tuple[int]) -> torch.Tensor:
    return IdentityAllreduce.apply(tensor, ranks)


class AllReduceAllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, ranks: Tuple[int]):
        ctx._ranks = ranks
        otensor = all_reduce(itensor, ranks)
        return otensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        grad = all_reduce(grad, ranks)
        return grad, None


def allreduce_allreduce(tensor: torch.Tensor, ranks: Tuple[int]) -> torch.Tensor:
    return AllReduceAllReduce.apply(tensor, ranks)


class ReduceScatterAllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._dim = dim
        return reduce_scatter(itensor, dim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        grad = all_gather(grad, dim, ranks)
        return grad, None, None


def reducescatter_allgather(tensor: torch.Tensor, dim: int, ranks: List[int]):
    return ReduceScatterAllGather.apply(tensor, dim, ranks)


class AllGatherReduceScatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._dim = dim
        return all_gather(itensor, dim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        grad = reduce_scatter(grad, dim, ranks)
        return grad, None, None


def allgather_reducescatter(tensor: torch.Tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    return AllGatherReduceScatter.apply(tensor, dim, ranks)


class AllGatherSplit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._dim = dim
        return all_gather(itensor, dim, ranks)      

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        return chunk(grad, dim, ranks), None, None


def allgather_split(tensor: torch.Tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    return AllGatherSplit.apply(tensor, dim, ranks)


class SplitAllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        """
        ranks should be the global rank
        """
        ctx._ranks = ranks
        ctx._dim = dim
        return chunk(itensor, dim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        grad = all_gather(grad, dim, ranks)
        return grad, None, None


def split_allgather(tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    return SplitAllGather.apply(tensor, dim, ranks)


class AllToAllAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, idim: int, odim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._idim = idim
        ctx._odim = odim
        return all_to_all(itensor, idim, odim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        idim, odim = ctx._idim, ctx._odim
        grad = all_to_all(grad, odim, idim, ranks)
        return grad, None, None, None


class AllToAllAllToAllSingle(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, idim: int, odim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._idim = idim
        ctx._odim = odim
        return all_to_all_single(itensor, idim, odim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        idim, odim = ctx._idim, ctx._odim
        grad = all_to_all_single(grad, odim, idim, ranks)
        return grad, None, None, None


def alltoall_alltoall(itensor: torch.Tensor, idim: int, odim: int, ranks: Tuple[int]) -> torch.Tensor:
    return AllToAllAllToAllSingle.apply(itensor, idim, odim, ranks)


class ReduceBroadcast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dst: int, ranks: List[int]):
        group = DeviceGroup().get_group(ranks)
        ctx._dst = dst
        ctx._group = group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_
        CudaTimer().start(field_name='comm', predefined=True)
        torch.distributed.reduce(input_, dst, group=group)
        CudaTimer().stop(field_name='comm', predefined=True)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        src = ctx._dst
        group = ctx._group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return grad_output, None, None
        CudaTimer().start(field_name='comm', predefined=True)
        torch.distributed.broadcast(grad_output, src, group=group)
        CudaTimer().stop(field_name='comm', predefined=True)
        return grad_output, None, None


class BroadcastReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, src: int, ranks: List[int]):
        group = DeviceGroup().get_group(ranks)
        ctx._src = src
        ctx._group = group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_
        CudaTimer().start(field_name='comm', predefined=True)
        torch.distributed.broadcast(input_, src, group=group)
        CudaTimer().stop(field_name='comm', predefined=True)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        dst = ctx._src
        group = ctx._group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return grad_output, None, None
        CudaTimer().start(field_name='comm', predefined=True)
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        torch.distributed.reduce(grad_output, dst, group=group)
        CudaTimer().stop(field_name='comm', predefined=True)
        return grad_output, None, None
