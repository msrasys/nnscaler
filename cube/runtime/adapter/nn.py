from typing import List, Tuple
import torch

from cube.profiler.timer import CudaTimer
from cube.runtime.device import DeviceGroup


def _allreduce(itensor: torch.Tensor, ranks: Tuple[int]) -> torch.Tensor:
    CudaTimer().start(field_name='comm', predefined=True)
    if not itensor.is_contiguous():
        itensor = itensor.contiguous()
    group = DeviceGroup().get_group(ranks)
    torch.distributed.all_reduce(itensor, group=group)
    CudaTimer().stop(field_name='comm', predefined=True)
    return itensor


def _allgather(itensor: torch.Tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    CudaTimer().start(field_name='comm', predefined=True)
    if not itensor.is_contiguous():
        itensor = itensor.contiguous()
    group = DeviceGroup().get_group(ranks)
    tensor_list = [torch.empty_like(itensor) for _ in ranks]
    tensor_list[torch.distributed.get_rank(group)] = itensor.data
    torch.distributed.all_gather(tensor_list, itensor, group=group)
    # concat
    otensor = torch.concat(tuple(tensor_list), dim=dim).requires_grad_()
    CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def _reducescatter(itensor: torch.Tensor, dim:int, ranks: Tuple[int]) -> torch.Tensor:
    CudaTimer().start(field_name='comm', predefined=True)
    itensors = list(itensor.chunk(len(ranks), dim))
    for idx, tensor in enumerate(itensors):
        if not tensor.is_contiguous():
            itensors[idx] = tensor.contiguous()
    group = DeviceGroup().get_group(ranks)
    otensor = torch.empty_like(itensors[0])
    torch.distributed.reduce_scatter(otensor, itensors, group=group)
    CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def _alltoall(itensor: torch.Tensor, idim: int, odim: int, ranks: Tuple[int]) -> torch.Tensor:
    CudaTimer().start(field_name='comm', predefined=True)
    itensors = list(itensor.chunk(len(ranks), dim=odim))
    for idx, tensor in enumerate(itensors):
        if not tensor.is_contiguous():
            itensors[idx] = tensor.contiguous()
    otensors = [torch.empty_like(t) for t in itensors]
    group = DeviceGroup().get_group(ranks)
    torch.distributed.all_to_all(otensors, itensors, group=group)
    otensor = torch.concat(tuple(otensors), dim=idim)
    CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def _chunk(itensor: torch.Tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    """
    split dimension in n chunks and take idx-th chunk

    ranks (Tuple[int]): the order of split tensor.
    """
    group = DeviceGroup().get_group(ranks)
    idx = torch.distributed.get_rank(group)
    return itensor.chunk(len(ranks), dim)[idx]


class AllReduceIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, ranks: Tuple[int]):
        return _allreduce(itensor, ranks)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def allreduce_identity(tensor: torch.Tensor, ranks: List[int]):
    return AllReduceIdentity.apply(tensor, ranks)


class IdentityAllreduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, ranks: Tuple[int]):
        ctx._ranks = ranks
        return itensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        grad = _allreduce(grad, ranks)
        return grad, None


def identity_allreduce(tensor: torch.Tensor, ranks: Tuple[int]) -> torch.Tensor:
    return IdentityAllreduce.apply(tensor, ranks)


class AllReduceAllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, ranks: Tuple[int]):
        ctx._ranks = ranks
        otensor = _allreduce(itensor, ranks)
        return otensor

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        grad = _allreduce(grad, ranks)
        return grad, None


def allreduce_allreduce(tensor: torch.Tensor, ranks: Tuple[int]) -> torch.Tensor:
    return AllReduceAllReduce.apply(tensor, ranks)


class ReduceScatterAllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._dim = dim
        return _reducescatter(itensor, dim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        grad = _allgather(grad, dim, ranks)
        return grad, None, None


def reducescatter_allgather(tensor: torch.Tensor, dim: int, ranks: List[int]):
    return ReduceScatterAllGather.apply(tensor, dim, ranks)


class AllGatherReduceScatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._dim = dim
        return _allgather(itensor, dim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        grad = _reducescatter(grad, dim, ranks)
        return grad, None, None


def allgather_reducescatter(tensor: torch.Tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    return AllGatherReduceScatter.apply(tensor, dim, ranks)


class AllGatherSplit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, dim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._dim = dim
        return _allgather(itensor, dim, ranks)      

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        return _chunk(grad, dim, ranks), None, None


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
        return _chunk(itensor, dim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        dim = ctx._dim
        grad = _allgather(grad, dim, ranks)
        return grad, None, None


def split_allgather(tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    return SplitAllGather.apply(tensor, dim, ranks)


class AllToAllAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, itensor: torch.Tensor, idim: int, odim: int, ranks: Tuple[int]):
        ctx._ranks = ranks
        ctx._idim = idim
        ctx._odim = odim
        return _alltoall(itensor, idim, odim, ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        ranks = ctx._ranks
        idim, odim = ctx._idim, ctx._odim
        grad = _alltoall(grad, odim, idim, ranks)
        return grad, None, None, None


def alltoall_alltoall(itensor: torch.Tensor, idim: int, odim: int, ranks: Tuple[int]) -> torch.Tensor:
    return AllToAllAllToAll.apply(itensor, idim, odim, ranks)


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
