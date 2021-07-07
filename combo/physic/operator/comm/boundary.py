"""
Autograd backward needs to return the same number of gradients as input,
even if they are not tensors.
"""

import torch

from combo.physical.device.group import DeviceGroup


__all__ = ['parallel_in', 'gather_out', 'scatter_in', 'reduce_out']


def _reduce(input_, group):
    """All-reduce the the input tensor across model parallel group."""

    # allreduce
    torch.distributed.all_reduce(input_, group=group)
    return input_


def _split(input_, dim, chunk_num, rank):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    # bypass the function if we are using only 1 GPU.
    if chunk_num == 1:
        return input_
    # split along specified dim
    if input_.size()[dim] % chunk_num != 0:
        raise RuntimeError("backward on Gather Out Error: un divideable")
    dim_size = input_.size()[dim] // chunk_num
    tensor_list = torch.split(input_, dim_size, dim=dim)
    # note: torch.split does not create contiguous tensors by default.
    output = tensor_list[rank].contiguous()
    return output


def _gather(input_, dim, group):
    """Gather tensors and concatinate along the last dimension."""

    world_size = torch.distributed.get_world_size(group)
    rank = torch.distributed.get_rank(group)
    # bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)
    # note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output


class _ParallelIn(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    
    @staticmethod
    def forward(ctx, input_, ranks):
        # record group
        group = DeviceGroup().get_group(ranks)
        ctx.constants = group
        # identitfy forward
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # allreduce
        group = ctx.constants
        return _reduce(grad_output, group), None


class _GatherOut(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""
    
    @staticmethod
    def forward(ctx, input_, dim, ranks):
        # record group
        group = DeviceGroup().get_group(ranks)
        ctx.constants = (group, dim)
        # allgather
        return _gather(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        group, dim = ctx.constants
        world_size = torch.distributed.get_world_size(group)
        rank = torch.distributed.get_rank(group)
        return _split(grad_output, dim, world_size, rank), None, None


class _ScatterIn(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def forward(ctx, input_, dim, ranks):
        group = DeviceGroup().get_group(ranks)
        world_size = torch.distributed.get_world_size(group)
        rank = torch.distributed.get_rank(group)
        ctx.constants = (group, dim)
        return _split(input_, dim, world_size, rank)

    @staticmethod
    def backward(ctx, grad_output):
        group, dim = ctx.constants
        return _gather(grad_output, dim, group), None, None


class _ReduceOut(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_, ranks):
        group = DeviceGroup().get_group(ranks)
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def parallel_in(input_, ranks):
    return _ParallelIn.apply(input_, ranks)


def gather_out(input_, dim, ranks):
    return _GatherOut.apply(input_, dim, ranks)


def scatter_in(input_, dim, ranks):
    return _ScatterIn.apply(input_, dim, ranks)


def reduce_out(input_, ranks):
    return _ReduceOut.apply(input_, ranks)
