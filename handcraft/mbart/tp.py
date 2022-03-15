from typing import Tuple
import torch


class AllReduceIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, group):
        torch.distributed.all_reduce(input, group=group)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class IdentityAllreduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, group):
        ctx._group = group
        return input
    
    @staticmethod
    def backward(ctx, grad_output):
        torch.distributed.all_reduce(grad_output, group=ctx._group)
        return grad_output, None


class AllGatherScatter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dim, group):
        ctx._group = group
        ctx._dim = dim
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input
        rank = torch.distributed.get_rank(group)
        tensor_list = [torch.empty_like(input) for _ in range(world_size)]
        tensor_list[rank] = input
        torch.distributed.all_gather(tensor_list, input, group=group)
        output = torch.cat(tensor_list, dim=dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = ctx._group
        dim = ctx._dim
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return grad_output
        input_list = grad_output.chunk(world_size, dim=dim)
        rank = torch.distributed.get_rank(group)
        grad = input_list[rank].contiguous()
        return grad, None, None


class ReduceBroadcast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dst: int, group=None):
        ctx._dst = dst
        ctx._group = group
        torch.distributed.reduce(input, dst, group=group)
        torch.cuda.synchronize()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        src = ctx._dst
        group = ctx._group
        torch.distributed.broadcast(grad_output, src, group=group)
        torch.cuda.synchronize()
        return grad_output, None, None


class BroadcastReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, src: int, group=None):
        ctx._src = src
        ctx._group = group
        torch.distributed.broadcast(input, src, group=group)
        torch.cuda.synchronize()
        return input

    @staticmethod
    def backward(ctx, grad_output):
        dst = ctx._src
        group = ctx._group
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        torch.distributed.reduce(grad_output, dst, group=group)
        torch.cuda.synchronize()
        return grad_output, None, None
