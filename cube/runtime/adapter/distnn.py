import torch
from cube.profiler.timer import CudaTimer


class AllReduceIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, group):
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input
        CudaTimer().start(field_name='comm')
        torch.distributed.all_reduce(input, group=group)
        CudaTimer().stop(field_name='comm')
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
        world_size = torch.distributed.get_world_size(ctx._group)
        if world_size == 1:
            return grad_output, None
        CudaTimer().start(field_name='comm')
        torch.distributed.all_reduce(grad_output, group=ctx._group)
        CudaTimer().stop(field_name='comm')
        return grad_output, None


class AllGatherSplit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dim, group):
        ctx._group = group
        ctx._dim = dim
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input
        CudaTimer().start(field_name='comm')
        rank = torch.distributed.get_rank(group)
        tensor_list = [torch.empty_like(input) for _ in range(world_size)]
        tensor_list[rank] = input
        torch.distributed.all_gather(tensor_list, input, group=group)
        output = torch.cat(tensor_list, dim=dim).contiguous()
        CudaTimer().stop(field_name='comm')
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = ctx._group
        dim = ctx._dim
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return grad_output
        CudaTimer().start(field_name='comm')
        input_list = grad_output.chunk(world_size, dim=dim)
        rank = torch.distributed.get_rank(group)
        grad = input_list[rank].contiguous()
        CudaTimer().stop(field_name='comm')
        return grad, None, None


class ReduceBroadcast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dst: int, group):
        ctx._dst = dst
        ctx._group = group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input
        CudaTimer().start(field_name='comm')
        torch.distributed.reduce(input, dst, group=group)
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm')
        return input

    @staticmethod
    def backward(ctx, grad_output):
        src = ctx._dst
        group = ctx._group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return grad_output, None, None
        CudaTimer().start(field_name='comm')
        torch.distributed.broadcast(grad_output, src, group=group)
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm')
        return grad_output, None, None


class BroadcastReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, src: int, group=None):
        ctx._src = src
        ctx._group = group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input
        CudaTimer().start(field_name='comm')
        torch.distributed.broadcast(input, src, group=group)
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm')
        return input

    @staticmethod
    def backward(ctx, grad_output):
        dst = ctx._src
        group = ctx._group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return grad_output, None, None
        CudaTimer().start(field_name='comm')
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        torch.distributed.reduce(grad_output, dst, group=group)
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm')
        return grad_output, None, None
