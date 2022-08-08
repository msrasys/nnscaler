from typing import List
import torch

from cube.profiler.timer import CudaTimer
from cube.runtime.device import DeviceGroup


class SendRecv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dst: int, group):
        CudaTimer().start(field_name='comm')
        ctx._tsize = input_.size()
        ctx._tdtype = input_.dtype
        ctx._src = dst
        if not input_.is_contiguous():
            input_ = input_.contiguous()
        sendop = torch.distributed.P2POp(
            torch.distributed.isend, input_, dst
        )
        reqs = torch.distributed.batch_isend_irecv([sendop])
        for req in reqs:
            req.wait()
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm')
        return input_

    @staticmethod
    def backward(ctx, _grad: torch.Tensor):
        CudaTimer().start(field_name='comm')
        size = ctx._tsize
        dtype = ctx._tdtype
        src = ctx._src
        grad = torch.empty(size, dtype=dtype, device=torch.cuda.current_device())
        recvop = torch.distributed.P2POp(
            torch.distributed.irecv, grad, src
        )
        reqs = torch.distributed.batch_isend_irecv([recvop])
        for req in reqs:
            req.wait()
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm')
        return grad, None, None


class RecvSend(torch.autograd.Function):

    @staticmethod
    def forward(ctx, size, dtype, src: int, ranks: List[int]):
        CudaTimer().start(field_name='comm')
        ctx._tsize = size
        ctx._tdtype = dtype
        ctx._dst = src
        input_ = torch.empty(
            size, dtype=dtype, device=torch.cuda.current_device(),
            requires_grad=True)
        recvop = torch.distributed.P2POp(
            torch.distributed.irecv, input_, src
        )
        reqs = torch.distributed.batch_isend_irecv([recvop])
        for req in reqs:
            req.wait()
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm')
        return input_

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        CudaTimer().start(field_name='comm')
        dst = ctx._dst
        if not grad.is_contiguous():
            grad = grad.contiguous()
        sendop = torch.distributed.P2POp(
            torch.distributed.isend, grad, dst
        )
        reqs = torch.distributed.batch_isend_irecv([sendop])
        for req in reqs:
            req.wait()
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm')
        return None, None, None, None


class AllReduceIdentity(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, group):
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_
        CudaTimer().start(field_name='comm')
        torch.distributed.all_reduce(input_, group=group)
        CudaTimer().stop(field_name='comm')
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class IdentityAllreduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_, group):
        ctx._group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        world_size = torch.distributed.get_world_size(ctx._group)
        if world_size == 1:
            return grad_output, None
        CudaTimer().start(field_name='comm')
        torch.distributed.all_reduce(grad_output, group=ctx._group)
        CudaTimer().stop(field_name='comm')
        return grad_output, None


class ReduceScatterAllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, group):
        ctx._group = group
        ctx._dim = dim
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_,
        CudaTimer().start(field_name='comm')
        input_tensors = input_.chunk(world_size, dim)
        rank = torch.distributed.get_rank(group)
        input_ = torch.empty_like(input_tensors[rank], requires_grad=True)
        torch.distributed.reduce_scatter(
            input_, input_tensors, group=group
        )
        CudaTimer().stop(field_name='comm')
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx._group
        dim = ctx._dim
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return grad_output
        CudaTimer().start(field_name='comm')
        rank = torch.distributed.get_rank(group)
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        tensor_list[rank] = grad_output
        torch.distributed.all_gather(tensor_list, grad_output, group=group)
        grad = torch.cat(tensor_list, dim=dim).contiguous()
        CudaTimer().stop(field_name='comm')
        return grad, None, None


class AllGatherSplit(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, group):
        ctx._group = group
        ctx._dim = dim
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_
        CudaTimer().start(field_name='comm')
        rank = torch.distributed.get_rank(group)
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_, group=group)
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


class SplitAllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int, group):
        ctx._group = group
        ctx._dim = dim
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_
        CudaTimer().start(field_name='comm')
        input_list = input_.chunk(world_size, dim=dim)
        rank = torch.distributed.get_rank(group)
        input_ = input_list[rank].contiguous()
        CudaTimer().stop(field_name='comm')
        return input_

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = ctx._group
        dim = ctx._dim
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return grad_output
        CudaTimer().start(field_name='comm')
        rank = torch.distributed.get_rank(group)
        tensor_list = [torch.empty_like(grad_output) for _ in range(world_size)]
        tensor_list[rank] = grad_output
        torch.distributed.all_gather(tensor_list, grad_output, group=group)
        grad = torch.cat(tensor_list, dim=dim).contiguous()
        CudaTimer().stop(field_name='comm')
        return grad, None, None


class ReduceBroadcast(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, dst: int, group):
        ctx._dst = dst
        ctx._group = group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_
        CudaTimer().start(field_name='comm')
        torch.distributed.reduce(input_, dst, group=group)
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm')
        return input_

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
    def forward(ctx, input_: torch.Tensor, src: int, group):
        ctx._src = src
        ctx._group = group
        world_size = torch.distributed.get_world_size(group)
        if world_size == 1:
            return input_
        CudaTimer().start(field_name='comm')
        torch.distributed.broadcast(input_, src, group=group)
        torch.cuda.synchronize()
        CudaTimer().stop(field_name='comm')
        return input_

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
