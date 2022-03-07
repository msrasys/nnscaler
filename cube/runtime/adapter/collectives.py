from typing import List
from unittest import defaultTestLoader
import torch

from cube.runtime.device import DeviceGroup
from cube.profiler.timer import CudaTimer, print_each_rank


def send(tensor: torch.Tensor, to_rank: int):
    """
    send tensor to the remote devices. Each tensor can be
    sent to multiple devices

    Args:
        tensors (List[torch.Tensor]): list of tensor to send
        tensor_devices (List[List[int]]): tensor sent devices
    """
    # print(f'{torch.distributed.get_rank()}: sending...')
    CudaTimer().start(field_name='comm')
    
    send_ops = list()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    send_op = torch.distributed.P2POp(
        torch.distributed.isend, tensor, to_rank
    )
    send_ops.append(send_op)
    reqs = torch.distributed.batch_isend_irecv(send_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')


def recv(shape: List[int], from_rank: int, dtype: torch.dtype):
    # print(f'{torch.distributed.get_rank()}: recving...')
    CudaTimer().start(field_name='comm')
    ## synthetic ##
    # for shape in shapes:
    #     recv_tensors.append(
    #         torch.ones(tuple(shape),
    #         device=torch.cuda.current_device()
    #     ))
    # 
    tensor = torch.empty(
        shape, requires_grad=True, dtype=dtype,
        device=torch.cuda.current_device()
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv, tensor, from_rank
    )
    reqs = torch.distributed.batch_isend_irecv([recv_op])
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return tensor


def sendrecv(input_tensors: List[torch.Tensor],
             output_shapes: List[List[int]],
             output_dtypes: List[torch.dtype],
             send_ranks: List[int],
             recv_ranks: List[int]) -> List[torch.Tensor]:
    CudaTimer().start(field_name='comm')
    # print('sending and recving...')
    ops = list()
    outputs = list()
    for tensor, rank in zip(input_tensors, send_ranks):
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"Expected {tensor} to be tensor")
        send_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor, rank
        )
        ops.append(send_op)
    for shape, dtype, rank in zip(output_shapes, output_dtypes, recv_ranks):
        tensor = torch.empty(
            shape, dtype=dtype,
            requires_grad=True, device=torch.cuda.current_device()
        )
        outputs.append(tensor)
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor, rank
        )
        ops.append(recv_op)
    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return outputs


### Collective Universal Interface ###
# def universal(input_tensors: List[torch.Tensor],
#               output_shapes: List[List[int]],
#               output_dtypes: List[torch.dtype],
#               ranks: List[int])


def all_reduce(input_tensors: List[torch.Tensor],
               output_shapes: List[List[int]],
               output_dtypes: List[torch.dtype],
               ranks: List[int]) -> torch.Tensor:
    """
    Allreduce
    """
    CudaTimer().start(field_name='comm')
    assert len(input_tensors) == 1
    tensor = input_tensors[0]
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    tensor = tensor.detach()
    tensor = tensor.requires_grad_()
    group = DeviceGroup().get_group(ranks)
    torch.distributed.all_reduce(tensor, group=group)

    CudaTimer().stop(field_name='comm')
    return tensor


def all_gather(input_tensors: List[torch.Tensor],
               output_shapes: List[List[int]],
               output_dtypes: List[torch.dtype],
               ranks: List[int]) -> List[torch.Tensor]:
    """
    Allgather
    """
    CudaTimer().start(field_name='comm')
    assert len(input_tensors) == 1
    tensor = input_tensors[0]
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    group = DeviceGroup().get_group(ranks)
    tensor_list = [torch.empty_like(tensor) for _ in ranks]
    idx = ranks.index(DeviceGroup().rank)
    tensor_list[idx] = tensor
    torch.distributed.all_gather(tensor_list, tensor, group=group)
    CudaTimer().stop(field_name='comm')
    return tensor_list


def reduce_scatter(input_tensors: List[torch.Tensor],
                   output_shapes: List[List[int]],
                   output_dtypes: List[torch.dtype],
                   ranks: List[int]) -> List[torch.Tensor]:
    """
    ReduceScatter
    """
    CudaTimer().start(field_name='comm')
    input_tensors = list(input_tensors)
    for idx, tensor in enumerate(input_tensors):
        if not tensor.is_contiguous():
            input_tensors[idx] = tensor.contiguous()
    group = DeviceGroup().get_group(ranks)
    idx = ranks.index(DeviceGroup().rank)
    output = torch.empty_like(input_tensors[idx], requires_grad=True)
    torch.distributed.reduce_scatter(
        output, input_tensors, group=group
    )
    CudaTimer().stop(field_name='comm')
    return output


def broadcast(input_tensors: List[torch.Tensor],
              output_shapes: List[List[int]],
              output_dtypes: List[torch.dtype],
              ranks: List[int]) -> List[torch.Tensor]:
    """
    Broadcast. ranks[0] is the root
    """
    CudaTimer().start(field_name='comm')
    assert len(input_tensors) == 1 or len(input_tensors) == 0
    if len(input_tensors) == 1:
        tensor: torch.Tensor = input_tensors[0]
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
    else:
        assert len(output_shapes) == 1
        assert len(output_dtypes) == 1
        shape = output_shapes[0]
        dtype = output_dtypes[0]
        tensor = torch.empty(shape, device=torch.cuda.current_device(), dtype=dtype)
    group = DeviceGroup().get_group(ranks)
    torch.distributed.broadcast(tensor, ranks[0], group=group)
    CudaTimer().stop(field_name='comm')
    return tensor


def gather(input_tensors: List[torch.Tensor],
           output_shapes: List[List[int]],
           output_dtypes: List[torch.dtype],
           ranks: List[int]) -> List[torch.Tensor]:
    """
    Gather. ranks[0] is the root
    """
    CudaTimer().start(field_name='comm')
    assert len(input_tensors) == 1
    input_tensor = input_tensors[0]
    dst = ranks[0]
    if DeviceGroup().rank == dst:
        # recv
        tensor_list = [input_tensor] + [torch.empty_like(input_tensor) for _ in range(len(ranks)-1)]
        ops = list()
        for rank, tensor in zip(ranks[1:], tensor_list[1:]):
            recv_op = torch.distributed.P2POp(
                torch.distributed.irecv, tensor, rank
            )
            ops.append(recv_op)
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    else:
        # send
        tensor_list = []
        send_op = torch.distributed.P2POp(
            torch.distributed.isend, input_tensor, ranks[0]
        )
        reqs = torch.distributed.batch_isend_irecv([send_op])
        for req in reqs:
            req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return tensor_list


def scatter(input_tensors: List[torch.Tensor],
            output_shapes: List[List[int]],
            output_dtypes: List[torch.dtype],
            ranks: List[int]) -> List[torch.Tensor]:
    CudaTimer().start(field_name='comm')
    output = None
    src = ranks[0]
    if DeviceGroup().rank == src:
        # send
        ops = list()
        for rank, tensor in zip(ranks, input_tensors):
            if rank == src:
                output = tensor
            else:
                if not tensor.is_contiguous():
                    with torch.no_grad():
                        tensor = tensor.contiguous()
                send_op = torch.distributed.P2POp(
                    torch.distributed.isend, tensor, rank
                )
                ops.append(send_op)
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
    else:
        # recv
        assert len(output_shapes) == 1 and len(output_dtypes) == 1
        output = torch.empty(
            output_shapes[0], dtype=output_dtypes[0],
            requires_grad=True, device=torch.cuda.current_device()
        )
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv, output, src
        )
        reqs = torch.distributed.batch_isend_irecv([recv_op])
        for req in reqs:
            req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm')
    return output
