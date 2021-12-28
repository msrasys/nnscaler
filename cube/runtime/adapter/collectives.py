from typing import List
import torch

from cube.runtime.device import DeviceGroup
from cube.profiler.timer import CudaTimer


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


def send_and_recv(send_tensors, to_ranks, recv_shapes, from_ranks):
    CudaTimer().start(field_name='comm')
    # print('sending and recving...')
    ops = list()
    recv_tensors = list()
    for tensor, ranks in zip(send_tensors, to_ranks):
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"Expected {tensor} to be tensor")
        for rank in ranks:
            send_op = torch.distributed.P2POp(
                torch.distributed.isend, tensor, rank
            )
            ops.append(send_op)
    for shape, ranks in zip(recv_shapes, from_ranks):
        if len(ranks) != 1:
            raise RuntimeError(
                "Not supported for recving same tensor from multiple devices"
            )
        rank = ranks[0]
        tensor = torch.empty(
            shape, requires_grad=True, device=torch.cuda.current_device()
        )
        recv_tensors.append(tensor)
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor, rank
        )
        ops.append(recv_op)
    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()

    CudaTimer().stop(field_name='comm')

    if    len(recv_tensors) == 0: return None
    elif  len(recv_tensors) == 1: return recv_tensors[0]
    else: return tuple(recv_tensors)


def all_reduce(tensors: List[torch.Tensor], ranks: List[int]):
    """
    Allreduce
    """
    CudaTimer().start(field_name='comm')
    # print(f'{torch.distributed.get_rank()}: all_reduce...')
    assert len(tensors) == 1
    tensor = tensors[0]
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    tensor = tensor.detach()
    tensor = tensor.requires_grad_()

    ### Bypass ###
    # return tensor

    group = DeviceGroup().get_group(ranks)
    torch.distributed.all_reduce(tensor, group=group)

    CudaTimer().stop(field_name='comm')
    return tensor


def all_gather(tensors: List[torch.Tensor], ranks: List[int]):
    """
    Allgather
    """
    # print(f'{torch.distributed.get_rank()}: all_gather...')
    CudaTimer().start(field_name='comm')

    assert len(tensors) == 1
    tensor = tensors[0]
    group = DeviceGroup().get_group(ranks)
    tensor_list = [torch.empty_like(tensor) for _ in ranks]
    idx = ranks.index(DeviceGroup().rank)
    tensor_list[idx] = tensor
    torch.distributed.all_gather(tensor_list, tensor, group=group)
    tensor_list = [t for oidx, t in enumerate(tensor_list) if oidx != idx]

    CudaTimer().stop(field_name='comm')
    if len(tensor_list) == 1:
        return tensor_list[0]
    else:
        return tensor_list


def reduce_scatter(tensors: List[torch.Tensor], ranks: List[int]):
    """
    ReduceScatter
    """
    # print(f'{torch.distributed.get_rank()}: reduce-scatter...')
    CudaTimer().start(field_name='comm')

    tensors = list(tensors)
    group = DeviceGroup().get_group(ranks)
    idx = ranks.index(DeviceGroup().rank)
    output = torch.empty_like(tensors[idx], requires_grad=True)
    torch.distributed.reduce_scatter(
        output, tensors, group=group
    )

    CudaTimer().stop(field_name='comm')
    return output


def broadcast(tensors: List[torch.Tensor], ranks: List[int], shape=None, dtype=None):
    """
    Broadcast. ranks[0] is the root
    """
    CudaTimer().start(field_name='comm')
    # print(f'{torch.distributed.get_rank()}: broadcast...')
    # FIXME: data type
    if len(tensors) == 1:
        tensor = tensors[0]
    else:
        tensor = torch.empty(shape, device=torch.cuda.current_device(), dtype=dtype)
        # tensor.requires_grad_()
    group = DeviceGroup().get_group(ranks)
    torch.distributed.broadcast(tensor, ranks[0], group=group)

    CudaTimer().stop(field_name='comm')
    return tensor
