from typing import List, Tuple, Optional
import torch

from cube.runtime.device import DeviceGroup
from cube.profiler.timer import CudaTimer, print_each_rank


def send(tensor: torch.Tensor, dst: int):
    """
    send tensor to the remote devices. Each tensor can be
    sent to multiple devices

    Args:
        tensors (List[torch.Tensor]): list of tensor to send
        tensor_devices (List[List[int]]): tensor sent devices
    """
    # print(f'{torch.distributed.get_rank()}: sending...')
    CudaTimer().start(field_name='comm', predefined=True)
    
    send_ops = list()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    send_op = torch.distributed.P2POp(
        torch.distributed.isend, tensor, dst
    )
    send_ops.append(send_op)
    reqs = torch.distributed.batch_isend_irecv(send_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def recv(tensors: List[torch.Tensor], shape: List[int], dtype: torch.dtype, src: int):
    # print(f'{torch.distributed.get_rank()}: recving...')
    CudaTimer().start(field_name='comm', predefined=True)
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
        torch.distributed.irecv, tensor, src
    )
    reqs = torch.distributed.batch_isend_irecv([recv_op])
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()
    CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


# def move(tensor: Optional[torch.Tensor], shape: Tuple[int], dtype: torch.dtype, src: int, dst: int):
#     rank = torch.distributed.get_rank()
#     if rank == src:
#         assert torch.is_tensor(tensor)
#         return send(tensor, dst)
#     else:
#         assert rank == dst
#         return recv(None, shape, dtype, src)


def move(tensor: Optional[torch.Tensor], shape: Tuple[int], dtype: torch.dtype, src: int, dst: int):
    """
    Move a tensor from source device to destination device.
    """
    CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    if rank == src:
        tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
        assert torch.is_tensor(tensor)
        torch.distributed.send(tensor, dst)
    else:
        assert rank == dst
        tensor = torch.empty(shape, dtype=dtype, 
            device=torch.cuda.current_device(), requires_grad=True
        )
        torch.distributed.recv(tensor, src)
    CudaTimer().stop(field_name='comm', predefined=True)
    return tensor



def sendrecv(input_tensors: List[torch.Tensor],
             output_shapes: List[List[int]],
             output_dtypes: List[torch.dtype],
             send_ranks: List[int],
             recv_ranks: List[int]) -> List[torch.Tensor]:
    CudaTimer().start(field_name='comm', predefined=True)
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
    CudaTimer().stop(field_name='comm', predefined=True)
    return outputs


def all_reduce(itensor: torch.Tensor,
               ranks: List[int]) -> torch.Tensor:
    """
    Allreduce
    """
    CudaTimer().start(field_name='comm', predefined=True)
    if not itensor.is_contiguous():
        itensor = itensor.contiguous()
    itensor = itensor.detach()
    group = DeviceGroup().get_group(ranks)
    torch.distributed.all_reduce(itensor, group=group)
    CudaTimer().stop(field_name='comm', predefined=True)
    return itensor


def all_gather(itensor: torch.Tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    """
    Allgather
    """
    CudaTimer().start(field_name='comm', predefined=True)
    if not itensor.is_contiguous():
        itensor = itensor.contiguous()
    group = DeviceGroup().get_group(ranks)
    tensor_list = [torch.empty_like(itensor) for _ in ranks]
    tensor_list[torch.distributed.get_rank(group)] = itensor.data
    torch.distributed.all_gather(tensor_list, itensor, group=group)
    # concat
    otensor = torch.concat(tuple(tensor_list), dim=dim)
    CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def reduce_scatter(itensor: torch.Tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    """
    ReduceScatter
    """
    CudaTimer().start(field_name='comm', predefined=True)
    itensors = list(itensor.chunk(len(ranks), dim))
    for idx, tensor in enumerate(itensors):
        if not tensor.is_contiguous():
            itensors[idx] = tensor.contiguous()
    group = DeviceGroup().get_group(ranks)
    otensor = torch.empty_like(itensors[0], requires_grad=False)
    torch.distributed.reduce_scatter(otensor, itensors, group=group)
    CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def all_to_all(itensor: torch.Tensor, idim: int, odim: int, ranks: Tuple[int]) -> torch.Tensor:
    """
    All-to-all
    """
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


def chunk(itensor: torch.Tensor, dim: int, ranks: Tuple[int]) -> torch.Tensor:
    """
    split dimension in n chunks and take idx-th chunk

    ranks (Tuple[int]): the order of split tensor.
    """
    group = DeviceGroup().get_group(ranks)
    idx = torch.distributed.get_rank(group)
    require_grad = itensor.requires_grad
    with torch.no_grad():
        otensor = itensor.chunk(len(ranks), dim)[idx]
        otensor = otensor.detach()
    if require_grad:
        otensor = otensor.requires_grad_()
    return otensor


def rdscatter(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
              dim: int, src: int, dsts: Tuple[int]):
    """
    RDScatter: split itensor at rank `src` along dim into `len(dsts)` chunks,
    and then send each chunk to `dst` devices.
    """
    CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    if rank == src:
        with torch.no_grad():
            otensors = itensor.chunk(len(dsts), dim)
            for dst, otensor in zip(dsts, otensors):
                otensor = otensor.contiguous() if not otensor.is_contiguous() else otensor
                torch.distributed.send(otensor, dst)
        otensor = itensor
    else:
        assert rank in dsts
        shape = list(shape)
        shape[dim] = shape[dim] // len(dsts)
        otensor = torch.empty(
            shape, requires_grad=True, dtype=dtype,
            device=torch.cuda.current_device()
        )
        torch.distributed.recv(otensor, src)
    CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def rvscatter(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
              dim: int, src: int, dsts: Tuple[int]):
    """
    src: global rank
    """
    CudaTimer().start(field_name='comm', predefined=True)
    group = DeviceGroup().get_group((src,) + dsts)
    rank = torch.distributed.get_rank()
    tensor: torch.Tensor = itensor / len(dsts) if src == rank else \
        torch.empty(shape, dtype=dtype, requires_grad = True)
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    torch.distributed.broadcast(tensor, src, group=group)
    CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def rdgather(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
             dim: int, srcs: Tuple[int], dst: int):
    """
    @param srcs Tuple[int]: global rank of each source device
    @param dst int: global rank of destination device
    """
    CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    if rank == dst:
        recv_tensors = []
        for src in srcs:
            tensor = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
            torch.distributed.recv(tensor, src)
            recv_tensors.append(tensor)
        otensor = torch.cat(tuple(recv_tensors), dim=dim)
        otensor = otensor.requires_grad_()
    else:
        assert rank in srcs
        otensor = itensor.contiguous() if not itensor.is_contiguous() else itensor
        torch.distributed.send(otensor, dst)
    CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def rvgather(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
             srcs: Tuple[int], dst: int):
    """
    @param srcs Tuple[int]: global rank of each source device
    @param dst int: global rank of destination device
    """
    CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    group = DeviceGroup().get_group(srcs + (dst,))
    tensor = torch.zeros(shape, dtype=dtype, requires_grad=True) if rank == dst else itensor
    torch.distributed.reduce(tensor, dst, group=group)
    CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def broadcast(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype, src: int, ranks: List[int]) -> torch.Tensor:
    """
    Broadcast
    @param src: the global rank that holds tensor for broadcasting
    """
    CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    group = DeviceGroup().get_group(ranks)
    if rank == src:
        tensor = itensor.contiguous() if not itensor.is_contiguous() else itensor
    else:
        assert rank in ranks
        tensor = torch.empty(shape, 
            device=torch.cuda.current_device(), requires_grad=True, dtype=dtype)
    torch.distributed.broadcast(tensor, src, group=group)
    CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def gather(input_tensors: List[torch.Tensor],
           output_shapes: List[List[int]],
           output_dtypes: List[torch.dtype],
           ranks: List[int]) -> List[torch.Tensor]:
    """
    Gather. ranks[0] is the root
    """
    CudaTimer().start(field_name='comm', predefined=True)
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
    CudaTimer().stop(field_name='comm', predefined=True)
    return tensor_list


def scatter(input_tensors: List[torch.Tensor],
            output_shapes: List[List[int]],
            output_dtypes: List[torch.dtype],
            ranks: List[int]) -> List[torch.Tensor]:
    CudaTimer().start(field_name='comm', predefined=True)
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
    CudaTimer().stop(field_name='comm', predefined=True)
    return output
   