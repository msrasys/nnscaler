from typing import List
import torch

from cube.runtime.device import DeviceGroup


def send(tensors, to_ranks: List[int]):
    """
    send tensor to the remote devices. Each tensor can be
    sent to multiple devices

    Args:
        tensors (List[torch.Tensor]): list of tensor to send
        tensor_devices (List[List[int]]): tensor sent devices
    """
    # print('sending...')
    send_ops = list()

    ## synthetic ##
    # return

    for tensor, rank in zip(tensors, to_ranks):
        send_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor, rank
        )
        send_ops.append(send_op)
    reqs = torch.distributed.batch_isend_irecv(send_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()


def recv(shapes: List[List[int]], from_ranks: List[int]):
    # print('recving...')
    recv_ops = list()
    recv_tensors = list()

    ## synthetic ##
    # for shape in shapes:
    #     recv_tensors.append(
    #         torch.ones(tuple(shape),
    #         device=torch.cuda.current_device()
    #     ))

    for shape, rank in zip(shapes, from_ranks):
        tensor = torch.empty(
            shape, requires_grad=True, device=torch.cuda.current_device()
        )
        recv_tensors.append(tensor)
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor, rank
        )
        recv_ops.append(recv_op)
    reqs = torch.distributed.batch_isend_irecv(recv_ops)
    for req in reqs:
        req.wait()
    torch.cuda.synchronize()

    if    len(recv_tensors) == 0: return None
    elif  len(recv_tensors) == 1: return recv_tensors[0]
    else: return tuple(recv_tensors)


def send_and_recv(send_tensors, to_ranks, recv_shapes, from_ranks):
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

    if    len(recv_tensors) == 0: return None
    elif  len(recv_tensors) == 1: return recv_tensors[0]
    else: return tuple(recv_tensors)


def all_reduce(tensors: List[torch.Tensor], ranks: List[int]):
    """
    Allreduce
    """
    assert len(tensors) == 1
    tensor = tensors[0]
    group = DeviceGroup().get_group(ranks)
    torch.distributed.all_reduce(tensor, group=group)
    return tensor


def all_gather(tensors: List[torch.Tensor], ranks: List[int]):
    """
    Allgather
    """
    assert len(tensors) == 1
    tensor = tensors[0]
    group = DeviceGroup().get_group(ranks)
    tensor_list = [torch.empty_like(tensor) for _ in ranks]
    idx = ranks.index(DeviceGroup().rank)
    tensor_list[idx] = tensor
    torch.distributed.all_gather(tensor_list, tensor, group=group)
    tensor_list = [t for oidx, t in enumerate(tensor_list) if oidx != idx]
    if len(tensor_list) == 1:
        return tensor_list[0]
    else:
        return tensor_list


def reduce_scatter(tensors: List[torch.Tensor], ranks: List[int]):
    """
    ReduceScatter
    """
    group = DeviceGroup().get_group(ranks)
    idx = ranks.index(DeviceGroup().rank)
    output = tensors[idx]
    torch.distributed.reduce_scatter(
        output, tensors, group=group
    )
    return output


def broadcast(tensors: List[torch.Tensor], ranks: List[int]):
    """
    Broadcast. ranks[0] is the root
    """
    assert len(tensors) == 1
    tensor = tensors[0]
    group = DeviceGroup().get_group(ranks)
    torch.distributed.broadcast(tensor, ranks[0], group=group)
    return tensor
