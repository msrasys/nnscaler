from typing import List

import torch


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
