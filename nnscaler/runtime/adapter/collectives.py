#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
This module offers the wrap of communication primitives
based on `torch.distributed`. The use of these primitives standalone is typically
for non-autograd (e.g., inference) scenarios.

Every collective is implemented using out-of-place semantics.
"""

import io
from typing import List, Tuple, Optional
import torch

from nnscaler.runtime.device import DeviceGroup
from nnscaler.profiler.timer import CudaTimer

from nnscaler.runtime.executor import (
    AsyncCommHandler,
    complete_deferred_pseudo_free_tensor,
    defer_pseudo_free_tensor,
    pseudo_free_tensor,
)


def _serialize_object(obj) -> bytes:
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def _deserialize_object(payload: bytes):
    # Object collectives otherwise restore nested CUDA tensors on the sender's
    # device, which is invalid when an IRObject crosses pipeline stages.
    def map_location(storage, location):
        if location.startswith('cuda'):
            return storage.cuda(torch.cuda.current_device())
        return None

    return torch.load(
        io.BytesIO(payload), map_location=map_location, weights_only=False)


def move(
    tensor: Optional[torch.Tensor],
    shape: Tuple[int],
    dtype: torch.dtype,
    src: int,
    dst: int,
    async_op=False,
    release_after_send: Optional[torch.Tensor] = None,
):
    """
    Move a tensor from source device to destination device.
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    work = None
    group, group_src, group_dst = DeviceGroup().get_p2p_group(src, dst)
    if rank == src:
        tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
        assert torch.is_tensor(tensor)
        if async_op:
            if group is None:
                work = torch.distributed.isend(tensor, dst)
            else:
                work = torch.distributed.isend(tensor, group=group, group_dst=group_dst)
            callback = None
            if release_after_send is not None:
                defer_pseudo_free_tensor(release_after_send)
                callback = lambda: complete_deferred_pseudo_free_tensor(release_after_send)
            AsyncCommHandler().hold_send(tensor, work, callback=callback)
        else:
            if group is None:
                torch.distributed.send(tensor, dst)
            else:
                torch.distributed.send(tensor, group=group, group_dst=group_dst)
            if release_after_send is not None:
                pseudo_free_tensor(release_after_send)
    else:
        assert rank == dst
        tensor = torch.empty(shape, dtype=dtype,
            device=torch.cuda.current_device()
        )
        if async_op:
            if group is None:
                work = torch.distributed.irecv(tensor, src)
            else:
                work = torch.distributed.irecv(tensor, group=group, group_src=group_src)
            AsyncCommHandler().submit(tensor, [work])
        else:
            if group is None:
                torch.distributed.recv(tensor, src)
            else:
                torch.distributed.recv(tensor, group=group, group_src=group_src)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def move_object(obj, src: int, dst: int, async_op=False):
    """
    Move a non-tensor object from source device to destination device
    using send_object_list / recv_object_list.
    """
    if async_op:
        raise NotImplementedError("Async move_object is not implemented yet")

    CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    if rank == src:
        torch.distributed.send_object_list([_serialize_object(obj)], dst=dst)
    else:
        assert rank == dst
        obj_list = [None]
        torch.distributed.recv_object_list(obj_list, src=src)
        obj = _deserialize_object(obj_list[0])
    CudaTimer().stop(field_name='comm', predefined=True)
    return obj


def all_reduce(tensor: torch.Tensor,
               ranks: List[int], async_op=False) -> torch.Tensor:
    """Allreduce"""
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    tensor = tensor.detach().clone()
    group = DeviceGroup().get_group(ranks)

    if async_op:
        work = torch.distributed.all_reduce(tensor, group=group, async_op=True)
        AsyncCommHandler().submit(tensor, [work])
    else:
        torch.distributed.all_reduce(tensor, group=group)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def all_gather(tensor: torch.Tensor, dim: int,
               ranks: Tuple[int], async_op=False) -> torch.Tensor:
    """Allgather"""
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    group = DeviceGroup().get_group(ranks)
    tensor_list = [torch.empty_like(tensor) for _ in ranks]
    tensor_list[torch.distributed.get_rank(group)] = tensor.data
    work = torch.distributed.all_gather(tensor_list, tensor, group=group, async_op=async_op)
    group_ranks = torch.distributed.get_process_group_ranks(group)
    gather_order = tuple(group_ranks.index(rank) for rank in ranks)

    def concat_gathered(_):
        return torch.concat(tuple(tensor_list[index] for index in gather_order), dim=dim)

    if work:
        AsyncCommHandler().submit(tensor, [work], concat_gathered)
        otensor = tensor
    else:
        otensor = concat_gathered(tensor)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def reduce_scatter(tensor: torch.Tensor, dim: int,
                   ranks: Tuple[int], async_op=False) -> torch.Tensor:
    """ReduceScatter"""
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    itensors = list(tensor.chunk(len(ranks), dim))
    for idx, t in enumerate(itensors):
        itensors[idx] = t.contiguous() if not t.is_contiguous() else t
    group = DeviceGroup().get_group(ranks)
    otensor = torch.empty_like(itensors[0], requires_grad=False)
    work = torch.distributed.reduce_scatter(otensor, itensors, group=group, async_op=async_op)
    if work:
        AsyncCommHandler().submit(otensor, [work])
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def all_to_all(tensor: torch.Tensor, idim: int, odim: int,
               ranks: Tuple[int, ...], async_op=False) -> torch.Tensor:
    """
    All-to-all (but different with torch.distributed.all_to_all)

    1. Each device will split the tensor into `len(ranks)` chunks on `odim`
    2. Send each chunk to the corresponding device with `torch.distributed.all_to_all`.
    3. Concatenate the received chunks on `idim`.

    So the overall work is to change the tensor partitioning from `idim` to `odim`.

    Args:
        tensor (torch.Tensor): input tensor
        idim (int): the dimension to concatenate the received chunks
        odim (int): the dimension to split the tensor
        ranks (Tuple[int]): the order of split tensor.
        async_op (bool): whether to use async communication

    Returns:
        torch.Tensor: the output tensor
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    itensors = list(tensor.chunk(len(ranks), dim=odim))
    for idx, itensor in enumerate(itensors):
        itensors[idx] = itensor.contiguous() if not itensor.is_contiguous() else itensor
    otensors = [torch.empty_like(t) for t in itensors]
    group = DeviceGroup().get_group(ranks)
    work = torch.distributed.all_to_all(otensors, itensors, group=group, async_op=async_op)
    if work:
        all2all_callback = lambda t: torch.concat(tuple(otensors), dim=idim)
        AsyncCommHandler().submit(tensor, [work], all2all_callback)
        otensor = tensor
    else:
        otensor = torch.concat(tuple(otensors), dim=idim)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def all_to_all_single(tensor: torch.Tensor, idim: int, odim: int,
                      ranks: Tuple[int], async_op: bool = False) -> torch.Tensor:
    """All-to-all for single tensor"""
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    tensor = tensor.transpose(0, odim) if odim != 0 else tensor
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    group = DeviceGroup().get_group(ranks)
    otensor = torch.empty_like(tensor)
    work = torch.distributed.all_to_all_single(otensor, tensor, group=group, async_op=async_op)

    def all2all_callback(t):
        t = t.transpose(0, odim) if odim != 0 else t
        return torch.concat(tuple(t.chunk(len(ranks), dim=odim)), dim=idim)

    if work:
        AsyncCommHandler().submit(tensor, [work], all2all_callback)
    else:
        otensor = all2all_callback(otensor)

    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def chunk(itensor: torch.Tensor, dim: int, ranks: Tuple[int], async_op=False) -> torch.Tensor:
    """
    split dimension in n chunks and take idx-th chunk

    ranks (Tuple[int]): the order of split tensor.
    """
    idx = tuple(ranks).index(torch.distributed.get_rank())
    with torch.no_grad():
        otensor = itensor.chunk(len(ranks), dim)[idx]
        otensor = otensor.detach()
    return otensor


def rdscatter(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
              dim: int, src: int, dsts: Tuple[int], async_op=False):
    """
    RDScatter: split itensor at rank `src` along dim into `len(dsts)` chunks,
    and then send each chunk to `dst` devices.
    """
    if async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    if rank == src:
        with torch.no_grad():
            otensors = itensor.chunk(len(dsts), dim)
            for dst, otensor in zip(dsts, otensors):
                otensor = otensor.contiguous() if not otensor.is_contiguous() else otensor
                if async_op:
                    work = torch.distributed.isend(otensor, dst)
                    AsyncCommHandler().hold_send(otensor, work)
                else:
                    torch.distributed.send(otensor, dst)
        otensor = itensor
    else:
        assert rank in dsts
        shape = list(shape)
        shape[dim] = shape[dim] // len(dsts)
        otensor = torch.empty(
            shape, requires_grad=False, dtype=dtype,
            device=torch.cuda.current_device()
        )
        if async_op:
            work = torch.distributed.irecv(otensor, src)
            AsyncCommHandler().submit(otensor, [work])
        else:
            torch.distributed.recv(otensor, src)
    if async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def rvscatter(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
              src: int, dsts: Tuple[int], async_op=False):
    """
    src: global rank
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    group = DeviceGroup().get_group((src,) + dsts)
    rank = torch.distributed.get_rank()
    tensor: torch.Tensor = itensor / len(dsts) if src == rank else torch.empty(
        shape,
        dtype=dtype,
        requires_grad=False,
        device=torch.cuda.current_device(),
    )
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    work = torch.distributed.broadcast(tensor, src, group=group, async_op=async_op)
    if work:
        if rank == src:
            AsyncCommHandler().hold_send(tensor, work)
        else:
            AsyncCommHandler().submit(tensor, [work])
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def rdgather(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
             dim: int, srcs: Tuple[int], dst: int, async_op=False):
    """
    @param srcs Tuple[int]: global rank of each source device
    @param dst int: global rank of destination device
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    if rank == dst:
        recv_tensors, works = [], []
        for src in srcs:
            tensor = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
            recv_tensors.append(tensor)
            group, group_src, _ = DeviceGroup().get_p2p_group(src, dst)
            if async_op:
                if group is None:
                    work = torch.distributed.irecv(tensor, src)
                else:
                    work = torch.distributed.irecv(
                        tensor, group=group, group_src=group_src)
                works.append(work)
            else:
                if group is None:
                    torch.distributed.recv(tensor, src)
                else:
                    torch.distributed.recv(
                        tensor, group=group, group_src=group_src)

        if async_op:
            rdgather_callback = lambda t: torch.cat(tuple(recv_tensors), dim=dim)
            otensor = recv_tensors[0]
            AsyncCommHandler().submit(otensor, works, rdgather_callback)
        else:
            otensor = torch.cat(tuple(recv_tensors), dim=dim)
    else:
        assert rank in srcs
        otensor = itensor.contiguous() if not itensor.is_contiguous() else itensor
        group, _, group_dst = DeviceGroup().get_p2p_group(rank, dst)
        if async_op:
            if group is None:
                work = torch.distributed.isend(otensor, dst)
            else:
                work = torch.distributed.isend(
                    otensor, group=group, group_dst=group_dst)
            AsyncCommHandler().hold_send(otensor, work)
        else:
            if group is None:
                torch.distributed.send(otensor, dst)
            else:
                torch.distributed.send(
                    otensor, group=group, group_dst=group_dst)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return otensor


def rvgather(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype,
             srcs: Tuple[int], dst: int, async_op=False):
    """
    @param srcs Tuple[int]: global rank of each source device
    @param dst int: global rank of destination device
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    group = DeviceGroup().get_group(srcs + (dst,))
    tensor = torch.zeros(
        shape,
        dtype=dtype,
        requires_grad=False,
        device=torch.cuda.current_device(),
    ) if rank == dst else itensor
    tensor = tensor.contiguous() if not tensor.is_contiguous() else tensor
    work = torch.distributed.reduce(tensor, dst, group=group, async_op=async_op)
    if work:
        if rank == dst:
            AsyncCommHandler().submit(tensor, [work])
        else:
            AsyncCommHandler().hold_send(tensor, work)
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def broadcast(itensor: torch.Tensor, shape: Tuple[int], dtype: torch.dtype, src: int, ranks: List[int], async_op=False) -> torch.Tensor:
    """
    Broadcast
    @param src: the global rank that holds tensor for broadcasting
    """
    if not async_op:
        CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    group = DeviceGroup().get_group(ranks)
    if rank == src:
        tensor = itensor.contiguous() if not itensor.is_contiguous() else itensor
    else:
        assert rank in ranks
        tensor = torch.empty(shape,
            device=torch.cuda.current_device(), requires_grad=False, dtype=dtype)
    work = torch.distributed.broadcast(tensor, src, group=group, async_op=async_op)
    if work:
        if rank == src:
            AsyncCommHandler().hold_send(tensor, work)
        else:
            AsyncCommHandler().submit(tensor, [work])
    if not async_op:
        CudaTimer().stop(field_name='comm', predefined=True)
    return tensor


def broadcast_object(obj, src: int, ranks: List[int], async_op=False):
    """
    Broadcast a non-tensor object using broadcast_object_list.
    """
    if async_op:
        raise NotImplementedError("Async broadcast_object is not implemented yet")

    if src not in ranks:
        raise ValueError(f"src {src} must be in ranks {ranks}")

    CudaTimer().start(field_name='comm', predefined=True)
    rank = torch.distributed.get_rank()
    group = DeviceGroup().get_group(ranks)
    if rank == src:
        torch.distributed.broadcast_object_list(
            [_serialize_object(obj)], src=src, group=group)
    else:
        assert rank in ranks
        obj_list = [None]
        torch.distributed.broadcast_object_list(obj_list, src=src, group=group)
        obj = _deserialize_object(obj_list[0])

    CudaTimer().stop(field_name='comm', predefined=True)
    return obj
