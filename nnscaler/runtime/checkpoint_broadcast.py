# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Bounded CPU staging for checkpoint tensors broadcast into final storage."""
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist


DEFAULT_CHUNK_BYTES = 64 * 1024 * 1024


def _flat_chunks(tensor: torch.Tensor, max_numel: int) -> Iterator[torch.Tensor]:
    """Yield logical contiguous chunks without flattening a large strided input."""
    if tensor.numel() == 0:
        return
    if tensor.is_contiguous():
        flat = tensor.view(-1)
        for offset in range(0, flat.numel(), max_numel):
            yield flat[offset:offset + max_numel]
    elif tensor.numel() <= max_numel:
        yield tensor.contiguous().view(-1)
    else:
        # Recurse until a view fits the staging budget. This path is uncommon
        # for flattened optimizer buckets, but preserves arbitrary strides.
        for index in range(tensor.shape[0]):
            yield from _flat_chunks(tensor.select(0, index), max_numel)


@torch.no_grad()
def broadcast_checkpoint_tensors(
    tensors: Sequence[torch.Tensor] | None,
    metadata: Sequence[tuple[tuple[int, ...], torch.dtype]],
    *,
    src: int,
    group: dist.ProcessGroup | None,
    device: torch.device,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> list[torch.Tensor]:
    """Broadcast checkpoint tensors directly into their destination storage.

    Metadata and ``chunk_bytes`` must agree on every member. CPU->CUDA copies
    use two reusable pinned buffers (at most 2*chunk_bytes total), a copy stream,
    and event dependencies on the caller's stream. NCCL operates directly on
    final tensor slices; no second full-size CUDA staging copy is allocated.
    At most two transfers are in flight. On return all copy/broadcast events
    have completed; buffers cannot outlive their CPU or CUDA source storage.

    For CPU process groups the same chunk schedule is synchronous. Singleton
    groups need no broadcast. Already resident source tensors are retained,
    as in the original loader; otherwise the destination has independent storage.
    """
    if chunk_bytes <= 0:
        raise ValueError('chunk_bytes must be positive')
    device = torch.device(device)
    rank = dist.get_rank()
    is_source = rank == src
    if is_source and (tensors is None or len(tensors) != len(metadata)):
        raise ValueError('The source must provide one tensor per metadata entry')
    if dist.get_world_size(group) == 1:
        return [tensor.to(device) for tensor in tensors]

    current = torch.cuda.current_stream(device) if device.type == 'cuda' else None
    copy_stream = None
    staging = []
    copied = [None, None]
    completed = [None, None]
    staging_bytes = min(chunk_bytes, max(
        (tensor.numel() * tensor.element_size() for tensor in tensors
         if tensor.device.type == 'cpu'), default=0)) if is_source else 0
    slot_index = 0
    transfer_index = 0
    result = []
    try:
        for index, (shape, dtype) in enumerate(metadata):
            source = tensors[index] if is_source else None
            if source is not None and (tuple(source.shape) != tuple(shape) or source.dtype != dtype):
                raise ValueError('Source tensor does not match checkpoint metadata')
            # A contiguous tensor already on the target device can be sent
            # directly, as in the original checkpoint loader.
            direct = source is not None and source.device == device and source.is_contiguous()
            target = source if direct else torch.empty(shape, dtype=dtype, device=device)
            result.append(target)
            capacity = max(1, chunk_bytes // target.element_size())
            if target.element_size() > chunk_bytes:
                raise ValueError('chunk_bytes must hold at least one tensor element')
            flat = target.view(-1)
            # Use identical offsets on sender/receivers even for strided input.
            # The source packs a fixed-size slice from bounded logical chunks.
            pieces = iter(_flat_chunks(source, capacity)) if is_source and not direct else None
            pending = None
            offset = 0
            while offset < flat.numel():
                transfer_slot = transfer_index % 2
                if completed[transfer_slot] is not None:
                    completed[transfer_slot].synchronize()
                count = min(capacity, flat.numel() - offset)
                destination = flat[offset:offset + count]
                if is_source and not direct:
                    use_pinned = source.device.type == 'cpu' and device.type == 'cuda'
                    if use_pinned:
                        if copy_stream is None:
                            copy_stream = torch.cuda.Stream(device=device)
                            staging = [torch.empty(staging_bytes, dtype=torch.uint8, pin_memory=True)
                                       for _ in range(2)]
                        if offset == 0:
                            # Account for destination allocator reuse on the
                            # caller's stream before writing from another one.
                            copy_stream.wait_stream(current)
                        slot = slot_index % 2
                        if copied[slot] is not None:
                            copied[slot].synchronize()  # CPU can now overwrite this slot.
                        packed = staging[slot][:count * target.element_size()].view(dtype)
                    else:
                        packed = destination
                    filled = 0
                    while filled < count:
                        if pending is None or pending.numel() == 0:
                            pending = next(pieces)
                        n = min(count - filled, pending.numel())
                        packed[filled:filled+n].copy_(pending[:n])
                        pending = pending[n:]
                        filled += n
                    if use_pinned:
                        with torch.cuda.stream(copy_stream):
                            destination.copy_(packed, non_blocking=True)
                            target.record_stream(copy_stream)
                            event = torch.cuda.Event()
                            event.record(copy_stream)
                        copied[slot] = event
                        current.wait_event(event)
                        slot_index += 1
                # Work.wait establishes the dependency on the current stream
                # for NCCL; it does not force a device-wide synchronize.
                work = dist.broadcast(destination, src=src, group=group, async_op=True)
                work.wait()
                if current is not None:
                    event = torch.cuda.Event()
                    event.record(current)
                    completed[transfer_slot] = event
                transfer_index += 1
                offset += count
    finally:
        # A failure must not release pinned buffers still read by a CUDA copy.
        for event in copied:
            if event is not None:
                event.synchronize()
        for event in completed:
            if event is not None:
                event.synchronize()
    return result
