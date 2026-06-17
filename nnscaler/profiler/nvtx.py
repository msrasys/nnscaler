#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Small opt-in NVTX helpers for generated profiling traces.

Label format
------------

Segment compute ranges:
    ``<action> r<rank> s<stage> mb<microbatch> <segment>``

Examples:
    ``FWD r3 s1 mb7 segment7207``
    ``BWD r4 s14 mb7 segment7898``

Communication adapter ranges:
    ``COMM/<scope> r<rank> [mb<microbatch>] <primitive> <peers/group> <bytes> [tensor=<summary>] [async] <adapter>``

Examples:
    ``COMM/P2P r3 mb7 move 3->5 96.0MB tensor=add_26:id=123:shape=(16384, 1536):dtype=float32 async adapter12606``
    ``COMM/COLL r5 mb7 move+all_gather 3->5+grp[4, 5] 96.0MB tensor=add_26:id=123:shape=(16384, 1536):dtype=float32 adapter12606``

Communication wait ranges:
    ``COMM/WAIT <communication-adapter-range>``

Examples:
    ``COMM/WAIT COMM/P2P r3 mb7 move 3->5 96.0MB tensor=add_26:id=123:shape=(16384, 1536):dtype=float32 async adapter12606``

Weight-reducer ranges:
    ``COMM/COLL r<rank> grad_reduce nparams=<count> <reducer>``

Examples:
    ``COMM/COLL r2 grad_reduce nparams=42 wreducer21820``

These labels describe the logical nnScaler operation. Use Nsight Systems
kernel names and kernel intervals as the source of truth for actual GPU
communication duration; use these labels for schedule context and dependency
attribution.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterable, Iterator, Optional
import os


_MESSAGE_POOL: dict[str, str] = {}
_RANGE_STACK: ContextVar[tuple[str, ...]] = ContextVar('nnscaler_nvtx_range_stack', default=())


def enabled() -> bool:
    value = os.environ.get('NNSCALER_NVTX_TRACE', '')
    return value.lower() not in ('', '0', 'false', 'no', 'off')


def _stable_message(message: str) -> str:
    return _MESSAGE_POOL.setdefault(message, message)


def current_label() -> Optional[str]:
    stack = _RANGE_STACK.get()
    if not stack:
        return None
    return stack[-1]


def join_label(parts: Iterable[object]) -> str:
    return ' '.join(str(part) for part in parts if part not in (None, ''))


def format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None or num_bytes <= 0:
        return ''
    units = ('B', 'KB', 'MB', 'GB', 'TB')
    value = float(num_bytes)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == 'B':
        return f'{int(value)}{unit}'
    return f'{value:.1f}{unit}'


def primitive_name(prim) -> str:
    return str(prim.signature).rsplit('.', 1)[-1]


def primitive_peers(prim) -> str:
    kwargs = prim.kwargs
    if 'src' in kwargs and 'dst' in kwargs:
        return f"{kwargs['src']}->{kwargs['dst']}"
    if 'src' in kwargs and 'dsts' in kwargs:
        return f"{kwargs['src']}->{list(kwargs['dsts'])}"
    if 'srcs' in kwargs and 'dst' in kwargs:
        return f"{list(kwargs['srcs'])}->{kwargs['dst']}"
    ranks = kwargs.get('ranks', None)
    if ranks is None:
        ranks = getattr(prim, 'device', None)
    if ranks is None:
        return ''
    return f'grp{list(ranks)}'


def primitive_bytes(prim) -> Optional[int]:
    try:
        volume = prim.volume()
    except Exception:
        return None
    if volume is None or volume <= 0:
        return None

    dtype = None
    for tensor in prim.inputs() + prim.outputs():
        dtype = getattr(tensor, 'dtype', None)
        if dtype is not None:
            break
    if dtype is None:
        return int(volume)

    try:
        import torch
        return int(volume) * torch.empty((), dtype=dtype).element_size()
    except Exception:
        return int(volume)


def tensor_summary(tensor) -> str:
    name = getattr(tensor, 'name', type(tensor).__name__)
    tid = getattr(tensor, 'tid', None)
    shape = getattr(tensor, 'shape', None)
    dtype = getattr(tensor, 'dtype', None)
    dtype_name = str(dtype).replace('torch.', '') if dtype is not None else ''
    parts = [str(name)]
    if tid is not None:
        parts.append(f'id={tid}')
    if shape is not None:
        parts.append(f'shape={tuple(shape)}')
    if dtype_name:
        parts.append(f'dtype={dtype_name}')
    return ':'.join(parts)


def primitive_tensors(prim, limit: int = 2) -> str:
    tensors = []
    seen = set()
    for tensor in prim.inputs() + prim.outputs():
        tid = getattr(tensor, 'tid', id(tensor))
        if tid in seen:
            continue
        seen.add(tid)
        tensors.append(tensor_summary(tensor))
        if len(tensors) >= limit:
            break
    return '|'.join(tensors)


def is_p2p_primitive(prim) -> bool:
    from nnscaler.ir.adapter.prim import MovePrim, ObjectMovePrim, RDGatherPrim, RDScatterPrim
    return isinstance(prim, (MovePrim, ObjectMovePrim, RDScatterPrim, RDGatherPrim))


def is_traceable_comm_primitive(prim) -> bool:
    from nnscaler.ir.adapter.prim import ChunkPrim, CommPrim, VChunkPrim
    return isinstance(prim, CommPrim) and not isinstance(prim, (ChunkPrim, VChunkPrim))


def adapter_uses_async(adapter) -> bool:
    from nnscaler.flags import CompileFlag
    from nnscaler.ir.adapter.prim import ChunkPrim, CommPrim, MovePrim, VChunkPrim

    has_p2p = any(isinstance(prim, MovePrim) for prim in adapter.prims)
    has_other_comm = any(
        isinstance(prim, CommPrim) and not isinstance(prim, (MovePrim, ChunkPrim, VChunkPrim))
        for prim in adapter.prims
    )
    return CompileFlag.async_comm or (has_p2p and not has_other_comm)


def segment_trace_label(
    action: str,
    rank: Optional[int],
    stage_id: Optional[int],
    micro_batch_id: Optional[int],
    segment_name: str,
) -> str:
    return join_label([
        action,
        f'r{rank}' if rank is not None else None,
        f's{stage_id}' if stage_id is not None else None,
        f'mb{micro_batch_id}' if micro_batch_id is not None else None,
        segment_name,
    ])


def adapter_trace_label(
    adapter,
    rank: Optional[int],
    adapter_name: str,
    micro_batch_id: Optional[int] = None,
) -> Optional[str]:
    prims = [prim for prim in adapter.prims if is_traceable_comm_primitive(prim)]
    if not prims:
        return None

    category = 'COMM/P2P' if all(is_p2p_primitive(prim) for prim in prims) else 'COMM/COLL'
    prim_names = '+'.join(dict.fromkeys(primitive_name(prim) for prim in prims))
    peer_labels = [primitive_peers(prim) for prim in prims]
    peers = '+'.join(dict.fromkeys(peer for peer in peer_labels if peer))
    total_bytes = 0
    has_bytes = False
    for prim in prims:
        prim_bytes = primitive_bytes(prim)
        if prim_bytes is not None:
            total_bytes += prim_bytes
            has_bytes = True
    tensor_labels = '+'.join(dict.fromkeys(label for label in (primitive_tensors(prim) for prim in prims) if label))

    return join_label([
        category,
        f'r{rank}' if rank is not None else None,
        f'mb{micro_batch_id}' if micro_batch_id is not None else None,
        prim_names,
        peers,
        format_bytes(total_bytes) if has_bytes else None,
        f'tensor={tensor_labels}' if tensor_labels else None,
        'async' if adapter_uses_async(adapter) else None,
        adapter_name,
    ])


def weight_reducer_trace_label(reducer, rank: Optional[int], reducer_name: str) -> str:
    return join_label([
        'COMM/COLL',
        f'r{rank}' if rank is not None else None,
        'grad_reduce',
        f'nparams={len(reducer.inputs())}',
        reducer_name,
    ])


def wait_trace_label(comm_label: Optional[str]) -> Optional[str]:
    if comm_label is None:
        return None
    return join_label(['COMM/WAIT', comm_label])


@contextmanager
def range(message: Optional[str]) -> Iterator[None]:
    if message is None or not enabled():
        yield
        return

    import torch

    stack = _RANGE_STACK.get()
    token = _RANGE_STACK.set(stack + (message,))
    torch.cuda.nvtx.range_push(_stable_message(message))
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()
        _RANGE_STACK.reset(token)
