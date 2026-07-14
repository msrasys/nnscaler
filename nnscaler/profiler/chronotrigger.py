#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""ChronoTrigger trace specifications derived from nnScaler IR adapters."""

from typing import Any, Iterable, Mapping, NamedTuple, Optional, Tuple

from nnscaler.ir.adapter.prim import (
    ChunkPrim,
    CommPrim,
    MovePrim,
    ObjectMovePrim,
    RDGatherPrim,
    RDScatterPrim,
    VChunkPrim,
)


class TraceSpec(NamedTuple):
    kind: str
    entity: str
    peer: Optional[int]


def p2p_trace_spec(
    rank: int,
    adapter_name: str,
    primitive_name: str,
    kwargs: Mapping[str, Any],
    tensor_ids: Iterable[int] = (),
) -> TraceSpec:
    """Return a stable, direction-aware P2P trace specification."""

    sources = _endpoints(kwargs, "src", "srcs")
    destinations = _endpoints(kwargs, "dst", "dsts")
    edge = f"{_format_endpoints(sources)}->{_format_endpoints(destinations)}"
    tensor_key = ",".join(str(tensor_id) for tensor_id in sorted(set(tensor_ids)))
    entity = f"{adapter_name}:{primitive_name}:{edge}"
    if tensor_key:
        entity = f"{entity}:t{tensor_key}"

    in_sources = rank in sources
    in_destinations = rank in destinations
    if in_sources and not in_destinations:
        return TraceSpec("SEND", entity, _single_peer(destinations))
    if in_destinations and not in_sources:
        return TraceSpec("RECV", entity, _single_peer(sources))
    return TraceSpec("COMM", entity, None)


def primitive_trace_spec(prim: Any, rank: int, adapter_name: str, index: int) -> Optional[TraceSpec]:
    """Map a dispatched communication primitive to a generated trace range."""

    if not isinstance(prim, CommPrim) or isinstance(prim, (ChunkPrim, VChunkPrim)):
        return None

    primitive_name = str(prim.signature).rsplit(".", 1)[-1]
    if isinstance(prim, (MovePrim, ObjectMovePrim, RDScatterPrim, RDGatherPrim)):
        tensor_ids = [
            tensor_id
            for tensor in prim.inputs() + prim.outputs()
            if (tensor_id := getattr(tensor, "tid", None)) is not None
        ]
        return p2p_trace_spec(
            rank,
            adapter_name,
            primitive_name,
            prim.kwargs,
            tensor_ids,
        )

    return TraceSpec("COMM", f"{adapter_name}:{primitive_name}:{index}", None)


def _endpoints(kwargs: Mapping[str, Any], singular: str, plural: str) -> Tuple[int, ...]:
    values = kwargs.get(plural)
    if values is None:
        value = kwargs.get(singular)
        values = () if value is None else (value,)
    elif not isinstance(values, (tuple, list, set)):
        values = (values,)
    return tuple(sorted({int(value) for value in values if value is not None}))


def _format_endpoints(endpoints: Tuple[int, ...]) -> str:
    return ",".join(str(endpoint) for endpoint in endpoints) if endpoints else "?"


def _single_peer(endpoints: Tuple[int, ...]) -> Optional[int]:
    return endpoints[0] if len(endpoints) == 1 else None