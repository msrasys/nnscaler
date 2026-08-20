#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Rank topology and differentiable collectives for nnScaler scale units.

The helpers distinguish communication within one scale unit from communication
across the same plan-local lane of adjacent scale units. All chunk/gather
operations require equal-sized shards.
"""

from typing import Optional

import torch
import torch.distributed as dist

import nnscaler
from nnscaler.runtime.adapter.collectives import all_gather, chunk


__all__ = [
    'scale_unit_ranks',
    'cross_scale_unit_ranks',
    'cross_scale_unit_lane_ranks',
    'inner_scale_unit_chunk',
    'inner_scale_unit_all_gather',
    'cross_scale_unit_chunk',
    'cross_scale_unit_all_gather',
]


def scale_unit_ranks(plan_ngpus: int, rank: Optional[int] = None) -> tuple[int, ...]:
    """
    Get the ranks of the current scale unit.

    Args:
        plan_ngpus (int): Number of ranks in one scale unit.
        rank (Optional[int]): Global rank whose scale unit is requested. Uses
            the current distributed rank when omitted.

    Returns:
        tuple[int, ...]: Ranks of the current scale unit.
    """
    rank = dist.get_rank() if rank is None else rank
    first_rank = rank // plan_ngpus * plan_ngpus
    return tuple(range(first_rank, first_rank + plan_ngpus))


def cross_scale_unit_ranks(
    plan_ngpus: int,
    num_scale_units: int,
    rank: Optional[int] = None,
) -> tuple[int, ...]:
    """
    Get the contiguous ranks in the current cross-scale-unit group.

    A group contains ``num_scale_units`` adjacent scale units, with
    ``plan_ngpus`` ranks in each unit. For example, with ``plan_ngpus=2`` and
    ``num_scale_units=2``, ranks 0-3 form ``(0, 1, 2, 3)`` and ranks 4-7 form
    ``(4, 5, 6, 7)``.

    Args:
        plan_ngpus (int): Number of ranks in one scale unit.
        num_scale_units (int): Number of adjacent scale units in one group.
        rank (Optional[int]): Global rank whose group is requested. Uses the
            current distributed rank when omitted.

    Returns:
        tuple[int, ...]: Global ranks in the current cross-scale-unit group.
    """
    rank = dist.get_rank() if rank is None else rank
    group_size = plan_ngpus * num_scale_units
    first_rank = rank // group_size * group_size
    return tuple(range(first_rank, first_rank + group_size))


def cross_scale_unit_lane_ranks(
    plan_ngpus: int,
    num_scale_units: int,
    rank: Optional[int] = None,
) -> tuple[int, ...]:
    """
    Get ranks at the same plan-local lane across scale units.

    A lane is a rank's position inside its scale unit, calculated as
    ``rank % plan_ngpus``. Equivalently, it is the device index in the compiled
    plan that the rank executes. Ranks in the same lane belong to different
    scale units but execute the same plan-device role.

    For example,
        with ``plan_ngpus=2`` and ``num_scale_units=4``, the scale units are::

            scale unit 0: ranks (0, 1)
            scale unit 1: ranks (2, 3)
            scale unit 2: ranks (4, 5)
            scale unit 3: ranks (6, 7)

        Lane 0 contains the first rank of every scale unit, ``(0, 2, 4, 6)``.
        Lane 1 contains the second rank, ``(1, 3, 5, 7)``.

    Args:
        plan_ngpus (int): Number of GPUs in the scale unit.
        num_scale_units (int): Number of adjacent scale units in one group.
        rank (Optional[int]): Global rank whose lane is requested. Uses the
            current distributed rank when omitted.

    Returns:
        tuple[int, ...]: Global ranks in the current rank's plan-local lane,
            ordered by scale-unit index.
    """
    rank = dist.get_rank() if rank is None else rank
    group_size = plan_ngpus * num_scale_units
    group_start = rank // group_size * group_size
    lane = rank % plan_ngpus
    return tuple(range(group_start + lane, group_start + group_size, plan_ngpus))


class _InnerScaleUnitChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, plan_ngpus: int, dim: int) -> torch.Tensor:
        ctx.ranks = scale_unit_ranks(plan_ngpus)
        ctx.dim = dim
        return chunk(x, dim=ctx.dim, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return all_gather(grad, dim=ctx.dim, ranks=ctx.ranks), None, None

    @staticmethod
    def fake_forward(x: torch.Tensor, plan_ngpus: int, dim: int) -> torch.Tensor:
        if x.shape[dim] % plan_ngpus != 0:
            raise ValueError('tensor dimension must be divisible by plan_ngpus')
        return x.chunk(plan_ngpus, dim=dim)[0]


class _InnerScaleUnitAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, plan_ngpus: int, dim: int) -> torch.Tensor:
        ctx.ranks = scale_unit_ranks(plan_ngpus)
        ctx.dim = dim
        return all_gather(x, dim=ctx.dim, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return chunk(grad, dim=ctx.dim, ranks=ctx.ranks), None, None

    @staticmethod
    def fake_forward(x: torch.Tensor, plan_ngpus: int, dim: int) -> torch.Tensor:
        return torch.cat([x] * plan_ngpus, dim=dim)


@nnscaler.register_op('? -> ?', fake_fn=_InnerScaleUnitChunk.fake_forward)
def inner_scale_unit_chunk(x: torch.Tensor, plan_ngpus: int, dim: int) -> torch.Tensor:
    """Split ``x`` across ranks in the current scale unit.

    Forward selects this rank's equal chunk along ``dim``; backward gathers all
    chunks to reconstruct the input gradient.

    Args:
        x (torch.Tensor): Full tensor available on every rank in the scale unit.
        plan_ngpus (int): Number of ranks in the scale unit.
        dim (int): Dimension to split.

    Returns:
        torch.Tensor: This rank's chunk of ``x``.
    """
    return _InnerScaleUnitChunk.apply(x, plan_ngpus, dim)


@nnscaler.register_op('? -> ?', fake_fn=_InnerScaleUnitAllGather.fake_forward)
def inner_scale_unit_all_gather(x: torch.Tensor, plan_ngpus: int, dim: int) -> torch.Tensor:
    """Gather equal chunks from ranks in the current scale unit.

    Forward concatenates along ``dim``; backward selects this rank's matching
    chunk from the full output gradient.

    Args:
        x (torch.Tensor): This rank's equal-sized local chunk.
        plan_ngpus (int): Number of ranks in the scale unit.
        dim (int): Dimension along which chunks are concatenated.

    Returns:
        torch.Tensor: Full tensor replicated on the scale-unit ranks.
    """
    return _InnerScaleUnitAllGather.apply(x, plan_ngpus, dim)


class _CrossScaleUnitChunk(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        plan_ngpus: int,
        num_scale_units: int,
        dim: int,
    ) -> torch.Tensor:
        ctx.ranks = cross_scale_unit_lane_ranks(
            plan_ngpus,
            num_scale_units,
        )
        ctx.dim = dim
        return chunk(x, dim=ctx.dim, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return all_gather(grad, dim=ctx.dim, ranks=ctx.ranks), None, None, None

    @staticmethod
    def fake_forward(x: torch.Tensor, plan_ngpus: int, num_scale_units: int, dim: int) -> torch.Tensor:
        if x.shape[dim] % num_scale_units != 0:
            raise ValueError('tensor dimension must be divisible by num_scale_units')
        return x.chunk(num_scale_units, dim=dim)[0]


class _CrossScaleUnitAllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        plan_ngpus: int,
        num_scale_units: int,
        dim: int,
    ) -> torch.Tensor:
        ctx.ranks = cross_scale_unit_lane_ranks(
            plan_ngpus,
            num_scale_units,
        )
        ctx.dim = dim
        return all_gather(x, dim=ctx.dim, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return chunk(grad, dim=ctx.dim, ranks=ctx.ranks), None, None, None

    @staticmethod
    def fake_forward(x: torch.Tensor, plan_ngpus: int, num_scale_units: int, dim: int) -> torch.Tensor:
        return torch.cat([x] * num_scale_units, dim=dim)


@nnscaler.register_op(
    '? -> ?',
    fake_fn=_CrossScaleUnitChunk.fake_forward,
)
def cross_scale_unit_chunk(
    x: torch.Tensor,
    plan_ngpus: int,
    num_scale_units: int,
    dim: int,
) -> torch.Tensor:
    """Split ``x`` across the same lane of adjacent scale units.

    The communication group is returned by :func:`cross_scale_unit_lane_ranks`.
    Forward selects this rank's equal chunk along ``dim``; backward gathers the
    chunks to reconstruct the input gradient.

    Args:
        x (torch.Tensor): Full tensor available at the same lane of every scale
            unit in the group.
        plan_ngpus (int): Number of ranks in one scale unit.
        num_scale_units (int): Number of adjacent scale units in the group.
        dim (int): Dimension to split.

    Returns:
        torch.Tensor: This scale unit's chunk for the current lane.
    """
    return _CrossScaleUnitChunk.apply(x, plan_ngpus, num_scale_units, dim)


@nnscaler.register_op(
    '? -> ?',
    fake_fn=_CrossScaleUnitAllGather.fake_forward,
)
def cross_scale_unit_all_gather(
    x: torch.Tensor,
    plan_ngpus: int,
    num_scale_units: int,
    dim: int,
) -> torch.Tensor:
    """Gather equal chunks across the same lane of adjacent scale units.

    Forward concatenates along ``dim``; backward selects this rank's matching
    chunk from the full output gradient.

    Args:
        x (torch.Tensor): Equal-sized local chunk for the current lane.
        plan_ngpus (int): Number of ranks in one scale unit.
        num_scale_units (int): Number of adjacent scale units in the group.
        dim (int): Dimension along which chunks are concatenated.

    Returns:
        torch.Tensor: Full tensor replicated across the same-lane ranks.
    """
    return _CrossScaleUnitAllGather.apply(x, plan_ngpus, num_scale_units, dim)
