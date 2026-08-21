#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""Parallelism helpers within and across nnScaler scale units.

An nnScaler plan describes ``plan_ngpus`` ranks and is repeated at runtime as
one or more scale units. This module supports two ways to use those ranks:

DP over scale units
-------------------
``dp_scale_unit_*`` treats every rank in ``num_scale_units`` adjacent scale
units as one data-parallel group. The group size is
``plan_ngpus * num_scale_units``. This is useful when a replicated or opaque
region should process one logical batch without repeating the same computation
on every rank. All ranks have the same computational role and own equivalent
copies of the region's parameters; only their data shards differ.

Use :func:`dp_scale_unit_chunk` before the region to split the chosen tensor
dimension over all ranks in the group, then use
:func:`dp_scale_unit_all_gather` to restore the full tensor afterward. Parameters
that consume these complementary data shards should be assigned the bucket
configuration returned by :func:`dp_scale_unit_param_config`, which preserves
the SUM of their gradient contributions.

EP across scale units
---------------------
``ep_scale_unit_*`` assumes each scale unit runs the same expert-parallel plan.
A rank's offset inside the plan is its *lane*; the same lane in different scale
units owns the same expert shard. Activation partitioning across scale units
therefore communicates only among corresponding lanes, using
:func:`ep_scale_unit_chunk` and :func:`ep_scale_unit_all_gather`. The complete
contiguous union of those scale units, returned by
:func:`ep_scale_unit_ranks`, can be used for context-wide communication.

Plan-level EP already determines the reducer groups for matching expert shards,
so :func:`ep_scale_unit_param_config` inherits the generated reducer settings.

Reducer semantics
-----------------
Parameter ownership and gradient duplication are separate concerns:

* Parameter ownership determines the reducer group. In DP, all ranks own the
    same complete parameters. In EP, different lanes own different expert shards,
    so only ranks in matching lanes reduce the same parameters.
* Gradient duplication determines ``reducer_nreplicas``. Contributions from
    disjoint samples, sequence shards, or routed tokens are complementary and
    must remain summed. Only completely repeated contributions should be divided
    by their repetition count.

The manual DP split is outside the compiled plan, so the compiler cannot infer
that formerly equivalent replicas now process complementary data. Therefore
:func:`dp_scale_unit_param_config` sets ``reducer_nreplicas=1``. In the EP case,
the plan already describes expert ownership and its generated reducers normally
have the correct divisor, so :func:`ep_scale_unit_param_config` leaves them
unchanged. This is not an unconditional EP rule: if matching expert replicas
repeat the same data and computation, their bucket still needs a divisor equal
to that actual repetition count.

All chunk/gather helpers are differentiable: chunk forward pairs with
all-gather backward, and all-gather forward pairs with chunk backward. They
require equal-sized shards, and every required process group must be created by
all ranks in the same deterministic order before execution.
"""

from typing import Optional, TYPE_CHECKING

import torch
import torch.distributed as dist

import nnscaler
from nnscaler.runtime.adapter.collectives import all_gather, chunk


if TYPE_CHECKING:
    from nnscaler.cli.trainer_args import TrainerArgs


__all__ = [
    'dp_scale_unit_ranks',
    'ep_scale_unit_ranks',
    'ep_scale_unit_lane_ranks',
    'dp_scale_unit_chunk',
    'dp_scale_unit_all_gather',
    'dp_scale_unit_param_config',
    'ep_scale_unit_chunk',
    'ep_scale_unit_all_gather',
    'ep_scale_unit_param_config',
]


def _validate_scale_unit_sizes(plan_ngpus: int, num_scale_units: int) -> None:
    if plan_ngpus <= 0:
        raise ValueError(f'plan_ngpus must be positive, but got {plan_ngpus}')
    if num_scale_units <= 0:
        raise ValueError(f'num_scale_units must be positive, but got {num_scale_units}')


def dp_scale_unit_ranks(plan_ngpus: int, num_scale_units: int = 1, rank: Optional[int] = None) -> tuple[int, ...]:
    """
    Get ranks in the current contiguous scale-unit DP group.

    The group contains every plan lane from ``num_scale_units`` adjacent scale
    units. For ``plan_ngpus=2`` and ``num_scale_units=2``, ranks 0-3 form one
    DP group and ranks 4-7 form the next group.

    Args:
        plan_ngpus (int): Number of ranks in one scale unit.
        num_scale_units (int): Number of adjacent scale units in one group.
        rank (Optional[int]): Global rank whose DP group is requested. Uses
            the current distributed rank when omitted.

    Returns:
        tuple[int, ...]: Global ranks in the current scale-unit DP group.
    """
    _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
    group_size = plan_ngpus * num_scale_units
    rank = dist.get_rank() if rank is None else rank
    first_rank = rank // group_size * group_size
    return tuple(range(first_rank, first_rank + group_size))


def ep_scale_unit_ranks(
    plan_ngpus: int,
    num_scale_units: int,
    rank: Optional[int] = None,
) -> tuple[int, ...]:
    """
    Get the contiguous ranks in the current expert-parallel scale-unit group.

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
        tuple[int, ...]: Global ranks in the current expert-parallel scale-unit group.
    """
    _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
    group_size = plan_ngpus * num_scale_units
    rank = dist.get_rank() if rank is None else rank
    first_rank = rank // group_size * group_size
    return tuple(range(first_rank, first_rank + group_size))


def ep_scale_unit_lane_ranks(
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
    _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
    group_size = plan_ngpus * num_scale_units
    rank = dist.get_rank() if rank is None else rank
    group_start = rank // group_size * group_size
    lane = rank % plan_ngpus
    return tuple(range(group_start + lane, group_start + group_size, plan_ngpus))


class _DpScaleUnitChunk(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, plan_ngpus: int, num_scale_units: int, dim: int) -> torch.Tensor:
        ctx.ranks = dp_scale_unit_ranks(plan_ngpus, num_scale_units)
        ctx.dim = dim
        return chunk(x, dim=ctx.dim, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return all_gather(grad, dim=ctx.dim, ranks=ctx.ranks), None, None, None

    @staticmethod
    def fake_forward(x: torch.Tensor, plan_ngpus: int, num_scale_units: int, dim: int) -> torch.Tensor:
        _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
        group_size = plan_ngpus * num_scale_units
        if x.shape[dim] % group_size != 0:
            raise ValueError('tensor dimension must be divisible by group_size')
        return x.chunk(group_size, dim=dim)[0]


class _DpScaleUnitAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, plan_ngpus: int, num_scale_units: int, dim: int) -> torch.Tensor:
        ctx.ranks = dp_scale_unit_ranks(plan_ngpus, num_scale_units)
        ctx.dim = dim
        return all_gather(x, dim=ctx.dim, ranks=ctx.ranks)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return chunk(grad, dim=ctx.dim, ranks=ctx.ranks), None, None, None

    @staticmethod
    def fake_forward(x: torch.Tensor, plan_ngpus: int, num_scale_units: int, dim: int) -> torch.Tensor:
        _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
        group_size = plan_ngpus * num_scale_units
        return torch.cat([x] * group_size, dim=dim)


@nnscaler.register_op('? -> ?', fake_fn=_DpScaleUnitChunk.fake_forward)
def dp_scale_unit_chunk(x: torch.Tensor, plan_ngpus: int, num_scale_units: int, dim: int) -> torch.Tensor:
    """Split ``x`` across a contiguous scale-unit DP group.

    Forward selects this rank's equal chunk along ``dim``; backward gathers all
    chunks to reconstruct the input gradient.

    Args:
        x (torch.Tensor): Full tensor available on every rank in the DP group.
        plan_ngpus (int): Number of ranks in one scale unit.
        num_scale_units (int): Number of adjacent scale units in the group.
        dim (int): Dimension to split.

    Returns:
        torch.Tensor: This rank's chunk of ``x``.
    """
    _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
    return _DpScaleUnitChunk.apply(x, plan_ngpus, num_scale_units, dim)


@nnscaler.register_op('? -> ?', fake_fn=_DpScaleUnitAllGather.fake_forward)
def dp_scale_unit_all_gather(x: torch.Tensor, plan_ngpus: int, num_scale_units: int, dim: int) -> torch.Tensor:
    """Gather equal chunks from a contiguous scale-unit DP group.

    Forward concatenates along ``dim``; backward selects this rank's matching
    chunk from the full output gradient.

    Args:
        x (torch.Tensor): This rank's equal-sized local chunk.
        plan_ngpus (int): Number of ranks in one scale unit.
        num_scale_units (int): Number of adjacent scale units in the group.
        dim (int): Dimension along which chunks are concatenated.

    Returns:
        torch.Tensor: Full tensor replicated on all ranks in the DP group.
    """
    _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
    return _DpScaleUnitAllGather.apply(x, plan_ngpus, num_scale_units, dim)


def dp_scale_unit_param_config(trainer_args: 'TrainerArgs') -> dict:
    """Return bucket settings for parameters consuming DP-group data shards.

    Ranks in the DP group process complementary data, so their gradient
    contributions must remain summed after the reducer collective.

    Args:
        trainer_args (TrainerArgs): Trainer configuration. Accepted so this
            helper can be called from a two-argument ``param_clss_fn``.

    Returns:
        dict: A :class:`ParamBucketConfig`-compatible mapping that preserves the
            collective SUM by setting ``reducer_nreplicas`` to 1.
    """
    return dict(reducer_nreplicas=1)


class _EpScaleUnitChunk(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        plan_ngpus: int,
        num_scale_units: int,
        dim: int,
    ) -> torch.Tensor:
        ctx.ranks = ep_scale_unit_lane_ranks(
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
        _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
        if x.shape[dim] % num_scale_units != 0:
            raise ValueError('tensor dimension must be divisible by num_scale_units')
        return x.chunk(num_scale_units, dim=dim)[0]


class _EpScaleUnitAllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        plan_ngpus: int,
        num_scale_units: int,
        dim: int,
    ) -> torch.Tensor:
        ctx.ranks = ep_scale_unit_lane_ranks(
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
        _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
        return torch.cat([x] * num_scale_units, dim=dim)


@nnscaler.register_op(
    '? -> ?',
    fake_fn=_EpScaleUnitChunk.fake_forward,
)
def ep_scale_unit_chunk(
    x: torch.Tensor,
    plan_ngpus: int,
    num_scale_units: int,
    dim: int,
) -> torch.Tensor:
    """Split ``x`` across the same lane of adjacent scale units.

    The communication group is returned by :func:`ep_scale_unit_lane_ranks`.
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
    _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
    return _EpScaleUnitChunk.apply(x, plan_ngpus, num_scale_units, dim)


@nnscaler.register_op(
    '? -> ?',
    fake_fn=_EpScaleUnitAllGather.fake_forward,
)
def ep_scale_unit_all_gather(
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
    _validate_scale_unit_sizes(plan_ngpus, num_scale_units)
    return _EpScaleUnitAllGather.apply(x, plan_ngpus, num_scale_units, dim)


def ep_scale_unit_param_config(trainer_args: 'TrainerArgs') -> dict:
    """Return bucket settings for EP activation sharding across scale units.

    Corresponding lanes in different scale units own the same plan-level expert
    shard. Splitting activations among those lanes does not change the parameter
    replica count inferred from the EP weight layout, so parameters should
    inherit their generated reducer configuration.

    Args:
        trainer_args (TrainerArgs): Trainer configuration. Accepted so this
            helper can be called from a two-argument ``param_clss_fn``.

    Returns:
        dict: An empty mapping, which inherits all reducer bucket defaults.
    """
    return {}
