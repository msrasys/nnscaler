import pytest
import torch

import nnscaler.cli.scale_unit_parallelism as scale_unit_parallelism
from nnscaler.cli.scale_unit_parallelism import (
    dp_scale_unit_all_gather,
    dp_scale_unit_chunk,
    dp_scale_unit_ranks,
    ep_scale_unit_all_gather,
    ep_scale_unit_chunk,
    ep_scale_unit_lane_ranks,
    ep_scale_unit_ranks,
)


_INVALID_SCALE_UNIT_SIZES = (
    (0, 1, 'plan_ngpus'),
    (-1, 1, 'plan_ngpus'),
    (1, 0, 'num_scale_units'),
    (1, -1, 'num_scale_units'),
)


@pytest.mark.parametrize(
    'rank_helper',
    (dp_scale_unit_ranks, ep_scale_unit_ranks, ep_scale_unit_lane_ranks),
)
@pytest.mark.parametrize(
    ('plan_ngpus', 'num_scale_units', 'invalid_name'),
    _INVALID_SCALE_UNIT_SIZES,
)
def test_scale_unit_rank_helpers_require_positive_sizes(
    rank_helper,
    plan_ngpus,
    num_scale_units,
    invalid_name,
):
    with pytest.raises(ValueError, match=rf'{invalid_name} must be positive'):
        rank_helper(plan_ngpus, num_scale_units, rank=0)


@pytest.mark.parametrize(
    'tensor_helper',
    (
        dp_scale_unit_chunk,
        dp_scale_unit_all_gather,
        ep_scale_unit_chunk,
        ep_scale_unit_all_gather,
    ),
)
@pytest.mark.parametrize(
    ('plan_ngpus', 'num_scale_units', 'invalid_name'),
    _INVALID_SCALE_UNIT_SIZES,
)
def test_scale_unit_tensor_helpers_require_positive_sizes(
    tensor_helper,
    plan_ngpus,
    num_scale_units,
    invalid_name,
):
    with pytest.raises(ValueError, match=rf'{invalid_name} must be positive'):
        tensor_helper(torch.ones(2), plan_ngpus, num_scale_units, dim=0)


@pytest.mark.parametrize(
    'fake_forward',
    (
        scale_unit_parallelism._DpScaleUnitChunk.fake_forward,
        scale_unit_parallelism._DpScaleUnitAllGather.fake_forward,
        scale_unit_parallelism._EpScaleUnitChunk.fake_forward,
        scale_unit_parallelism._EpScaleUnitAllGather.fake_forward,
    ),
)
@pytest.mark.parametrize(
    ('plan_ngpus', 'num_scale_units', 'invalid_name'),
    _INVALID_SCALE_UNIT_SIZES,
)
def test_scale_unit_fake_helpers_require_positive_sizes(
    fake_forward,
    plan_ngpus,
    num_scale_units,
    invalid_name,
):
    with pytest.raises(ValueError, match=rf'{invalid_name} must be positive'):
        fake_forward(torch.ones(2), plan_ngpus, num_scale_units, dim=0)