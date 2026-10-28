# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU review evidence: omitted optimizer ranks versus legitimate replica dedup."""
import pytest
import torch

from nnscaler.runtime.module import AttrMeta, ParallelModule


def _meta(rank, start, stop):
    return {f'p_{rank}': AttrMeta(1, True, 'p', (4,), (slice(start, stop),),
                                1, torch.float32, (stop - start,))}


def _state(value):
    return {'state': {0: {'momentum': torch.tensor(value, dtype=torch.float32)}},
            'param_groups': [{'params': [0]}]}


@pytest.mark.parametrize('duplicate_first_shard', [False, True])
def test_entire_missing_optimizer_shard_is_rejected(duplicate_first_shard):
    fullmaps = [_meta(0, 0, 2), _meta(1, 2, 4)]
    model = [{'p_0': torch.tensor([1., 2.])}, {'p_1': torch.tensor([3., 4.])}]
    optimizers = [_state([11., 12.])]
    if duplicate_first_shard:
        fullmaps = [_meta(0, 0, 2), _meta(1, 0, 2), _meta(2, 2, 4)]
        model = [{'p_0': torch.tensor([1., 2.])}, {'p_1': torch.tensor([1., 2.])},
                 {'p_2': torch.tensor([3., 4.])}]
        optimizers = [_state([11., 12.]), _state([11., 12.])]
    with pytest.raises(ValueError, match='Incomplete optimizer state'):
        ParallelModule.merge_state_dicts(fullmaps, model, optimizers)


def test_whole_replica_optimizer_dedup_remains_valid():
    fullmaps = [_meta(0, 0, 4), _meta(1, 0, 4)]
    model = [{f'p_{rank}': torch.tensor([1., 2., 3., 4.])} for rank in range(2)]
    _, optimizer = ParallelModule.merge_state_dicts(fullmaps, model, [_state([11., 12., 13., 14.])])
    torch.testing.assert_close(optimizer['state'][0]['momentum'], torch.tensor([11., 12., 13., 14.]))


def test_absent_optimizer_parameter_state_remains_valid():
    fullmaps = [_meta(0, 0, 2), _meta(1, 2, 4)]
    model = [{'p_0': torch.tensor([1., 2.])}, {'p_1': torch.tensor([3., 4.])}]
    empty = {'state': {}, 'param_groups': [{'params': [0]}]}
    _, optimizer = ParallelModule.merge_state_dicts(fullmaps, model, [empty])
    assert optimizer['state'] == {}


@pytest.mark.parametrize('zero', [1, 3])
def test_entire_missing_zero_optimizer_bucket_rank_is_rejected(zero):
    fullmaps = [_meta(0, 0, 4), _meta(1, 0, 4)]
    end = 4 if zero == 1 else 2
    zero_map = ({0: (0, 0, end, (4,))}, {0: ([0, 1], 4)}, zero, None)
    with pytest.raises(ValueError, match='Inconsistent optimizer state for bucket'):
        ParallelModule.merge_opt_state_dicts(fullmaps, [_state([11., 12.])], [zero_map, zero_map])


@pytest.mark.parametrize('zero', [1, 3])
def test_complete_zero_optimizer_replica_dedup_remains_valid(zero):
    fullmaps = [_meta(0, 0, 4), _meta(1, 0, 4)]
    zero_maps = [({0: (0, 0, 4, (4,))}, {0: ([rank], 4)}, zero, None) for rank in range(2)]
    optimizer = ParallelModule.merge_opt_state_dicts(
        fullmaps, [_state([11., 12., 13., 14.])], zero_maps,
    )
    torch.testing.assert_close(optimizer['state'][0]['momentum'], torch.tensor([11., 12., 13., 14.]))


@pytest.mark.parametrize('zero', [1, 3])
def test_spatial_shard_coverage_includes_omitted_zero_rank(zero):
    fullmaps = [_meta(0, 0, 2), _meta(1, 2, 4)]
    # This parameter has no reducer of its own in a module using ZeRO.
    zero_map = ({0: 0}, {0: None}, zero, None)
    with pytest.raises(ValueError, match='Incomplete optimizer state'):
        ParallelModule.merge_opt_state_dicts(fullmaps, [_state([11., 12.])], [zero_map, zero_map])
