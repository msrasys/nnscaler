# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ZeRO-3 optimizer padding must not be treated as another parameter replica."""
import copy
import math

import pytest
import torch

from nnscaler.runtime.module import AttrMeta, ParallelModule


def _checkpoint(shape, ranks, *, metadata=True, ranges=None):
    numel = math.prod(shape)
    chunk_size = math.ceil(numel / ranks)
    ranges = ranges or [(rank * chunk_size, min((rank + 1) * chunk_size, numel))
                        for rank in range(ranks)]
    fullmaps, states, zero_maps = [], [], []
    expected = torch.arange(1, numel + 1, dtype=torch.float32)
    # The parameter is inside a bucket, with unrelated data on both sides.
    bucket_size, offset = 8 * ranks, 3
    for rank, (start, end) in enumerate(ranges):
        name = f'p_{rank}'
        fullmaps.append({name: AttrMeta(0, True, 'p', shape,
                         tuple(slice(0, size) for size in shape), 1, torch.float32, shape)})
        state = {'step': torch.tensor(3.)}
        for key, multiplier in [('exp_avg', 1), ('exp_avg_sq', 2)]:
            # Deliberately nonzero padding catches accidental copying/comparison.
            value = torch.full((bucket_size,), 1000. + rank)
            if end > start:
                value[offset:offset + end - start] = expected[start:end] * multiplier
            state[key] = value
        states.append({'state': {0: state}, 'param_groups': [{'params': [0]}]})
        zero3 = {name: {'orig_name': 'p', 'attr_name': name, 'start': start,
                       'end': end, 'chunk_size': chunk_size}} if metadata else None
        zero_maps.append(({0: (0, offset, offset + chunk_size, (chunk_size,))},
                          {0: (list(range(ranks)), bucket_size)}, 3, zero3))
    return fullmaps, states, zero_maps, expected.reshape(shape)


@pytest.mark.parametrize('shape,ranks', [((), 2), ((), 4), ((3,), 2), ((3,), 4), ((5,), 4), ((2, 3), 4)])
@pytest.mark.parametrize('metadata', [False, True])
def test_zero3_merge_skips_full_and_partial_padding(shape, ranks, metadata):
    fullmaps, states, zero_maps, expected = _checkpoint(shape, ranks, metadata=metadata)
    original = copy.deepcopy(states)
    merged = ParallelModule.merge_opt_state_dicts(fullmaps, states, zero_maps)
    for key, multiplier in [('exp_avg', 1), ('exp_avg_sq', 2)]:
        torch.testing.assert_close(merged['state'][0][key], expected * multiplier, atol=0, rtol=0)
        for before, after in zip(original, states):
            assert torch.equal(before['state'][0][key], after['state'][0][key])


@pytest.mark.parametrize('conflict', [False, True])
def test_zero3_metadata_distinguishes_replicas_from_padding(conflict):
    # Two actual replicas of a scalar have identical real ranges [0, 1).
    fullmaps, states, zero_maps, expected = _checkpoint((), 2, ranges=[(0, 1), (0, 1)])
    if conflict:
        states[1]['state'][0]['exp_avg'][3] += 10
        with pytest.raises(ValueError, match='Conflict in merging optimizer state'):
            ParallelModule.merge_opt_state_dicts(fullmaps, states, zero_maps)
    else:
        merged = ParallelModule.merge_opt_state_dicts(fullmaps, states, zero_maps)
        torch.testing.assert_close(merged['state'][0]['exp_avg'], expected, atol=0, rtol=0)


def test_zero3_metadata_rejects_missing_shard_despite_present_bucket_ranks():
    # Every rank has a bucket, but both claim the same first spatial chunk.
    fullmaps, states, zero_maps, _ = _checkpoint((3,), 2, ranges=[(0, 2), (0, 2)])
    with pytest.raises(ValueError, match='Incomplete optimizer state'):
        ParallelModule.merge_opt_state_dicts(fullmaps, states, zero_maps)


def test_legacy_zero3_rejects_insufficient_chunk_coverage():
    fullmaps, states, zero_maps, _ = _checkpoint((3,), 2, metadata=False)
    for model_indices, _, _, _ in zero_maps:
        # Both bucket ranks exist, but two one-element chunks cannot cover p.
        model_indices[0] = (0, 3, 4, (1,))
    with pytest.raises(ValueError, match='Incomplete optimizer state'):
        ParallelModule.merge_opt_state_dicts(fullmaps, states, zero_maps)


def test_legacy_zero3_without_parameter_metadata_skips_padding():
    fullmaps, states, zero_maps, expected = _checkpoint((), 2, metadata=False)
    merged = ParallelModule.merge_opt_state_dicts(fullmaps, states, [info[:3] for info in zero_maps])
    torch.testing.assert_close(merged['state'][0]['exp_avg'], expected, atol=0, rtol=0)


@pytest.mark.parametrize('metadata', [False, True])
def test_zero3_missing_bucket_rank_is_still_rejected(metadata):
    fullmaps, states, zero_maps, _ = _checkpoint((3,), 2, metadata=metadata)
    with pytest.raises(ValueError, match='Inconsistent optimizer state for bucket'):
        ParallelModule.merge_opt_state_dicts(fullmaps, states[:1], zero_maps)
