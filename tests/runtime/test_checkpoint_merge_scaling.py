# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import time

import pytest
import torch

from nnscaler.runtime.module import AttrMeta, ParallelModule


class CountedMap(dict):
    def __init__(self, counter):
        super().__init__()
        self.counter = counter

    def items(self):
        for item in super().items():
            self.counter[0] += 1
            yield item

    def values(self):
        for value in super().values():
            self.counter[0] += 1
            yield value


@pytest.mark.parametrize('parameter_count', [64, 128])
@pytest.mark.parametrize('deduped', [False, True])
def test_optimizer_merge_scans_metadata_linearly(parameter_count, deduped):
    rank_count = 8
    counter = [0]
    fullmaps, states = [], []
    for rank in range(rank_count):
        fullmap = CountedMap(counter)
        fullmap['buffer'] = AttrMeta(-1, False, 'buffer', (1,), (slice(0, 1),), 1, torch.float32, (1,))
        state = {}
        for index in range(parameter_count):
            shape = (1,) if deduped else (rank_count,)
            offset = 0 if deduped else rank
            fullmap[f'p{index}_{rank}'] = AttrMeta(index, True, f'p{index}', shape,
                (slice(offset, offset + 1),), 1, torch.float32, (1,))
            if index % 17:
                state[index] = {'momentum': torch.tensor([float(index + offset)]), 'updates': 3}
        fullmaps.append(fullmap)
        states.append({'state': state, 'param_groups': [{'params': list(range(parameter_count))}]})
    if deduped:
        states = states[:1]

    start = time.perf_counter()
    merged = ParallelModule.merge_opt_state_dicts(fullmaps, states)
    elapsed = time.perf_counter() - start
    expected_indices = {index for index in range(parameter_count) if index % 17}
    assert set(merged['state']) == expected_indices
    assert merged['param_groups'][0]['params'] == list(range(parameter_count))
    for index in expected_indices:
        torch.testing.assert_close(merged['state'][index]['momentum'],
                                   torch.arange(1 if deduped else rank_count, dtype=torch.float32) + index,
                                   atol=0, rtol=0)
        assert merged['state'][index]['updates'] == 3
    print(json.dumps(dict(ranks=rank_count, optimizer_ranks=len(states), parameters=parameter_count,
                          metadata_visits=counter[0], seconds=elapsed,
                          max_absolute_error=0.0)), flush=True)
    # One pass establishes original order and one pre-indexes shards. Leave
    # room for another linear validation pass, never one scan per parameter.
    assert counter[0] <= 3 * rank_count * (parameter_count + 1)
