#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

from unittest.mock import patch

import pytest
import torch

from nnscaler.runtime.adapter.reducer import Reducer
from tests.utils import mock_reducer_env


@mock_reducer_env(0, 2)
def test_uneven_scheduled_contributions_wait_for_complete_gradients():
    reused = torch.nn.Parameter(torch.zeros(2))
    single = torch.nn.Parameter(torch.zeros(2))
    reducer = Reducer([0, 1], async_op=True)
    reducer.add_param(reused)
    reducer.add_param(single)
    reducer.build_buckets()

    for _ in range(2):
        with patch('torch.cuda.synchronize'):
            reducer.zero_grad()
        # train_step installs a uniform microbatch count before the schedule.
        reducer.grad_accumulation_steps = 1
        reducer.set_async_grad_expected_counts({reused: 2, single: 1})
        with patch('torch.distributed.all_reduce') as reduce:
            single.sum().backward()
            reused.sum().backward()
            reduce.assert_not_called()
            (reused * 2).sum().backward()
            reduce.assert_called_once()
            reducer.sync_grads()
        torch.testing.assert_close(reused.grad, torch.full_like(reused, 3))
        torch.testing.assert_close(single.grad, torch.ones_like(single))


@mock_reducer_env(0, 2)
def test_extra_contribution_cannot_substitute_for_another_parameter():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(2))
    reducer = Reducer([0, 1], async_op=True)
    reducer.add_param(first)
    reducer.add_param(second)
    reducer.build_buckets()
    reducer.set_async_grad_expected_counts({first: 1, second: 2})

    with patch('torch.distributed.all_reduce') as reduce:
        first.sum().backward()
        with pytest.raises(RuntimeError, match='2/1 expected'):
            first.sum().backward()
        reduce.assert_not_called()
