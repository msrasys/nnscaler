# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import Mock, patch

import pytest
import torch

from nnscaler.flags import RuntimeFlag
from nnscaler.runtime.adapter.reducer import Reducer, accumulate_reducer_grad
from tests.utils import mock_reducer_env


@pytest.mark.parametrize('uniform', [False, True])
@pytest.mark.parametrize('zero', [0, 1])
@mock_reducer_env(0, 2)
def test_excess_parameter_gradient_rejected_before_buffer_write(uniform, zero):
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(2))
    reducer = Reducer([0, 1], async_op=True, zero=zero)
    reducer.add_param(first)
    reducer.add_param(second)
    reducer.build_buckets()
    bucket = reducer.buckets[0]
    bucket._launch_async_reduce = Mock()
    if uniform:
        reducer.grad_accumulation_steps = 2
        expected = 2
    else:
        reducer.set_async_grad_expected_counts({first: 1, second: 2})
        expected = 1
    with patch.object(RuntimeFlag, 'skip_reducer', False):
        for _ in range(expected):
            first.sum().backward()
        before = bucket._contiguous_grads.clone()
        count = bucket._async_param_cnt
        seen = dict(bucket._async_seen_param_cnt)
        with pytest.raises(RuntimeError, match='more async reducer gradient contributions'):
            first.sum().backward()
    torch.testing.assert_close(bucket._contiguous_grads, before, rtol=0, atol=0)
    assert bucket._async_param_cnt == count
    assert bucket._async_seen_param_cnt == seen
    bucket._launch_async_reduce.assert_not_called()


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('skip_reducer', [False, True])
@mock_reducer_env(0, 2)
def test_inflight_reduction_buffer_cannot_be_modified(manual, skip_reducer):
    param = torch.nn.Parameter(torch.zeros(2))
    reducer = Reducer([0, 1], async_op=True)
    reducer.add_param(param)
    reducer.build_buckets()
    bucket = reducer.buckets[0]

    def launch():
        bucket._async_handle = Mock()

    bucket._launch_async_reduce = Mock(side_effect=launch)
    with patch.object(RuntimeFlag, 'skip_reducer', False):
        param.sum().backward()
    before = bucket._contiguous_grads.clone()
    with patch.object(RuntimeFlag, 'skip_reducer', skip_reducer):
        with pytest.raises(RuntimeError, match='more async reducer gradient contributions'):
            if manual:
                accumulate_reducer_grad(param, torch.ones(2))
            else:
                param.sum().backward()
    torch.testing.assert_close(bucket._contiguous_grads, before, rtol=0, atol=0)
    bucket._launch_async_reduce.assert_called_once_with()


@mock_reducer_env(0, 2)
def test_manual_slice_rejected_before_excess_write():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(2))
    reducer = Reducer([0, 1], async_op=True)
    reducer.add_param(first)
    reducer.add_param(second)
    reducer.build_buckets()
    bucket = reducer.buckets[0]
    with patch.object(RuntimeFlag, 'skip_reducer', False):
        first.sum().backward()
        before = bucket._contiguous_grads.clone()
        with pytest.raises(RuntimeError, match='more async reducer gradient contributions'):
            accumulate_reducer_grad(first, torch.ones(1), offset=1)
    torch.testing.assert_close(bucket._contiguous_grads, before, rtol=0, atol=0)
