#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
pytest unit_tests/runtime/test_reducer.py
"""
import io
import os
import torch
import logging
from functools import partial
from unittest.mock import ANY, Mock, patch

import pytest

import nnscaler
from nnscaler.compiler import compile
from nnscaler.utils import load_model
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.flags import CompileFlag
from nnscaler.runtime.adapter.reducer import (
    Reducer,
    accumulate_reducer_grad,
    has_reducer_grad_accumulator,
    mark_reducer_grad_ready,
)
from nnscaler.runtime.device import DeviceGroup
from ..launch_torchrun import launch_torchrun, torchrun
from ..utils import catch_log, init_parameter, assert_parity, mock_reducer_env


class MLP(torch.nn.Module):
    def __init__(self, dim=512, nlayers=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(nlayers):
            self.layers.append(torch.nn.Linear(dim, dim, bias=False))

    def forward(self, data):
        x = data
        for layer in self.layers:
            x = layer(x)
        loss = torch.sum(x)
        return loss


def get_dummy_data(batch_size: int = 256):
    torch.random.manual_seed(0)
    return torch.randn(
        [batch_size, 512], dtype=torch.float32,
        device=torch.cuda.current_device())


def baseline():

    model = MLP()
    init_parameter(model)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    losses = []
    for _ in range(3):
        x = get_dummy_data()
        loss = model(x)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0
        losses.append(loss)

    return losses


def reducer(use_zero: bool, async_reducer: bool):

    CompileFlag.use_zero = use_zero
    CompileFlag.async_reducer = async_reducer

    model = MLP()
    init_parameter(model)

    def policy(graph: IRGraph, resource):

        def tensor_parallelism(node, idx, dim, num):
            sub_nodes = graph.partition(
                node, node.algorithm('dim'), idx=idx, dim=dim, num=num)
            for idx, sub_node in enumerate(sub_nodes):
                graph.assign(sub_node, idx)
            return sub_nodes

        l1, l2, l3, l4 = graph.select(name='linear')

        # l1 data parallelism
        tensor_parallelism(l1, idx=0, dim=0, num=resource.ngpus)
        # l2 data parallelism
        tensor_parallelism(l2, idx=0, dim=0, num=resource.ngpus)
        # l3, l4 replicate

        for node in graph.select(ntype=IRFwOperation):
            if len(node.device) == 0:
                sub_nodes = graph.replicate(node, times=resource.ngpus)
                for idx, sub_node in enumerate(sub_nodes):
                    graph.assign(sub_node, idx)
        return graph

    x = get_dummy_data()

    @compile(model, x, PAS=policy)
    def train_iter(model, x):
        loss = model(x)
        loss.backward()
        return loss

    model = load_model()
    optimizer = torch.optim.Adam(model.parameters_for_optimizer(), lr=0.01)

    losses = []
    for _ in range(3):
        x = get_dummy_data()
        loss = train_iter(model, x)
        optimizer.step()
        optimizer.zero_grad()
        ## === neccessary for zero ===
        model.gather_params()
        ## ===========================
        loss = loss.item()
        while abs(loss) > 10.0:
            loss /= 10.0
        losses.append(loss)

    return losses


def reducer_test():
    nnscaler.init()
    CompileFlag.disable_code_line_info = True  # speedup parse
    print('starting zero=True, async=True')
    assert_parity(baseline, partial(reducer, True, True))
    print('starting zero=True, async=False')
    assert_parity(baseline, partial(reducer, True, False))
    print('starting zero=False, async=True')
    assert_parity(baseline, partial(reducer, False, True))
    print('starting zero=False, async=False')
    assert_parity(baseline, partial(reducer, False, False))

test_reducer_2gpu = partial(torchrun, 2, reducer_test)


@mock_reducer_env(0, 2)
def test_reducer_build():
    reducer = Reducer([0, 1], max_bucket_size_bytes=48)  # 24 bytes means 12 float32
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 2)))  # 4 floats # small at first <bucket 0>
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 14))) # 16 floats # bigger than max_bucket_size_bytes <bucket 1>
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 5)))  # 8 floats # small again <bucket 2>
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 5)))  # 8 floats # small again <bucket 3>
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 1)))  # 4 floats small again <bucket 3>
    reducer.add_param(torch.nn.Parameter(torch.randn(1, 1)))  # 4 floats small again <bucket 4>
    reducer.build_buckets()
    assert len(reducer.buckets) == 5
    buckets = list(reversed(reducer.buckets))
    assert buckets[0].numel == 2
    assert buckets[0]._aligned_numel == 4
    assert buckets[1].numel == 14
    assert buckets[1]._aligned_numel == 16
    assert buckets[2].numel == 5
    assert buckets[2]._aligned_numel == 8
    assert buckets[3].numel == 6
    assert buckets[3]._aligned_numel == 12
    assert buckets[4].numel == 1
    assert buckets[4]._aligned_numel == 4


def _add_scalar_params(reducer, num_params, param_clss=None, param_cls=None):
    # NOTE: all params will be aligned to 16 bytes.
    # So even 1 float32 param will take 16 bytes in the bucket.
    params = []
    for _ in range(num_params):
        param = torch.nn.Parameter(torch.randn(1))
        reducer.add_param(param)
        params.append(param)
        if param_clss is not None:
            param_clss[param] = param_cls
    return params


class _ManualReducerGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, param, chunks):
        ctx.save_for_backward(param)
        ctx.chunks = chunks
        return param.sum()

    @staticmethod
    def backward(ctx, grad_output):
        param, = ctx.saved_tensors
        for grad, offset in ctx.chunks:
            assert accumulate_reducer_grad(param, grad, offset)
        assert mark_reducer_grad_ready(param)
        return None, None


@mock_reducer_env(0, 8)
def test_reducer_build_zero_param_level_sharding_merges_single_param_tail():
    reducer = Reducer(
        list(range(8)),
        max_bucket_size_bytes=128,
        zero=1,
        zero_param_level_sharding=True,
    )
    # max_bucket_size_bytes = 128 bytes means 8 float32 params (taking account of alignment) can be put in one bucket,
    # so the first 8 params will fill one bucket, and the last 1 param will be merged to the first bucket because it's smaller than zero_size(8)
    # 8 will be put in the first bucket,
    # and the last 1 param will be merged to the first bucket
    # because it's smaller than zero_size(8)
    _add_scalar_params(reducer, 9)

    reducer.build_buckets()

    buckets = list(reversed(reducer.buckets))
    assert [len(bucket.params) for bucket in buckets] == [9]


@mock_reducer_env(0, 8)
def test_reducer_build_zero_param_level_sharding_merges_small_tail_to_previous_bucket():
    reducer = Reducer(
        list(range(8)),
        max_bucket_size_bytes=128,
        zero=1,
        zero_param_level_sharding=True,
    )
    _add_scalar_params(reducer, 17)

    reducer.build_buckets()

    buckets = list(reversed(reducer.buckets))
    assert [len(bucket.params) for bucket in buckets] == [8, 9]


@mock_reducer_env(0, 8)
def test_reducer_build_zero_param_level_sharding_keeps_param_classes_separate():
    reducer = Reducer(
        list(range(8)),
        max_bucket_size_bytes=128,
        zero=1,
        zero_param_level_sharding=True,
    )
    param_clss = {}
    _add_scalar_params(reducer, 9, param_clss, 0)
    _add_scalar_params(reducer, 9, param_clss, 1)

    reducer.build_buckets(param_clss=param_clss)

    buckets = list(reversed(reducer.buckets))
    assert [len(bucket.params) for bucket in buckets] == [9, 9]
    assert [bucket.param_cls[0] for bucket in buckets] == [0, 1]


@mock_reducer_env(0, 8)
def test_reducer_build_zero_param_level_sharding_rejects_too_few_params():
    reducer = Reducer(
        list(range(8)),
        max_bucket_size_bytes=128,
        zero=1,
        zero_param_level_sharding=True,
    )
    _add_scalar_params(reducer, 7)

    with pytest.raises(RuntimeError, match="Please disable ZeRO or disable parameter-level sharding or increase bucket size."):
        reducer.build_buckets()


@mock_reducer_env(0, 8)
def test_reducer_build_zero_param_level_sharding_waste_warning():
    reducer = Reducer(
        list(range(8)),
        max_bucket_size_bytes=128,
        zero=1,
        zero_param_level_sharding=True,
    )
    _add_scalar_params(reducer, 7)
    param = torch.nn.Parameter(torch.randn(100))
    reducer.add_param(param)

    from nnscaler.runtime.adapter.reducer import _logger
    with catch_log(_logger) as log_stream:
        reducer.build_buckets()
        logs = log_stream.getvalue()
        assert "which may cause memory waste" in logs


@mock_reducer_env(0, 16)
def test_reducer_build_zero_param_level_sharding_zero_ngroups_chunk_by_zero_subgroup():
    DeviceGroup().get_group([0, 1])
    reducer = Reducer(
        list(range(16)),
        max_bucket_size_bytes=128,
        zero=1,
        zero_ngroups=8,
        zero_param_level_sharding=True,
    )
    _add_scalar_params(reducer, 2)

    reducer.build_buckets()

    buckets = list(reversed(reducer.buckets))
    assert buckets[0]._contiguous_grads.numel() == 8
    assert buckets[0]._flatten_param_info.opt_num_chunks == 2


@mock_reducer_env(0, 2)
def test_manual_grad_slice_accumulates_in_reducer_buffer_without_param_grad():
    param = torch.nn.Parameter(torch.zeros(3, 2))
    reducer = Reducer([0, 1])
    reducer.add_param(param)
    reducer.build_buckets()
    bucket = reducer.buckets[0]

    assert has_reducer_grad_accumulator(param)
    _ManualReducerGrad.apply(param, [
        (torch.tensor([[1.0, 2.0]]), 2),
        (torch.tensor([[3.0, 4.0]]), 4),
    ]).backward()

    assert param.grad is None
    offset = bucket._pofset[param]
    torch.testing.assert_close(
        bucket._contiguous_grads[offset:offset + param.numel()],
        torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0, 4.0]),
    )


@mock_reducer_env(0, 2)
def test_reducer_grad_registration_keeps_parameter_picklable():
    param = torch.nn.Parameter(torch.zeros(2))
    reducer = Reducer([0, 1])
    reducer.add_param(param)
    reducer.build_buckets()

    assert has_reducer_grad_accumulator(param)
    buffer = io.BytesIO()
    torch.save(param, buffer)


@mock_reducer_env(0, 2)
def test_manual_grad_ready_uses_async_vpp_expected_contribution_count():
    param = torch.nn.Parameter(torch.zeros(2))
    reducer = Reducer([0, 1], async_op=True)
    reducer.add_param(param)
    reducer.build_buckets()
    reducer.set_async_grad_expected_counts({param: 2})
    bucket = reducer.buckets[0]
    bucket._launch_async_reduce = Mock()

    _ManualReducerGrad.apply(
        param, [(torch.tensor([1.0, 2.0]), 0)]
    ).backward()
    bucket._launch_async_reduce.assert_not_called()

    _ManualReducerGrad.apply(
        param, [(torch.tensor([3.0, 4.0]), 0)]
    ).backward()
    bucket._launch_async_reduce.assert_called_once_with()
    assert bucket._async_seen_param_cnt[param] == 2
    assert bucket._async_param_cnt == bucket._async_expected_total == 2
    assert param.grad is None


@mock_reducer_env(0, 2)
def test_manual_grad_combines_with_regular_autograd_path_before_async_ready():
    param = torch.nn.Parameter(torch.zeros(2))
    reducer = Reducer([0, 1], async_op=True)
    reducer.add_param(param)
    reducer.build_buckets()
    bucket = reducer.buckets[0]
    bucket._launch_async_reduce = Mock()

    loss = _ManualReducerGrad.apply(
        param, [(torch.tensor([1.0, 2.0]), 0)]
    ) + (param * 3.0).sum()
    loss.backward()

    offset = bucket._pofset[param]
    torch.testing.assert_close(
        bucket._contiguous_grads[offset:offset + param.numel()],
        torch.tensor([4.0, 5.0]),
    )
    bucket._launch_async_reduce.assert_called_once_with()
    assert bucket._async_param_cnt == 1
    assert param.grad is None


@mock_reducer_env(0, 2)
def test_manual_grad_slice_zero3_reduces_by_owner_and_postevicts_param():
    param = torch.nn.Parameter(torch.zeros(8))
    reducer = Reducer([0, 1], zero=3)
    reducer.add_param(param)
    reducer.build_buckets()
    bucket = reducer.buckets[0]
    info = reducer.get_z3_info(param)

    # Emulate the full parameter materialized for forward/backward. The mock
    # process group cannot execute collectives, so retain the local input in
    # the patched all-reduce call (the rank-0 result for a single local source).
    param.data = torch.zeros(info.shape)
    with patch('torch.distributed.all_reduce') as all_reduce:
        _ManualReducerGrad.apply(
            param, [(torch.arange(1.0, 7.0), 0)]
        ).backward()
        all_reduce.assert_called_once_with(
            ANY,
            op=bucket._reduce_op,
            group=bucket._zero_subgroup,
        )

    offset = bucket._pofset[param]
    torch.testing.assert_close(
        bucket._contiguous_grads[offset:offset + info.numel()],
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    assert param.grad is None
    assert param.shape == (info.numel_with_padding(),)


def _manual_grad_slice_zero3_distributed_worker():
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')

    rank = torch.distributed.get_rank()
    param = torch.nn.Parameter(torch.zeros(8, device=local_rank))
    reducer = Reducer([0, 1], zero=3)
    reducer.add_param(param)
    reducer.build_buckets()
    bucket = reducer.buckets[0]
    info = reducer.get_z3_info(param)

    # Materialize the full parameter as a normal ZeRO-3 forward would do.
    param.data = torch.zeros(info.shape, device=local_rank)
    base_grad = torch.arange(1.0, 7.0, device=local_rank)
    contribution = base_grad * (1 if rank == 0 else 10)
    _ManualReducerGrad.apply(param, [(contribution, 0)]).backward()

    reduced = base_grad * 11
    expected_full = torch.cat((reduced, torch.zeros(2, device=local_rank)))
    expected_local = expected_full[info.start:info.end]
    offset = bucket._pofset[param]
    torch.testing.assert_close(
        bucket._contiguous_grads[offset:offset + info.numel()],
        expected_local,
    )
    assert param.grad is None
    assert param.shape == (info.numel_with_padding(),)

    reducer.sync_grads()
    torch.testing.assert_close(param.grad[:info.numel()], expected_local)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason='requires two CUDA devices',
)
def test_manual_grad_slice_zero3_real_collective_2gpu():
    launch_torchrun(2, _manual_grad_slice_zero3_distributed_worker)
