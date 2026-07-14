#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

"""
pytest unit_tests/runtime/test_reducer.py
"""
import torch
import logging
from functools import partial

import pytest

import nnscaler
from nnscaler.compiler import compile
from nnscaler.utils import load_model
from nnscaler.graph import IRGraph
from nnscaler.ir.operator import IRFwOperation
from nnscaler.flags import CompileFlag
from nnscaler.runtime.adapter.reducer import FlattenParamInfo, Reducer, ReducerParamInfo
from nnscaler.runtime.device import DeviceGroup
from ..launch_torchrun import torchrun
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


def test_flatten_param_info_dtype():
    params = [
        torch.nn.Parameter(torch.zeros(2, dtype=torch.float16)),
        torch.nn.Parameter(torch.zeros(2, dtype=torch.float16)),
    ]
    flatten_info = FlattenParamInfo(
        zero=0,
        params_info={
            params[0]: ReducerParamInfo(
                shape=params[0].shape,
                start=0,
                end=2,
                bucket_param_buffer_start=0,
                bucket_param_buffer_end=2,
            ),
            params[1]: ReducerParamInfo(
                shape=params[1].shape,
                start=0,
                end=2,
                bucket_param_buffer_start=2,
                bucket_param_buffer_end=4,
            ),
        },
        opt_numel=4,
        opt_num_chunks=1,
        opt_chunk_index=0,
    )

    tensor = torch.tensor([1, 2], dtype=torch.float32)
    flattened = flatten_info.flatten([None, tensor])
    assert flattened.device.type == 'cpu'
    assert flattened.dtype == torch.float32
    assert flattened.equal(torch.tensor([0, 0, 1, 2], dtype=torch.float32))

    flattened = flatten_info.flatten([None, tensor], dtype=torch.float64, device='cpu')
    assert flattened.dtype == torch.float64
    assert flattened.equal(torch.tensor([0, 0, 1, 2], dtype=torch.float64))

    flattened = flatten_info.flatten([None, None], dtype=torch.float64, device='cpu')
    assert flattened.dtype == torch.float64
    assert flattened.equal(torch.zeros(4, dtype=torch.float64))

    flattened = flatten_info.flatten([None, None], device='cpu')
    assert flattened.dtype == torch.float16  # default to first param's dtype
    assert flattened.equal(torch.zeros(4, dtype=torch.float16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
def test_flatten_param_info_cuda_device_and_synchronization(monkeypatch):
    param = torch.nn.Parameter(torch.zeros(2))
    flatten_info = FlattenParamInfo(
        zero=0,
        params_info={
            param: ReducerParamInfo(
                shape=param.shape,
                start=0,
                end=2,
                bucket_param_buffer_start=0,
                bucket_param_buffer_end=2,
            ),
        },
        opt_numel=2,
        opt_num_chunks=1,
        opt_chunk_index=0,
    )

    synchronize_calls = 0
    cuda_synchronize = torch.cuda.synchronize

    def record_synchronize():
        nonlocal synchronize_calls
        synchronize_calls += 1
        cuda_synchronize()

    monkeypatch.setattr(torch.cuda, 'synchronize', record_synchronize)

    tensor = torch.tensor([1, 2], dtype=torch.float32)
    cuda_flattened = flatten_info.flatten([tensor], device=0)

    assert cuda_flattened.device == torch.device(0)
    assert synchronize_calls == 0

    cpu_flattened = flatten_info.flatten([cuda_flattened], device='cpu')

    assert synchronize_calls == 1
    assert cpu_flattened.equal(tensor)


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


@mock_reducer_env(0, 2)
def test_reducer_nreplicas():
    """Test that nreplicas is correctly passed to buckets"""
    # Test nreplicas defaults to 1
    reducer = Reducer([0, 1])
    assert reducer._nreplicas == 1

    # Test nreplicas is set correctly
    reducer = Reducer([0, 1], nreplicas=2)
    assert reducer._nreplicas == 2

    # Test nreplicas is passed to buckets
    reducer = Reducer([0, 1], nreplicas=3)
    reducer.add_param(torch.nn.Parameter(torch.randn(4)))
    reducer.build_buckets()
    for bucket in reducer.buckets:
        assert bucket._nreplicas == 3


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
def test_reducer_build_zero_param_level_sharding_pads_too_few_params():
    """When fewer params than zero_size, padding is added so that some ranks get empty shards."""
    reducer = Reducer(
        list(range(8)),
        max_bucket_size_bytes=128,
        zero=1,
        zero_param_level_sharding=True,
    )
    _add_scalar_params(reducer, 7)

    from nnscaler.runtime.adapter.reducer import _logger
    with catch_log(_logger) as log_stream:
        reducer.build_buckets()
        logs = log_stream.getvalue()
        assert "Padding will be added so that some ranks have empty shards" in logs

    buckets = list(reversed(reducer.buckets))
    assert len(buckets) == 1
    assert len(buckets[0].params) == 7
    # buffer should be max_group_size * zero_size = 4 * 8 = 32
    # (each scalar param is aligned to 4 float32 = 16 bytes)
    assert buckets[0]._contiguous_params.numel() == 4 * 8


@mock_reducer_env(0, 8)
def test_reducer_build_zero_param_level_sharding_pads_single_param():
    """A single param with zero_param_level_sharding should pad to zero_size chunks."""
    reducer = Reducer(
        list(range(8)),
        max_bucket_size_bytes=128,
        zero=1,
        zero_param_level_sharding=True,
    )
    _add_scalar_params(reducer, 1)

    reducer.build_buckets()

    buckets = list(reversed(reducer.buckets))
    assert len(buckets) == 1
    assert len(buckets[0].params) == 1
    # buffer = max_group_size(4) * zero_size(8) = 32
    assert buckets[0]._contiguous_params.numel() == 4 * 8


@mock_reducer_env(0, 8)
def test_reducer_build_zero_param_level_sharding_pads_with_classes():
    """Two param classes each with fewer params than zero_size should each get their own padded bucket."""
    reducer = Reducer(
        list(range(8)),
        max_bucket_size_bytes=128,
        zero=1,
        zero_param_level_sharding=True,
    )
    param_clss = {}
    _add_scalar_params(reducer, 3, param_clss, 0)
    _add_scalar_params(reducer, 5, param_clss, 1)

    reducer.build_buckets(param_clss=param_clss)

    buckets = list(reversed(reducer.buckets))
    assert len(buckets) == 2
    assert [len(b.params) for b in buckets] == [3, 5]
    assert [b.param_cls[0] for b in buckets] == [0, 1]
    # each bucket buffer = 4 * 8 = 32
    assert buckets[0]._contiguous_params.numel() == 4 * 8
    assert buckets[1]._contiguous_params.numel() == 4 * 8


@mock_reducer_env(0, 4)
def test_reducer_build_zero_param_level_sharding_pads_mixed_sizes():
    """Params with different sizes, fewer than zero_size, should pad using max aligned size."""
    reducer = Reducer(
        list(range(4)),
        max_bucket_size_bytes=1024,
        zero=1,
        zero_param_level_sharding=True,
    )
    # Add 2 params of different sizes: 1 float32 (aligned to 4) and 5 float32 (aligned to 8)
    p1 = torch.nn.Parameter(torch.randn(1))
    p2 = torch.nn.Parameter(torch.randn(5))
    reducer.add_param(p1)
    reducer.add_param(p2)

    reducer.build_buckets()

    buckets = list(reversed(reducer.buckets))
    assert len(buckets) == 1
    assert len(buckets[0].params) == 2
    # max aligned size is 8 (from param of size 5), buffer = 8 * 4 = 32
    assert buckets[0]._contiguous_params.numel() == 8 * 4


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
