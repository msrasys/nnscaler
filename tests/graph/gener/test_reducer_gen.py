#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
from pathlib import Path

import nnscaler
from nnscaler.graph.gener.gen import IRAdapterGener
from nnscaler.graph import IRGraph
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.parser.converter import convert_model
from nnscaler.ir.operator import IRFwOperation
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.ir.adapter import IRWeightReducer
from nnscaler.flags import CompileFlag
from nnscaler.parallel import ComputeConfig, _load_parallel_module_class, parallelize
from ...utils import new_empty

import torch
import tempfile
import importlib

from ...utils import replace_all_device_with


def make_param(shape, dtype) -> IRFullTensor:
    param = IRFullTensor(shape=shape, dtype=dtype, requires_grad=True)
    param.as_param()
    return param


class ReducerModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float32))
        self.param2 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float32))

    def forward(self, x):
        x = torch.matmul(x, self.param1)
        x = torch.matmul(x, self.param2)
        x = x + self.param1
        x = torch.sum(x)
        return x


def build_graph(model_cls=ReducerModule):
    # build graph
    model = model_cls()
    with tempfile.TemporaryDirectory() as tempdir:
        graph = convert_model(
            model,
            {'x': torch.randn([128, 128], dtype=torch.float32)},
            attr_savedir=tempdir,
            constant_folding=True
        )
    graph.backward(graph.output(0))
    return graph


@replace_all_device_with('cpu')
def test_cross_segment_weight_reducer():

    graph = build_graph()
    [matmul1, matmul2, add, sum] = graph.select(ntype=IRFwOperation)
    graph.group([matmul1, matmul2])
    graph.group([add, sum])

    for idx, segment in enumerate(graph.select(ntype=IRSegment, flatten=False)):
        if not segment.isfw():
            continue
        for node in segment.nodes():
            graph.assign(node, idx)

    print(graph.extra_repr())

    # build reducer
    graph = IRAdapterGener.gen_weight(graph)
    print(graph.extra_repr())
    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 1
    assert len(reducers[0].inputs()) == 1
    assert reducers[0].input(0) == matmul1.input(1)
    assert reducers[0].device == (0, 1)
    assert reducers[0].nreplicas == 1


@replace_all_device_with('cpu')
def test_replicate_shared_param():
    # default: reducer_replicated_params=False, no reducers for replicated weights
    assert not CompileFlag.reducer_replicated_params
    graph = build_graph()
    for node in graph.select(ntype=IRFwOperation):
        sn1, sn2 = graph.replicate(node, 2)
        graph.assign(sn1, 0)
        graph.assign(sn2, 1)

    graph = IRAdapterGener.gen_weight(graph)
    print(graph.extra_repr())

    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 0

    old_flag = CompileFlag.reducer_replicated_params
    try:
        # with reducer_replicated_params=True, reducers are generated with nreplicas
        CompileFlag.reducer_replicated_params = True
        graph = IRAdapterGener.gen_weight(graph)
        reducers = graph.select(ntype=IRWeightReducer)
        assert len(reducers) > 0
        for reducer in reducers:
            assert reducer.nreplicas == 2
            assert reducer.device == (0, 1)
    finally:
        CompileFlag.reducer_replicated_params = old_flag


class SimpleModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float32))
        self.param2 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float32))

    def forward(self, x):
        x = torch.matmul(x, self.param1)
        x = torch.matmul(x, self.param2)
        x = torch.sum(x)
        return x


@replace_all_device_with('cpu')
def test_tp_partitioned_weight():
    """
    All Params are partitioned.
    """
    def _get_reducers(reducer_replicated_params=False):
        old_flag = CompileFlag.reducer_replicated_params
        try:
            CompileFlag.reducer_replicated_params = reducer_replicated_params
            graph = build_graph(model_cls=SimpleModule)
            [matmul1, matmul2, sum] = graph.select(ntype=IRFwOperation)
            for node in graph.select(ntype=IRFwOperation):
                if node == matmul1:
                    nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=1, num=2)
                else:
                    nodes = graph.replicate(node, 2)
                sn1, sn2 = nodes
                graph.assign(sn1, 0)
                graph.assign(sn2, 1)

            graph = IRAdapterGener.gen_weight(graph)
            reducers = graph.select(ntype=IRWeightReducer)
            return reducers, [matmul1, matmul2, sum]
        finally:
            CompileFlag.reducer_replicated_params = old_flag

    reducers, [matmul1, matmul2, sum] = _get_reducers(reducer_replicated_params=False)
    assert len(reducers) == 0

    reducers, [matmul1, matmul2, sum] = _get_reducers(reducer_replicated_params=True)
    assert len(reducers) == 1
    assert reducers[0].nreplicas == 2
    assert reducers[0].device == (0, 1)
    # param2 is replicated. reducer is generated when reducer_replicated_params is True
    assert reducers[0].input(0).parent == matmul2.input(1).parent


class SimpleModule2ConsumersTp(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float32))
        self.param2 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float32))

    def forward(self, x):
        x = torch.matmul(x, self.param1)
        x = x + self.param1
        x = torch.matmul(x, self.param2)
        x = torch.sum(x)
        return x


@replace_all_device_with('cpu')
def test_tp_partitioned_weight_2_consumers():
    # partition param1
    def _get_reducers1(reducer_replicated_params=False):
        old_flag = CompileFlag.reducer_replicated_params
        try:
            CompileFlag.reducer_replicated_params = reducer_replicated_params
            graph = build_graph(model_cls=SimpleModule2ConsumersTp)
            [matmul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
            for node in graph.select(ntype=IRFwOperation):
                if node == matmul1:
                    nodes = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=2)
                elif node == add:
                    nodes = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=2)
                else:
                    nodes = graph.replicate(node, 2)
                sn1, sn2 = nodes
                graph.assign(sn1, 0)
                graph.assign(sn2, 1)

            graph = IRAdapterGener.gen_weight(graph)
            reducers = graph.select(ntype=IRWeightReducer)
            return reducers, [matmul1, add, matmul2, sum]
        finally:
            CompileFlag.reducer_replicated_params = old_flag

    reducers, [matmul1, add, matmul2, sum] = _get_reducers1(reducer_replicated_params=False)
    assert len(reducers) == 0
    reducers, [matmul1, add, matmul2, sum] = _get_reducers1(reducer_replicated_params=True)
    assert len(reducers) == 1
    assert reducers[0].nreplicas == 2
    assert reducers[0].device == (0, 1)
    # param2 is replicated. reducer is generated when reducer_replicated_params is True
    assert reducers[0].input(0).parent == matmul2.input(1).parent

    # partition param2
    def _get_reducers2(reducer_replicated_params=False):
        old_flag = CompileFlag.reducer_replicated_params
        try:
            CompileFlag.reducer_replicated_params = reducer_replicated_params
            graph = build_graph(model_cls=SimpleModule2ConsumersTp)
            [matmul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
            for node in graph.select(ntype=IRFwOperation):
                if node == matmul2:
                    nodes = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=2)
                else:
                    nodes = graph.replicate(node, 2)
                sn1, sn2 = nodes
                graph.assign(sn1, 0)
                graph.assign(sn2, 1)

            graph = IRAdapterGener.gen_weight(graph)
            reducers = graph.select(ntype=IRWeightReducer)
            return reducers, [matmul1, add, matmul2, sum]
        finally:
            CompileFlag.reducer_replicated_params = old_flag
    reducers, [matmul1, add, matmul2, sum] = _get_reducers2(reducer_replicated_params=False)
    assert len(reducers) == 0
    reducers, [matmul1, add, matmul2, sum] = _get_reducers2(reducer_replicated_params=True)
    assert len(reducers) == 1
    assert reducers[0].nreplicas == 2
    assert reducers[0].device == (0, 1)
    # param2 is replicated. reducer is generated when reducer_replicated_params is True
    assert reducers[0].input(0).parent == matmul1.input(1).parent


@replace_all_device_with('cpu')
def test_replicated_2_consumers_1gpu():
    graph = build_graph(model_cls=SimpleModule2ConsumersTp)
    [matmul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
    # partition param1
    for node in graph.select(ntype=IRFwOperation):
        if node == matmul1:
            nodes = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=1)
        elif node == add:
            nodes = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=1)
        else:
            nodes = graph.replicate(node, 1)
        sn1, = nodes
        graph.assign(sn1, 0)

    graph = IRAdapterGener.gen_weight(graph)
    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 0

    # partition param2
    graph = build_graph(model_cls=SimpleModule2ConsumersTp)
    [matmul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
    for node in graph.select(ntype=IRFwOperation):
        if node == matmul2:
            nodes = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=1)
        else:
            nodes = graph.replicate(node, 1)
        sn1, = nodes
        graph.assign(sn1, 0)

    graph = IRAdapterGener.gen_weight(graph)
    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 0


class SimpleModule2ConsumersSP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.zeros([128], dtype=torch.float32))
        self.param2 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float32))

    def forward(self, x):
        x = x * self.param1
        x = x + self.param1
        x = torch.matmul(x, self.param2)
        x = torch.sum(x)
        return x


@replace_all_device_with('cpu')
def test_tp_partitioned_input_2_consumers_allreduce():
    # partition input
    graph = build_graph(model_cls=SimpleModule2ConsumersSP)
    [mul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
    for node in graph.select(ntype=IRFwOperation):
        if node == mul1 or node == matmul2:
            nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=2)
        elif node == add:
            nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=2)
        else:
            nodes = graph.replicate(node, 2)
        sn1, sn2 = nodes
        graph.assign(sn1, 0)
        graph.assign(sn2, 1)

    graph = IRAdapterGener.gen_weight(graph)
    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 2
    for reducer in reducers:
        assert reducer.device == (0, 1)
        assert reducer.nreplicas == 1
    assert set([reducers[0].input(0).parent, reducers[1].input(0).parent]) ==\
        set([mul1.input(1).parent, matmul2.input(1).parent])


@nnscaler.register_op('a b, b c : / -> a c')
def no_grad_matmul(x, w):
    """Custom op that asks nnscaler to skip grad all-reduce on input 0."""
    return x @ w


@nnscaler.register_op('a b, b : / -> a b')
def no_grad_add(x, w):
    """Custom op that asks nnscaler to skip grad all-reduce on input 0."""
    return x + w


@nnscaler.register_op('a b, b : / -> a b')
def no_grad_mul(x, w):
    """Custom op that asks nnscaler to skip grad all-reduce on input 0."""
    return x * w


class SimpleModuleNoReduce(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.zeros([128], dtype=torch.float32))
        self.param2 = torch.nn.Parameter(torch.zeros([128, 128], dtype=torch.float32))

    def forward(self, x):
        x = no_grad_mul(x, self.param1)
        x = no_grad_add(x, self.param1)
        x = no_grad_matmul(x, self.param2)
        x = torch.sum(x)
        return x


@replace_all_device_with('cpu')
def test_tp_partitioned_input_2_consumers_noreduce():
    def _get_reducers(reducer_replicated_params=False):
        old_flag = CompileFlag.reducer_replicated_params
        try:
            CompileFlag.reducer_replicated_params = reducer_replicated_params
            graph = build_graph(model_cls=SimpleModuleNoReduce)
            [mul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
            for node in graph.select(ntype=IRFwOperation):
                if node == mul1 or node == matmul2:
                    nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=2)
                elif node == add:
                    nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=2)
                else:
                    nodes = graph.replicate(node, 2)
                sn1, sn2 = nodes
                graph.assign(sn1, 0)
                graph.assign(sn2, 1)

            graph = IRAdapterGener.gen_weight(graph)
            reducers = graph.select(ntype=IRWeightReducer)
            return reducers, [mul1, add, matmul2, sum]
        finally:
            CompileFlag.reducer_replicated_params = old_flag

    # partition input
    reducers, _ = _get_reducers(reducer_replicated_params=False)
    assert len(reducers) == 0

    reducers, [mul1, add, matmul2, sum] = _get_reducers(reducer_replicated_params=True)

    assert len(reducers) == 2
    for reducer in reducers:
        assert reducer.device == (0, 1)
        assert reducer.nreplicas == 2
    assert set([reducers[0].input(0).parent, reducers[1].input(0).parent]) ==\
        set([mul1.input(1).parent, matmul2.input(1).parent])


@replace_all_device_with('cpu')
def test_pp_no_shared_tp():
    graph = build_graph(model_cls=SimpleModule2ConsumersTp)
    [matmul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
    graph.group([matmul1, add])
    graph.group([matmul2, sum])

    for idx, segment in enumerate(graph.select(ntype=IRSegment, flatten=False)):
        if not segment.isfw():
            continue
        for node in segment.nodes():
            if node == matmul1:
                nodes = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=2)
            elif node == add:
                nodes = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=2)
            else:
                nodes = graph.replicate(node, 2)
            sn1, sn2 = nodes
            sn1, sn2 = nodes
            graph.assign(sn1, idx * 2)
            graph.assign(sn2, idx * 2 + 1)

    # build reducer
    graph = IRAdapterGener.gen_weight(graph)
    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 0


@replace_all_device_with('cpu')
def test_pp_shared_tp():
    graph = build_graph(model_cls=SimpleModule2ConsumersTp)
    [matmul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
    graph.group([matmul1])
    graph.group([add, matmul2, sum])

    for idx, segment in enumerate(graph.select(ntype=IRSegment, flatten=False)):
        if not segment.isfw():
            continue
        for node in segment.nodes():
            if node == matmul1:
                nodes = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=2)
            elif node == add:
                nodes = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=2)
            else:
                nodes = graph.replicate(node, 2)
            sn1, sn2 = nodes
            graph.assign(sn1, idx * 2)
            graph.assign(sn2, idx * 2 + 1)

    # build reducer
    graph = IRAdapterGener.gen_weight(graph)
    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 2
    assert reducers[0].nreplicas == 1
    assert reducers[1].nreplicas == 1
    assert reducers[0].device == (0, 2)
    assert reducers[1].device == (1, 3)
    assert reducers[0].input(0).parent == reducers[1].input(0).parent


@replace_all_device_with('cpu')
def test_pp_shared_replicated():
    graph = build_graph(model_cls=SimpleModule2ConsumersTp)
    [matmul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
    graph.group([matmul1])
    graph.group([add, matmul2, sum])

    for idx, segment in enumerate(graph.select(ntype=IRSegment, flatten=False)):
        if not segment.isfw():
            continue
        for node in segment.nodes():
            nodes = graph.replicate(node, 2)
            sn1, sn2 = nodes
            graph.assign(sn1, idx * 2)
            graph.assign(sn2, idx * 2 + 1)

    # build reducer
    with pytest.raises(RuntimeError):
        # we should be able to suppor this case.
        # but currently the pre-check in gen_weights will raise error.
        graph = IRAdapterGener.gen_weight(graph)
        # reducers = graph.select(ntype=IRWeightReducer)
        # assert len(reducers) == 2
        # assert reducers[0].nreplicas == 1
        # assert reducers[1].nreplicas == 1
        # assert reducers[0].device == (0, 2)
        # assert reducers[1].device == (1, 3)
        # assert reducers[0].input(0).parent == reducers[1].input(0).parent


@replace_all_device_with('cpu')
def test_pp_shared_no_reduce():
    def _get_reducers(reducer_replicated_params=False):
        old_flag = CompileFlag.reducer_replicated_params
        try:
            CompileFlag.reducer_replicated_params = reducer_replicated_params
            graph = build_graph(model_cls=SimpleModuleNoReduce)
            [mul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
            graph.group([mul1])
            graph.group([add, matmul2, sum])

            for idx, segment in enumerate(graph.select(ntype=IRSegment, flatten=False)):
                if not segment.isfw():
                    continue
                for node in segment.nodes():
                    if node == mul1 or node == matmul2 or node == add:
                        nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=2)
                    else:
                        nodes = graph.replicate(node, 2)
                    sn1, sn2 = nodes
                    graph.assign(sn1, idx * 2)
                    graph.assign(sn2, idx * 2 + 1)

            graph = IRAdapterGener.gen_weight(graph)
            reducers = graph.select(ntype=IRWeightReducer)
            return reducers, [mul1, add, matmul2, sum]
        finally:
            CompileFlag.reducer_replicated_params = old_flag

    reducers = _get_reducers(reducer_replicated_params=False)[0]
    assert len(reducers) == 2
    assert reducers[0].nreplicas == 1
    assert reducers[1].nreplicas == 1
    assert reducers[0].device == (0, 2)
    assert reducers[1].device == (1, 3)
    assert reducers[0].input(0).parent == reducers[1].input(0).parent

    reducers = _get_reducers(reducer_replicated_params=True)[0]
    assert len(reducers) == 2
    assert reducers[0].nreplicas == 2
    assert reducers[1].nreplicas == 2
    assert reducers[0].device == (0, 1, 2, 3)
    assert reducers[1].device == (2, 3)
    assert reducers[0].input(0).parent != reducers[1].input(0).parent


@replace_all_device_with('cpu')
def test_pp_no_shared_allreduce():
    graph = build_graph(model_cls=SimpleModule2ConsumersSP)
    [mul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
    graph.group([mul1, add])
    graph.group([matmul2, sum])

    for idx, segment in enumerate(graph.select(ntype=IRSegment, flatten=False)):
        if not segment.isfw():
            continue
        for node in segment.nodes():
            if node == mul1 or node == matmul2 or node == add:
                nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=2)
            else:
                nodes = graph.replicate(node, 2)
            sn1, sn2 = nodes
            sn1, sn2 = nodes
            graph.assign(sn1, idx * 2)
            graph.assign(sn2, idx * 2 + 1)

    # build reducer
    graph = IRAdapterGener.gen_weight(graph)
    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 2
    for reducer in reducers:
        assert reducer.nreplicas == 1
    assert set([reducers[0].device, reducers[1].device]) == set([(0, 1), (2, 3)])
    assert set([reducers[0].input(0).parent, reducers[1].input(0).parent]) ==\
        set([mul1.input(1).parent, matmul2.input(1).parent])


class SimpleModule2ConsumersSP3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param1 = torch.nn.Parameter(torch.zeros([128], dtype=torch.float32))

    def forward(self, x):
        x = x * self.param1
        x = x + self.param1
        x = x / self.param1
        x = torch.sum(x)
        return x


@replace_all_device_with('cpu')
def test_pp_shared_allreduce():
    # partition input
    graph = build_graph(model_cls=SimpleModule2ConsumersSP3)
    [mul1, add, div, sum] = graph.select(ntype=IRFwOperation)
    graph.group([mul1, add])
    graph.group([div, sum])
    for idx, segment in enumerate(graph.select(ntype=IRSegment, flatten=False)):
        if not segment.isfw():
            continue
        for node in segment.nodes():
            if node == mul1 or node == add:
                nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=2)
            elif node == div:
                nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=2)
            else:
                nodes = graph.replicate(node, 2)
            sn1, sn2 = nodes
            graph.assign(sn1, idx * 2)
            graph.assign(sn2, idx * 2 + 1)

    graph = IRAdapterGener.gen_weight(graph)
    reducers = graph.select(ntype=IRWeightReducer)
    assert len(reducers) == 1
    assert reducers[0].nreplicas == 1
    assert reducers[0].device == (0, 1, 2, 3)


@replace_all_device_with('cpu')
def test_pp_no_shared_noreduce():
    def _get_reducers(reducer_replicated_params=False):
        old_flag = CompileFlag.reducer_replicated_params
        try:
            CompileFlag.reducer_replicated_params = reducer_replicated_params
            graph = build_graph(model_cls=SimpleModuleNoReduce)
            [mul1, add, matmul2, sum] = graph.select(ntype=IRFwOperation)
            graph.group([mul1, add])
            graph.group([matmul2, sum])

            for idx, segment in enumerate(graph.select(ntype=IRSegment, flatten=False)):
                if not segment.isfw():
                    continue
                for node in segment.nodes():
                    if node == mul1 or node == matmul2 or node == add:
                        nodes = graph.partition(node, node.algorithm('dim'), idx=0, dim=0, num=2)
                    else:
                        nodes = graph.replicate(node, 2)
                    sn1, sn2 = nodes
                    graph.assign(sn1, idx * 2)
                    graph.assign(sn2, idx * 2 + 1)

            graph = IRAdapterGener.gen_weight(graph)
            reducers = graph.select(ntype=IRWeightReducer)
            return reducers, [mul1, add, matmul2, sum]
        finally:
            CompileFlag.reducer_replicated_params = old_flag

    # partition input
    reducers, _ = _get_reducers(reducer_replicated_params=False)
    assert len(reducers) == 0

    reducers, [mul1, add, matmul2, sum] = _get_reducers(reducer_replicated_params=True)

    assert len(reducers) == 2
    for reducer in reducers:
        assert reducer.nreplicas == 2
    assert set([reducers[0].device, reducers[1].device]) == set([(0, 1), (2, 3)])
    assert set([reducers[0].input(0).parent, reducers[1].input(0).parent]) ==\
        set([mul1.input(1).parent, matmul2.input(1).parent])


@replace_all_device_with('cpu')
def test_reducer_partially_shared_part():
    graph = build_graph()
    [matmul1, matmul2, add, sum] = graph.select(ntype=IRFwOperation)

    m1, m2 = graph.partition(matmul1, matmul1.algorithm('dim'), idx=0, dim=1, num=2)
    graph.assign(m1, 0)
    graph.assign(m2, 1)

    add1, add2 = graph.partition(add, add.algorithm('dim'), idx=0, dim=1, num=2)
    graph.assign(add1, 0)
    graph.assign(add2, 1)

    for node in [matmul2, sum]:
        sn1, sn2 = graph.replicate(node, 2)
        graph.assign(sn1, 0)
        graph.assign(sn2, 1)

    print(graph.extra_repr())

    with pytest.raises(RuntimeError):
        graph = IRAdapterGener.gen_weight(graph)
    print(graph.extra_repr())


def pas_intra_reducer(graph: IRGraph, config: ComputeConfig):
    dataloader = graph.nodes()[0]
    sn0, sn1 = graph.replicate(dataloader, 2)
    graph.assign(sn0, 0)
    graph.assign(sn1, 1)

    fw_nodes = graph.select(ntype=IRFwOperation)

    for i, node in enumerate(fw_nodes):
        if i == 1:
            sn0, sn1 = graph.partition(node, node.algorithm('dim'), idx=1, dim=0, num=2)
        else:
            sn0, sn1 = graph.replicate(node, 2)
        graph.assign(sn0, 0)
        graph.assign(sn1, 1)
    return graph


@replace_all_device_with('cpu')
def test_intra_scale_unit_reducers():
    compute_config = ComputeConfig(
        plan_ngpus=2,
        runtime_ngpus=4,
        constant_folding=True,
        use_zero=True,
        use_end2end=True,
    )
    model = ReducerModule()
    with tempfile.TemporaryDirectory() as tempdir:
        parallelize(
            model,
            {'x': torch.randn([128, 128], dtype=torch.float32)},
            pas_intra_reducer,
            compute_config,
            gen_savedir=tempdir,
            reuse='match',
            load_module=False,
        )
        for i in range(4):
            module_class = _load_parallel_module_class(ReducerModule, gen_savedir=Path(tempdir), rank=i)
            m = new_empty(module_class)
            assert len(m.reducers) == 2
            reducer0, reducer1 = m.reducers
            assert len(reducer0.params) == 1
            assert reducer0.params[0].shape == torch.Size([128, 128])
            assert len(reducer1.params) == 1
            assert reducer1.params[0].shape == torch.Size([64, 128])
