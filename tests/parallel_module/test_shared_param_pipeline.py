#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
import torch.nn as nn
import tempfile
import shutil
import contextlib
import copy
import pytest
from pathlib import Path


import nnscaler
import nnscaler.graph.function.function as F
from nnscaler.ir.tensor import IRFullTensor
from nnscaler.graph import IRGraph
from nnscaler.ir.adapter import IRAdapter
from nnscaler.parallel import ComputeConfig, parallelize, build_optimizer
from nnscaler.policies import OpPartition, OpPlan, get_pas_ops
from nnscaler.ir.operator import IRFwOperation, IRDataOperation
from nnscaler.graph.segment import IRSegment
from nnscaler.graph.schedule.predefined import PredefinedSched
from tests.utils import clear_dir_on_rank0, init_random, raises_with_cause, PYTEST_RUN_ID
from tests.launch_torchrun import torchrun
from tests.parallel_module.test_gencode import _gencode_contains, print_gencode


# This test file demonstrates when to use multiref for shared parameters in pipeline parallelism.
# The criteria is simple, if we can insert reducers across stages to sync gradients, multiref is
# not needed. Otherwise, multiref is inserted into the graph so that gradients sync is achieved by
# combination of multiref and communications.
# The fundamental reason is that nnScaler's reducer requires a shared parameter should be ALL
# partitioned or ALL replicated, check `gen_weight` in IRAdapterGener for more details.


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(x, self.weight)
        return x.sum()


class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.weight = nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(x, self.weight)
        l = x.sum()
        return l, l.data


class Model3(torch.nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.weight = nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(x, self.weight)
        x = torch.matmul(x, self.weight)
        return x.sum()


class Model4(torch.nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.weight = nn.Parameter(torch.randn(16, 16))

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(x, self.weight)
        x = torch.matmul(x, self.weight)
        l = x.sum()
        return l, l.data


def policy_easy_no_multiref(graph, cfg):
    data_loader, fc1, fc2, loss = graph.nodes()[:4]
    graph.staging([fc1, fc2])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    ngpus = cfg.plan_ngpus
    sub_nodes = graph.replicate(data_loader, ngpus)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    if ngpus == 2:
        graph.assign(fc1, 0)

        identity = stages[1].nodes()[0]
        graph.assign(identity, 1)
        graph.assign(fc2, 1)
        graph.assign(loss, 1)
    elif ngpus == 4:
        sub_nodes = graph.partition(fc1, fc1.algorithm('dim'), idx=0, dim=0, num=2)
        graph.assign(sub_nodes[0], 0)
        graph.assign(sub_nodes[1], 1)

        identity = stages[1].nodes()[0]
        sub_nodes = graph.replicate(identity, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.partition(fc2, fc2.algorithm('dim'), idx=0, dim=0, num=2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.partition(loss, loss.algorithm('dim'), idx=0, dim=0, num=2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)
    else:
        raise NotImplementedError

    PredefinedSched.sched_1f1b(graph, 4, len(stages))

    return graph


def policy_hard_no_multiref(graph, cfg):
    data_loader, fc1, fc2, fc3, loss = graph.nodes()[:5]
    graph.staging([fc1, fc2, fc3])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    ngpus = cfg.plan_ngpus
    sub_nodes = graph.replicate(data_loader, ngpus)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    if ngpus == 4:
        graph.assign(fc1, 0)

        identity = stages[1].nodes()[0]
        graph.assign(identity, 1)
        graph.assign(fc2, 1)

        identity = stages[2].nodes()[0]
        sub_nodes = graph.replicate(identity, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.partition(fc3, fc3.algorithm('dim'), idx=0, dim=0, num=2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.partition(loss, loss.algorithm('dim'), idx=0, dim=0, num=2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)
    else:
        raise NotImplementedError

    PredefinedSched.sched_1f1b(graph, 4, len(stages))

    return graph


def policy_hard_multiref(graph, cfg):
    data_loader, fc1, fc2, fc3, loss = graph.nodes()[:5]

    # need multiref here
    param = fc1.inputs()[1].parent
    graph.multiref(param)

    graph.staging([fc1, fc2, fc3])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    ngpus = cfg.plan_ngpus
    sub_nodes = graph.replicate(data_loader, ngpus)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    if ngpus == 4:
        multiref = stages[0].nodes()[0]
        graph.assign(multiref, 0)
        graph.assign(fc1, 0)

        identity1, identity2, identity3 = stages[1].nodes()[:3]
        graph.assign(identity1, 1)
        graph.assign(identity2, 1)
        graph.assign(identity3, 1)
        graph.assign(fc2, 1)

        identity1, identity2 = stages[2].nodes()[:2]
        sub_nodes = graph.replicate(identity1, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)
        sub_nodes = graph.replicate(identity2, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.replicate(fc3, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.replicate(loss, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)
    else:
        raise NotImplementedError

    PredefinedSched.sched_1f1b(graph, 4, len(stages))

    return graph


def policy_hard_multiref2(graph, cfg):
    data_loader, fc1, fc2, loss = graph.nodes()[:4]

    # need multiref here
    param = fc1.inputs()[1].parent
    graph.multiref(param)

    graph.staging([fc1, fc2])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    ngpus = cfg.plan_ngpus
    sub_nodes = graph.replicate(data_loader, ngpus)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    if ngpus == 4:
        sub_nodes = graph.partition(fc1, fc1.algorithm('dim'), idx=0, dim=0, num=2)
        graph.assign(sub_nodes[0], 0)
        graph.assign(sub_nodes[1], 1)

        identity1, identity2 = stages[1].nodes()[:2]
        sub_nodes = graph.replicate(identity1, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)
        sub_nodes = graph.replicate(identity2, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.replicate(fc2, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.replicate(loss, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)
    else:
        raise NotImplementedError

    PredefinedSched.sched_1f1b(graph, 4, len(stages))

    return graph


def worker_pipeline(model_cls, pas, plan_ngpus, checker):
    nnscaler.init()
    m = model_cls()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2, 16], dtype=torch.float32, device=torch.cuda.current_device())

    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'test_detach_loss_pp_2_{PYTEST_RUN_ID}') as tempdir:
        pm = parallelize(
                m,
                {'x': trace_data},
                pas,
                ComputeConfig(plan_ngpus, plan_ngpus, use_end2end=True),
                reuse='override',
                gen_savedir=tempdir,
            )
        pm.to(torch.cuda.current_device())

        # print_gencode(tempdir, model_cls, pm.rank)
        checker(model_cls, pm, tempdir)
        samples = [torch.randn([2, 16], dtype=torch.float32, device=torch.cuda.current_device()) for _ in range(4)]
        ret = pm.train_step(samples)


def checker_no_multiref(model_cls, pm, tempdir):
    assert len(pm.reducers) == 1
    assert len(pm.reducers[0].params) == 1
    assert pm.reducers[0].params[0].shape == torch.Size([16, 16])


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('model_cls', [Model, Model2])
@pytest.mark.parametrize('plan_ngpus', [2, 4])
def test_shared_param_pipeline_no_multiref_easy(model_cls, plan_ngpus):
    torchrun(plan_ngpus, worker_pipeline, model_cls, policy_easy_no_multiref, plan_ngpus, checker_no_multiref)
    # should not raise any exception
    assert True


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('model_cls', [Model3, Model4])
@pytest.mark.parametrize('plan_ngpus', [4])
def test_shared_param_pipeline_no_multiref_hard(model_cls, plan_ngpus):
    torchrun(plan_ngpus, worker_pipeline, model_cls, policy_hard_no_multiref, plan_ngpus, checker_no_multiref)
    # should not raise any exception
    assert True


def checker_multiref(model_cls, pm, tempdir):
    # no reducer should be created in any rank
    # gradient accumulation and sync is achieved by multiref and communications
    assert not pm.reducers
    all_params = list(pm.parameters())
    if pm.rank == 0:
        assert len(all_params) == 1
        assert all_params[0].shape == torch.Size([16, 16])
        assert len(_gencode_contains(tempdir, model_cls, pm.rank, r'multiref\(')) == 1
    else:
        assert not all_params
        assert not _gencode_contains(tempdir, model_cls, pm.rank, r'multiref\(')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('model_cls', [Model3, Model4])
@pytest.mark.parametrize('plan_ngpus', [4])
def test_shared_param_pipeline_multiref_hard(model_cls, plan_ngpus):
    torchrun(plan_ngpus, worker_pipeline, model_cls, policy_hard_multiref, plan_ngpus, checker_multiref)
    # should not raise any exception
    assert True


def checker_multiref2(model_cls, pm, tempdir):
    # no reducer should be created in any rank
    # gradient accumulation and sync is achieved by multiref and communications
    # print_gencode(tempdir, model_cls, pm.rank)
    assert not pm.reducers
    all_params = list(pm.parameters())
    if pm.rank in [0, 1]:
        assert len(all_params) == 1
        assert all_params[0].shape == torch.Size([16, 16])
        assert len(_gencode_contains(tempdir, model_cls, pm.rank, r'multiref\(')) == 1
    else:
        assert not all_params
        assert not _gencode_contains(tempdir, model_cls, pm.rank, r'multiref\(')


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('model_cls', [Model, Model2])
@pytest.mark.parametrize('plan_ngpus', [4])
def test_shared_param_pipeline_multiref_hard2(model_cls, plan_ngpus):
    torchrun(plan_ngpus, worker_pipeline, model_cls, policy_hard_multiref2, plan_ngpus, checker_multiref2)
    # should not raise any exception
    assert True


def policy_hard_multiref_error(graph, cfg):
    data_loader, fc1, fc2, fc3, loss = graph.nodes()[:5]

    graph.staging([fc1, fc2, fc3])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    ngpus = cfg.plan_ngpus
    sub_nodes = graph.replicate(data_loader, ngpus)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    if ngpus == 4:
        multiref = stages[0].nodes()[0]
        graph.assign(multiref, 0)
        graph.assign(fc1, 0)

        identity = stages[1].nodes()[0]
        graph.assign(identity, 1)
        graph.assign(fc2, 1)

        identity = stages[2].nodes()[0]
        sub_nodes = graph.replicate(identity, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.replicate(fc3, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.replicate(loss, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)
    else:
        raise NotImplementedError

    PredefinedSched.sched_1f1b(graph, 4, len(stages))

    return graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
@pytest.mark.parametrize('model_cls', [Model3, Model4])
def test_shared_param_error(model_cls):
    m = model_cls()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2, 16], dtype=torch.float32, device=torch.cuda.current_device())

    with tempfile.TemporaryDirectory() as tempdir:
        with raises_with_cause(RuntimeError, match='The weight consumers can either be ALL replicated or ALL partitioned'):
            parallelize(
                m,
                {'x': trace_data},
                policy_hard_multiref_error,
                ComputeConfig(4, 4, use_end2end=True),
                reuse='override',
                gen_savedir=tempdir,
                load_module=False,
            )


def policy_hard_multiref2_error(graph, cfg):
    data_loader, fc1, fc2, loss = graph.nodes()[:4]

    graph.staging([fc1, fc2])
    stages = graph.select(ntype=IRSegment, flatten=False)
    stages = [s for s in stages if s.isfw()]

    ngpus = cfg.plan_ngpus
    sub_nodes = graph.replicate(data_loader, ngpus)
    for i, sub_node in enumerate(sub_nodes):
        graph.assign(sub_node, i)

    if ngpus == 4:
        sub_nodes = graph.partition(fc1, fc1.algorithm('dim'), idx=0, dim=0, num=2)
        graph.assign(sub_nodes[0], 0)
        graph.assign(sub_nodes[1], 1)

        identity = stages[1].nodes()[0]
        sub_nodes = graph.replicate(identity, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.replicate(fc2, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)

        sub_nodes = graph.replicate(loss, 2)
        graph.assign(sub_nodes[0], 2)
        graph.assign(sub_nodes[1], 3)
    else:
        raise NotImplementedError

    PredefinedSched.sched_1f1b(graph, 4, len(stages))

    return graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason='lack of gpu devices')
@pytest.mark.parametrize('model_cls', [Model, Model2])
def test_shared_param_error2(model_cls):
    m = model_cls()
    m.train()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    trace_data = torch.randn([2, 16], dtype=torch.float32, device=torch.cuda.current_device())

    with tempfile.TemporaryDirectory() as tempdir:
        with raises_with_cause(RuntimeError, match='The weight consumers can either be ALL replicated or ALL partitioned'):
            parallelize(
                m,
                {'x': trace_data},
                policy_hard_multiref2_error,
                ComputeConfig(4, 4, use_end2end=True),
                reuse='override',
                gen_savedir=tempdir,
                load_module=False,
            )


class ColocatedSharedModel(Model):
    def __init__(self, repeat_first):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(16))
        self.repeat_first = repeat_first

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        if self.repeat_first:
            x = torch.matmul(x, self.weight)
        x = x + self.bias
        x = torch.matmul(x, self.weight)
        return x.sum()


def policy_colocated_shared_param(graph, cfg):
    stage = 0
    for node in get_pas_ops(graph):
        if node.fn == torch.add:
            stage = 1
        elif node.fn == torch.matmul and stage == 1:
            stage = 2
        elif node.name == 'sum':
            stage = 3
        yield OpPlan(node, stage_id=stage, partition=OpPartition(0, 0))


def check_pipeline_training(pm, reference):
    optimizer = build_optimizer(pm, torch.optim.SGD, lr=0.001)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    reference_params = dict(reference.named_parameters())
    for step in range(2):
        samples = [
            torch.arange(64, device='cuda', dtype=torch.float64).reshape(4, 16) / 64
            + micro * 0.1 + step * 0.01
            for micro in range(4)
        ]
        reference_optimizer.zero_grad()
        expected_losses = [reference(sample) for sample in samples]
        torch.stack(expected_losses).sum().backward()
        losses = pm.train_step(samples)
        assert len(losses) == len(expected_losses)
        for actual, expected in zip(losses, expected_losses):
            torch.testing.assert_close(actual, expected)
        for name, meta in pm.fullmap.items():
            torch.testing.assert_close(
                getattr(pm, name).grad,
                reference_params[meta.orig_name].grad[meta.slicers],
            )
        optimizer.step()
        reference_optimizer.step()
        for name, meta in pm.fullmap.items():
            torch.testing.assert_close(
                getattr(pm, name),
                reference_params[meta.orig_name][meta.slicers],
            )
        optimizer.zero_grad()


def worker_colocated_shared_param(multiref, repeat_first):
    nnscaler.init()
    torch.manual_seed(0)
    model = ColocatedSharedModel(repeat_first).double()
    reference = copy.deepcopy(model).cuda()
    config = ComputeConfig(
        4, 4, use_end2end=True,
        pas_config={
            'pipeline_size': 2,
            'pipeline_nmicros': 4,
            'pipeline_scheduler': '1f1b_interleaved',
            'pipeline_multiref_replicated_params': multiref,
        },
    )
    directory = Path(tempfile.gettempdir()) / f'test_colocated_shared_param_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        pm = parallelize(
            model, {'x': torch.ones(4, 16, dtype=torch.float64)},
            policy_colocated_shared_param, config,
            gen_savedir=tempdir, reuse='override',
        ).cuda()
        expected_name = 'weight' if pm.rank < 2 else 'bias'
        assert {meta.orig_name for meta in pm.fullmap.values()} == {expected_name}
        check_pipeline_training(pm, reference)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('multiref', [False, True])
@pytest.mark.parametrize('repeat_first', [False, True])
def test_colocated_shared_param_pipeline(multiref, repeat_first):
    torchrun(4, worker_colocated_shared_param, multiref, repeat_first)


def worker_shared_param_gradients(plan_ngpus, partition_input, partition_dim):
    nnscaler.init()
    torch.manual_seed(0)
    model = Model().double()
    reference = copy.deepcopy(model).cuda()

    def policy(graph, cfg):
        stage = -1
        for node in get_pas_ops(graph):
            partition = 'auto'
            if node.fn == torch.matmul:
                stage += 1
                partition = OpPartition(partition_input, partition_dim)
            yield OpPlan(node, stage_id=stage, partition=partition)

    config = ComputeConfig(
        plan_ngpus, plan_ngpus, use_end2end=True,
        pas_config={
            'pipeline_size': 2,
            'pipeline_nmicros': 4,
            'pipeline_multiref_replicated_params': False,
        },
    )
    directory = Path(tempfile.gettempdir()) / f'test_shared_param_gradients_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        pm = parallelize(
            model, {'x': torch.ones(4, 16, dtype=torch.float64)}, policy, config,
            gen_savedir=tempdir, reuse='override',
        ).cuda()
        assert len(pm.reducers) == 1
        check_pipeline_training(pm, reference)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('plan_ngpus,partition_input,partition_dim', [
    (2, 0, 0), (4, 0, 0), (4, 1, 0), (4, 1, 1),
])
def test_shared_param_pipeline_gradients(plan_ngpus, partition_input, partition_dim):
    torchrun(plan_ngpus, worker_shared_param_gradients, plan_ngpus, partition_input, partition_dim)
