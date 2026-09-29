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
from nnscaler.graph.schedule.schedplan import SchedulePlan
from tests.utils import clear_dir_on_rank0, init_random, raises_with_cause, replace_all_device_with, PYTEST_RUN_ID
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
        scaling_factor = pm.compute_config.runtime_ngpus // pm.compute_config.plan_ngpus
        (torch.stack(expected_losses).sum() * scaling_factor).backward()
        losses = pm.train_step(samples)
        assert len(losses) == len(expected_losses)
        for actual, expected in zip(losses, expected_losses):
            torch.testing.assert_close(actual, expected)
        for name, meta in pm.fullmap.items():
            expected_grad = reference_params[meta.orig_name].grad
            if expected_grad is None:
                assert getattr(pm, name).grad is None
                continue
            torch.testing.assert_close(
                getattr(pm, name).grad,
                expected_grad[meta.slicers],
            )
        optimizer.step()
        reference_optimizer.step()
        for name, meta in pm.fullmap.items():
            torch.testing.assert_close(
                getattr(pm, name),
                reference_params[meta.orig_name][meta.slicers],
            )
        optimizer.zero_grad()

    with torch.inference_mode():
        pm.eval()
        reference.eval()
        torch.testing.assert_close(pm.infer_step(samples), [reference(sample) for sample in samples])


def worker_colocated_shared_param(multiref, repeat_first, async_reducer=False, use_fbw=False):
    nnscaler.init()
    torch.manual_seed(0)
    model = ColocatedSharedModel(repeat_first).double()
    reference = copy.deepcopy(model).cuda()
    pas_config = {
        'pipeline_size': 2,
        'pipeline_nmicros': 4,
        'pipeline_scheduler': '1f1b_interleaved',
    }
    if multiref != 'default':
        pas_config['pipeline_multiref_replicated_params'] = multiref
    config = ComputeConfig(
        4, 4, use_end2end=True, use_async_reducer=async_reducer, use_fbw=use_fbw,
        pas_config=pas_config,
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
        assert len(pm.reducers) == (0 if multiref is True and pm.rank < 2 else 1)
        for reducer in pm.reducers:
            assert set(reducer._param_num_segments.values()) == {2 if pm.rank < 2 else 1}
        check_pipeline_training(pm, reference)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('multiref', ['default', None, False, True])
@pytest.mark.parametrize('repeat_first', [False, True])
@pytest.mark.parametrize('async_reducer', [False, True])
def test_colocated_shared_param_pipeline(multiref, repeat_first, async_reducer):
    torchrun(4, worker_colocated_shared_param, multiref, repeat_first, async_reducer)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_colocated_shared_param_pipeline_async_fbw():
    torchrun(4, worker_colocated_shared_param, None, True, True, True)


class UnevenColocatedSharedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(16, 16))
        self.bias = nn.Parameter(torch.randn(16))

    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.matmul(x, self.weight)
        x = torch.matmul(x, self.weight)
        return (x + self.bias).sum()


def policy_uneven_colocated_shared_param(graph, cfg):
    stage = -1
    for node in get_pas_ops(graph):
        if node.fn in (torch.matmul, torch.add):
            stage += 1
        yield OpPlan(node, stage_id=stage, partition=OpPartition(0, 0))


def worker_uneven_colocated_shared_param(async_reducer):
    nnscaler.init()
    torch.manual_seed(0)
    model = UnevenColocatedSharedModel().double()
    reference = copy.deepcopy(model).cuda()
    config = ComputeConfig(
        2, 2, use_end2end=True, use_async_reducer=async_reducer,
        pas_config={
            'pipeline_size': 1,
            'pipeline_nmicros': 4,
            'pipeline_scheduler': '1f1b_interleaved',
        },
    )
    directory = Path(tempfile.gettempdir()) / f'test_uneven_colocated_shared_param_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        pm = parallelize(
            model, {'x': torch.ones(4, 16, dtype=torch.float64)},
            policy_uneven_colocated_shared_param, config,
            gen_savedir=tempdir, reuse='override',
        ).cuda()
        assert len(pm.reducers) == 1
        reducer = pm.reducers[0]
        registered_counts = {
            meta.orig_name: reducer._param_num_segments[getattr(pm, name)]
            for name, meta in pm.fullmap.items()
        }
        assert registered_counts == {'weight': 3, 'bias': 1}
        check_pipeline_training(pm, reference)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('async_reducer', [False, True])
def test_uneven_colocated_shared_param_pipeline(async_reducer):
    torchrun(2, worker_uneven_colocated_shared_param, async_reducer)


class SharedShardModel(torch.nn.Module):
    def __init__(self, uses):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(8, 8) / 8)
        self.uses = uses

    def forward(self, x):
        for _ in range(self.uses):
            x = torch.matmul(x, self.weight)
        return (x * x).sum()


def policy_shared_shards(graph, cfg):
    matmuls = [node for node in graph.select(ntype=IRFwOperation) if node.fn == torch.matmul]
    # Rank 1 owns the right half in stages on (0, 1), and the left half
    # in the stage on (1, 2). Both local parameters have the same shape.
    starts = matmuls[:2] if cfg.pas_config['repeat_in_segment'] else matmuls
    graph.staging(starts)
    stages = [node for node in graph.select(ntype=IRSegment, flatten=False) if node.isfw()]
    for index, stage in enumerate(stages):
        devices = (0, 1) if index % 2 == 0 else (1, 2)
        for node in list(stage.nodes()):
            if node.fn == torch.matmul:
                parts = graph.partition(node, node.algorithm('dim'), idx=1, dim=1, num=2)
            else:
                parts = graph.partition(node, node.algorithm('dim'), idx=0, dim=1, num=2)
            for part, device in zip(parts, devices):
                graph.assign(part, device)
    for node in graph.select(ntype=IRDataOperation):
        for device, replica in enumerate(graph.replicate(node, cfg.plan_ngpus)):
            graph.assign(replica, device)
    nmicros = cfg.pas_config['pipeline_nmicros']
    schedule = SchedulePlan(graph, nmicros)
    # Serialize overlapping stages so they never compete for the same rank.
    sequence = stages + [stage.mirror for stage in reversed(stages)]
    for micro in range(nmicros):
        for step, stage in enumerate(sequence):
            schedule.add_segment(stage, micro, micro * len(sequence) + step)
    schedule.finish()
    graph.bind_schedule(schedule)
    return graph


def shared_shard_config(async_reducer, repeat_in_segment, runtime_ngpus=3):
    return ComputeConfig(
        3, runtime_ngpus, use_end2end=True, use_async_reducer=async_reducer,
        pas_config={'pipeline_nmicros': 2, 'repeat_in_segment': repeat_in_segment},
    )


def assert_shared_shard_counts(module, rank, uses, repeat_in_segment):
    rank %= 3
    counts = {}
    for reducer in module.reducers:
        for name, meta in module.fullmap.items():
            parameter = getattr(module, name)
            if parameter in reducer._param_num_segments:
                assert meta.orig_name == 'weight'
                assert tuple(parameter.shape) == (8, 4)
                counts[meta.slicers[1].start] = reducer._param_num_segments[parameter]
    first_group_count = 2 if uses == 3 and not repeat_in_segment else 1
    expected = {0: first_group_count} if rank == 0 else {4: 1} if rank == 2 else {
        4: first_group_count, 0: 1,
    }
    assert counts == expected


@replace_all_device_with('cpu')
@pytest.mark.parametrize('uses,repeat_in_segment', [(2, False), (3, False), (3, True)])
def test_shared_shard_counts_codegen(tmp_path, uses, repeat_in_segment):
    from nnscaler.parallel import _load_parallel_module_class
    from tests.utils import new_empty

    instance_name = f'shards_{uses}_{repeat_in_segment}'
    parallelize(
        SharedShardModel(uses), {'x': torch.ones(4, 8)}, policy_shared_shards,
        shared_shard_config(True, repeat_in_segment),
        gen_savedir=tmp_path, load_module=False, reuse='override', instance_name=instance_name,
    )
    for rank in range(3):
        cls = _load_parallel_module_class(
            SharedShardModel, gen_savedir=tmp_path, rank=rank, instance_name=instance_name,
        )
        module = new_empty(cls, device='cpu')
        assert_shared_shard_counts(module, rank, uses, repeat_in_segment)


def worker_shared_shard_counts(async_reducer, uses, repeat_in_segment):
    nnscaler.init()
    torch.manual_seed(0)
    source = SharedShardModel(uses).double()
    reference = copy.deepcopy(source).cuda()
    directory = Path(tempfile.gettempdir()) / f'shared_shard_counts_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        module = parallelize(
            source, {'x': torch.ones(4, 8, dtype=torch.float64)}, policy_shared_shards,
            shared_shard_config(async_reducer, repeat_in_segment, torch.distributed.get_world_size()),
            gen_savedir=tempdir, reuse='override',
        ).cuda()
        rank = torch.distributed.get_rank()
        assert_shared_shard_counts(module, rank, uses, repeat_in_segment)
        optimizer = build_optimizer(module, torch.optim.SGD, lr=0.001)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
        for step in range(2):
            samples = [
                torch.arange(32, device='cuda', dtype=torch.float64).reshape(4, 8) / 32
                + micro * 0.1 + step * 0.01
                for micro in range(2)
            ]
            reference_optimizer.zero_grad()
            losses = [reference(sample) for sample in samples]
            scaling_factor = module.compute_config.runtime_ngpus // module.compute_config.plan_ngpus
            (torch.stack(losses).sum() * scaling_factor).backward()
            torch.testing.assert_close(module.train_step(samples), losses)
            for name, meta in module.fullmap.items():
                torch.testing.assert_close(getattr(module, name).grad, reference.weight.grad[meta.slicers])
            optimizer.step()
            reference_optimizer.step()
            for name, meta in module.fullmap.items():
                torch.testing.assert_close(getattr(module, name), reference.weight[meta.slicers])
            optimizer.zero_grad()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 6, reason='lack of gpu devices')
@pytest.mark.parametrize('async_reducer', [False, True])
@pytest.mark.parametrize('uses,repeat_in_segment', [(2, False), (3, False), (3, True)])
def test_shared_shard_counts_runtime(async_reducer, uses, repeat_in_segment):
    # DP replicas give gradient reducers different groups from activation
    # collectives in this deliberately overlapping pipeline layout.
    torchrun(6, worker_shared_shard_counts, async_reducer, uses, repeat_in_segment)


class ColocatedNoGradModel(Model):
    def __init__(self, train_first):
        super().__init__()
        self.other = nn.Parameter(torch.randn(16, 16))
        self.unused = nn.Parameter(torch.randn(16, 16))
        self.bias = nn.Parameter(torch.randn(16))
        self.train_first = train_first

    def forward(self, x):
        with torch.no_grad():
            auxiliary = x @ self.weight + x @ self.unused
        y = x @ self.other
        if self.train_first:
            y = y + x @ self.weight
        x = torch.sigmoid(y + auxiliary) + self.bias
        x = x @ self.weight
        return x.sum()


def policy_colocated_no_grad(graph, cfg):
    stage = 0
    for node in get_pas_ops(graph):
        if node.fn == torch.sigmoid:
            stage = 1
        elif node.fn == torch.matmul and stage == 1:
            stage = 2
        elif node.name == 'sum':
            stage = 3
        yield OpPlan(node, stage_id=stage, partition=OpPartition(0, 0))


def worker_colocated_no_grad(plan_ngpus, async_reducer, train_first, use_zero, use_fbw):
    nnscaler.init()
    torch.manual_seed(0)
    model = ColocatedNoGradModel(train_first).double()
    reference = copy.deepcopy(model).cuda()
    config = ComputeConfig(
        plan_ngpus, 4, use_end2end=True, use_async_reducer=async_reducer,
        use_zero=use_zero, use_fbw=use_fbw,
        pas_config={
            'pipeline_size': 2,
            'pipeline_nmicros': 4,
            'pipeline_scheduler': '1f1b_interleaved',
        },
    )
    directory = Path(tempfile.gettempdir()) / f'test_colocated_no_grad_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        pm = parallelize(
            model, {'x': torch.ones(4, 16, dtype=torch.float64)},
            policy_colocated_no_grad, config,
            gen_savedir=tempdir, reuse='override',
        ).cuda()
        check_pipeline_training(pm, reference)
        names = {getattr(pm, name): meta.orig_name for name, meta in pm.fullmap.items()}
        counts = {}
        for reducer in pm.reducers:
            assert reducer._nreplicas == 1
            counts.update({
                names[param]: count for param, count in reducer._param_num_segments.items()
            })
        expected = {'other': 1, 'weight': 2 if train_first else 1} if 'weight' in names.values() else {'bias': 1}
        assert counts == expected


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('plan_ngpus,async_reducer,train_first,use_zero,use_fbw', [
    (4, False, False, False, False),
    (4, True, False, False, False),
    (4, True, True, False, False),
    (2, False, False, False, False),
    (2, True, False, False, False),
    (2, True, True, True, False),
    (4, True, False, True, True),
])
def test_colocated_no_grad_pipeline(plan_ngpus, async_reducer, train_first, use_zero, use_fbw):
    torchrun(4, worker_colocated_no_grad, plan_ngpus, async_reducer, train_first, use_zero, use_fbw)


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
