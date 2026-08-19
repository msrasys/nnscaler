#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging
import os
from types import SimpleNamespace
import unittest.mock
import pytest
import torch
import nnscaler.autodist.pipeline_solver
from nnscaler.autodist.descs import MeshDesc, SPMDSearchOutput, TensorParallelDesc
from nnscaler.cli.trainer import Trainer
from nnscaler.cli.trainer_args import *

from .. import utils


@pytest.mark.parametrize(
    ('uniform_tp', 'expected_tp_degrees'),
    [(False, [8, 4, 2, 2]), (True, [4, 4, 4, 4])],
)
def test_stages_use_same_tp_degree(monkeypatch, uniform_tp, expected_tp_degrees):
    op_costs = (8, 4, 2, 2)

    def fake_compute_tp_info(model_graph, cfg, legal_tp_degrees):
        tp_info = {}
        for tp_degree in legal_tp_degrees:
            for stage_num in range(1, model_graph.op_num + 1):
                for start in range(model_graph.op_num):
                    for end in range(start, model_graph.op_num):
                        desc = TensorParallelDesc({}, [], MeshDesc(1, tp_degree), {})
                        all_time = sum(op_costs[start:end + 1]) / tp_degree
                        tp_info[(tp_degree, stage_num, start, end)] = SPMDSearchOutput(
                            desc, memory=0, all_time=all_time, comp_time=all_time
                        )
        return tp_info

    monkeypatch.setattr(nnscaler.autodist.pipeline_solver, '_compute_tp_info', fake_compute_tp_info)
    model_graph = SimpleNamespace(
        op_num=len(op_costs),
        get_pipeline_pivots=lambda: [1, 2, 3],
    )
    config = SimpleNamespace(
        mesh_desc=MeshDesc(2, 8),
        pipeline_nstages=4,
        update_freq=4,
    )

    if uniform_tp:
        plan = nnscaler.autodist.pipeline_solver.calc_optimal_pp_plan(
            model_graph, config, uniform_tp=True
        )
    else:
        # The default preserves the original heterogeneous-stage search.
        plan = nnscaler.autodist.pipeline_solver.calc_optimal_pp_plan(model_graph, config)

    assert [desc.mesh_desc.ngpus for desc in plan.desc.spmd_descs] == expected_tp_degrees


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
def test_1_stage(tmp_path):
    _compile(tmp_path, 1)
    # for TP, the scripts should be identical (except tensor names)
    lines0 = _count_gencode_lines(tmp_path, 0)
    lines1 = _count_gencode_lines(tmp_path, 1)
    assert lines0 == lines1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
def test_2_stages(tmp_path):
    _compile(tmp_path, 2)
    # for PP, since we have 3 linears, the scripts should be different
    lines0 = _count_gencode_lines(tmp_path, 0)
    lines1 = _count_gencode_lines(tmp_path, 1)
    assert lines0 != lines1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
def test_auto_stages(tmp_path):
    _compile(tmp_path, 'auto')
    # just check it does not throw
    # because both results are possible theoretically


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no gpu')
def test_small(tmp_path):
    # check it spreads the model to all gpus even when less are required

    # the graph is as follow:
    #   [0] data['x']
    #   [1] linear1
    #   [2] linear2
    #   [3] linear3
    #   [4] data['y']
    #   [5] cross_entroy
    # we assume linear costs 0.2, cross_entroy costs 0.3, getitem costs 0, unavoidable overhead costs 0.1,
    # and ngpus does not affect the time
    # since tp has no gain in our "profiling", the algorithm will tend to use 1 gpu per stage

    # the "best" result can be T[2,2,1,0] <- min(T[1,1,1,3]=tp_info[1,1,3,5], tp_info[1,2,0,2])
    # what we expect is T[2,4,2,0] <- min(T[1,2,2,3]=tp_info[2,1,3,5], tp_info[2,2,2,2])

    costs = [0.0, 0.2, 0.2, 0.2, 0.0, 0.3]

    orig_compute_tp_info = nnscaler.autodist.pipeline_solver._compute_tp_info

    def patched_compute_tp_info(model_graph, cfg, legal_tp_degrees):
        tp_info = orig_compute_tp_info(model_graph, cfg, legal_tp_degrees)
        for k, v in tp_info.items():
            _ngpus, _nstages, start, end = k
            v.all_time = 0.1 + sum(costs[i] for i in range(start, end + 1))
        return tp_info

    patch = unittest.mock.patch('nnscaler.autodist.pipeline_solver._compute_tp_info', patched_compute_tp_info)

    with utils.catch_log(nnscaler.autodist.pipeline_solver._logger, 'WARNING') as log:
        with patch:
            _compile(tmp_path, nstages=2, ngpus=4)

        assert 'model is too small' in log.getvalue()


def _compile(tmp_path, nstages, ngpus=2):
    trainer_args = TrainerArgs(
        compute_config=ComputeConfig(
            plan_ngpus=ngpus,
            runtime_ngpus=ngpus,
            use_end2end=True,
            pas_config={
                'pipeline_pivots': 'Linear' if nstages != 1 else '',
                'pipeline_nstages': nstages,
                'max_pipeline_bubble_ratio': 0.99,  # force autodist to accept unbalanced stages
                'max_pipeline_unbalance_ratio': 0.01,
            },
        ),
        gen_savedir=tmp_path/'src',
        gen_reuse='override',
        broadcast_strategy='none',
        run_mode='compile',
        model=ModelConfig(type=Model),
        optimizer=OptimizerConfig(type=torch.optim.AdamW),
        dataset=DatasetConfig(type=RandomDataset, train_args={'length': 100}),
        max_train_steps=1,
    )
    trainer = Trainer(train_args=trainer_args)
    trainer.run()


def _count_gencode_lines(tmp_path, index):
    script = 'tests/autodist/test_pipeline_nstages'
    path = f'_parallel_modules/{script}/Model/_/gencode{index}.py'
    text = Path(tmp_path, 'src', path).read_text()
    return text.count('\n')


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, 10)

    def forward(self, data):
        x = data['x']
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return torch.nn.functional.cross_entropy(x, data['y'])


class RandomDataset:
    def __init__(self, length):
        self.length = length

    def __getitem__(self, i):
        return {
            'x': torch.rand(10),
            'y': torch.randint(10, tuple()),
        }

    def __len__(self):
        return self.length
