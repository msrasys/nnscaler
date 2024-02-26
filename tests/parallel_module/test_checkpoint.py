import tempfile
import itertools
import re
from pathlib import Path
import shutil
import pytest
from typing import Dict, Tuple, List
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from cube.parallel import ComputeConfig, parallelize, build_optimizer, merge_state_dicts, load_merged_state_dicts
from cube.runtime.module import ParallelModule, ExtraState
from cube.runtime.gnorm import calcuate_gnorm

from .common import PASRandomSPMD, PASData, CubeLinear, init_random, init_distributed, clear_dir_on_rank0
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively


class FcRelu(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.fc1 = CubeLinear(in_features, in_features, bias=bias)
        self.relu1 = nn.ReLU()
        self.fc2 = CubeLinear(in_features, out_features, bias=bias)
        self.relu2 = nn.ReLU()


    def forward(self, x):
        return self.relu2(self.fc2(self.relu1(self.fc1(x))))


class FcRelu_4_4(FcRelu):
    def __init__(self):
        super().__init__(4, 4)


def _to_cube_model(module, pas, compute_config, cube_savedir, instance_name):
    return parallelize(
        module,
        {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
        pas,
        compute_config,
        cube_savedir=cube_savedir,
        instance_name=instance_name
    )


def _create_cube_module(pas, compute_config, cube_savedir, module_type='whole'):
    init_random()
    if module_type == 'whole':
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 4)
                self.fc_relu1 = FcRelu_4_4()
                self.linear2 = nn.Linear(4, 4)
                self.fc_relu2 = FcRelu_4_4()
                self.linear3 = nn.Linear(4, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.fc_relu1(x)
                x = self.linear2(x)
                x = self.fc_relu2(x)
                x = self.linear3(x)
                x = self.sigmoid(x)
                return x
        CompiledModule = _to_cube_model(CompiledModule, pas, compute_config, cube_savedir, 'whole')
    else:
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 4)
                self.fc_relu1 = _to_cube_model(FcRelu_4_4(), pas, compute_config, cube_savedir, 'fc_relu1')
                self.linear2 = nn.Linear(4, 4)
                self.fc_relu2 = _to_cube_model(FcRelu_4_4(), pas, compute_config, cube_savedir, 'fc_relu2')
                self.linear3 = nn.Linear(4, 1)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.fc_relu1(x)
                x = self.linear2(x)
                x = self.fc_relu2(x)
                x = self.linear3(x)
                x = self.sigmoid(x)
                return x
    init_random()
    compiled_module = CompiledModule().cuda()
    return compiled_module

DATA_SIZE = 256

@dataclass
class StepResult:
    pred: torch.Tensor
    loss: torch.Tensor
    grads: Dict[str, torch.Tensor]
    gnorm: torch.Tensor
    weights: Dict[str, torch.Tensor]


def _train(model: torch.nn.Module, num_replicas, rank, start, end, ckpt_dir):
    ckpt_file_template = 'ckpt_{rank}_{start}.pth'
    ckpt_merged_file_template = 'ckpt_merged_{start}.pth'
    ckpt_start_file = ckpt_dir / ckpt_file_template.format(
        rank=torch.distributed.get_rank(),
        start=start
    )
    ckpt_start_merged_file = ckpt_dir / ckpt_merged_file_template.format(
        start=start
    )
    init_random()

    loss_fn = nn.BCELoss()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    if ckpt_start_file.exists():
        ckpt_dict = torch.load(ckpt_start_file)
        model_state_dict = ckpt_dict['model']
        for name, m in model.named_modules():
            prefix = f'{name}.' if name else ''
            if isinstance(m, ParallelModule):
                assert f'{prefix}CUBE_EXTRA_STATE' in model_state_dict
        optimizer_state_dict = ckpt_dict['optimizer']
        assert 'CUBE_EXTRA_STATE' in optimizer_state_dict
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        assert ckpt_start_merged_file.exists()
        merged_ckpt_dict = torch.load(ckpt_start_merged_file)
        merged_model_state_dict = merged_ckpt_dict['model']
        model_from_merged = load_merged_state_dicts(type(model)(), merged_model_state_dict)

        # check merged model
        result_orig_model_state_dict = model.state_dict()
        result_merged_model_state_dict = model_from_merged.state_dict()
        assert set(result_orig_model_state_dict.keys()) == set(result_merged_model_state_dict.keys())
        for k in result_orig_model_state_dict.keys():
            if k.endswith('CUBE_EXTRA_STATE'):
                continue
            assert torch.equal(result_orig_model_state_dict[k], result_merged_model_state_dict[k])

        # TODO: check merged optimizer
        # merged_optimizer_state_dict = merged_ckpt_dict['optimizer']

    data = []
    init_random()
    for _ in range(DATA_SIZE):
        data.append((
            torch.randn((2, 4), device='cuda', dtype=torch.float32),
            torch.randn((2, 1), device='cuda', dtype=torch.float32),
        ))
    data = data[start:end]  # continue from last training
    data = [data[i] for i in range(rank, len(data), num_replicas)]
    results = []
    for i, (x, y) in enumerate(data):
        model.train()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        grads = {n: p.grad for n, p in model.named_parameters()}
        gnorm = optimizer.clip_gnorm()
        results.append(clone_to_cpu_recursively([y_pred, loss, grads, gnorm]))
        optimizer.zero_grad()
        weights = {n: p.data for n, p in model.named_parameters()}
        results[-1].append(clone_to_cpu_recursively(weights))
        results[-1] = StepResult(*results[-1])

    ckpt_file = ckpt_dir / ckpt_file_template.format(
        rank=torch.distributed.get_rank(),
        start=end
    )
    ckpt_merged_file = ckpt_dir / ckpt_merged_file_template.format(
        start=end
    )
    model_state_dict = model.state_dict()
    for name, m in model.named_modules():
        if isinstance(m, ParallelModule):
            prefix = f'{name}.' if name else ''
            assert f'{prefix}CUBE_EXTRA_STATE' in model_state_dict
            extra_state1 = ExtraState(**model_state_dict[f'{prefix}CUBE_EXTRA_STATE'])
            assert extra_state1.compute_config
            assert extra_state1.model_idx2opt_idx
            assert extra_state1.opt_idx2ranks
            assert extra_state1.origin_param_names
    optimizer_state_dict = optimizer.state_dict()
    assert 'CUBE_EXTRA_STATE' in optimizer_state_dict
    torch.save({
        'model': model_state_dict,
        'optimizer': optimizer_state_dict
    }, ckpt_file)
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        ckpt_files = [ckpt_dir / ckpt_file_template.format(rank=i, start=end) for i in range(torch.distributed.get_world_size())]
        ckpt_state_dicts = [torch.load(f) for f in ckpt_files]
        model_state_dicts = [ckpt['model'] for ckpt in ckpt_state_dicts]
        optimizer_state_dicts = [ckpt['optimizer'] for ckpt in ckpt_state_dicts]
        merged_model_state_dicts, merged_optimizer_state_dict = merge_state_dicts(model_state_dicts, optimizer_state_dicts)
        torch.save({
            'model': merged_model_state_dicts,
            'optimizer': merged_optimizer_state_dict
        }, ckpt_merged_file)
    return results


def _gpu_worker(module_type, pas, plan_ngpus, runtime_ngpus, per_resume_update_count, resume_count):
    init_distributed()
    compiled_results = []
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / 'cube_test_ckpt') as tempdir:
        for i in range(resume_count):
            start = i * per_resume_update_count
            end = (i + 1) * per_resume_update_count
            compiled_module = _create_cube_module(pas,
                ComputeConfig(plan_ngpus, runtime_ngpus, use_zero=True),
                tempdir,
                module_type,
            )
            compiled_results.extend(_train(
                compiled_module,
                runtime_ngpus // plan_ngpus,
                torch.distributed.get_rank() // plan_ngpus,
                start, end, tempdir
            ))
        return compiled_results

@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('module_type', ['sub', 'whole'])
def test_checkpoint(module_type):
    cube_results = launch_torchrun(4, _gpu_worker, module_type, PASRandomSPMD, 2, 4, 32, 1)
    rcube_results = launch_torchrun(4, _gpu_worker, module_type, PASRandomSPMD, 2, 4, 16, 2)

    results0, results1,  results2, results3 = cube_results[0], cube_results[1], cube_results[2], cube_results[3]
    rresults0, rresults1,  rresults2, rresults3 = rcube_results[0], rcube_results[1], rcube_results[2], rcube_results[3]

    # pred, loss
    for r0, r1 in [(results0, results1), (results2, results3),
                   (rresults0, rresults1), (rresults2, rresults3),
                   (results0, rresults0), (results2, rresults2)
        ]:
        # have the same input
        assert len(r0) == len(r1)  # iteration count
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.pred, b.pred)  # pred
            assert torch.equal(a.loss, b.loss)  # loss
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm

    # grad, weights
    for r0, r1 in [(results0, results2), (results1, results3),
                   (rresults0, rresults2), (rresults1, rresults3),
                   (results0, rresults0), (results1, rresults1)
        ]:
        # in the same shard, grads and weights are the same
        assert len(r0) == len(r1)
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm
            for k in a.grads.keys(): # grad
                assert torch.equal(a.grads[k], b.grads[k])
            for k in a.weights.keys():  # weights
                assert torch.equal(a.weights[k], b.weights[k])
