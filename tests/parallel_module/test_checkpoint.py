#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import tempfile
import itertools
import re
from pathlib import Path
import shutil
import pytest
from typing import Dict, Tuple, List
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import numpy as np

from nnscaler.parallel import (
    ComputeConfig, parallelize,
    build_optimizer,
    merge_state_dicts,
    load_merged_state_dict,
    load_merged_state_dict_from_rank,
    trimmed_broadcast_merged_state_dict,
    gather_full_model_state_dict_from_files,
)
from nnscaler.runtime.module import ParallelModule, ExtraState
from nnscaler.runtime.gnorm import calcuate_gnorm

from .common import CubeLinear, init_random, init_distributed, PASMegatron, assert_equal
from ..launch_torchrun import launch_torchrun, clone_to_cpu_recursively
from ..utils import replace_all_device_with, clear_dir_on_rank0, PYTEST_RUN_ID


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
        self.register_buffer('buffer', torch.ones(1, 4))
    def forward(self, x):
        return super().forward(x + self.buffer)


def _to_cube_model(module, pas, compute_config, cube_savedir, instance_name, dummy_input = None):
    return parallelize(
        module,
        dummy_input if dummy_input is not None else {'x': torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])},
        pas,
        compute_config,
        gen_savedir=cube_savedir,
        instance_name=instance_name
    )


def pipeline_dummy_data():
    return {
        'data': torch.randn(
            2, 16, device=torch.cuda.current_device()),
        'target': torch.rand(
            2, 16, device=torch.cuda.current_device())
    }


class End2EndMLP(nn.Module):
    def __init__(self):
        init_random()
        super().__init__()
        self.layers = torch.nn.ModuleList([])
        for _ in range(8):
            self.layers.append(nn.Linear(16, 16, bias=False))
        self.loss_fn = nn.BCELoss()

    def forward(self, data: Dict[str, torch.Tensor]):
        x = data['data']
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(x)
        loss = self.loss_fn(x, data['target'])
        return loss

    @classmethod
    def to_pipeline_module(cls, compute_config: ComputeConfig, cube_savedir,
        instance_name='pipeline', scheduler='1f1b'
    ):
        assert compute_config.runtime_ngpus == 4
        assert compute_config.plan_ngpus == 2
        compute_config = replace(compute_config,
            use_end2end=True,
            pas_config=dict(
                pipeline_nmicros=2,
                pipeline_nstages=2,
                pipeline_scheduler=scheduler
            )
        )
        return parallelize(
            cls,
            {'data': pipeline_dummy_data()},
            PASMegatron,
            compute_config,
            gen_savedir=cube_savedir,
            instance_name=instance_name
        )

    @classmethod
    def gen_pipeline_data(cls, data_size, start, end, rank, num_replicas):
        data = []
        for _ in range(data_size):
            data.append(pipeline_dummy_data())
        data = data[start:end]
        data = [data[i] for i in range(rank, len(data), num_replicas)]
        data = [(data[i:i + 2], None) for i in range(0, len(data), 2)]
        return data

    @classmethod
    def gen_raw_data(cls, data_size, start, end, rank, num_replicas):
        data = []
        for _ in range(data_size):
            data.append(pipeline_dummy_data())
        data = data[start:end]
        data = [(data[i], None) for i in range(rank, len(data), num_replicas)]
        return data


class End2EndMLPWithUnusedAndShared(End2EndMLP):
    def __init__(self):
        super().__init__()
        self.linear0_unused = nn.Linear(4, 4)  # unused weights
        self.layers[5].weight = self.layers[0].weight  # shared weights across stages


def train_step(model, x, y, optimizer):
    model.train()
    if isinstance(model, ParallelModule) and model.use_scheduler:
        # actually train_step will return two losses (for each input)
        # here we fake one loss to y_pred, so we don't need to change the check logic
        y_pred, loss = model.train_step(x)
        # workaround scalar tensor bug
        y_pred = y_pred.reshape(())
        loss = loss.reshape(())
    elif isinstance(model, End2EndMLP):
        y_pred = model(x)
        loss = y_pred
        loss.backward()
    else:
        loss_fn = nn.BCELoss()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
    optimizer.step()
    return y_pred, loss


def gendata(model, data_size, start, end, rank, num_replicas):
    data = []
    init_random()
    if isinstance(model, ParallelModule) and model.use_scheduler:
        data = End2EndMLP.gen_pipeline_data(data_size, start, end, rank, num_replicas)
    elif isinstance(model, End2EndMLP):
        data = End2EndMLP.gen_raw_data(data_size, start, end, rank, num_replicas)
    else:
        for _ in range(data_size):
            data.append((
                torch.randn((2, 4), device='cuda', dtype=torch.float32),
                torch.rand((2, 1), device='cuda', dtype=torch.float32),
            ))
        data = data[start:end]  # continue from last training
        data = [data[i] for i in range(rank, len(data), num_replicas)]
    return data


def _create_cube_module(pas, compute_config: ComputeConfig, cube_savedir, module_type='whole'):
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
        CompiledModule = _to_cube_model(CompiledModule, pas, compute_config, cube_savedir, f'whole-{compute_config.inference_only}')
    elif module_type == 'pipeline':
        CompiledModule = End2EndMLP.to_pipeline_module(compute_config, cube_savedir,
            f'pipeline-{compute_config.inference_only}',
            scheduler='infer_pipe' if compute_config.inference_only else '1f1b'
        )
    elif module_type == 'sub':
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(4, 4)
                self.fc_relu1 = _to_cube_model(FcRelu_4_4(), pas, compute_config, cube_savedir, f'fc_relu1-{compute_config.inference_only}')
                self.linear2 = nn.Linear(4, 4)
                self.fc_relu2 = _to_cube_model(FcRelu_4_4(), pas, compute_config, cube_savedir, f'fc_relu2-{compute_config.inference_only}')
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
    elif module_type == 'start':
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = _to_cube_model(CubeLinear(4, 4, bias=True),
                    pas, compute_config, cube_savedir, f'start_linear1-{compute_config.inference_only}'
                )
                self.linear2 = CubeLinear(4, 1, bias=True)
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sigmoid(x)
                return x
    elif module_type == 'end':
        # parallel module as the last module
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = CubeLinear(4, 4, bias=True)
                self.linear2 = _to_cube_model(CubeLinear(4, 4, bias=True),
                    pas, compute_config, cube_savedir, f'end_linear2-{compute_config.inference_only}'
                )
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = torch.sum(x, dim=1, keepdim=True)
                x = self.sigmoid(x)
                return x
    elif module_type == 'small':
        # num of parameter elements is small
        class CompiledModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = CubeLinear(4, 4, bias=True)
                self.linear2 = _to_cube_model(CubeLinear(4, 1, bias=True),
                    pas, compute_config, cube_savedir, f'small_linear2-{compute_config.inference_only}'
                )
                # the following tests depend on the rngstate in PASRandomSPMD
                if not compute_config.inference_only:
                    assert len(self.linear2.reducers) == 1
                    assert len(self.linear2.reducers[0].ranks) == 4
                self.sigmoid = nn.Sigmoid()
            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
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


def assert_model_state_dict_equal(state_dict1: dict, state_dict2: dict):
    assert set(state_dict1.keys()) == set(state_dict2.keys())
    for index in state_dict1.keys():
        if index.endswith('CUBE_EXTRA_STATE'):
            continue
        assert torch.equal(state_dict1[index].cpu(), state_dict2[index].cpu())


def _train(model: torch.nn.Module, num_replicas, rank, start, end, ckpt_dir, inference_module: torch.nn.Module = None, check_merge_log=False):
    ckpt_file_template = 'ckpt_{rank}_{start}.pth'
    ckpt_merged_file_template = 'ckpt_merged_{start}.pth'
    temp_inferenece_ckpt_file_template = 'inference-{rank}.pth'
    ckpt_start_file = ckpt_dir / ckpt_file_template.format(
        rank=torch.distributed.get_rank(),
        start=start
    )
    ckpt_start_merged_file = ckpt_dir / ckpt_merged_file_template.format(
        start=start
    )
    temp_inferenece_ckpt_file = ckpt_dir / temp_inferenece_ckpt_file_template.format(rank=torch.distributed.get_rank())

    init_random()

    loss_fn = nn.BCELoss()
    optimizer = build_optimizer(model, torch.optim.Adam, lr=0.01)
    if ckpt_start_file.exists():
        ckpt_dict = torch.load(ckpt_start_file, weights_only=False)
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
        merged_opt_state_dict = merged_ckpt_dict['optimizer']

        # In most cases, we can't load state_dict directly
        # because they are different models, and the names of parameters are changed.
        # inference_module.load_state_dict(model_state_dict, strict=False)
        # assert not check_model_state_dict_equal(inference_module.state_dict(), model_state_dict)

        # inference model can be loaded from merged state_dict
        load_merged_state_dict(inference_module, merged_model_state_dict)
        torch.save(inference_module.state_dict(), temp_inferenece_ckpt_file)
        torch.distributed.barrier()
        inference_ckpt_files = [ckpt_dir / temp_inferenece_ckpt_file_template.format(rank=i) for i in range(torch.distributed.get_world_size())]
        inference_state_dicts = [torch.load(f, weights_only=False) for f in inference_ckpt_files]
        merged_inference_state_dict, _ = merge_state_dicts(inference_state_dicts)
        assert_model_state_dict_equal(merged_model_state_dict, merged_inference_state_dict)

        model_from_merged = type(model)()
        optimizer_from_merged = build_optimizer(model_from_merged, torch.optim.Adam, lr=0.01)
        load_merged_state_dict(
            model_from_merged, merged_model_state_dict,
            optimizer_from_merged, merged_opt_state_dict,
        )

        model_from_merged_rank = type(model)()
        optimizer_from_merged_rank = build_optimizer(model_from_merged_rank, torch.optim.Adam, lr=0.01)
        load_merged_state_dict_from_rank(
            model_from_merged_rank, merged_model_state_dict if torch.distributed.get_rank() == 0 else None,
            optimizer_from_merged_rank, merged_opt_state_dict if torch.distributed.get_rank() == 0 else None,
        )
        assert_equal(model_from_merged.state_dict(), model_from_merged_rank.state_dict())
        assert_equal(optimizer_from_merged.state_dict(), optimizer_from_merged_rank.state_dict())

        trimmed_model_state_dict, trimmed_opt_state_dict = trimmed_broadcast_merged_state_dict(
            model_from_merged_rank, merged_model_state_dict if torch.distributed.get_rank() == 0 else None,
            optimizer_from_merged_rank, merged_opt_state_dict if torch.distributed.get_rank() == 0 else None,
        )
        assert_equal(dict(model_from_merged.state_dict()), trimmed_model_state_dict)
        assert_equal(optimizer_from_merged.state_dict()['state'], trimmed_opt_state_dict['state'])
        assert_equal(optimizer_from_merged.state_dict()['param_groups'], trimmed_opt_state_dict['param_groups'])

        # check merged model
        result_orig_model_state_dict = model.state_dict()
        result_merged_model_state_dict = model_from_merged.state_dict()
        assert_model_state_dict_equal(result_orig_model_state_dict, result_merged_model_state_dict)

        result_orig_opt_state_dict = optimizer.state_dict()
        result_merged_opt_state_dict = optimizer_from_merged.state_dict()
        assert set(result_orig_opt_state_dict.keys()) == set(result_merged_opt_state_dict.keys())
        assert result_orig_opt_state_dict['CUBE_EXTRA_STATE'] == result_merged_opt_state_dict['CUBE_EXTRA_STATE']
        assert result_orig_opt_state_dict['param_groups'] == result_merged_opt_state_dict['param_groups']
        assert set(result_orig_opt_state_dict['state']) == set(result_merged_opt_state_dict['state'])
        for index in result_orig_opt_state_dict['state']:
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                assert_equal(result_orig_opt_state_dict['state'][index][key], result_merged_opt_state_dict['state'][index][key])
    torch.distributed.barrier()
    data = gendata(model, DATA_SIZE, start, end, rank, num_replicas)
    results = []
    for i, (x, y) in enumerate(data):
        y_pred, loss = train_step(model, x, y, optimizer)
        grads = {n: p.grad.clone() for n, p in model.named_parameters()}
        gnorm = optimizer.clip_gnorm()
        results.append(clone_to_cpu_recursively([y_pred, loss, grads, gnorm]))
        optimizer.zero_grad()
        weights = {n: p.data.clone() for n, p in model.named_parameters()}
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
            if extra_state1.compute_config.use_zero:
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
        ckpt_state_dicts = [torch.load(f, weights_only=False) for f in ckpt_files]
        model_state_dicts = [ckpt['model'] for ckpt in ckpt_state_dicts]
        optimizer_state_dicts = [ckpt['optimizer'] for ckpt in ckpt_state_dicts]
        if check_merge_log:
            from nnscaler.runtime.module import _logger
            import logging
            from io import StringIO
            string_stream = StringIO()
            old = _logger.level
            _logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler(string_stream)
            handler.setLevel(logging.DEBUG)
            _logger.addHandler(handler)
            merged_model_state_dicts, merged_optimizer_state_dict = merge_state_dicts(model_state_dicts, optimizer_state_dicts)
            logs = string_stream.getvalue()
            # check some zero merging is skipped due to replicate
            assert 'skip merging duplicated optimizer state for param' in logs
            assert 'skip merging duplicated model state for param' in logs
            _logger.removeHandler(handler)
            _logger.setLevel(old)
        else:
            merged_model_state_dicts, merged_optimizer_state_dict = merge_state_dicts(model_state_dicts, optimizer_state_dicts)
        torch.save({
            'model': merged_model_state_dicts,
            'optimizer': merged_optimizer_state_dict
        }, ckpt_merged_file)
        from nnscaler.runtime.serialization import convert, load
        from contextlib import ExitStack
        ckpt_st_file_template = 'ckpt_{rank}_{start}.safetensors'
        ckpt_st_files = [ckpt_dir / ckpt_st_file_template.format(rank=i, start=end) for i in range(torch.distributed.get_world_size())]
        for pt, st in zip(ckpt_files, ckpt_st_files):
            convert(pt, st, src_format='pt', dst_format='safetensors')
        ckpt_st_state_dict_loaders = [load(f, lazy=True) for f in ckpt_st_files]
        with ExitStack() as stack:
            ckpt_st_state_dicts = []
            for f in ckpt_st_state_dict_loaders:
                ckpt_st_state_dicts.append(stack.enter_context(f).get_lazy_data())
            model_st_state_dicts = [ckpt['model'] for ckpt in ckpt_st_state_dicts]
            optimizer_st_state_dicts = [ckpt['optimizer'] for ckpt in ckpt_st_state_dicts]
            merged_model_st_state_dicts, merged_optimizer_st_state_dict = merge_state_dicts(
                model_st_state_dicts, optimizer_st_state_dicts
            )
            assert_equal(merged_model_state_dicts, merged_model_st_state_dicts)
            assert_equal(merged_optimizer_state_dict, merged_optimizer_st_state_dict)

    torch.distributed.barrier()
    return results


def _gpu_worker(module_type, use_zero, pas, plan_ngpus, runtime_ngpus, per_resume_update_count, resume_count, check_module=None):
    init_distributed()
    compiled_results = []
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'cube_test_ckpt_{PYTEST_RUN_ID}') as tempdir:
        for i in range(resume_count):
            start = i * per_resume_update_count
            end = (i + 1) * per_resume_update_count
            compiled_module = _create_cube_module(pas,
                ComputeConfig(plan_ngpus, runtime_ngpus, use_zero=use_zero),
                tempdir,
                module_type,
            )
            compiled_inference_module = _create_cube_module(pas,
                ComputeConfig(plan_ngpus, runtime_ngpus, use_zero=use_zero, inference_only=True),
                tempdir,
                module_type,
            )
            if check_module:
                check_module(compiled_module)
            compiled_results.extend(_train(
                compiled_module,
                runtime_ngpus // plan_ngpus,
                torch.distributed.get_rank() // plan_ngpus,
                start, end, tempdir,
                inference_module=compiled_inference_module
            ))
        return compiled_results


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('module_type', ['sub', 'whole', 'start', 'end', 'small', 'pipeline'])
@pytest.mark.parametrize('use_zero', [True, False])
def test_checkpoint(module_type, use_zero):
    plan_ngpus = 2
    runtime_ngpus = 4
    cube_results = launch_torchrun(4, _gpu_worker, module_type, use_zero, 'tp', plan_ngpus, runtime_ngpus, 32, 1)
    rcube_results = launch_torchrun(4, _gpu_worker, module_type, use_zero, 'tp', plan_ngpus, runtime_ngpus, 16, 2)

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


def assert_intra_reducer(module: ParallelModule):
    assert module.compute_config.plan_ngpus == module.compute_config.runtime_ngpus
    assert len(module.reducers) > 0
    # so we have both parameters in reducers and not in reducers
    # (assume one reducer gives one bucket, which is true in general.)
    assert len(module.parameters_for_optimizer()) > len(module.reducers)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason='lack of gpu devices')
@pytest.mark.parametrize('module_type', ['whole'])
@pytest.mark.parametrize('use_zero', [True, False])
def test_checkpoint_intra_reducer(module_type, use_zero):
    """
    Test when:
    Some of the parameters will be added to reducers,
    but some of the parameters are not.
    """
    plan_ngpus = 2
    runtime_ngpus = 2
    cube_results = launch_torchrun(2, _gpu_worker, module_type, use_zero, 'tp', plan_ngpus, runtime_ngpus, 32, 1, assert_intra_reducer)
    rcube_results = launch_torchrun(2, _gpu_worker, module_type, use_zero, 'tp', plan_ngpus, runtime_ngpus, 16, 2, assert_intra_reducer)
    results0 = cube_results[0]
    rresults0 = rcube_results[0]

    # pred, loss
    for r0, r1 in [(results0, rresults0)]:
        # have the same input
        assert len(r0) == len(r1)  # iteration count
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.pred, b.pred)  # pred
            assert torch.equal(a.loss, b.loss)  # loss
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm

    # grad, weights
    for r0, r1 in [(results0, rresults0)]:
        # in the same shard, grads and weights are the same
        assert len(r0) == len(r1)
        for i in range(len(r0)):
            a, b = r0[i], r1[i]
            assert torch.equal(a.gnorm, b.gnorm)  # gnorm
            for k in a.grads.keys(): # grad
                assert torch.equal(a.grads[k], b.grads[k])
            for k in a.weights.keys():  # weights
                assert torch.equal(a.weights[k], b.weights[k])


def _gpu_merge_worker():
    init_distributed()
    with clear_dir_on_rank0(Path(tempfile.gettempdir()) / f'cube_test_ckpt_merge_{PYTEST_RUN_ID}') as tempdir:
        compiled_module = _create_cube_module('data',
            ComputeConfig(2, 4, use_zero=True),
            tempdir,
            'whole',
        )
        _train(
            compiled_module,
            1,
            0,
            0,
            8,
            tempdir,
            check_merge_log=True
        )


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_checkpoint_merge():
    launch_torchrun(4, _gpu_merge_worker)


def _gather_full_model_state_dict_worker(tmp_path, use_zero):
    from .test_end2end import MLP, dummy_data
    from nnscaler.parallel import gather_full_model_state_dict, merge_state_dicts
    init_distributed()

    model = MLP()
    model = parallelize(
        model,
        {'data': dummy_data()},
        pas_policy='tp',
        compute_config= ComputeConfig(
            2, 4,
            use_end2end=True,
            use_zero=use_zero,
        ),
        gen_savedir=tmp_path,
    )
    model.cuda()
    rank = torch.distributed.get_rank()
    torch.save(model.state_dict(), tmp_path / f'{rank}.pt')
    torch.distributed.barrier()
    merged_state_dict = merge_state_dicts(
        [torch.load(tmp_path / f'{i}.pt', weights_only=False) for i in range(torch.distributed.get_world_size())]
    )
    full_state_dict = gather_full_model_state_dict(model)
    assert_equal(merged_state_dict, full_state_dict)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('use_zero', [0, 1, 3])
def test_gather_full_model_state_dict(tmp_path, use_zero):
    launch_torchrun(4, _gather_full_model_state_dict_worker, tmp_path, use_zero)


def _perf_gather_full_model_state_dict_worker(tmp_path, use_zero, warmup_iters, bench_iters):
    import time
    from .test_end2end import MLP, dummy_data
    from nnscaler.parallel import (
        gather_full_model_state_dict, merge_state_dicts,
        deduped_state_dict,
    )
    from nnscaler.utils import gather_mixed_data, broadcast_mixed_data
    from nnscaler.runtime.module import ParallelModule
    from nnscaler.runtime.device import DeviceGroup
    init_distributed()

    model = MLP()
    model = parallelize(
        model,
        {'data': dummy_data()},
        pas_policy='tp',
        compute_config=ComputeConfig(
            2, 4,
            use_end2end=True,
            use_zero=use_zero,
        ),
        gen_savedir=tmp_path,
    )
    model.cuda()

    rank = torch.distributed.get_rank()

    # Resolve groups (same logic as gather_full_model_state_dict)
    parallel_modules = [m for m in model.modules() if isinstance(m, ParallelModule)]
    compute_config = parallel_modules[0].compute_config
    num_involved_ranks = compute_config.module_dedup_group_size
    involved_group = DeviceGroup().get_group(list(range(num_involved_ranks)))

    # --- warmup ---
    for _ in range(warmup_iters):
        gather_full_model_state_dict(model)
        torch.distributed.barrier()

    torch.cuda.synchronize()
    torch.distributed.barrier()

    # --- per-phase benchmark ---
    phase_times = {
        'deduped_state_dict': [],
        'gather_mixed_data': [],
        'merge_state_dicts': [],
        'broadcast_mixed_data': [],
        'barrier_final': [],
        'total': [],
    }
    full_state_dict = None

    for _ in range(bench_iters):
        torch.cuda.synchronize()
        torch.distributed.barrier()

        t_total_start = time.perf_counter()

        # Phase 1: deduped_state_dict
        t0 = time.perf_counter()
        if rank < num_involved_ranks:
            local_state_dict, _ = deduped_state_dict(model, optimizer=None)
        t1 = time.perf_counter()
        phase_times['deduped_state_dict'].append(t1 - t0)

        # Phase 2: gather_mixed_data
        t0 = time.perf_counter()
        if rank < num_involved_ranks:
            state_dicts = gather_mixed_data(local_state_dict, src_rank=0, group=involved_group, device='cpu')
        t1 = time.perf_counter()
        phase_times['gather_mixed_data'].append(t1 - t0)

        # Phase 3: merge_state_dicts (rank 0 only)
        t0 = time.perf_counter()
        if rank < num_involved_ranks and rank == 0:
            merge_state_dict = merge_state_dicts(state_dicts)
        else:
            merge_state_dict = None
        t1 = time.perf_counter()
        phase_times['merge_state_dicts'].append(t1 - t0)

        # Phase 4: broadcast_mixed_data
        t0 = time.perf_counter()
        full_state_dict = broadcast_mixed_data(merge_state_dict, src_rank=0, device='cpu')
        t1 = time.perf_counter()
        phase_times['broadcast_mixed_data'].append(t1 - t0)

        # Phase 5: final barrier
        t0 = time.perf_counter()
        torch.distributed.barrier()
        t1 = time.perf_counter()
        phase_times['barrier_final'].append(t1 - t0)

        t_total_end = time.perf_counter()
        phase_times['total'].append(t_total_end - t_total_start)

    # --- compute full_state_dict size ---
    full_state_dict_bytes = 0
    full_state_dict_num_keys = 0
    # merge_state_dicts returns (model_dict, optimizer_dict), so full_state_dict is a tuple
    model_state_dict_result = full_state_dict[0] if isinstance(full_state_dict, tuple) else full_state_dict
    if model_state_dict_result is not None:
        full_state_dict_num_keys = len(model_state_dict_result)
        for v in model_state_dict_result.values():
            if isinstance(v, torch.Tensor):
                full_state_dict_bytes += v.nelement() * v.element_size()

    # --- file-based merge_state_dicts baseline (rank 0 only) ---
    torch.save(model.state_dict(), tmp_path / f'{rank}.pt')
    torch.distributed.barrier()

    file_merge_times = []
    if rank == 0:
        all_state_dicts = [
            torch.load(tmp_path / f'{i}.pt', weights_only=False)
            for i in range(torch.distributed.get_world_size())
        ]
        for _ in range(warmup_iters):
            merge_state_dicts(all_state_dicts)
        for _ in range(bench_iters):
            t0 = time.perf_counter()
            merge_state_dicts(all_state_dicts)
            t1 = time.perf_counter()
            file_merge_times.append(t1 - t0)

    torch.distributed.barrier()

    # Build per-phase stats
    phase_stats = {}
    for phase, times in phase_times.items():
        phase_stats[phase] = {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'times': times,
        }

    return {
        'rank': rank,
        'use_zero': use_zero,
        'full_state_dict_bytes': full_state_dict_bytes,
        'full_state_dict_num_keys': full_state_dict_num_keys,
        'phase_stats': phase_stats,
        'file_merge_times': file_merge_times,
        'file_merge_mean': float(np.mean(file_merge_times)) if file_merge_times else None,
    }


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('use_zero', [0, 1, 3])
def test_perf_gather_full_model_state_dict(tmp_path, use_zero):
    """Performance test for gather_full_model_state_dict with per-phase breakdown."""
    warmup_iters = 2
    bench_iters = 5
    results = launch_torchrun(
        4, _perf_gather_full_model_state_dict_worker,
        tmp_path, use_zero, warmup_iters, bench_iters,
    )

    # Print full_state_dict size from rank 0
    rank0_res = results[0]
    size_bytes = rank0_res['full_state_dict_bytes']
    num_keys = rank0_res['full_state_dict_num_keys']
    if size_bytes >= 1 << 20:
        size_str = f"{size_bytes / (1 << 20):.2f} MB"
    elif size_bytes >= 1 << 10:
        size_str = f"{size_bytes / (1 << 10):.2f} KB"
    else:
        size_str = f"{size_bytes} B"
    print(f"\n[use_zero={use_zero}] full_state_dict: {num_keys} keys, {size_str} ({size_bytes} bytes)")

    # Print per-phase breakdown for each rank
    phase_order = ['deduped_state_dict', 'gather_mixed_data', 'merge_state_dicts', 'broadcast_mixed_data', 'barrier_final', 'total']
    for rank in sorted(results.keys()):
        res = results[rank]
        print(f"\n[use_zero={use_zero}] rank {rank} phase breakdown (mean over {bench_iters} iters):")
        total_mean = res['phase_stats']['total']['mean']
        for phase in phase_order:
            s = res['phase_stats'][phase]
            pct = (s['mean'] / total_mean * 100) if total_mean > 0 else 0
            print(f"  {phase:30s}  mean={s['mean']:.6f}s  std={s['std']:.6f}s  "
                  f"min={s['min']:.6f}s  max={s['max']:.6f}s  ({pct:5.1f}%)")
        if res['file_merge_mean'] is not None:
            print(f"  {'file-based merge (baseline)':30s}  mean={res['file_merge_mean']:.6f}s")

    # Sanity: all total times should be positive and finite
    for res in results.values():
        for t in res['phase_stats']['total']['times']:
            assert t > 0 and np.isfinite(t)


def _perf_gather_roundrobin_worker(tmp_path, use_zero, warmup_iters, bench_iters):
    """Worker that benchmarks gather_full_model_state_dict_roundrobin end-to-end."""
    import time
    from .test_end2end import MLP, dummy_data
    from nnscaler.parallel import (
        gather_full_model_state_dict,
        gather_full_model_state_dict_roundrobin,
    )
    init_distributed()

    model = MLP()
    model = parallelize(
        model,
        {'data': dummy_data()},
        pas_policy='tp',
        compute_config=ComputeConfig(
            2, 4,
            use_end2end=True,
            use_zero=use_zero,
        ),
        gen_savedir=tmp_path,
    )
    model.cuda()

    rank = torch.distributed.get_rank()

    # --- warmup both variants ---
    for _ in range(warmup_iters):
        gather_full_model_state_dict(model)
        torch.distributed.barrier()
    for _ in range(warmup_iters):
        gather_full_model_state_dict_roundrobin(model)
        torch.distributed.barrier()

    torch.cuda.synchronize()
    torch.distributed.barrier()

    # --- benchmark original ---
    orig_times = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        t0 = time.perf_counter()
        orig_result = gather_full_model_state_dict(model)
        torch.distributed.barrier()
        t1 = time.perf_counter()
        orig_times.append(t1 - t0)

    # --- benchmark roundrobin ---
    rr_times = []
    for _ in range(bench_iters):
        torch.cuda.synchronize()
        torch.distributed.barrier()
        t0 = time.perf_counter()
        rr_result = gather_full_model_state_dict_roundrobin(model)
        torch.distributed.barrier()
        t1 = time.perf_counter()
        rr_times.append(t1 - t0)

    # --- correctness check ---
    orig_model_dict = orig_result[0] if isinstance(orig_result, tuple) else orig_result
    rr_model_dict = rr_result[0] if isinstance(rr_result, tuple) else rr_result

    keys_match = set(orig_model_dict.keys()) == set(rr_model_dict.keys())
    values_match = True
    if keys_match:
        for k in orig_model_dict:
            if isinstance(orig_model_dict[k], torch.Tensor) and isinstance(rr_model_dict[k], torch.Tensor):
                if not torch.equal(orig_model_dict[k].cpu(), rr_model_dict[k].cpu()):
                    values_match = False
                    break

    return {
        'rank': rank,
        'orig_times': orig_times,
        'orig_mean': float(np.mean(orig_times)),
        'rr_times': rr_times,
        'rr_mean': float(np.mean(rr_times)),
        'keys_match': keys_match,
        'values_match': values_match,
        'num_keys': len(orig_model_dict),
    }


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('use_zero', [0, 1, 3])
def test_perf_gather_roundrobin(tmp_path, use_zero):
    """Benchmark gather_full_model_state_dict_roundrobin vs original."""
    warmup_iters = 2
    bench_iters = 5
    results = launch_torchrun(
        4, _perf_gather_roundrobin_worker,
        tmp_path, use_zero, warmup_iters, bench_iters,
    )

    print(f"\n[use_zero={use_zero}] gather_full_model_state_dict: original vs round-robin")
    for rank in sorted(results.keys()):
        res = results[rank]
        speedup = res['orig_mean'] / res['rr_mean'] if res['rr_mean'] > 0 else float('inf')
        print(f"  rank {rank}: orig={res['orig_mean']:.6f}s  rr={res['rr_mean']:.6f}s  "
              f"speedup={speedup:.2f}x  keys_match={res['keys_match']}  values_match={res['values_match']}")

    # Assert correctness on all ranks
    for rank, res in results.items():
        assert res['keys_match'], f"Rank {rank}: key mismatch between original and round-robin"
        assert res['values_match'], f"Rank {rank}: value mismatch between original and round-robin"


# ---------------------------------------------------------------------------
# gather_full_model_state_dict_from_files tests
# ---------------------------------------------------------------------------

def _gather_from_files_worker(tmp_path, use_zero):
    """
    Worker: parallelise a small MLP, save per-rank deduped state dicts to
    files, then call gather_full_model_state_dict_from_files and verify the
    result matches a live gather_full_model_state_dict.
    """
    import os
    from .test_end2end import MLP, dummy_data
    from nnscaler.parallel import (
        gather_full_model_state_dict,
        deduped_state_dict,
    )
    init_distributed()

    model = MLP()
    model = parallelize(
        model,
        {'data': dummy_data()},
        pas_policy='tp',
        compute_config=ComputeConfig(
            2, 4,
            use_end2end=True,
            use_zero=use_zero,
        ),
        gen_savedir=tmp_path,
    )
    model.cuda()

    rank = torch.distributed.get_rank()

    # --- Reference: gather from live model ---
    ref_result = gather_full_model_state_dict(model)
    ref_dict = ref_result[0] if isinstance(ref_result, tuple) else ref_result

    # --- Save per-rank deduped state dict to files ---
    ckpt_dir = os.path.join(tmp_path, 'ckpt_files')
    os.makedirs(ckpt_dir, exist_ok=True)
    local_sd, _ = deduped_state_dict(model, optimizer=None)
    torch.save(local_sd, os.path.join(ckpt_dir, f'{rank}.ckpt.model'))
    torch.distributed.barrier()

    # --- Gather from files ---
    file_result = gather_full_model_state_dict_from_files(ckpt_dir)
    file_dict = file_result[0] if isinstance(file_result, tuple) else file_result

    # --- Correctness check ---
    keys_match = set(ref_dict.keys()) == set(file_dict.keys())
    values_match = True
    mismatched_keys = []
    if keys_match:
        for k in ref_dict:
            if isinstance(ref_dict[k], torch.Tensor) and isinstance(file_dict[k], torch.Tensor):
                if not torch.equal(ref_dict[k].cpu(), file_dict[k].cpu()):
                    values_match = False
                    mismatched_keys.append(k)
            elif not isinstance(ref_dict[k], torch.Tensor) and not isinstance(file_dict[k], torch.Tensor):
                pass  # skip non-tensor keys (CUBE_EXTRA_STATE etc.)
    else:
        ref_only = set(ref_dict.keys()) - set(file_dict.keys())
        file_only = set(file_dict.keys()) - set(ref_dict.keys())
        mismatched_keys = list(ref_only | file_only)

    return {
        'rank': rank,
        'keys_match': keys_match,
        'values_match': values_match,
        'mismatched_keys': mismatched_keys,
        'num_ref_keys': len(ref_dict),
        'num_file_keys': len(file_dict),
    }


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
@pytest.mark.parametrize('use_zero', [0, 1, 3])
def test_gather_from_files(tmp_path, use_zero):
    """Correctness test: gather_full_model_state_dict_from_files vs live gather."""
    results = launch_torchrun(4, _gather_from_files_worker, tmp_path, use_zero)
    for rank in sorted(results.keys()):
        res = results[rank]
        print(f"  rank {rank}: keys={res['num_ref_keys']}/{res['num_file_keys']} "
              f"keys_match={res['keys_match']} values_match={res['values_match']}")
        if res['mismatched_keys']:
            print(f"    mismatched: {res['mismatched_keys'][:5]}")
    for rank, res in results.items():
        assert res['keys_match'], f"Rank {rank}: key mismatch (ref={res['num_ref_keys']}, file={res['num_file_keys']})"
        assert res['values_match'], f"Rank {rank}: value mismatch on keys {res['mismatched_keys'][:5]}"


# ---------------------------------------------------------------------------
# Real .ckpt.model file test
# ---------------------------------------------------------------------------

CKPT_MODEL_DIR = 'tmp/0000-129375'

def _gather_from_ckpt_model_worker(ckpt_dir):
    """
    Worker: each rank loads {rank}.ckpt.model from *ckpt_dir*,
    then gather_full_model_state_dict_from_files merges them.
    """
    import time
    init_distributed()

    rank = torch.distributed.get_rank()

    t0 = time.perf_counter()
    result = gather_full_model_state_dict_from_files(ckpt_dir)
    t1 = time.perf_counter()

    model_dict = result[0] if isinstance(result, tuple) else result

    # Compute summary stats
    num_keys = len(model_dict)
    num_tensors = sum(1 for v in model_dict.values() if isinstance(v, torch.Tensor))
    total_bytes = sum(
        v.nelement() * v.element_size()
        for v in model_dict.values()
        if isinstance(v, torch.Tensor)
    )

    return {
        'rank': rank,
        'elapsed': t1 - t0,
        'num_keys': num_keys,
        'num_tensors': num_tensors,
        'total_bytes': total_bytes,
    }


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_gather_from_ckpt_model_files():
    """
    Load per-rank .ckpt.model files and gather/merge into a full model state dict.

    Uses tmp/0000-129375/ which contains 512 .ckpt.model files
    (plan_ngpus=8, runtime_ngpus=512).
    With 4 GPUs, rank 0 loads the extra files (ranks 4-7) locally.
    """
    import os
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', CKPT_MODEL_DIR)
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        pytest.skip(f"Checkpoint directory not found: {ckpt_dir}")

    # Check that at least file 0 exists
    if not os.path.exists(os.path.join(ckpt_dir, '0.ckpt.model')):
        pytest.skip(f"No 0.ckpt.model in {ckpt_dir}")

    nproc = min(torch.cuda.device_count(), 4)
    results = launch_torchrun(nproc, _gather_from_ckpt_model_worker, ckpt_dir)

    def _fmt_bytes(b):
        for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
            if b < 1024:
                return f'{b:.2f} {unit}'
            b /= 1024
        return f'{b:.2f} PB'

    print(f"\n[test_gather_from_ckpt_model_files] nproc={nproc}")
    for rank in sorted(results.keys()):
        res = results[rank]
        print(f"  rank {rank}: elapsed={res['elapsed']:.2f}s  "
              f"keys={res['num_keys']}  tensors={res['num_tensors']}  "
              f"size={_fmt_bytes(res['total_bytes'])}")

    # Basic sanity: all ranks should get the same number of keys
    all_keys = [results[r]['num_keys'] for r in results]
    assert len(set(all_keys)) == 1, f"Key count mismatch across ranks: {all_keys}"
    assert all_keys[0] > 0, "Merged state dict is empty"


# ---------------------------------------------------------------------------
# Profile gather_full_model_state_dict_from_files on real .ckpt.model files
# ---------------------------------------------------------------------------

def _profile_gather_from_files_worker(ckpt_dir, suffix='.ckpt.model'):
    """
    Worker: profile each phase of gather_full_model_state_dict_from_files
    separately so we can identify bottlenecks.

    Phases:
      1. load_own_file   – torch.load of rank's own .ckpt.model
      2. gather           – gather_mixed_data (send local dicts to rank 0)
      3. load_extra_files – rank 0 loads files for ranks [world_size..num_involved)
      4. merge            – merge_state_dicts on rank 0
      5. broadcast        – broadcast_mixed_data from rank 0
      6. barrier          – final barrier
    """
    import time
    from pathlib import Path
    from nnscaler.parallel import (
        merge_state_dicts,
        _sanitize_extra_state_in_state_dict,
        ComputeConfig,
    )
    from nnscaler.utils import gather_mixed_data, broadcast_mixed_data
    from nnscaler.runtime.module import ParallelModule, ExtraState
    from nnscaler.runtime.device import DeviceGroup

    init_distributed()

    ckpt_dir = Path(ckpt_dir)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    def _load_ckpt(path):
        sd = torch.load(path, map_location='cpu', weights_only=False)
        return _sanitize_extra_state_in_state_dict(sd)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    # ------ Phase 1: load own file ------
    t0 = time.perf_counter()
    ckpt_file = ckpt_dir / f'{rank}{suffix}'
    local_state_dict = _load_ckpt(ckpt_file)
    t_load_own = time.perf_counter() - t0

    # Parse metadata
    extra_state_key = None
    for k in local_state_dict:
        if k.split('.')[-1] == ParallelModule.EXTRA_STATE_KEY:
            extra_state_key = k
            break
    extra_state = ExtraState(**local_state_dict[extra_state_key])
    num_involved_ranks = extra_state.compute_config.module_dedup_group_size

    local_bytes = sum(
        v.nelement() * v.element_size()
        for v in local_state_dict.values()
        if isinstance(v, torch.Tensor)
    )

    torch.cuda.synchronize()
    torch.distributed.barrier()

    # ------ Phase 2: gather_mixed_data ------
    t0 = time.perf_counter()
    state_dicts = gather_mixed_data(
        local_state_dict, src_rank=0, group=None, device='cpu',
    )
    torch.cuda.synchronize()
    t_gather = time.perf_counter() - t0

    # ------ Phase 3: load extra files (rank 0 only) ------
    t0 = time.perf_counter()
    if rank == 0 and world_size < num_involved_ranks:
        for i in range(world_size, num_involved_ranks):
            state_dicts.append(_load_ckpt(ckpt_dir / f'{i}{suffix}'))
    t_load_extra = time.perf_counter() - t0

    torch.distributed.barrier()

    # ------ Phase 4: merge_state_dicts (rank 0 only) ------
    t0 = time.perf_counter()
    if rank == 0:
        merge_result = merge_state_dicts(state_dicts)
    else:
        merge_result = None
    t_merge = time.perf_counter() - t0

    torch.distributed.barrier()

    # ------ Phase 5: broadcast_mixed_data ------
    t0 = time.perf_counter()
    merge_result = broadcast_mixed_data(merge_result, src_rank=0, device='cpu')
    torch.cuda.synchronize()
    t_broadcast = time.perf_counter() - t0

    # ------ Phase 6: barrier ------
    t0 = time.perf_counter()
    torch.distributed.barrier()
    t_barrier = time.perf_counter() - t0

    # Summarize result
    model_dict = merge_result[0] if isinstance(merge_result, tuple) else merge_result
    merged_bytes = sum(
        v.nelement() * v.element_size()
        for v in model_dict.values()
        if isinstance(v, torch.Tensor)
    )

    total = t_load_own + t_gather + t_load_extra + t_merge + t_broadcast + t_barrier

    return {
        'rank': rank,
        'world_size': world_size,
        'num_involved_ranks': num_involved_ranks,
        'local_keys': len(local_state_dict),
        'local_bytes': local_bytes,
        'merged_keys': len(model_dict),
        'merged_bytes': merged_bytes,
        'phases': {
            'load_own_file': t_load_own,
            'gather': t_gather,
            'load_extra_files': t_load_extra,
            'merge': t_merge,
            'broadcast': t_broadcast,
            'barrier': t_barrier,
        },
        'total': total,
    }


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_profile_gather_from_ckpt_model_files():
    """Profile each phase of gather_full_model_state_dict_from_files on real data."""
    import os
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', CKPT_MODEL_DIR)
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        pytest.skip(f"Checkpoint directory not found: {ckpt_dir}")
    if not os.path.exists(os.path.join(ckpt_dir, '0.ckpt.model')):
        pytest.skip(f"No 0.ckpt.model in {ckpt_dir}")

    nproc = min(torch.cuda.device_count(), 4)
    results = launch_torchrun(nproc, _profile_gather_from_files_worker, ckpt_dir)

    def _fmt(b):
        for u in ('B', 'KB', 'MB', 'GB', 'TB'):
            if b < 1024:
                return f'{b:.2f} {u}'
            b /= 1024
        return f'{b:.2f} PB'

    print(f"\n{'='*80}")
    print(f"Profile: gather_full_model_state_dict_from_files")
    print(f"  ckpt_dir = {ckpt_dir}")
    print(f"  nproc = {nproc}")
    print(f"{'='*80}")

    for rank in sorted(results.keys()):
        r = results[rank]
        print(f"\n--- Rank {rank} ---")
        print(f"  local: {r['local_keys']} keys, {_fmt(r['local_bytes'])}")
        print(f"  merged: {r['merged_keys']} keys, {_fmt(r['merged_bytes'])}")
        print(f"  num_involved_ranks: {r['num_involved_ranks']}")
        print(f"  {'Phase':<20s} {'Time (s)':>10s} {'% Total':>8s}")
        print(f"  {'-'*40}")
        for phase, t in r['phases'].items():
            pct = 100.0 * t / r['total'] if r['total'] > 0 else 0
            print(f"  {phase:<20s} {t:>10.2f}  {pct:>7.1f}%")
        print(f"  {'-'*40}")
        print(f"  {'TOTAL':<20s} {r['total']:>10.2f}")

    # Sanity checks
    all_keys = [results[r]['merged_keys'] for r in results]
    assert len(set(all_keys)) == 1, f"Key count mismatch: {all_keys}"
    assert all_keys[0] > 0


# ---------------------------------------------------------------------------
# Profile broadcast_mixed_data internals on real .ckpt.model files
# ---------------------------------------------------------------------------

def _profile_broadcast_worker(ckpt_dir, suffix='.ckpt.model'):
    """
    Worker: load & merge state dicts, then profile broadcast_mixed_data
    broken down into its internal phases:

      1. extract_tensors     – separate skeleton from tensors
      2. broadcast_object    – broadcast skeleton + meta tensors
      3. tensor_h2d          – .cuda() copy on src rank (cumulative)
      4. tensor_broadcast    – dist.broadcast per tensor (cumulative)
      5. tensor_d2h          – .to(device) on receivers (cumulative)
      6. cuda_sync           – final torch.cuda.synchronize()
      7. refill              – refill_tensors

    Also reports per-tensor size distribution and top-N slowest tensors.
    """
    import time
    from pathlib import Path
    from collections import defaultdict
    from nnscaler.parallel import merge_state_dicts, _sanitize_extra_state_in_state_dict
    from nnscaler.utils import (
        extract_tensors, refill_tensors, gather_mixed_data,
    )
    from nnscaler.runtime.module import ParallelModule, ExtraState
    from nnscaler.runtime.device import DeviceGroup

    init_distributed()

    ckpt_dir = Path(ckpt_dir)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    def _load_ckpt(path):
        sd = torch.load(path, map_location='cpu', weights_only=False)
        return _sanitize_extra_state_in_state_dict(sd)

    # --- Prepare data: load, gather, merge (same as real pipeline) ---
    local_sd = _load_ckpt(ckpt_dir / f'{rank}{suffix}')
    extra_key = next(k for k in local_sd if k.split('.')[-1] == ParallelModule.EXTRA_STATE_KEY)
    extra_state = ExtraState(**local_sd[extra_key])
    num_involved = extra_state.compute_config.module_dedup_group_size

    state_dicts = gather_mixed_data(local_sd, src_rank=0, device='cpu')
    if rank == 0 and world_size < num_involved:
        for i in range(world_size, num_involved):
            state_dicts.append(_load_ckpt(ckpt_dir / f'{i}{suffix}'))

    if rank == 0:
        merge_result = merge_state_dicts(state_dicts)
    else:
        merge_result = None

    torch.distributed.barrier()

    # --- Now profile broadcast_mixed_data internals ---
    device = torch.device('cpu')

    # Phase 1: extract_tensors
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if rank == 0:
        skeleton, tensors = extract_tensors(merge_result)
        meta_tensors = [t.to('meta') for t in tensors]
        sent = [(skeleton, meta_tensors)]
    else:
        skeleton, tensors, meta_tensors = None, None, None
        sent = [None]
    t_extract = time.perf_counter() - t0

    # Phase 2: broadcast_object_list (metadata)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    t0 = time.perf_counter()
    torch.distributed.broadcast_object_list(sent, src=0)
    torch.cuda.synchronize()
    t_bcast_obj = time.perf_counter() - t0

    skeleton, meta_tensors = sent[0]
    if rank != 0:
        tensors = [None] * len(meta_tensors)

    n_tensors = len(meta_tensors)

    # Collect per-tensor timings
    per_tensor_h2d = []      # src only
    per_tensor_bcast = []    # all ranks
    per_tensor_d2h = []      # receiver only
    per_tensor_sizes = []    # bytes

    torch.cuda.synchronize()
    torch.distributed.barrier()
    t_loop_start = time.perf_counter()

    for i in range(n_tensors):
        nbytes = meta_tensors[i].numel() * meta_tensors[i].element_size()
        per_tensor_sizes.append(nbytes)

        # H2D (src rank)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if rank != 0:
            tensor = torch.empty_like(meta_tensors[i], device='cuda')
        else:
            tensor = tensors[i].cuda()
        torch.cuda.synchronize()
        t_h2d = time.perf_counter() - t0
        per_tensor_h2d.append(t_h2d)

        # Broadcast
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        torch.distributed.broadcast(tensor, src=0)
        torch.cuda.synchronize()
        t_bc = time.perf_counter() - t0
        per_tensor_bcast.append(t_bc)

        # D2H / device placement
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if rank != 0:
            tensors[i] = tensor.to(device, non_blocking=True)
        else:
            if tensor.device == device:
                tensors[i] = tensor
            else:
                tensors[i] = tensors[i].to(device, non_blocking=True)
        # no sync yet – non_blocking
        per_tensor_d2h.append(time.perf_counter() - t0)

    # Phase 6: final cuda sync
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    t_cuda_sync = time.perf_counter() - t0

    t_loop_total = time.perf_counter() - t_loop_start

    # Phase 7: refill
    t0 = time.perf_counter()
    result = refill_tensors(skeleton, tensors)
    t_refill = time.perf_counter() - t0

    # --- Aggregate stats ---
    total_h2d = sum(per_tensor_h2d)
    total_bcast = sum(per_tensor_bcast)
    total_d2h = sum(per_tensor_d2h)
    total_bytes = sum(per_tensor_sizes)
    grand_total = t_extract + t_bcast_obj + t_loop_total + t_cuda_sync + t_refill

    # Size distribution buckets
    size_buckets = defaultdict(lambda: {'count': 0, 'bytes': 0, 'h2d': 0.0, 'bcast': 0.0, 'd2h': 0.0})
    bucket_names = ['<1KB', '1KB-1MB', '1MB-100MB', '100MB-1GB', '>1GB']
    bucket_bounds = [1024, 1024**2, 100*1024**2, 1024**3, float('inf')]
    for i in range(n_tensors):
        sz = per_tensor_sizes[i]
        for bname, bbound in zip(bucket_names, bucket_bounds):
            if sz < bbound:
                b = size_buckets[bname]
                b['count'] += 1
                b['bytes'] += sz
                b['h2d'] += per_tensor_h2d[i]
                b['bcast'] += per_tensor_bcast[i]
                b['d2h'] += per_tensor_d2h[i]
                break

    # Top-10 slowest by (h2d + bcast) time
    combined = [(per_tensor_h2d[i] + per_tensor_bcast[i], per_tensor_sizes[i],
                 per_tensor_h2d[i], per_tensor_bcast[i], per_tensor_d2h[i], i)
                for i in range(n_tensors)]
    combined.sort(reverse=True)
    top_slow = combined[:10]

    return {
        'rank': rank,
        'n_tensors': n_tensors,
        'total_bytes': total_bytes,
        'phases': {
            'extract_tensors': t_extract,
            'broadcast_object': t_bcast_obj,
            'loop_h2d': total_h2d,
            'loop_broadcast': total_bcast,
            'loop_d2h': total_d2h,
            'loop_total': t_loop_total,
            'cuda_sync': t_cuda_sync,
            'refill': t_refill,
        },
        'grand_total': grand_total,
        'size_buckets': dict(size_buckets),
        'top_slow': top_slow,
    }


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_profile_broadcast_mixed_data():
    """Profile broadcast_mixed_data internals on real checkpoint data."""
    import os
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', CKPT_MODEL_DIR)
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        pytest.skip(f"Checkpoint directory not found: {ckpt_dir}")
    if not os.path.exists(os.path.join(ckpt_dir, '0.ckpt.model')):
        pytest.skip(f"No 0.ckpt.model in {ckpt_dir}")

    nproc = min(torch.cuda.device_count(), 4)
    results = launch_torchrun(nproc, _profile_broadcast_worker, ckpt_dir)

    def _fmt(b):
        for u in ('B', 'KB', 'MB', 'GB', 'TB'):
            if b < 1024:
                return f'{b:.2f} {u}'
            b /= 1024
        return f'{b:.2f} PB'

    for rank_id in sorted(results.keys()):
        r = results[rank_id]
        print(f"\n{'='*80}")
        print(f"Rank {rank_id}: broadcast_mixed_data profile")
        print(f"  {r['n_tensors']} tensors, {_fmt(r['total_bytes'])} total")
        print(f"{'='*80}")
        print(f"  {'Phase':<22s} {'Time (s)':>10s} {'%':>7s}")
        print(f"  {'-'*42}")
        for phase, t in r['phases'].items():
            pct = 100.0 * t / r['grand_total'] if r['grand_total'] > 0 else 0
            print(f"  {phase:<22s} {t:>10.3f}  {pct:>6.1f}%")
        print(f"  {'-'*42}")
        print(f"  {'GRAND TOTAL':<22s} {r['grand_total']:>10.3f}")

        # Size distribution
        print(f"\n  Size distribution:")
        print(f"  {'Bucket':<14s} {'Count':>6s} {'Bytes':>12s} {'H2D (s)':>10s} {'Bcast (s)':>10s} {'D2H (s)':>10s}")
        for bname in ['<1KB', '1KB-1MB', '1MB-100MB', '100MB-1GB', '>1GB']:
            if bname in r['size_buckets']:
                b = r['size_buckets'][bname]
                print(f"  {bname:<14s} {b['count']:>6d} {_fmt(b['bytes']):>12s} "
                      f"{b['h2d']:>10.3f} {b['bcast']:>10.3f} {b['d2h']:>10.3f}")

        # Top-10 slowest tensors
        print(f"\n  Top-10 slowest tensors (by h2d + bcast):")
        print(f"  {'Idx':>5s} {'Size':>12s} {'H2D (s)':>10s} {'Bcast (s)':>10s} {'D2H (s)':>10s} {'Total (s)':>10s}")
        for total_t, sz, h2d, bcast, d2h, idx in r['top_slow']:
            print(f"  {idx:>5d} {_fmt(sz):>12s} {h2d:>10.4f} {bcast:>10.4f} {d2h:>10.4f} {total_t:>10.4f}")


# ---------------------------------------------------------------------------
# Compare broadcast_mixed_data (NCCL) vs broadcast_mixed_data_gloo (gloo)
# ---------------------------------------------------------------------------

def _bench_broadcast_gloo_worker(ckpt_dir, suffix='.ckpt.model'):
    """
    Worker: prepare real merged state_dict, then run both
    broadcast_mixed_data (NCCL) and broadcast_mixed_data_gloo (gloo)
    and report timings.
    """
    import time
    from pathlib import Path
    from nnscaler.parallel import merge_state_dicts, _sanitize_extra_state_in_state_dict
    from nnscaler.utils import (
        broadcast_mixed_data, broadcast_mixed_data_gloo,
        gather_mixed_data,
    )
    from nnscaler.runtime.module import ParallelModule, ExtraState

    init_distributed()

    ckpt_dir = Path(ckpt_dir)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    def _load_ckpt(path):
        sd = torch.load(path, map_location='cpu', weights_only=False)
        return _sanitize_extra_state_in_state_dict(sd)

    # --- Prepare data: load, gather, merge ---
    local_sd = _load_ckpt(ckpt_dir / f'{rank}{suffix}')
    extra_key = next(k for k in local_sd if k.split('.')[-1] == ParallelModule.EXTRA_STATE_KEY)
    extra_state = ExtraState(**local_sd[extra_key])
    num_involved = extra_state.compute_config.module_dedup_group_size

    state_dicts = gather_mixed_data(local_sd, src_rank=0, device='cpu')
    if rank == 0 and world_size < num_involved:
        for i in range(world_size, num_involved):
            state_dicts.append(_load_ckpt(ckpt_dir / f'{i}{suffix}'))

    if rank == 0:
        merge_result = merge_state_dicts(state_dicts)
    else:
        merge_result = None

    torch.distributed.barrier()

    # --- Benchmark 1: original NCCL broadcast ---
    torch.cuda.synchronize()
    torch.distributed.barrier()
    t0 = time.perf_counter()
    result_nccl = broadcast_mixed_data(merge_result, src_rank=0, device='cpu')
    torch.cuda.synchronize()
    t_nccl = time.perf_counter() - t0

    # Verify and get stats
    model_nccl = result_nccl[0] if isinstance(result_nccl, tuple) else result_nccl
    nccl_keys = len(model_nccl)
    nccl_bytes = sum(v.nelement() * v.element_size() for v in model_nccl.values() if isinstance(v, torch.Tensor))
    del result_nccl, model_nccl

    torch.distributed.barrier()

    # --- Benchmark 2: gloo broadcast ---
    torch.cuda.synchronize()
    torch.distributed.barrier()
    t0 = time.perf_counter()
    result_gloo = broadcast_mixed_data_gloo(merge_result, src_rank=0, device='cpu')
    t_gloo = time.perf_counter() - t0

    model_gloo = result_gloo[0] if isinstance(result_gloo, tuple) else result_gloo
    gloo_keys = len(model_gloo)
    gloo_bytes = sum(v.nelement() * v.element_size() for v in model_gloo.values() if isinstance(v, torch.Tensor))
    del result_gloo, model_gloo

    speedup = t_nccl / t_gloo if t_gloo > 0 else float('inf')

    return {
        'rank': rank,
        't_nccl': t_nccl,
        't_gloo': t_gloo,
        'speedup': speedup,
        'nccl_keys': nccl_keys,
        'nccl_bytes': nccl_bytes,
        'gloo_keys': gloo_keys,
        'gloo_bytes': gloo_bytes,
    }


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_bench_broadcast_gloo():
    """Benchmark broadcast_mixed_data (NCCL) vs broadcast_mixed_data_gloo."""
    import os
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', CKPT_MODEL_DIR)
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        pytest.skip(f"Checkpoint directory not found: {ckpt_dir}")
    if not os.path.exists(os.path.join(ckpt_dir, '0.ckpt.model')):
        pytest.skip(f"No 0.ckpt.model in {ckpt_dir}")

    nproc = min(torch.cuda.device_count(), 4)
    results = launch_torchrun(nproc, _bench_broadcast_gloo_worker, ckpt_dir)

    def _fmt(b):
        for u in ('B', 'KB', 'MB', 'GB', 'TB'):
            if b < 1024:
                return f'{b:.2f} {u}'
            b /= 1024
        return f'{b:.2f} PB'

    print(f"\n{'='*70}")
    print(f"Benchmark: broadcast_mixed_data (NCCL) vs gloo")
    print(f"{'='*70}")
    print(f"  {'Rank':>4s}  {'NCCL (s)':>10s}  {'Gloo (s)':>10s}  {'Speedup':>8s}  {'Keys':>6s}  {'Size':>12s}")
    print(f"  {'-'*56}")
    for rank_id in sorted(results.keys()):
        r = results[rank_id]
        print(f"  {r['rank']:>4d}  {r['t_nccl']:>10.2f}  {r['t_gloo']:>10.2f}  "
              f"{r['speedup']:>7.2f}x  {r['gloo_keys']:>6d}  {_fmt(r['gloo_bytes']):>12s}")

    # Correctness: both should produce same number of keys and bytes
    for rank_id in results:
        r = results[rank_id]
        assert r['nccl_keys'] == r['gloo_keys'], f"Rank {rank_id}: key mismatch {r['nccl_keys']} vs {r['gloo_keys']}"
        assert r['nccl_bytes'] == r['gloo_bytes'], f"Rank {rank_id}: byte mismatch"


# ---------------------------------------------------------------------------
# Compare broadcast_mixed_data (NCCL per-tensor) vs coalesced NCCL
# ---------------------------------------------------------------------------

def _bench_broadcast_coalesced_worker(ckpt_dir, suffix='.ckpt.model'):
    """
    Worker: prepare real merged state_dict, then benchmark
    broadcast_mixed_data (per-tensor NCCL) vs broadcast_mixed_data_coalesced.
    """
    import time
    from pathlib import Path
    from nnscaler.parallel import merge_state_dicts, _sanitize_extra_state_in_state_dict
    from nnscaler.utils import (
        broadcast_mixed_data, broadcast_mixed_data_coalesced,
        gather_mixed_data,
    )
    from nnscaler.runtime.module import ParallelModule, ExtraState

    init_distributed()

    ckpt_dir = Path(ckpt_dir)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    def _load_ckpt(path):
        sd = torch.load(path, map_location='cpu', weights_only=False)
        return _sanitize_extra_state_in_state_dict(sd)

    # --- Prepare: load, gather, merge ---
    local_sd = _load_ckpt(ckpt_dir / f'{rank}{suffix}')
    extra_key = next(k for k in local_sd if k.split('.')[-1] == ParallelModule.EXTRA_STATE_KEY)
    extra_state = ExtraState(**local_sd[extra_key])
    num_involved = extra_state.compute_config.module_dedup_group_size

    state_dicts = gather_mixed_data(local_sd, src_rank=0, device='cpu')
    if rank == 0 and world_size < num_involved:
        for i in range(world_size, num_involved):
            state_dicts.append(_load_ckpt(ckpt_dir / f'{i}{suffix}'))

    if rank == 0:
        merge_result = merge_state_dicts(state_dicts)
    else:
        merge_result = None

    torch.distributed.barrier()

    # --- Benchmark 1: original per-tensor NCCL broadcast ---
    torch.cuda.synchronize()
    torch.distributed.barrier()
    t0 = time.perf_counter()
    result_orig = broadcast_mixed_data(merge_result, src_rank=0, device='cpu')
    torch.cuda.synchronize()
    t_orig = time.perf_counter() - t0

    model_orig = result_orig[0] if isinstance(result_orig, tuple) else result_orig
    orig_keys = len(model_orig)
    orig_bytes = sum(v.nelement() * v.element_size() for v in model_orig.values() if isinstance(v, torch.Tensor))
    del result_orig, model_orig

    torch.distributed.barrier()

    # --- Benchmark 2: coalesced NCCL broadcast ---
    torch.cuda.synchronize()
    torch.distributed.barrier()
    t0 = time.perf_counter()
    result_coal = broadcast_mixed_data_coalesced(merge_result, src_rank=0, device='cpu')
    torch.cuda.synchronize()
    t_coal = time.perf_counter() - t0

    model_coal = result_coal[0] if isinstance(result_coal, tuple) else result_coal
    coal_keys = len(model_coal)
    coal_bytes = sum(v.nelement() * v.element_size() for v in model_coal.values() if isinstance(v, torch.Tensor))
    del result_coal, model_coal

    speedup = t_orig / t_coal if t_coal > 0 else float('inf')

    return {
        'rank': rank,
        't_orig': t_orig,
        't_coalesced': t_coal,
        'speedup': speedup,
        'orig_keys': orig_keys,
        'orig_bytes': orig_bytes,
        'coal_keys': coal_keys,
        'coal_bytes': coal_bytes,
    }


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4, reason='lack of gpu devices')
def test_bench_broadcast_coalesced():
    """Benchmark per-tensor NCCL vs coalesced NCCL broadcast."""
    import os
    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', '..', CKPT_MODEL_DIR)
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        pytest.skip(f"Checkpoint directory not found: {ckpt_dir}")
    if not os.path.exists(os.path.join(ckpt_dir, '0.ckpt.model')):
        pytest.skip(f"No 0.ckpt.model in {ckpt_dir}")

    nproc = min(torch.cuda.device_count(), 4)
    results = launch_torchrun(nproc, _bench_broadcast_coalesced_worker, ckpt_dir)

    def _fmt(b):
        for u in ('B', 'KB', 'MB', 'GB', 'TB'):
            if b < 1024:
                return f'{b:.2f} {u}'
            b /= 1024
        return f'{b:.2f} PB'

    print(f"\n{'='*70}")
    print(f"Benchmark: broadcast_mixed_data (per-tensor) vs coalesced NCCL")
    print(f"{'='*70}")
    print(f"  {'Rank':>4s}  {'Original (s)':>12s}  {'Coalesced (s)':>13s}  {'Speedup':>8s}  {'Size':>12s}")
    print(f"  {'-'*56}")
    for rank_id in sorted(results.keys()):
        r = results[rank_id]
        print(f"  {r['rank']:>4d}  {r['t_orig']:>12.2f}  {r['t_coalesced']:>13.2f}  "
              f"{r['speedup']:>7.2f}x  {_fmt(r['coal_bytes']):>12s}")

    # Correctness
    for rank_id in results:
        r = results[rank_id]
        assert r['orig_keys'] == r['coal_keys'], f"Rank {rank_id}: key mismatch"
        assert r['orig_bytes'] == r['coal_bytes'], f"Rank {rank_id}: byte mismatch"