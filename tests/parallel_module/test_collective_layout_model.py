# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import copy
from dataclasses import replace
import json
import tempfile
from pathlib import Path

import pytest
import torch
import nnscaler
from nnscaler import ComputeConfig, parallelize, build_optimizer
from nnscaler.ir.operator import IRDataOperation
from nnscaler.policies import get_pas_ops, OpPlan
from tests.launch_torchrun import launch_torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0, replace_all_device_with


class PipelineMLP(torch.nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(8, 8, bias=False) for _ in range(num_layers)
        ])

    def forward(self, sample):
        x = sample['x']
        for layer in self.layers:
            x = layer(x)
        return x.square().sum()


def pipeline_policy(graph, cfg):
    # fn assigns devices. The user policy never specifies a rank permutation.
    for op in get_pas_ops(graph):
        stage = -1
        if torch.nn.Linear in op.module_class_chain:
            stage = int(op.get_module_fqn(torch.nn.Linear).split('.')[-1])
        yield OpPlan(op, stage_id=stage, partition=None)


def pipeline_config():
    return ComputeConfig(8, 8, use_end2end=True, constant_folding=False,
                         pas_config={'pipeline_size': 2, 'pipeline_nmicros': 4,
                                     'pipeline_scheduler': '1f1b_interleaved'})


def mesh_policy(graph, cfg):
    # A 2x2 tensor-parallel first layer followed by a replicated second layer.
    # Assign tiles to increasing ranks; RVD chooses the adapter ranks.
    for op in [*get_pas_ops(graph), *graph.select(ntype=IRDataOperation)]:
        if (torch.nn.Linear in op.module_class_chain
                and op.get_module_fqn(torch.nn.Linear) == 'layers.0'):
            input_shards = graph.partition(op, op.algorithm('dim'), idx=1, dim=1, num=2)
            nodes = [piece for shard in input_shards for piece in
                     graph.partition(shard, shard.algorithm('dim'), idx=1, dim=0, num=2)]
        else:
            nodes = graph.replicate(op, cfg.plan_ngpus)
        for rank, node in enumerate(nodes):
            graph.assign(node, rank)
    return graph


def model_case(ngpus):
    if ngpus == 4:
        return (PipelineMLP(2), mesh_policy,
                ComputeConfig(4, 4, use_end2end=True, constant_folding=False))
    return PipelineMLP(), pipeline_policy, pipeline_config()


@pytest.mark.parametrize('ngpus', [4, 8])
@replace_all_device_with('cpu')
def test_fn_pipeline_generates_permuted_collectives(tmp_path, ngpus):
    source, policy, config = model_case(ngpus)
    parallelize(source, {'sample': {'x': torch.ones(8, 8)}},
                policy, config, gen_savedir=tmp_path,
                load_module=False, reuse='override')
    collectives = set()
    for path in tmp_path.rglob('gencode*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('all_gather', 'chunk'):
                    collectives.update((node.func.attr, tuple(ast.literal_eval(k.value)))
                                       for k in node.keywords if k.arg == 'ranks')
    if ngpus == 4:
        # The generated rank-0 adapter includes (tensor IDs may vary):
        # linear_21 = nnscaler.runtime.adapter.all_gather(linear_82, dim=1, ranks=(0, 2, 1, 3))
        assert ('all_gather', (0, 2, 1, 3)) in collectives
    else:
        # Either stage group may get the permuted placement among equal-cost plans.
        for kind in ('chunk', 'all_gather'):
            assert any(name == kind and ranks != tuple(sorted(ranks))
                       for name, ranks in collectives)


@replace_all_device_with('cpu')
def test_mesh_dp_generates_shifted_collective_layouts(tmp_path):
    source, policy, config = model_case(4)
    parallelize(source, {'sample': {'x': torch.ones(8, 8)}}, policy,
                replace(config, runtime_ngpus=8), gen_savedir=tmp_path,
                load_module=False, reuse='override')
    layouts = set()
    for path in tmp_path.rglob('gencode*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'all_gather':
                layouts.update(tuple(ast.literal_eval(k.value)) for k in node.keywords if k.arg == 'ranks')
    assert {(0, 2, 1, 3), (4, 6, 5, 7)} <= layouts


def pipeline_worker(ngpus=8, runtime_ngpus=None):
    nnscaler.init()
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(7)
    source, policy, config = model_case(ngpus)
    runtime_ngpus = runtime_ngpus or ngpus
    config = replace(config, runtime_ngpus=runtime_ngpus)
    replicas = runtime_ngpus // ngpus
    replica = torch.distributed.get_rank() // ngpus
    errors = {'loss_max_abs': 0., 'gradient_max_abs': 0., 'weight_max_abs': 0.}
    reference = copy.deepcopy(source).cuda()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    reference_params = dict(reference.named_parameters())
    directory = Path(tempfile.gettempdir()) / f'fn_collective_mlp_{ngpus}_{runtime_ngpus}_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        model = parallelize(source, {'sample': {'x': torch.ones(8, 8)}},
                            policy, config,
                            gen_savedir=tempdir, reuse='override').cuda()
        if ngpus == 8 and nnscaler.flags.CompileFlag.async_comm:
            # Prove the ASYNC_COMM acceptance run actually generated async
            # pipeline communication instead of silently exercising a sync path.
            assert any('async_op=True' in path.read_text() for path in tempdir.rglob('gencode*.py'))
        optimizer = build_optimizer(model, torch.optim.SGD, lr=0.001)
        for step in range(2):
            replica_samples = [[
                {'x': torch.arange(64, device='cuda').reshape(8, 8).float() / 64 + micro + step + dp / 4}
                for micro in range(4)] for dp in range(replicas)]
            reference_optimizer.zero_grad()
            replica_losses = [[reference(sample) for sample in samples] for samples in replica_samples]
            expected = replica_losses[replica]
            # nnScaler's default DP reducer sums different data replicas; loss
            # normalization is the caller's scale_fn responsibility.
            torch.stack([loss for losses in replica_losses for loss in losses]).sum().backward()
            actual = model.train_step(replica_samples[replica])
            torch.testing.assert_close(actual, expected)
            errors['loss_max_abs'] = max(errors['loss_max_abs'], max(
                (left - right).abs().max().item() for left, right in zip(actual, expected)))
            for name, meta in model.fullmap.items():
                actual_grad = getattr(model, name).grad
                expected_grad = reference_params[meta.orig_name].grad[meta.slicers]
                torch.testing.assert_close(actual_grad, expected_grad)
                errors['gradient_max_abs'] = max(errors['gradient_max_abs'],
                                                 (actual_grad - expected_grad).abs().max().item())
            optimizer.step()
            reference_optimizer.step()
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(getattr(model, name),
                                           reference_params[meta.orig_name][meta.slicers])
                errors['weight_max_abs'] = max(errors['weight_max_abs'],
                    (getattr(model, name) - reference_params[meta.orig_name][meta.slicers]).abs().max().item())
            optimizer.zero_grad()
    print('NNSCALER_TEST_METRICS ' + json.dumps({
        'case': 'collective_layout_model', 'plan_ngpus': ngpus, 'runtime_ngpus': runtime_ngpus,
        'rank': torch.distributed.get_rank(), 'async_comm': nnscaler.flags.CompileFlag.async_comm, **errors,
    }))
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
def test_fn_mesh_matches_eager():
    assert all(launch_torchrun(4, pipeline_worker, 4).values())


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason='requires 8 GPUs')
def test_fn_pipeline_matches_eager():
    assert all(launch_torchrun(8, pipeline_worker).values())


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason='requires 8 GPUs')
def test_fn_mesh_dp_scaling_matches_eager():
    assert all(launch_torchrun(8, pipeline_worker, 4, 8).values())
