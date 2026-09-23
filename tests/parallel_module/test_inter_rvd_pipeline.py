# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import ast
import copy
import tempfile
from pathlib import Path

import pytest
import torch
import nnscaler
from nnscaler import ComputeConfig, parallelize, build_optimizer
from nnscaler.policies import get_pas_ops, OpPlan
from tests.launch_torchrun import launch_torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0, replace_all_device_with


class InterStageMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(8, 8, bias=False) for _ in range(4)
        ])

    def forward(self, sample):
        x = sample['x']
        for layer in self.layers:
            x = layer(x)
        return x.square().sum()


def inter_stage_policy(graph, cfg):
    # fn assigns devices. The user policy never specifies a rank permutation.
    for op in get_pas_ops(graph):
        stage = -1
        if torch.nn.Linear in op.module_class_chain:
            stage = int(op.get_module_fqn(torch.nn.Linear).split('.')[-1])
        yield OpPlan(op, stage_id=stage, partition=None)


def inter_stage_config():
    return ComputeConfig(8, 8, use_end2end=True, constant_folding=False,
                         pas_config={'pipeline_size': 2, 'pipeline_nmicros': 4,
                                     'pipeline_scheduler': '1f1b_interleaved'})


@replace_all_device_with('cpu', force=True)
def test_fn_inter_stage_gencode_prefers_matching_peers(tmp_path):
    """Check the placement preference in generated communication for a real model."""
    parallelize(InterStageMLP(), {'sample': {'x': torch.ones(8, 8)}},
                inter_stage_policy, inter_stage_config(),
                gen_savedir=tmp_path, reuse='override', load_module=False)
    peers = set()
    files = list(tmp_path.rglob('gencode*.py'))
    assert len(files) == 8
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and ast.unparse(node.func) == 'nnscaler.runtime.adapter.move':
                kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords
                          if kw.arg in ('src', 'dst')}
                if kwargs['src'] < 4:
                    peers.add((kwargs['src'], kwargs['dst']))
    # This model permits matching relative ranks for transfers from 0..3 to
    # 4..7. Other boundaries may require a permutation to align shard layouts.
    assert peers == {(r, r + 4) for r in range(4)}


def inter_stage_worker():
    nnscaler.init()
    torch.manual_seed(7)
    source = InterStageMLP()
    reference = copy.deepcopy(source).cuda()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
    reference_params = dict(reference.named_parameters())
    directory = Path(tempfile.gettempdir()) / f'inter_stage_mlp_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        model = parallelize(source, {'sample': {'x': torch.ones(8, 8)}},
                            inter_stage_policy, inter_stage_config(),
                            gen_savedir=tempdir, reuse='override').cuda()
        optimizer = build_optimizer(model, torch.optim.SGD, lr=0.001)
        for step in range(2):
            samples = [{'x': torch.arange(64, device='cuda').reshape(8, 8).float() / 64 + micro + step}
                       for micro in range(4)]
            reference_optimizer.zero_grad()
            expected = [reference(sample) for sample in samples]
            torch.stack(expected).sum().backward()
            actual = model.train_step(samples)
            torch.testing.assert_close(actual, expected)
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(getattr(model, name).grad,
                                           reference_params[meta.orig_name].grad[meta.slicers])
            optimizer.step()
            reference_optimizer.step()
            for name, meta in model.fullmap.items():
                torch.testing.assert_close(getattr(model, name),
                                           reference_params[meta.orig_name][meta.slicers])
            optimizer.zero_grad()
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason='requires 8 GPUs')
def test_fn_inter_stage_layout_matches_eager():
    """Check two-step losses, gradients and updates; peer order is tested above."""
    assert all(launch_torchrun(8, inter_stage_worker).values())
