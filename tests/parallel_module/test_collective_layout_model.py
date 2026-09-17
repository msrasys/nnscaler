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
from nnscaler.ir.operator import IRDataOperation, IRFwOperation
from nnscaler.runtime.device import DeviceGroup
from tests.launch_torchrun import launch_torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0, replace_all_device_with


RANK_ORDER = (0, 2, 1, 3)


class ReorderedMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 8, bias=False)
        self.second = torch.nn.Linear(8, 8, bias=False)

    def forward(self, x):
        y = self.second(self.first(x))
        return (y * y).sum()


def reordered_policy(graph, cfg):
    layout = cfg.pas_config['layout']
    for node in list(graph.select(ntype=(IRDataOperation, IRFwOperation))):
        name = node.input(1).name if node.fn == torch.nn.functional.linear else None
        partition = (name == 'first.weight' and layout != 'split') or (name == 'second.weight' and layout != 'gather')
        if partition:
            dim = 1 if layout == 'split' else 0
            parts = graph.partition(node, node.algorithm('dim'), idx=1, dim=dim, num=4)
            for rank, part in zip(RANK_ORDER, parts):
                graph.assign(part, rank)
        else:
            for rank, part in enumerate(graph.replicate(node, 4)):
                graph.assign(part, rank)
    return graph


@replace_all_device_with('cpu')
@pytest.mark.parametrize('layout', ['gather', 'gather_reduce', 'split'])
def test_reordered_mlp_codegen(tmp_path, layout):
    parallelize(ReorderedMLP().double(), {'x': torch.ones(3, 4, dtype=torch.float64)},
                reordered_policy, ComputeConfig(4, 4, use_end2end=True, pas_config={'layout': layout}),
                gen_savedir=tmp_path, load_module=False, reuse='override')
    gathers = []
    for path in tmp_path.rglob('gencode*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('all_gather', 'allgather_split', 'allgather_reducescatter', 'split_allgather'):
                    gathers.extend(ast.literal_eval(k.value) for k in node.keywords if k.arg == 'ranks')
    assert gathers and all(tuple(ranks) == RANK_ORDER for ranks in gathers)


def reordered_worker(layout):
    nnscaler.init()
    assert DeviceGroup().get_group(RANK_ORDER) is None  # reuse WORLD
    torch.manual_seed(17)
    source = ReorderedMLP().double()
    reference = copy.deepcopy(source).cuda()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    reference_params = dict(reference.named_parameters())
    directory = Path(tempfile.gettempdir()) / f'reordered_mlp_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        model = parallelize(source, {'x': torch.ones(3, 4, dtype=torch.float64)},
                            reordered_policy, ComputeConfig(4, 4, use_end2end=True, pas_config={'layout': layout}),
                            gen_savedir=tempdir, reuse='override').cuda()
        optimizer = build_optimizer(model, torch.optim.SGD, lr=0.01)
        for step in range(2):
            samples = [(torch.arange(12, device='cuda', dtype=torch.float64).reshape(3, 4)
                        + micro + step) / 12 for micro in range(2)]
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


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason='requires 4 GPUs')
@pytest.mark.parametrize('layout', ['gather', 'gather_reduce', 'split'])
def test_reordered_mlp_matches_eager(layout):
    assert all(launch_torchrun(4, reordered_worker, layout).values())
