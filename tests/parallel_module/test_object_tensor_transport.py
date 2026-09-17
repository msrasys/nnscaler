# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import nnscaler
from nnscaler import ComputeConfig, parallelize, build_optimizer
from nnscaler.policies import OpPlan, get_pas_ops
from tests.launch_torchrun import launch_torchrun
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0, replace_all_device_with


@nnscaler.register_op('? -> ?')
def make_metadata(x):
    return SimpleNamespace(scale=float(x.mean().item()) + 2.0, label='sample')


@nnscaler.register_op('n d, ? -> n d')
def apply_metadata(x, metadata):
    assert isinstance(metadata.scale, float) and metadata.label == 'sample'
    return x * metadata.scale


class TensorAndMetadata(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.output = torch.nn.Linear(4, 2, bias=False)

    def forward(self, x):
        metadata = make_metadata(x)
        y = self.linear(x)
        return self.output(apply_metadata(y, metadata)).sum()


def metadata_policy(graph, cfg):
    stage = 0
    for node in get_pas_ops(graph):
        if node.signature.endswith('.apply_metadata'):
            stage = 1
        yield OpPlan(node, stage_id=stage, partition=None)


def config():
    return ComputeConfig(2, 2, use_end2end=True, constant_folding=False,
                         pas_config={'pipeline_nmicros': 2, 'pipeline_scheduler': '1f1b'})


@replace_all_device_with('cpu')
def test_tensor_and_metadata_use_distinct_primitives(tmp_path):
    parallelize(TensorAndMetadata().double(), {'x': torch.ones(2, 4, dtype=torch.float64)},
                metadata_policy, config(), gen_savedir=tmp_path, load_module=False, reuse='override')
    source = '\n'.join(p.read_text() for p in tmp_path.rglob('gencode*.py'))
    assert 'nnscaler.runtime.adapter.move(' in source
    assert 'nnscaler.runtime.adapter.move_object(' in source


def transport_worker():
    nnscaler.init()
    torch.manual_seed(23)
    source = TensorAndMetadata().double()
    reference = copy.deepcopy(source).cuda()
    directory = Path(tempfile.gettempdir()) / f'tensor_metadata_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir:
        model = parallelize(source, {'x': torch.ones(2, 4, dtype=torch.float64)},
                            metadata_policy, config(), gen_savedir=tempdir, reuse='override').cuda()
        optimizer = build_optimizer(model, torch.optim.SGD, lr=0.01)
        eager_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        eager_params = dict(reference.named_parameters())
        original_send = torch.distributed.send_object_list
        sent = []

        def send_metadata(objects, *args, **kwargs):
            for obj in objects:
                assert isinstance(obj, SimpleNamespace)
                assert isinstance(obj.scale, float) and obj.label == 'sample'
                sent.append(obj.scale)
            return original_send(objects, *args, **kwargs)

        with patch.object(torch.distributed, 'send_object_list', send_metadata):
            for step in range(2):
                samples = [(torch.arange(8, device='cuda', dtype=torch.float64).reshape(2, 4)
                            + step + micro) / 8 for micro in range(2)]
                eager_optimizer.zero_grad()
                expected = [reference(x) for x in samples]
                torch.stack(expected).sum().backward()
                actual = model.train_step(samples)
                torch.testing.assert_close(actual, expected)
                for name, meta in model.fullmap.items():
                    torch.testing.assert_close(getattr(model, name).grad,
                                               eager_params[meta.orig_name].grad[meta.slicers])
                optimizer.step()
                eager_optimizer.step()
                for name, meta in model.fullmap.items():
                    torch.testing.assert_close(getattr(model, name), eager_params[meta.orig_name][meta.slicers])
                optimizer.zero_grad()
        if torch.distributed.get_rank() == 0:
            assert len(sent) == 4
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires 2 GPUs')
def test_tensor_and_metadata_pipeline_matches_eager():
    assert all(launch_torchrun(2, transport_worker).values())
