# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import copy
from collections import Counter
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import nnscaler
from nnscaler import ComputeConfig, parallelize, build_optimizer
from nnscaler.flags import CompileFlag
from nnscaler.ir.operator import IRFwOperation, IRDataOperation
from nnscaler.policies import OpPlan, get_pas_ops
from tests.launch_torchrun import launch_torchrun
from tests.parallel_module.test_gencode import _gencode_contains
from tests.utils import PYTEST_RUN_ID, clear_dir_on_rank0, replace_all_device_with


_FANOUT_CASES = [
    pytest.param(False, False, (1, 2), id='inter-rvd-broadcast'),
    pytest.param(True, True, (1, 2), id='unfused-moves'),
    pytest.param(False, True, (1, 2), id='general-broadcast'),
    pytest.param(False, True, (0, 1, 2), id='source-is-consumer'),
]


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


def broadcast_policy(graph, cfg):
    # fn/OpPlan assigns uniform per-stage device groups, not per-op rank subsets.
    # Explicit placement preserves independently selected producer/consumer ranks.
    ranks = cfg.pas_config['consumer_ranks']
    for node in graph.select(ntype=IRFwOperation):
        if node.signature.endswith('.make_metadata'):
            producer_ranks = cfg.pas_config.get('producer_ranks', (0,))
            for rank, replica in zip(producer_ranks, graph.replicate(node, len(producer_ranks))):
                graph.assign(replica, rank)
        else:
            for rank, replica in zip(ranks, graph.replicate(node, len(ranks))):
                graph.assign(replica, rank)
    for node in graph.select(ntype=IRDataOperation):
        for rank, replica in enumerate(graph.replicate(node, cfg.plan_ngpus)):
            graph.assign(replica, rank)
    return graph


@replace_all_device_with('cpu')
@pytest.mark.parametrize('producer_ranks,consumer_ranks', [
    ([0], [1, 2]),
    ([0, 2], [1, 3, 4]),
    ([0, 1, 2], [3, 4, 5, 6, 7]),
    ([0, 4], list(range(8))),
    ([4, 0], [7, 1, 6, 2, 5, 3]),
    ([0, 1, 2], [2, 3]),
    ([0, 0], [0, 1, 1, 2]),
    ([0, 1], [1]),
    ([0], [0]),
])
@pytest.mark.parametrize('disable_fusion', [False, True])
def test_object_adapter_covers_remote_consumers(tmp_path, monkeypatch, producer_ranks, consumer_ranks, disable_fusion):
    # Duplicate placements represent the same replica on a device.
    producer_ranks = tuple(dict.fromkeys(producer_ranks))
    consumer_ranks = tuple(dict.fromkeys(consumer_ranks))
    ngpus = max(producer_ranks + consumer_ranks) + 1
    monkeypatch.setattr(CompileFlag, 'disable_comm_fusion', disable_fusion)
    parallelize(
        TensorAndMetadata().double(), {'x': torch.ones(2, 4, dtype=torch.float64)},
        broadcast_policy,
        ComputeConfig(ngpus, ngpus, use_end2end=True, constant_folding=False,
                      pas_config={'producer_ranks': producer_ranks, 'consumer_ranks': consumer_ranks}),
        gen_savedir=tmp_path, load_module=False, reuse='override',
    )
    received = Counter()
    for rank in range(ngpus):
        moves = _gencode_contains(
            tmp_path, TensorAndMetadata, rank,
            r'nnscaler\.runtime\.adapter\.move_object\([^\n]*?src=(\d+), dst=(\d+)\)',
        )
        broadcasts = _gencode_contains(
            tmp_path, TensorAndMetadata, rank,
            r'nnscaler\.runtime\.adapter\.broadcast_object\([^\n]*?src=(\d+), ranks=[\[(]([\d, ]+)[\])]\)',
        )
        for src, dst in moves:
            assert int(src) in producer_ranks
            if rank != int(src):
                assert int(dst) == rank
                received[rank] += 1
        for src, ranks in broadcasts:
            assert int(src) in producer_ranks
            if rank != int(src):
                assert rank in [int(value) for value in ranks.split(',') if value.strip()]
                received[rank] += 1
    assert received == Counter({rank: 1 for rank in set(consumer_ranks) - set(producer_ranks)})


@replace_all_device_with('cpu')
def test_tensor_and_metadata_use_distinct_primitives(tmp_path):
    parallelize(TensorAndMetadata().double(), {'x': torch.ones(2, 4, dtype=torch.float64)},
                metadata_policy, config(), gen_savedir=tmp_path, load_module=False, reuse='override')
    for primitive in ('move', 'move_object'):
        assert any(_gencode_contains(tmp_path, TensorAndMetadata, rank,
                                     rf'nnscaler\.runtime\.adapter\.{primitive}\(')
                   for rank in range(2))


@replace_all_device_with('cpu')
@pytest.mark.parametrize('disable_fusion,disable_inter_rvd,consumer_ranks', _FANOUT_CASES)
def test_metadata_fanout_codegen(tmp_path, monkeypatch, disable_fusion, disable_inter_rvd, consumer_ranks):
    monkeypatch.setattr(CompileFlag, 'disable_comm_fusion', disable_fusion)
    # The existing planner's fusion flag only controls the general fallback, not inter-RVD.
    monkeypatch.setattr(CompileFlag, 'disable_inter_rvd', disable_inter_rvd)
    parallelize(TensorAndMetadata().double(), {'x': torch.ones(2, 4, dtype=torch.float64)},
                broadcast_policy, ComputeConfig(3, 3, use_end2end=True, constant_folding=False,
                                                pas_config={'consumer_ranks': consumer_ranks}),
                gen_savedir=tmp_path, load_module=False, reuse='override')
    primitive = 'move_object' if disable_fusion else 'broadcast_object'
    for rank in range(3):
        assert _gencode_contains(tmp_path, TensorAndMetadata, rank,
                                 rf'nnscaler\.runtime\.adapter\.{primitive}\(')


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


def broadcast_worker(disable_fusion, disable_inter_rvd, consumer_ranks):
    nnscaler.init()
    torch.manual_seed(23)
    source = TensorAndMetadata().double()
    reference = copy.deepcopy(source).cuda()
    directory = Path(tempfile.gettempdir()) / f'metadata_fanout_{PYTEST_RUN_ID}'
    with clear_dir_on_rank0(directory) as tempdir, \
            patch.object(CompileFlag, 'disable_comm_fusion', disable_fusion), \
            patch.object(CompileFlag, 'disable_inter_rvd', disable_inter_rvd):
        model = parallelize(source, {'x': torch.ones(2, 4, dtype=torch.float64)},
                            broadcast_policy, ComputeConfig(3, 3, use_end2end=True, constant_folding=False,
                                                            pas_config={'consumer_ranks': consumer_ranks}),
                            gen_savedir=tempdir, reuse='override').cuda()
        with torch.no_grad():
            for step in range(2):
                x = (torch.arange(8, device='cuda', dtype=torch.float64).reshape(2, 4) + step) / 8
                torch.testing.assert_close(model.infer_step([x]), [reference(x)])
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 3, reason='requires 3 GPUs')
@pytest.mark.parametrize('disable_fusion,disable_inter_rvd,consumer_ranks', _FANOUT_CASES)
def test_metadata_fanout_matches_eager(disable_fusion, disable_inter_rvd, consumer_ranks):
    assert all(launch_torchrun(3, broadcast_worker, disable_fusion, disable_inter_rvd, consumer_ranks).values())
