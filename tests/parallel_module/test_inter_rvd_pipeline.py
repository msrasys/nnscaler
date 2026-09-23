# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from nnscaler import ComputeConfig, parallelize
from nnscaler.policies import get_pas_ops, OpPlan
from tests.utils import replace_all_device_with

from .test_gencode import _gencode_contains


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


@replace_all_device_with('cpu', force=True)
def test_fn_inter_stage_gencode_prefers_matching_peers(tmp_path):
    """Check the placement preference in generated communication for a real model."""
    config = ComputeConfig(8, 8, use_end2end=True, constant_folding=False,
                           pas_config={'pipeline_size': 2, 'pipeline_nmicros': 4,
                                       'pipeline_scheduler': '1f1b_interleaved'})
    parallelize(InterStageMLP(), {'sample': {'x': torch.ones(8, 8)}},
                inter_stage_policy, config,
                gen_savedir=tmp_path, reuse='override', load_module=False)
    # This model permits matching relative ranks for transfers from 0..3 to
    # 4..7. Other boundaries may require a permutation to align shard layouts.
    for rank in range(8):
        peers = _gencode_contains(
            tmp_path, InterStageMLP, rank,
            r'nnscaler\.runtime\.adapter\.move\([^\n]*\bsrc=([0-3]), dst=(\d+)\b')
        assert set(peers) == {(str(rank % 4), str(rank % 4 + 4))}
