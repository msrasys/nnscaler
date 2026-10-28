# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from unittest.mock import patch

import pytest
import torch

from nnscaler import ComputeConfig
from nnscaler.algorithm.ops.dimops import DimSplitEinops
from nnscaler.graph import IRGraph
from nnscaler.graph.parser import convert_model
from nnscaler.graph.segment import IRSegment
from nnscaler.policies import fn, pas_hybrid
from tests.test_policies import FnPolicyModuleList, megatron_ffn_policy_list
from tests.utils import replace_all_device_with


@pytest.mark.parametrize('settings,ngpus,nstages,match', [
    ({'pp_size': 0}, 4, 4, 'pp_size'),
    ({'pipeline_size': 0}, 4, 4, 'pp_size'),
    ({'pipeline_size': -2}, 4, 4, 'pp_size'),
    ({'pipeline_size': 3}, 4, 4, 'nstages'),
    ({'pipeline_size': 3}, 4, 6, 'ngpus'),
    ({'pipeline_size': 2, 'pp_size': 4}, 4, 4, 'must match'),
])
def test_hybrid_validates_sizes_before_mutating_graph(settings, ngpus, nstages, match):
    graph = IRGraph([], [], [], 'invalid_pipeline')
    cfg = ComputeConfig(ngpus, ngpus, use_end2end=True, pas_config={
        'pipeline_nstages': nstages, 'pipeline_nmicros': 4, **settings,
    })
    with patch('nnscaler.policies.auto_multiref') as mutate:
        with pytest.raises(ValueError, match=match):
            pas_hybrid(graph, cfg)
        mutate.assert_not_called()


@replace_all_device_with('cpu')
@pytest.mark.parametrize('settings', [
    {'pp_size': 2}, {'pipeline_size': 2}, {'pp_size': 2, 'pipeline_size': 2},
])
def test_hybrid_aliases_place_virtual_stages_on_same_physical_groups(tmp_path, settings):
    graph = convert_model(FnPolicyModuleList(ffn_layers=4), {'x': torch.randn(4, 4)}, tmp_path)
    graph.backward(graph.output(0))
    cfg = ComputeConfig(4, 4, use_end2end=True, pas_config={
        'pipeline_nstages': 4, 'pipeline_nmicros': 4,
        'pipeline_scheduler': '1f1b_interleaved', **settings,
    })
    result = pas_hybrid(graph, cfg)
    stages = [stage for stage in result.select(ntype=IRSegment, flatten=False) if stage.isfw()]
    assert [tuple(stage.device) for stage in stages] == [(0, 1), (2, 3), (0, 1), (2, 3)]


@replace_all_device_with('cpu')
@pytest.mark.parametrize('key', ['pp_size', 'pipeline_size'])
def test_fn_rejects_logical_stage_mismatch_before_partition(tmp_path, key):
    graph = convert_model(FnPolicyModuleList(ffn_layers=4), {'x': torch.randn(4, 4)}, tmp_path)
    cfg = ComputeConfig(6, 6, use_end2end=True, pas_config={
        key: 3, 'pipeline_nmicros': 4, 'pipeline_scheduler': '1f1b_interleaved',
    })
    with patch.object(DimSplitEinops, 'instantiate', autospec=True, side_effect=DimSplitEinops.instantiate) as instantiate:
        with pytest.raises(ValueError, match='nstages 4'):
            fn(graph, cfg, megatron_ffn_policy_list)
        instantiate.assert_not_called()
