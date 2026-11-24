# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Trace-cache invalidation when compiler metadata was absent in older graphs."""

import importlib
from dataclasses import asdict, replace

import pytest
import torch

from nnscaler.graph import IRGraph
from nnscaler.graph import parser
from nnscaler.ir.operator import IRFwOperation
from nnscaler.parallel import ComputeConfig, ReuseType, parallelize
from nnscaler.runtime.module import ParallelModule
from tests.utils import replace_all_device_with
from tests.parallel_module.test_reuse_diagnostics import Model, package


parallel = importlib.import_module('nnscaler.parallel')
CURRENT_VERSION = parallel._TRACE_METADATA_VERSION


def test_recompute_merge_switch_roundtrips_and_invalidates_codegen(tmp_path):
    disabled = ComputeConfig(1, 1)
    enabled = replace(disabled, recompute_merge_adjacent=True)
    assert not ComputeConfig.safe_equals(disabled, enabled)
    assert disabled.graph_config == enabled.graph_config
    for config in (disabled, enabled):
        path = tmp_path / f'{config.recompute_merge_adjacent}.pt'
        ComputeConfig.safe_dump_to_file(config, path)
        assert ComputeConfig.safe_equals(config, ComputeConfig.safe_load_from_file(path))


def _serialized_config(config, serialized_type, *, legacy=False):
    saved = asdict(config)
    if legacy:
        del saved['_trace_metadata_version']
    if serialized_type == 'object':
        old_object = object.__new__(ComputeConfig)
        old_object.__dict__.update(saved)
        return old_object
    return saved


@pytest.mark.parametrize('serialized_type', ['dict', 'object'])
def test_missing_trace_version_loads_as_legacy_even_with_class_default(tmp_path, serialized_type):
    current = ComputeConfig(1, 1)
    saved = _serialized_config(current, serialized_type, legacy=True)
    if serialized_type == 'object':
        assert '_trace_metadata_version' not in vars(saved)
        assert saved._trace_metadata_version == CURRENT_VERSION  # The class default must not win.
        assert not ComputeConfig.safe_equals(saved, current)
        assert saved.graph_config != current.graph_config
    path = tmp_path / 'legacy.pt'
    torch.save(saved, path)
    restored = ComputeConfig.safe_load_from_file(path, return_none_on_error=False)
    assert restored._trace_metadata_version == 0
    assert restored.cpu_offload is None
    assert restored.graph_config != current.graph_config
    assert not ComputeConfig.safe_equals(restored, current)
    assert ComputeConfig.safe_equals(restored, replace(current, _trace_metadata_version=0))
    ComputeConfig.safe_dump_to_file(restored, path)
    assert ComputeConfig.safe_load_from_file(path)._trace_metadata_version == 0


@pytest.mark.parametrize('serialized_type, version', [
    ('dict', 0), ('object', CURRENT_VERSION), ('dict', CURRENT_VERSION + 1),
])
def test_explicit_trace_version_roundtrips(tmp_path, serialized_type, version):
    config = ComputeConfig(1, 1, _trace_metadata_version=version)
    path = tmp_path / 'config.pt'
    torch.save(_serialized_config(config, serialized_type), path)
    restored = ComputeConfig.safe_load_from_file(path, return_none_on_error=False)
    assert restored._trace_metadata_version == version
    assert ComputeConfig.safe_equals(config, restored)
    assert restored.graph_config == config.graph_config


@pytest.mark.parametrize('version', [-1, True, 1.0])
def test_trace_version_requires_a_nonnegative_integer(version):
    with pytest.raises(ValueError, match='_trace_metadata_version'):
        ComputeConfig(1, 1, _trace_metadata_version=version)


@pytest.mark.parametrize('reuse, version', [
    (ReuseType.MOO, 0), (ReuseType.GRAPH, CURRENT_VERSION - 1),
])
def test_legacy_config_equality_cannot_reuse_legacy_artifact(tmp_path, reuse, version):
    legacy = ComputeConfig(1, 1, _trace_metadata_version=version)
    folder = package(tmp_path, legacy)
    assert parallel._prepare_and_check_reusable(
        tmp_path, Model, legacy, 'test', reuse) == (folder, False)
    assert not (folder / parallel._GRAPH_DUMP_FILE).exists()


class EffectMetadataModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 4))

    def forward(self, x):
        return torch.sin(x @ self.weight)


@replace_all_device_with('cpu')
@pytest.mark.parametrize('reuse, serialized_type, input_config', [
    (ReuseType.MOO, 'dict', 'current'),
    (ReuseType.GRAPH, 'object', 'current'),
    (ReuseType.MOO, 'object', 'legacy'),
    (ReuseType.GRAPH, 'dict', 'future'),
])
def test_old_effectless_graph_is_retraced_then_current_graph_is_reused(
        tmp_path, monkeypatch, reuse, serialized_type, input_config):
    traces = []
    original_trace = parser.to_fx_graph

    def trace_with_known_effects(*args, **kwargs):
        traced = original_trace(*args, **kwargs)
        sine, = [node for node in traced.graph.nodes
                 if node.op == 'call_function' and node.target is torch.sin]
        sine.meta['nnscaler_recompute_barriers'] = ('custom_state',)
        traces.append(traced)
        return traced

    monkeypatch.setattr(parser, 'to_fx_graph', trace_with_known_effects)
    initial = ComputeConfig(1, 1, trace_strategy='cpu', pas_config={'_test_policy_version': 1})
    inputs = {'x': torch.randn(4, 4)}

    def compile_model(config, mode):
        parallelize(EffectMetadataModel, inputs, 'data', config, gen_savedir=tmp_path,
                    instance_name='trace_version', reuse=mode, load_module=False)

    compile_model(initial, ReuseType.MOO)
    assert len(traces) == 1
    _, folder = parallel._prepare_namespace(tmp_path, EffectMetadataModel, 'trace_version')
    graph_path = folder / parallel._GRAPH_DUMP_FILE
    config_path = folder / ParallelModule.COMPUTE_CONFIG_FILE
    graph = IRGraph.load(graph_path)
    sine, = [op for op in graph.select(ntype=IRFwOperation) if op.signature == 'torch.sin']
    assert sine.recompute_barriers == ('custom_state',)

    # Model a graph serialized before the parser recorded effect facts.
    for op in graph.select(ntype=IRFwOperation):
        op.__dict__.pop('_recompute_barriers', None)
    graph.dump(graph_path)
    torch.save(_serialized_config(initial, serialized_type, legacy=True), config_path)

    if input_config == 'legacy':
        current = ComputeConfig.safe_load_from_file(config_path, return_none_on_error=False)
    else:
        # A policy-only change must still retrace the metadata-free graph.
        current = replace(initial, pas_config={'_test_policy_version': 2},
                          _trace_metadata_version=CURRENT_VERSION + 1 if input_config == 'future' else CURRENT_VERSION)
    input_version = current._trace_metadata_version
    compile_model(current, reuse)
    assert current._trace_metadata_version == input_version  # Do not mutate the caller.
    assert len(traces) == 2
    refreshed = IRGraph.load(graph_path)
    sine, = [op for op in refreshed.select(ntype=IRFwOperation) if op.signature == 'torch.sin']
    assert sine.recompute_barriers == ('custom_state',)
    assert ComputeConfig.safe_load_from_file(config_path)._trace_metadata_version == CURRENT_VERSION

    current_dump = graph_path.read_bytes()
    compile_model(current, reuse)
    assert current._trace_metadata_version == input_version
    assert len(traces) == 2
    assert graph_path.read_bytes() == current_dump

