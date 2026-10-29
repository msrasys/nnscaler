# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
from dataclasses import asdict, replace

import pytest
import torch

from nnscaler.parallel import ComputeConfig
from nnscaler.runtime.cpu_offloading import CPUOffloadConfig


@pytest.mark.parametrize('offload', [None, CPUOffloadConfig(max_cpu_bytes=4096)])
@pytest.mark.parametrize('release', [False, True])
def test_pipeline_lifetime_is_independent_of_activation_storage(offload, release, tmp_path):
    config = ComputeConfig(2, 2, use_end2end=True, cpu_offload=offload,
                           release_pipeline_inputs=release)
    assert config.release_pipeline_inputs is release
    assert config.cpu_offload == offload
    path = tmp_path / 'config.pt'
    config.safe_dump_to_file(config, path)
    restored = config.safe_load_from_file(path, return_none_on_error=False)
    assert config.safe_equals(config, restored)
    assert not config.safe_equals(config, replace(config, release_pipeline_inputs=not release))


@pytest.mark.parametrize('serialized_type', ['dict', 'object'])
def test_legacy_nested_setting_migrates_once_without_mutating_caller(tmp_path, serialized_type):
    legacy = CPUOffloadConfig(max_cpu_bytes=4096, release_pipeline_inputs=True)
    with pytest.warns(DeprecationWarning, match='ComputeConfig.release_pipeline_inputs'):
        config = ComputeConfig(2, 2, cpu_offload=legacy)
    assert config.release_pipeline_inputs is True
    assert legacy.release_pipeline_inputs is True
    assert config.cpu_offload == replace(legacy, release_pipeline_inputs=False)
    # Re-saving uses the canonical field, so subsequent loads need no migration.
    saved = asdict(config)
    assert saved['release_pipeline_inputs'] is True
    assert saved['cpu_offload']['release_pipeline_inputs'] is False
    old = dict(saved)
    old.pop('release_pipeline_inputs')
    old['cpu_offload'] = asdict(legacy)
    if serialized_type == 'object':
        old_object = object.__new__(ComputeConfig)
        old_object.__dict__.update(old)
        old_object.__dict__['cpu_offload'] = legacy
        old = old_object
    path = tmp_path / 'old-config.pt'
    torch.save(old, path)
    with pytest.warns(DeprecationWarning):
        restored = ComputeConfig.safe_load_from_file(path, return_none_on_error=False)
    assert ComputeConfig.safe_equals(restored, config)


@pytest.mark.parametrize('release', [False, True])
def test_explicit_canonical_setting_wins_over_legacy_alias(release):
    with pytest.warns(DeprecationWarning):
        config = ComputeConfig(2, 2, release_pipeline_inputs=release,
                               cpu_offload={'release_pipeline_inputs': True})
    assert config.release_pipeline_inputs is release
    assert not config.cpu_offload.release_pipeline_inputs


@pytest.mark.parametrize('value', [1, 'true', []])
def test_pipeline_release_rejects_non_boolean_values(value):
    with pytest.raises(ValueError, match='must be a bool'):
        ComputeConfig(2, 2, release_pipeline_inputs=value)


def test_default_is_disabled_without_cpu_offload():
    config = ComputeConfig(2, 2)
    assert config.release_pipeline_inputs is False and config.cpu_offload is None


@pytest.mark.parametrize('release_inputs', [False, True])
def test_output_lifetime_has_an_independent_serialized_setting(release_inputs, tmp_path):
    config = ComputeConfig(2, 2, use_end2end=True, release_pipeline_inputs=release_inputs,
                           release_pipeline_outputs=True)
    path = tmp_path / 'outputs.pt'
    ComputeConfig.safe_dump_to_file(config, path)
    assert ComputeConfig.safe_equals(config, ComputeConfig.safe_load_from_file(path))
    assert not ComputeConfig.safe_equals(config, replace(config, release_pipeline_outputs=False))
    with pytest.raises(ValueError, match='not supported with split backward'):
        replace(config, use_fbw=True)


@pytest.mark.parametrize('value', [1, 'true', None])
def test_output_release_requires_a_boolean(value):
    with pytest.raises(ValueError, match='release_pipeline_outputs must be a bool'):
        ComputeConfig(2, 2, release_pipeline_outputs=value)
