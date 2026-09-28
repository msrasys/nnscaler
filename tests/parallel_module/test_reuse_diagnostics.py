# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
from dataclasses import asdict, replace

import pytest
import torch

from nnscaler.graph.parser import FxModuleParser
from nnscaler.parallel import ComputeConfig, ReuseType, _prepare_and_check_reusable
from nnscaler.runtime.module import ParallelModule

parallel = importlib.import_module('nnscaler.parallel')


class Model(torch.nn.Module):
    pass


def package(tmp_path, config, legacy=False):
    folder, _ = _prepare_and_check_reusable(tmp_path, Model, config, 'test', ReuseType.MATCH)
    names = [
        *(parallel._GENCODE_FILE_TEMPLATE.format(rank) for rank in range(config.runtime_ngpus)),
        FxModuleParser.ATTR_CONTENT_FILE_0,
        FxModuleParser.ATTR_CONTENT_INDEX_FILE,
        FxModuleParser.ATTR_MAP_FILE,
        parallel._GRAPH_DUMP_FILE,
        parallel._FORWARD_ARGS_DUMP_FILE,
        ParallelModule.ORIGIN_MODULE_METADATA_FILE,
        FxModuleParser.NON_PERSISTENT_BUFFER_FILE,
    ]
    names += ([ParallelModule.ATTR_META_FILE_TEMPLATE.format(rank)
               for rank in range(config.runtime_ngpus)] if legacy else [ParallelModule.ATTR_META_FILE])
    for name in names:
        (folder / name).touch()
    ComputeConfig.safe_dump_to_file(config, folder / ParallelModule.COMPUTE_CONFIG_FILE)
    return folder


@pytest.mark.parametrize('legacy', [False, True])
def test_reuse_reports_staged_file_mismatch(tmp_path, legacy):
    config = ComputeConfig(1, 2)
    folder = package(tmp_path, config, legacy)
    assert _prepare_and_check_reusable(tmp_path, Model, config, 'test', ReuseType.MATCH)[1]
    # An interrupted copy or a copy over an older package can leave both kinds.
    (folder / 'old_metadata.pkl').touch()
    (folder / 'gencode1.py').unlink()
    with pytest.raises(RuntimeError) as error:
        _prepare_and_check_reusable(tmp_path, Model, config, 'test', ReuseType.MATCH)
    message = str(error.value)
    assert 'ComputeConfig match: True.' in message
    assert 'Missing files: gencode1.py.' in message
    assert 'Unexpected files: old_metadata.pkl.' in message
    assert (folder / 'old_metadata.pkl').exists()


def test_reuse_reports_nested_config_difference_without_values(tmp_path):
    config = ComputeConfig(1, 1, user_config={'model_args': {'api_key': 'private-value'}})
    package(tmp_path, config)
    current = replace(config, user_config={'model_args': {'api_key': 'another-value'}})
    with pytest.raises(RuntimeError) as error:
        _prepare_and_check_reusable(tmp_path, Model, current, 'test', ReuseType.MATCH)
    message = str(error.value)
    assert 'ComputeConfig match: False.' in message
    assert 'Differing config fields: user_config.model_args.api_key.' in message
    assert 'private-value' not in message and 'another-value' not in message
    assert 'Missing files:' not in message and 'Unexpected files:' not in message


def test_reuse_reports_unreadable_config(tmp_path):
    config = ComputeConfig(1, 1)
    folder = package(tmp_path, config)
    (folder / ParallelModule.COMPUTE_CONFIG_FILE).unlink()
    with pytest.raises(RuntimeError, match='Saved compute_config.pt is missing or could not be loaded'):
        _prepare_and_check_reusable(tmp_path, Model, config, 'test', ReuseType.MATCH)


def vl_config(path, **model_overrides):
    return ComputeConfig(1, 2, user_config={
        '__from_trainer_args': {
            'model_args': {
                'model_args': {
                    'vision_encoder_path': path,
                    'vision_freeze': True,
                    'vision_patch_size': 14,
                    'd_model': 3072,
                    **model_overrides,
                },
            },
        },
    })


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('reuse', [ReuseType.MATCH, ReuseType.MOO])
def test_reuse_accepts_relocated_vision_checkpoint(tmp_path, legacy, reuse):
    saved = vl_config('/data/.cache/vision-checkpoints/Kimi-K3-ViT-init-3072')
    current = vl_config('/tmp/job-123/vision-checkpoints/Kimi-K3-ViT-init-3072')
    folder = package(tmp_path, saved, legacy)
    before_files = {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    before_saved, before_current = asdict(saved), asdict(current)

    # The legacy package still records the original location; no migration is needed.
    assert _prepare_and_check_reusable(tmp_path, Model, current, 'test', reuse) == (folder, True)
    assert ComputeConfig.safe_equals(saved, current)
    assert ComputeConfig.safe_equals(current, saved)
    assert saved.graph_config == current.graph_config
    assert asdict(saved) == before_saved
    assert asdict(current) == before_current
    assert {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()} == before_files


@pytest.mark.parametrize('field,value', [
    ('d_model', 4096),
    ('vision_freeze', False),
    ('vision_patch_size', 16),
])
def test_relocated_vision_checkpoint_still_checks_model_config(tmp_path, field, value):
    saved = vl_config('/old/vision')
    current = vl_config('/new/vision', **{field: value})
    package(tmp_path, saved)

    assert not ComputeConfig.safe_equals(saved, current)
    assert saved.graph_config != current.graph_config
    with pytest.raises(RuntimeError) as error:
        _prepare_and_check_reusable(tmp_path, Model, current, 'test', ReuseType.MATCH)
    assert f'user_config.__from_trainer_args.model_args.model_args.{field}' in str(error.value)
    assert 'vision_encoder_path' not in str(error.value)


def test_relocated_vision_checkpoint_still_checks_topology_and_files(tmp_path):
    saved = vl_config('/old/vision')
    current = vl_config('/new/vision')
    folder = package(tmp_path, saved)
    with pytest.raises(RuntimeError, match='Differing config fields: plan_ngpus'):
        _prepare_and_check_reusable(
            tmp_path, Model, replace(current, plan_ngpus=2), 'test', ReuseType.MATCH)

    (folder / 'gencode1.py').unlink()
    with pytest.raises(RuntimeError, match='Missing files: gencode1.py'):
        _prepare_and_check_reusable(tmp_path, Model, current, 'test', ReuseType.MATCH)


@pytest.mark.parametrize('nested', [False, True])
def test_reuse_does_not_ignore_vision_path_in_arbitrary_user_config(nested):
    saved = {'vision_encoder_path': '/old/vision'}
    current = {'vision_encoder_path': '/new/vision'}
    if nested:
        saved, current = {'model_args': saved}, {'model_args': current}
    saved = ComputeConfig(1, 1, user_config=saved)
    current = ComputeConfig(1, 1, user_config=current)
    assert not ComputeConfig.safe_equals(saved, current)
    assert saved.graph_config != current.graph_config


def test_graph_reuse_preserves_trace_for_relocated_vision_checkpoint(tmp_path):
    saved = vl_config('/old/vision')
    current = vl_config('/new/vision')
    folder = package(tmp_path, saved)
    graph_file = folder / parallel._GRAPH_DUMP_FILE
    graph_file.write_bytes(b'saved trace')

    assert _prepare_and_check_reusable(
        tmp_path, Model, current, 'test', ReuseType.GRAPH) == (folder, False)
    assert graph_file.read_bytes() == b'saved trace'
    assert not (folder / 'gencode0.py').exists()
