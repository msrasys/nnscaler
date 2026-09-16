# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
from dataclasses import replace

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
