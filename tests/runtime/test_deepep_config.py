import copy
import json
import os
import sys
from types import SimpleNamespace

import pytest

from nnscaler.runtime.deepep import CONFIG_FIELDS, configure_from_file, get_config_override, validate_configs

SPEC = {'2': {'dispatch': [96, 128, 512, 6, 128], 'combine': [96, 64, 512, 6, 128]}}


def test_override_only_applies_to_configured_groups(monkeypatch):
    monkeypatch.setenv('NNSCALER_DEEPEP_CONFIG', json.dumps(SPEC))
    monkeypatch.setitem(sys.modules, 'deep_ep', SimpleNamespace(Config=lambda *args: args))
    monkeypatch.setattr('torch.cuda.current_device', lambda: 0)
    monkeypatch.setattr(
        'torch.cuda.get_device_properties',
        lambda _: SimpleNamespace(multi_processor_count=148),
    )
    assert get_config_override(2, 'dispatch') == (96, 128, 512, 6, 128)
    assert get_config_override(2, 'combine') == (96, 64, 512, 6, 128)
    assert get_config_override(8, 'dispatch') is None
    monkeypatch.delenv('NNSCALER_DEEPEP_CONFIG')
    assert get_config_override(2, 'dispatch') is None


@pytest.mark.parametrize(
    ('field', 'index', 'value'),
    [
        ('dispatch', 0, 149),
        ('dispatch', 0, 0),
        ('dispatch', 0, True),
        ('combine', 0, 48),
        ('combine', 2, 256),
        ('dispatch', 1, 1024),
    ],
)
def test_invalid_or_incompatible_configs_fail_before_kernel_launch(field, index, value):
    spec = copy.deepcopy(SPEC)
    spec['2'][field][index] = value
    with pytest.raises(ValueError):
        validate_configs(spec, 148)


def test_training_config_file_keeps_all_five_values(tmp_path, monkeypatch):
    path = tmp_path / 'deepep.json'
    named = {'2': {op: dict(zip(CONFIG_FIELDS, values)) for op, values in SPEC['2'].items()}}
    path.write_text(json.dumps(named))
    monkeypatch.setenv('NNSCALER_DEEPEP_CONFIG', '{}')
    assert configure_from_file(path) == validate_configs(SPEC, 148)
    monkeypatch.setitem(sys.modules, 'deep_ep', SimpleNamespace(Config=lambda *args: args))
    monkeypatch.setattr('torch.cuda.current_device', lambda: 0)
    monkeypatch.setattr('torch.cuda.get_device_properties', lambda _: SimpleNamespace(multi_processor_count=148))
    assert get_config_override(2, 'dispatch') == tuple(SPEC['2']['dispatch'])
    assert get_config_override(2, 'combine') == tuple(SPEC['2']['combine'])
    named['2']['dispatch']['misspelled'] = 1
    path.write_text(json.dumps(named))
    previous = os.environ['NNSCALER_DEEPEP_CONFIG']
    with pytest.raises(ValueError):
        configure_from_file(path)
    assert os.environ['NNSCALER_DEEPEP_CONFIG'] == previous
