import copy
import json
import sys
from types import SimpleNamespace

import pytest

from nnscaler.runtime.deepep import get_config_override, validate_configs

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
