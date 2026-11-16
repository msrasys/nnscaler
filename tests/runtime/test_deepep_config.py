import copy

import pytest

from nnscaler.runtime.deepep import install_defaults, validate_configs


SPEC = {'2': {'dispatch': [96, 128, 512, 6, 128], 'combine': [96, 64, 512, 6, 128]}}


def test_default_override_is_scoped_and_can_be_replaced():
    class Buffer:
        get_dispatch_config = staticmethod(lambda ranks: ('original_dispatch', ranks))
        get_combine_config = staticmethod(lambda ranks: ('original_combine', ranks))

    configs = validate_configs(SPEC, 148)
    install_defaults(Buffer, lambda *args: args, configs)
    assert Buffer.get_dispatch_config(2) == (96, 128, 512, 6, 128)
    assert Buffer.get_combine_config(2) == (96, 64, 512, 6, 128)
    assert Buffer.get_dispatch_config(8) == ('original_dispatch', 8)
    install_defaults(Buffer, lambda *args: args, {})
    assert Buffer.get_dispatch_config(2) == ('original_dispatch', 2)
    assert Buffer.get_combine_config(2) == ('original_combine', 2)


@pytest.mark.parametrize(('field', 'index', 'value'), [
    ('dispatch', 0, 149), ('dispatch', 0, 0), ('dispatch', 0, True),
    ('combine', 0, 48), ('combine', 2, 256), ('dispatch', 1, 1024),
])
def test_invalid_or_incompatible_configs_fail_before_kernel_launch(field, index, value):
    spec = copy.deepcopy(SPEC)
    spec['2'][field][index] = value
    with pytest.raises(ValueError):
        validate_configs(spec, 148)
