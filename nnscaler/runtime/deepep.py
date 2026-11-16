# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Opt-in tuning of DeepEP's default intranode communication configurations.

Explicit configs passed to DeepEP retain precedence. No model functions or
installed library files are changed. Configure before constructing any Buffer.
"""

import json
import logging
import os


ENV_NAME = 'NNSCALER_DEEPEP_CONFIG'


def validate_configs(configs, max_sms):
    if not isinstance(configs, dict):
        raise ValueError(f'{ENV_NAME} must map EP sizes to dispatch/combine configs')
    result = {}
    for ranks, pair in configs.items():
        ranks = int(ranks)
        if ranks not in (2, 4, 8):
            raise ValueError('DeepEP default tuning currently supports intranode EP2/4/8')
        if not isinstance(pair, dict) or set(pair) != {'dispatch', 'combine'}:
            raise ValueError('Each EP size requires exactly dispatch and combine')
        converted = {}
        for operation, values in pair.items():
            if not isinstance(values, (list, tuple)) or len(values) != 5:
                raise ValueError('DeepEP configs contain five integers: SMs, NVL send/receive, RDMA send/receive')
            if any(type(value) is not int or value <= 0 for value in values):
                raise ValueError('DeepEP configuration values must be positive integers')
            sms, nvl_send, nvl_recv, rdma_send, rdma_recv = values
            if sms % 2 or sms > max_sms:
                raise ValueError(f'DeepEP SM count must be even and at most {max_sms}')
            if nvl_send > nvl_recv or rdma_send > rdma_recv:
                raise ValueError('DeepEP send chunks must fit their receive buffers')
            converted[operation] = tuple(values)
        if converted['dispatch'][0] != converted['combine'][0]:
            raise ValueError('Dispatch and combine must use the same channel/SM count')
        if converted['dispatch'][2] != converted['combine'][2]:
            raise ValueError('Dispatch and combine must use the same NVL receive capacity')
        result[ranks] = converted
    return result


def install_defaults(buffer_type, config_type, configs):
    """Install validated defaults once; subsequent calls replace the overrides."""
    if not hasattr(buffer_type, '_nnscaler_original_dispatch_config'):
        buffer_type._nnscaler_original_dispatch_config = staticmethod(buffer_type.get_dispatch_config)
        buffer_type._nnscaler_original_combine_config = staticmethod(buffer_type.get_combine_config)

        def dispatch(ranks):
            pair = buffer_type._nnscaler_config_overrides.get(ranks)
            return (config_type(*pair['dispatch']) if pair is not None
                    else buffer_type._nnscaler_original_dispatch_config(ranks))

        def combine(ranks):
            pair = buffer_type._nnscaler_config_overrides.get(ranks)
            return (config_type(*pair['combine']) if pair is not None
                    else buffer_type._nnscaler_original_combine_config(ranks))

        buffer_type.get_dispatch_config = staticmethod(dispatch)
        buffer_type.get_combine_config = staticmethod(combine)
    buffer_type._nnscaler_config_overrides = configs


def configure_from_env():
    value = os.environ.get(ENV_NAME)
    if value is None:
        return
    import torch
    from deep_ep import Buffer, Config
    configs = validate_configs(json.loads(value), torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
    install_defaults(Buffer, Config, configs)
    logging.getLogger(__name__).info('DeepEP intranode default overrides: %s', configs)
