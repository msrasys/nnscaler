# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit configuration overrides for the installed DeepEP library."""

import json
from functools import lru_cache

import torch
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


@lru_cache(maxsize=8)
def _configs(value, max_sms):
    return validate_configs(json.loads(value), max_sms)


def get_config_override(num_ranks, operation):
    """Return an explicit DeepEP config override, without modifying the library."""
    value = os.environ.get(ENV_NAME)
    if value is None:
        return None
    if operation not in ('dispatch', 'combine'):
        raise ValueError('Expected dispatch or combine configuration')
    from deep_ep import Config

    pair = _configs(
        value,
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
    ).get(num_ranks)
    return Config(*pair[operation]) if pair is not None else None
