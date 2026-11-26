# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Explicit configuration overrides for the installed DeepEP library."""

import json
from functools import lru_cache

import torch
import os
from pathlib import Path


def gpu_counts_enabled():
    """Whether the backend should keep expert-layout metadata on the GPU."""
    value = os.environ.get('NNSCALER_DEEPEP_GPU_COUNTS', '0')
    if value not in ('0', '1'):
        raise ValueError('NNSCALER_DEEPEP_GPU_COUNTS must be 0 or 1')
    return value == '1'


ENV_NAME = 'NNSCALER_DEEPEP_CONFIG'
CONFIG_FIELDS = ('num_sms', 'nvl_chunk_size', 'nvl_buffer_size',
                 'rdma_chunk_size', 'rdma_buffer_size')


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
            if isinstance(values, dict):
                if set(values) != set(CONFIG_FIELDS):
                    raise ValueError(f'DeepEP named configs require exactly {CONFIG_FIELDS}')
                values = [values[field] for field in CONFIG_FIELDS]
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


def configure_from_file(path):
    """Load explicit runtime settings before any DeepEP buffer is initialized.

    Accept either five-value arrays or the named fields used by training job
    files. The device's actual SM limit is checked when the config is consumed.
    """
    configs = validate_configs(json.loads(Path(path).read_text()), float('inf'))
    os.environ[ENV_NAME] = json.dumps(configs)
    return configs


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
