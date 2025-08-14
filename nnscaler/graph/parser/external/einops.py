#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import logging

import torch


_logger = logging.getLogger(__name__)

try:
    import einops

    # trigger einops initialization
    einops.rearrange(torch.arange(1), '(a b c) -> a b c', a=1, b=1, c=1)
except ImportError as e:
    _logger.debug("Einops is not installed")
    pass
