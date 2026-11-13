# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Compatibility exports for the original SM-only transport interface."""
from .activation_cuda_copy import (
    _extension,
    copy_d2h,
    copy_h2d,
    prepare_cuda_copy as prepare_sm_copy,
)

prepare_sm_h2d = prepare_sm_copy
