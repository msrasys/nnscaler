# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import warnings

from nnscaler.graph.tracer.metadata import AutocastInfo


def test_autocast_info_uses_current_torch_api():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        info = AutocastInfo.from_context()

    assert isinstance(info.cpu_enabled, bool)
    assert isinstance(info.cuda_enabled, bool)
