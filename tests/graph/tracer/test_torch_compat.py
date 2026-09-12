#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch
from torch.fx import Graph

from nnscaler.graph.tracer import wrap_utils
from nnscaler.graph.tracer.torch_fx_patcher import TorchFXPatcher


def test_torch_assert_wrapper():
    assert wrap_utils.torch_assert_wrapper(True, 'must be true') is None
    with pytest.raises(AssertionError, match='must be true'):
        wrap_utils.torch_assert_wrapper(False, 'must be true')


def test_torch_fx_patcher_preserves_effectful_ops():
    graph = Graph()
    node = graph.call_function(torch.ops.aten._print.default, ('side effect',))
    graph.output(None)

    assert node.is_impure()
    with TorchFXPatcher():
        assert node.is_impure()
