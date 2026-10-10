#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import pytest
import torch

from nnscaler.flags import CompileFlag
from nnscaler.graph.parser.converter import to_fx_graph

from ...utils import replace_all_device_with


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.dropout(x, 0.1, self.training)


@replace_all_device_with('cpu')
def test_getattr_from_root():
    model = SimpleModel()
    dummy_input = {'x': torch.rand(10)}
    traced_graph = to_fx_graph(model, dummy_input)
    traced_graph(**dummy_input)


class TensorAttributeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.constant = torch.rand(4)
        self.child = torch.nn.Module()
        self.child.offset = torch.rand(4)

    def forward(self, x):
        return x + self.constant + self.child.offset


@pytest.mark.parametrize('strategy', ['cpu', 'cuda', 'cuda_run_cpu_offload', 'reuse_cache'])
def test_tensor_attribute_names(monkeypatch, strategy):
    if strategy != 'cpu' and not torch.cuda.is_available():
        pytest.skip('requires CUDA')
    monkeypatch.setattr(CompileFlag, 'trace_strategy', strategy)
    model = TensorAttributeModel()
    dummy_input = {'x': torch.rand(4)}
    expected = model(**dummy_input)
    traced = to_fx_graph(model, dummy_input)
    targets = {node.target for node in traced.graph.nodes if node.op == 'get_attr'}
    assert targets == {'constant', 'child.offset'}
    assert torch.equal(traced(**dummy_input), expected)
