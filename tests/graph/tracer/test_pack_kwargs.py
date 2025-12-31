#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.

import torch
from torch.nn import Module

from nnscaler.graph.tracer import concrete_trace
from ...utils import replace_all_device_with


class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, **kwargs):
        return self.linear(kwargs['input'])


@replace_all_device_with('cpu')
def test_pack_kwargs():
    model = Model()
    example_inputs = {'input': torch.randn(1, 10)}
    traced_model = concrete_trace(model, example_inputs)
    assert list(traced_model.graph.nodes)[0].target == '**kwargs'


@replace_all_device_with('cpu')
def test_direct_kwargs():
    model = Model()
    example_inputs = {'**kwargs': {'input': torch.randn(1, 10)}}
    traced_model = concrete_trace(model, example_inputs)
    assert list(traced_model.graph.nodes)[0].target == '**kwargs'
