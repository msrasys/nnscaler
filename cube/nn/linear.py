import torch
from cube.tensor.logic.tensor import LogicalTensor
import cube.operator.logic as logic_op

import math

from torch import nn

class Linear(nn.Module):

    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = LogicalTensor((out_features, in_features))
        if bias:
            self.bias = LogicalTensor((out_features,))
        self.reset_parameters()
        # Actually here we can pass shapes
        self.op = logic_op.Linear()

    def reset_parameters(self) -> None:
        pass

    def forward(self, input: LogicalTensor) -> LogicalTensor:
        return self.op(input, self.weight, self.bias)
