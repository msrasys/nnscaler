from cube.operator.physic.generics import GenericPhysicOp

import torch


class Linear(GenericPhysicOp):
    """
    Apply matmul: Out = input * weight^T + bias
    """

    def __init__(self, placement=None):
        super().__init__(torch._C._nn.linear, placement)
