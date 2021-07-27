from cube.operator.holist.generics import GenericHolisticOp

import cube.operator.physic.linear as phy_linear

from cube.tensor.logic.tensor import LogicalTensor
import cube.tensor.logic.segment as sg
from cube.tensor.community import Community

# Debug
from cube.device.physic.group import DeviceGroup
import torch

# expert space to declare all kinds of holistic operators


__all__ = ['kHolistLinearSets']


class LinearColumnWeight(GenericHolisticOp):
    """
    Perform Y = XW + b -> Y = X[W1,W2] + [b1,b2]
    Split W and b on the last dimension
    """

    def __init__(self):

        inputs_layout = sg.outline.Full(
            reduction=sg.ReductionOp.Replica
        )

        weight_layout = sg.outline.SplitAxis(
            axis=0, chunk_num=None, overlap=0, uniform=False,
            reduction=sg.ReductionOp.Replica
        )

        bias_layout = weight_layout

        output_layout = sg.outline.SplitAxis(
            axis=1, chunk_num=weight_layout.chunk_num, overlap=0, uniform=False,
            reduction=sg.ReductionOp.Replica
        )

        super().__init__(
            input_layout=[inputs_layout, weight_layout, bias_layout],
            output_layout=[output_layout,],
            input_format=[None, None, None],
            output_format=[None]
        )

    def forward(self, input, weight, bias):
        outputs = list()
        # TODO: handle bias is None
        physical_input = input.get_physical_tensor(0)
        for cid in range(len(weight)):
            # output = physic_op.linear(inputs, weight, bias)
            #TODO: TensorContainer to enable op placement + tensor movement
            #TODO: ExecutionScheduler to handle re-compute / swap
            #TODO: nested hybrid call to enable hybrid-parallelisms
            #TODO: double-check necessety of stateful physical operator
            physical_weight = weight.get_physical_tensor(cid)
            # if DeviceGroup().rank == 0:
            #     print(physical_weight)
            physical_bias = bias.get_physical_tensor(cid)
            # TODO: this is the policy decision
            phy_op = phy_linear.Linear(placement=weight.get_community(cid).placement)
            output = phy_op(physical_input, physical_weight, physical_bias)
            # if DeviceGroup().rank == 0:
            #     print(output)
            outputs.append(output)
        return outputs


class LinearColumnInputRowWeight(GenericHolisticOp):
    """
    Perform 
        Y = XW + b 
            -> Y = [X1,X2] * [W1//W2] + b]
            -> Y = X1W1 + X2W2 + b
    Split X (inputs) in column major (last dim),
    Split W (weights) in row major (first dim)
    """

    def __init__(self):

        inputs_layout = sg.outline.SplitAxis(
            axis=-1, chunk_num=None, overlap=0,
            reduction=sg.ReductionOp.Replica)

        weight_layout = sg.outline.SplitAxis(
            axis=1, chunk_num=inputs_layout.chunk_num, overlap=0,
            reduction=sg.ReductionOp.Replica)

        bias_layout = sg.outline.Full(reduction=sg.ReductionOp.Sum)

        output_layout = sg.outline.Full(reduction=sg.ReductionOp.Sum)

        super().__init__(
            input_layout=[inputs_layout, weight_layout, bias_layout],
            output_layout=[output_layout,],
            input_format=[None, None, None],
            output_format=[None, None, None]
        )
    
    def forward(self, inputs, weight, bias):
        output = physic_op.linear(inputs, weight, bias)
        return [output,]


kHolistLinearSets = [LinearColumnWeight(), LinearColumnInputRowWeight()]