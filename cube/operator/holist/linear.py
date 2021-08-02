from cube.operator.holist.generics import GenericHolisticOp

import cube.operator.physic.linear as phy_linear
from cube.operator.physic.comm.mapreduce import PartialSum

from cube.tensor.logic.tensor import LogicalTensor
import cube.tensor.logic.outline as outline

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

    def __init__(self, shapes):

        super().__init__(shapes)

        # input layouts
        input_layout = outline.Full(
            self.solver, self.shapes[0]
        )
        weight_layout = outline.SplitAxis(
            self.solver, self.shapes[1],
            axis=0, chunk_num=None, overlap=0
        )
        bias_layout = outline.SplitAxis(
            self.solver, self.shapes[2],
            axis=0, chunk_num=weight_layout.chunk_num, overlap=0
        )
        # output layouts
        output_layout = outline.SplitAxis(
            self.solver, self.shapes[3],
            axis=1, chunk_num=weight_layout.chunk_num, overlap=0
        )

        self.set_input_layouts([input_layout, weight_layout, bias_layout])
        self.set_output_layouts([output_layout])

    def forward(self, input, weight, bias):
        """
        input: list[Segment] of input
        weight: list[Segment] of weight
        bias: list[Segment] of bias
        """
        outputs = list()
        # TODO: handle bias is None
        physical_input = input[0].get_physical_tensor()
        for weight_seg, bias_seg in zip(weight, bias):
            # output = physic_op.linear(inputs, weight, bias)
            #TODO: TensorContainer to enable op placement + tensor movement
            #TODO: ExecutionScheduler to handle re-compute / swap
            #TODO: nested hybrid call to enable hybrid-parallelisms
            #TODO: double-check necessety of stateful physical operator
            physical_weight = weight_seg.get_physical_tensor()
            # if DeviceGroup().rank == 0:
            #     print(physical_weight)
            physical_bias = bias_seg.get_physical_tensor()
            # TODO: this is the policy decision
            phy_op = phy_linear.Linear(placement=weight_seg.placement)
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

    def __init__(self, shapes):

        super().__init__(shapes)

        input_layout = outline.SplitAxis(
            self.solver, self.shapes,
            axis=-1, chunk_num=None, overlap=0,
        )

        weight_layout = outline.SplitAxis(
            self.solver, self.shapes,
            axis=1, chunk_num=input_layout.chunk_num, overlap=0,
        )

        bias_layout = outline.SplitValue(
            self.solver, self.shapes,
            chunk_num=input_layout.chunk_num,
            val_op=PartialSum
        )

        # output layout will only use reduce op
        output_layout = outline.SplitValue(
            self.solver, self.shapes,
            chunk_num=input_layout.chunk_num,
            val_op=PartialSum
        )

        self.set_input_layouts([input_layout, weight_layout, bias_layout])
        self.set_output_layouts([output_layout])
    
    def forward(self, input, weight, bias):
        outputs = list()
        for input_seg, weight_seg, bias_seg in zip(input, weight, bias):
            phy_op = phy_linear.Linear(placement=weight_seg.placement)
            output = phy_op(
                input_seg.get_physical_tensor(), 
                weight.get_physical_tensor(), 
                bias.get_physical_tensor()
            )
            outputs.append(output)
        return outputs


kHolistLinearSets = [LinearColumnWeight, LinearColumnInputRowWeight]