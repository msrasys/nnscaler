from cube.operator.holist.generics import GenericHolisticOp

import cube.operator.physic as physic_op

from cube.tensor.logic.tensor import LogicalTensor
import cube.tensor.logic.segment.outline as outline
from cube.tensor.community import Community

# expert space to declare all kinds of holistic operators


__all__ = ['kHolistLinearSets']


class LinearColumnWeight(GenericHolisticOp):
    """
    Perform Y = XW + b -> Y = X[W1,W2] + [b1,b2]
    Split W and b on the last dimension
    """

    def __init__(self):

        # TODO
        inputs_layout = outline.Full
        # TODO
        weight_layout = outline.SplitAxis(axis=0, chunk_num=None, overlap=0)
        # TODO
        bias_layout = outline.SplitAxis(axis=0, chunk_num=None, overlap=0)
        # TODO
        output_layout = weight_layout

        super().__init__(
            input_layout=[inputs_layout, weight_layout, bias_layout],
            output_layout=[output_layout,]
        )

    def forward(self, inputs, weight, bias):
        outputs = list()
        # TODO: handle bias is None
        for pw, pb in zip(weight, bias):
            output = physic_op.linear(inputs, weight, bias)
            outputs.append(outputs)
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

        # TODO
        inputs_layout = outline.SplitAxis(axis=-1, chunk_num=None, overlap=0)
        # TODO
        align = outline.Align(inputs_layout.chunk_num)
        weight_layout = outline.SplitAxis(axis=1, chunk_num=align, overlap=0)
        # TODO
        bias_layout = outline.Full(reduce=ReductionOpPool.Sum)
        # TODO
        output_layout = outline.Full(reduce=ReductionOpPool.Sum)

        super().__init__(
            input_layout=[inputs_layout, weight_layout, bias_layout],
            output_layout=[output_layout,]
        )
    
    def forward(self, inputs, weight, bias):
        output = physic_op.linear(inputs, weight, bias)
        return [output]


kHolistLinearSets = [LinearColumnWeight(), LinearColumnInputRowWeight()]