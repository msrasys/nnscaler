from combo.operator.holist.generics import GenericHolisticOp

import combo.operator.physic as physic_op

from combo.tensor.logic.tensor import LogicalTensor
from combo.tensor.logic.segment import TileSegment
from combo.tensor.community import Community

# expert space to declare all kinds of holistic operators

__all__ = ['kHolistLinearSets']

class LinearColumnWeight(GenericHolisticOp):
    """
    Perform Y = XW + b -> Y = X[W1,W2] + [b1,b2]
    Split W and b on the last dimension
    """

    def __init__(self):

        # TODO
        inputs_layout = None
        # TODO
        weight_layout = None
        # TODO
        bias_layout = None
        # TODO
        output_layout = None

        super().__init__(
            input_layout=(inputs_layout, weight_layout),
            output_layout=(output_layout,)
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
        inputs_layout = None
        # TODO
        weight_layout = None
        # TODO
        bias_layout = None
        # TODO
        output_layout = None

        super().__init__(
            input_layout=(inputs_layout, weight_layout),
            output_layout=(output_layout,)
        )
    
    def forward(self, inputs, weight, bias):
        #TODO: semantic errors on bias
        output = physic_op.linear(inputs, weight, bias)
        return [output]


kHolistLinearSets = [LinearColumnWeight(), LinearColumnInputRowWeight()]