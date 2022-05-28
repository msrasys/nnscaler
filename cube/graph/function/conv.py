from typing import List

from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRTensor


class IRConv2D(IRFwOperation):

    def __init__(self, signature: str, inputs: List[IRTensor], name: str,
                 **kwargs):
        signature = 'cube.runtime.function.conv2d'
        assert len(inputs) == 3, "Expected only input, weight, bias as inputs"
        assert len(kwargs) == 4, "Expected 4 kwargs: stride, padding, dialation, groups"
        super().__init__(name, signature, 3, 1)
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        self.kwargs.update(kwargs)

    def infer_shape(self) -> bool:
        """
        Output shape inference given the input shapes
        """
        if len(self.inputs(0).shape) == 0 or len(self.inputs(1).shape) == 0:
            return False
        N = self.inputs(0).shape[0]
        iH, iW = self.inputs(0).shape[2:4]
        oC = self.inputs(1).shape[0]
        stride = self.kwargs['stride']
        padding = self.kwargs['padding']
        dilation = self.kwargs['dilation']
        dH = self.inputs(1).shape[2]
        dW = self.inputs(1).shape[3]
        oH = (iH + padding[0] + padding[1] - dilation[0] * (dH - 1) - 1) // stride[0] + 1
        oW = (iW + padding[2] + padding[3] - dilation[1] * (dW - 1) - 1) // stride[1] + 1
        shape = [N, oC, oH, oW]
        self.outputs(0).shape = shape
        return True

    def new(self, inputs: List, outputs: List):
        """
        construct a new operator sharing same kwargs with new inputs
        and outputs
        """
        stride = self.kwargs['stride']
        padding = self.kwargs['padding']
        dilation = self.kwargs['dilation']
        groups = self.kwargs['groups']
        op = IRConv2D(self.signature, inputs, self.name,
                      stride=stride, padding=padding, dilation=dilation, groups=groups)
        assert len(outputs) == 1
        op.set_output(0, outputs[0])
        op.infer_shape()
        return op



class IRConv3D(IRFwOperation):

    def __init__(self, signature: str, inputs: List[IRTensor], name: str,
                 **kwargs):
        signature = 'cube.runtime.function.conv3d'
        assert len(inputs) == 3, "Expected only input, weight, bias as inputs"
        assert len(kwargs) == 4, "Expected 4 kwargs: stride, padding, dialation, groups"
        super().__init__(name, signature, 3, 1)
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        self.kwargs.update(kwargs)

    def infer_shape(self) -> bool:
        """
        Output shape inference given the input shapes
        """
        if len(self.inputs(0).shape) == 0 or len(self.inputs(1).shape) == 0:
            return False
        N = self.inputs(0).shape[0]
        iC = self.inputs(0).shape[1]
        iT, iH, iW = self.inputs(0).shape[2:5]

        oC = self.inputs(1).shape[0]
        stride = self.kwargs['stride']
        padding = self.kwargs['padding']
        dilation = self.kwargs['dilation']
        dT = self.inputs(1).shape[2]
        dH = self.inputs(1).shape[3]
        dW = self.inputs(1).shape[4]

        oT = (iT + 2 * padding[0] - dilation[0] * (dT - 1) - 1) // stride[0] + 1
        oH = (iH + 2 * padding[1] - dilation[1] * (dH - 1) - 1) // stride[1] + 1
        oW = (iW + 2 * padding[2] - dilation[2] * (dW - 1) - 1) // stride[2] + 1
        shape = [N, oC, oT, oH, oW]

        self.outputs(0).shape = shape
        return True

    def new(self, inputs: List, outputs: List):
        """
        construct a new operator sharing same kwargs with new inputs
        and outputs
        """
        stride = self.kwargs['stride']
        padding = self.kwargs['padding']
        dilation = self.kwargs['dilation']
        groups = self.kwargs['groups']
        op = IRConv3D(self.signature, inputs, self.name,
                      stride=stride, padding=padding, dilation=dilation, groups=groups)
        assert len(outputs) == 1
        op.set_output(0, outputs[0])
        op.infer_shape()
        return op