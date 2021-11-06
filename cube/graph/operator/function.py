import copy

from cube.graph.operator import IRFwOperation
from cube.ir.cten import IRTensor


class Linear(IRFwOperation):

    def __init__(self, signature, inputs, name='linear', **kwargs):

        input, weight, bias = inputs
        super().__init__(
            name, signature,
            input_length=3,
            output_length=1
        )
        self.set_input(0, input)
        self.set_input(1, weight)
        self.set_input(2, bias)

    def infer_shape(self):
        """
        input:  [(D), M, K]
        weight: [N, K]
        bias:   [N,]
        """
        if len(self.inputs(0).shape) != 0 and len(self.inputs(1).shape) != 0:
            shape = self.inputs(0).shape[:-1] + self.inputs(1).shape[:1]
            self._outputs[0].shape = shape
            return True
        return False


class ElementWise(IRFwOperation):
    """
    Functions like torch.add (tensor1 + tensor2 / scaler)
    """

    def __init__(self, signature, inputs, name='elementwise', **kwargs):

        super().__init__(
            name, signature,
            input_length=len(inputs),
            output_length=1
        )
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)

    def infer_shape(self):
        for input in self.inputs():
            if isinstance(input, IRTensor):
                if len(input.shape) != 0:
                    self._outputs[0].shape = copy.copy(input.shape)
                    return True
                return False
        return False


class ElementWiseActivation(IRFwOperation):
    """
    functions like GELU, RELU, Dropout.

    Exclude softmax
    """

    def __init__(self, signature, inputs, name='elementwise_activation', **kwargs):

        super().__init__(
            name, signature,
            input_length=len(inputs),
            output_length=1
        )
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)

    def infer_shape(self):
        for input in self.inputs():
            if isinstance(input, IRTensor):
                if len(input.shape) != 0:
                    self._outputs[0].shape = copy.copy(input.shape)
                    return True
                return False
        return False


class Reduce(IRFwOperation):
    """
    functions like sum, mean, cross_entropy
    """
    def __init__(self, signature, inputs, name='reduce', **kwargs):
        super().__init__(
            name, signature,
            input_length=len(inputs),
            output_length=1
        )
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)

    def infer_shape(self):
        self._outputs[0].shape = [1]
        return True


class UnkownOperator(IRFwOperation):

    def __init__(self, signature, inputs, name='unknown_op', n_output=None):

        super().__init__(
            name, signature=signature,
            input_length=len(inputs),
            output_length=n_output,
        )
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)

    def infer_shape(self):
        return False
