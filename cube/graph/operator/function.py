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
        if self.inputs(0).shape is None or self.inputs(1).shape is None:
            return False
        shape = self.inputs(0).shape[:-1] + self.inputs(1).shape[:1]
        self._outputs[0].shape = shape
        return True


class BatchLinear(IRFwOperation):
    """
    Inputs:
        input1: [B, N, M]
        input2: [B, M, P]

    Outputs:
        output: [B, N, P]
    """

    def __init__(self, signature, inputs, name='bmm', **kwargs):

        if len(inputs) != 2:
            raise TypeError(f"Requires 2 inputs. But got {inputs}")
        input1, input2 = inputs
        super().__init__(
            name, signature,
            input_length=2,
            output_length=1
        )
        self.set_input(0, input1)
        self.set_input(1, input2)

    def infer_shape(self):
        if self.inputs(0).shape is None or self.inputs(1).shape is None:
            return False
        b1, n1, m1 = self.inputs(0).shape
        b2, m2, p2 = self.inputs(1).shape
        if m1 != m2 or b1 != b2:
            raise RuntimeError("Unmatch {b1} != {b2} or {m1} != {m2}")
        shape = [b1, n1, p2]
        self._outputs[0].shape = shape
        return True


class ElementWise(IRFwOperation):
    """
    Functions like torch.add, torch.mul, torch.sub, etc.
    """

    def __init__(self, signature, inputs, name='elementwise', **kwargs):
        """
        Inputs:
            inputs[0]: IRTensor
            inputs[1]: other (IRTensor or Number)
        Outputs:
            same shape as inputs[0]
        """

        if len(inputs) != 2:
            raise TypeError(f"Expected 2 inputs but got {inputs}")
        super().__init__(
            name, signature,
            input_length=2,
            output_length=1
        )
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        shape = copy.copy(self.inputs(0).shape)
        self._outputs[0].shape = shape
        return True


class Add(ElementWise):
    """
    torch.add
    """
    def __init__(self, signature, inputs, name='add', **kwargs):
        """
        Inputs:
            inputs[0]: IRTensor
            inputs[1]: other (IRTensor or Number)
            inputs[2]: alpha (Number)
        Outputs:
            same shape as inputs[0]
        """
        if len(inputs) != 3:
            raise TypeError(
                f"Add expected 3 inputs: [tensor, other, alpha], but got {inputs}"
            )
        super().__init__(signature, inputs[:2], name=name)
        alpha = inputs[2]
        if alpha != 1:
            self.kwargs['alpha'] = alpha


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


class Sum(IRFwOperation):
    """
    torch.sum
    """
    def __init__(self, signature, inputs, name='sum', **kwargs):

        if len(inputs) <= 1:
            raise TypeError(f"Expected at least 2 inputs, but got {inputs}")
        if inputs[1] is not None and not isinstance(inputs[1], int):
            raise TypeError(f"Expected inputs[1] to be None or int, but got {type(inputs[1])}")

        super().__init__(
            name, signature,
            input_length=1,
            output_length=1
        )
        self.set_input(0, inputs[0])
        if inputs[1] is not None:
            self.kwargs['dim'] = inputs[1]
            if len(inputs) > 2:
                self.kwargs['keepdim'] = inputs[2]
            else:
                self.kwargs['keepdim'] = False

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        shape = list()
        if 'dim' in self.kwargs:
            dim = [self.kwargs['dim']]
            keepdim = self.kwargs['keepdim']
            for idx, nele in enumerate(self.inputs(0).shape):
                if idx in dim:
                    if not keepdim:
                        continue
                    nele = 1
                shape.append(nele)
        else:
            shape = [1]
        self._outputs[0].shape = shape
        return True


class Softmax(IRFwOperation):

    def __init__(self, signature, inputs, name='softmax', **kwargs):
        
        if len(inputs) != 4:
            raise TypeError(f"Expected 4 inputs, but got: {inputs}")
        
        tensor, dim, stacklevel, dtype = inputs[0], inputs[1], inputs[2], inputs[3]
        super().__init__(
            name, signature, input_length=1, output_length=1
        )
        self.set_input(0, tensor)
        self.kwargs['dim'] = dim
        self.kwargs['_stacklevel'] = stacklevel
        self.kwargs['dtype'] = dtype

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        dim = self.kwargs['dim']
        shape = [
            nele for idx, nele in enumerate(self.inputs(0).shape) if idx != dim
        ]
        self._outputs[0].shape = shape
        return True


class Transpose(IRFwOperation):
    """
    torch.transpose
    """
    def __init__(self, signature, inputs, name='transpose', **kwargs):

        if len(inputs) != 3:
            raise RuntimeError("expected 3 inputs <tensor, dim1, dim2>")

        if not isinstance(inputs[1], int):
            raise TypeError(f"Expected 1st input: int, but got {type(inputs[1])}")
        if not isinstance(inputs[2], int):
            raise TypeError(f"Expected 1st input: int, but got {type(inputs[2])}")

        super().__init__(
            name, signature,
            input_length=1,
            output_length=1
        )
        self.set_input(0, inputs[0])
        self.kwargs['dim0'] = inputs[1]
        self.kwargs['dim1'] = inputs[2]

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        dim1 = self.kwargs['dim0']
        dim2 = self.kwargs['dim1']
        shape = copy.copy(list(self.inputs(0).shape))
        shape[dim1], shape[dim2] = shape[dim2], shape[dim1]
        self._outputs[0].shape = shape
        return True

# ===================== Cube Complex Operation =======================

class CubeComplexToQKV(IRFwOperation):
    """
    Inputs:
        hidden_state: [L, N, E]
        weight: [3 * (num_heads * dim_head), E]
        num_heads: int

    where L = sequence length, N = batch size, E = num_heads * dim_head

    Returns:
        Q: [L, N * num_heads, dim_head]
        K: [L, N * num_heads, dim_head]
        V: [L, N * num_heads, dim_head]
    """
    def __init__(self, signature, inputs, name='toqkv', **kwargs):
        if len(inputs) != 3:
            raise TypeError(f"Expected 3 arguments but goit {inputs}")
        qkv, weight = inputs[0], inputs[1]
        super().__init__(
            name, signature,
            input_length=2,
            output_length=3
        )
        self.set_input(0, qkv)
        self.set_input(1, weight)
        self.kwargs['num_heads'] = inputs[2]

    def infer_shape(self):
        if self.inputs(0).shape is None or self.inputs(1) is None:
            return False
        seqlen = self.inputs(0).shape[0]
        bs = self.inputs(0).shape[1]
        num_heads = self.kwargs['num_heads']
        dim_head = self.inputs(1).shape[0] // 3 // num_heads

        shape = [seqlen, bs * num_heads, dim_head]
        for output in self.outputs():
            output.shape = shape
        return True


class CubeComplexTrilMask(IRFwOperation):
    """
    Inputs:
        input: [N * num_heads, L, L]
        num_head: int
    
    Returns:
        output: [N * num_heads, L, L]
    """
    def __init__(self, signature, inputs, name='trilmask', **kwargs):
        if len(inputs) != 2:
            raise TypeError("Expected 2 input")
        tensor, num_heads = inputs[0], inputs[1]
        super().__init__(
            name, signature,
            input_length=1,
            output_length=1
        )
        self.set_input(0, tensor)
        self.kwargs['num_heads'] = num_heads

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        self._outputs[0].shape = self.inputs(0).shape
        return True


class CubeComplexAttnView(IRFwOperation):
    """
    Inputs:
        [N * num_heads, L, dim_head]

    Outputs:
        [L, N, num_heads * dim_head]
    """
    def __init__(self, signature, inputs, name='attn_view', **kwargs):
        if len(inputs) != 2:
            raise TypeError("Expected 2 input")
        tensor, num_heads = inputs[0], inputs[1]
        super().__init__(
            name, signature,
            input_length=1,
            output_length=1
        )
        self.set_input(0, tensor)
        self.kwargs['num_heads'] = num_heads

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        num_heads = self.kwargs['num_heads']
        bs = self.inputs(0).shape[0] // num_heads
        seqlen = self.inputs(0).shape[1]
        dim_head = self.inputs(0).shape[2]
        shape = [seqlen, bs, num_heads * dim_head]
        self._outputs[0].shape = shape
        return True


class UnkownOperator(IRFwOperation):

    def __init__(self, signature, inputs, name='unknown_op', n_outputs=None):

        super().__init__(
            name, signature=signature,
            input_length=len(inputs),
            output_length=n_outputs,
        )
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)

    def infer_shape(self):
        return False
