import copy

from cube.graph.operator import IRFwOperation


class Linear(IRFwOperation):
    """
    Input:
        input: [N, *, in_features]
        weight: [out_features, in_features]
        bias: [out_features,]
    
    Output:
        [N, *, in_features]
    """
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

# ============================= Elementwise ============================

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
        self.kwargs['alpha'] = alpha


class LayerNorm(IRFwOperation):

    def __init__(self, signature, inputs, name='layernorm', **kwargs):

        if len(inputs) != 5:
            raise TypeError(f"Expected 5 inputs, but got: {inputs}")
        input = inputs[0]
        normalized_shape = inputs[1]
        if not isinstance(normalized_shape, list):
            raise TypeError(f"Expected list of int, but got: {type(normalized_shape)}")
        weight = inputs[2]
        bias = inputs[3]
        eps = inputs[4]

        inputs = [input, normalized_shape, weight, bias]
        super().__init__(name, signature, input_length=4, output_length=1)
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        self.kwargs['eps'] = eps

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        self.outputs(0).shape = self.inputs(0).shape
        return True


# ============================= Activation ============================

class Activation(IRFwOperation):
    """
    functions like GELU, RELU, Dropout.

    Exclude softmax
    """

    def __init__(self, signature, inputs, name='activation', **kwargs):

        if len(inputs) != 1:
            raise TypeError("Expected single tensor input")

        super().__init__(
            name, signature,
            input_length=1,
            output_length=1
        )
        self.set_input(0, inputs[0])
        # this is for partitioning indicator
        self.stay_dims = list()

    def infer_shape(self):
        input = self.inputs(0)
        if input.shape is None:
            return False
        self._outputs[0].shape = input.shape
        return True


class Dropout(Activation):
    """
    torch.nn.functional.dropout
    """
    def __init__(self, signature, inputs, name='dropout', **kwargs):

        if len(inputs) != 4:
            raise TypeError(f"Expected 4 inputs but got {inputs}")
        super().__init__(signature, [inputs[0]], name)
        self.set_input(0, inputs[0])
        self.kwargs['p'] = inputs[1]
        self.kwargs['training'] = inputs[2]
        self.kwargs['inplace'] = inputs[3]


class Softmax(Activation):

    def __init__(self, signature, inputs, name='softmax', **kwargs):
        
        if len(inputs) != 4:
            raise TypeError(f"Expected 4 inputs, but got: {inputs}")
        
        tensor, dim, stacklevel, dtype = inputs[0], inputs[1], inputs[2], inputs[3]
        super().__init__(signature, inputs=[inputs[0]], name=name)
        self.set_input(0, tensor)
        self.kwargs['dim'] = dim
        self.kwargs['_stacklevel'] = stacklevel
        self.kwargs['dtype'] = dtype
        self.stay_dims.append(dim)


# ===================== Loss Computation (Reduce) =========================

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

        # change dim to positive value
        ndim = len(self.inputs(0).shape)
        if 'dim' in self.kwargs:
            dim = self.kwargs['dim']
            dim = ndim + dim if dim < 0 else dim
            self.kwargs['dim'] = dim
            reduce_dims = [dim]
        else:
            reduce_dims = list(range(ndim))

        if 'keepdim' in self.kwargs:
            keepdim = self.kwargs['keepdim']
        else:
            keepdim = False

        shape = list()
        for dim, nele in enumerate(self.inputs(0).shape):
            if dim in reduce_dims:
                if keepdim:
                    shape.append(1)
            else:
                shape.append(nele)
        if len(shape) == 0:
            shape = [1]
        self._outputs[0].shape = shape
        return True

# ========================= Memory Operation ==========================

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
        ndim = len(self.inputs(0).shape)
        dim0 = self.kwargs['dim0']
        if dim0 < 0:
            dim0 = ndim + dim0
            self.kwargs['dim0'] = dim0
        dim1 = self.kwargs['dim1']
        if dim1 < 0:
            dim1 = ndim + dim1
            self.kwargs['dim1'] = dim1
        shape = list(self.inputs(0).shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        self._outputs[0].shape = shape
        return True

# ===================== Cube Complex Operation =======================

class CubeComplexToQKV(IRFwOperation):
    """
    Inputs:
        hidden_state: [L, N, E]
        weight: [3 * (num_head * dim_head), E]
        num_head: int
        dim_head: int 

    where L = sequence length, N = batch size, E = num_head * dim_head

    Returns:
        Q: [L, N * num_head, dim_head]
        K: [L, N * num_head, dim_head]
        V: [L, N * num_head, dim_head]
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
        self.kwargs['num_head'] = inputs[2]

    def infer_shape(self):
        if self.inputs(0).shape is None or self.inputs(1) is None:
            return False
        seqlen = self.inputs(0).shape[0]
        bs = self.inputs(0).shape[1]
        num_head = self.kwargs['num_head']
        dim_head = self.inputs(1).shape[0] // 3 // num_head

        shape = [seqlen, bs * num_head, dim_head]
        for output in self.outputs():
            output.shape = shape
        return True


class CubeComplexTrilMask(IRFwOperation):
    """
    Inputs:
        input: [N * num_head, L, L]
        num_head: int
    
    Returns:
        output: [N * num_head, L, L]
    """
    def __init__(self, signature, inputs, name='trilmask', **kwargs):
        if len(inputs) != 2:
            raise TypeError("Expected 2 input")
        tensor, num_head = inputs[0], inputs[1]
        super().__init__(
            name, signature,
            input_length=1,
            output_length=1
        )
        self.set_input(0, tensor)
        self.kwargs['num_head'] = num_head

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        self._outputs[0].shape = self.inputs(0).shape
        return True


class CubeComplexAttnView(IRFwOperation):
    """
    Inputs:
        [N * num_head, L, dim_head]

    Outputs:
        [L, N, num_head * dim_head]
    """
    def __init__(self, signature, inputs, name='attn_view', **kwargs):
        if len(inputs) != 2:
            raise TypeError("Expected 2 input")
        tensor, num_head = inputs[0], inputs[1]
        super().__init__(
            name, signature,
            input_length=1,
            output_length=1
        )
        self.set_input(0, tensor)
        self.kwargs['num_head'] = num_head

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        num_head = self.kwargs['num_head']
        bs = self.inputs(0).shape[0] // num_head
        seqlen = self.inputs(0).shape[1]
        dim_head = self.inputs(0).shape[2]
        shape = [seqlen, bs, num_head * dim_head]
        self._outputs[0].shape = shape
        return True


class CubeComplexSelfAttention(IRFwOperation):
    """
    Multi-Head Self-Attention.

    L: sequence length
    N: batch size
    E: embedding size
    
    Inputs:
        hidden_state: [L, N, E]
        w_qkv       : [3 * num_head * dim_head, E]
        w_out       : [E, E]
        num_head: int
        dim_head: int
        dropout_p: float

    Outputs:
        hidden_state: [L, N, E]
    """
    def __init__(self, signature, inputs, name='selfattn', **kwargs):
        if len(inputs) != 6:
            raise RuntimeError(f"Expected 6 inputs but got {input}")
        num_head: int = inputs[3]
        dim_head: int = inputs[4]
        dropout_p: float = inputs[5]
        super().__init__(
            name, signature,
            input_length = 3,
            output_length = 1
        )
        for idx, tensor in enumerate(inputs[:3]):
            self.set_input(idx, tensor)
        self.kwargs['num_head'] = num_head
        self.kwargs['dim_head'] = dim_head
        self.kwargs['dropout_p'] = dropout_p

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        self.outputs(0).shape = self.inputs(0).shape
        return True


class CubeComplexFeedForward(IRFwOperation):
    """
    FeedForward

    Inputs:
        hidden_state: [L, N, E]
        w_proj1: [4 * E, E]
        w_bias1: [4 * E,]
        w_porj2: [E, 4 * E]
        w_bias2: [E,]

    Outputs:
        hidden_state: [L, N, E]
    """
    def __init__(self, signature, inputs, name='selfattn', **kwargs):
        if len(inputs) != 5:
            raise RuntimeError(f"Expected 6 inputs but got {input}")
        super().__init__(
            name, signature,
            input_length = 5,
            output_length = 1
        )
        for idx, tensor in enumerate(inputs):
            self.set_input(idx, tensor)

    def infer_shape(self):
        if self.inputs(0).shape is None:
            return False
        self.outputs(0).shape = self.inputs(0).shape
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
