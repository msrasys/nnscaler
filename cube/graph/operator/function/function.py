import copy
from typing import List
import string

from cube.graph.operator import IRFwOperation
from cube.graph.operator.function.einops import EinDim, IREinops
from cube.ir.cten import IRTensor


class Linear(IREinops):
    """
    b * k, n k -> b * n
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

    def make_expression(self):
        expr = 'b * k, n k, n -> b * n'
        [idims, wdims, bdims], [odims] = self.parse(expr)
        if len(self.inputs(0).shape) == 2:
            idims = [idims[0], idims[2]]
            odims = [odims[0], odims[2]]
        else:
            extra_dims = list()
            num_extra_dim = len(self.inputs(0).shape) - 2
            dims = [c for c in string.ascii_lowercase if c not in 'bkn']
            for num in range(num_extra_dim):
                extra_dims.append(EinDim(dims[num]))
            idims = [idims[0]] + extra_dims + [idims[-1]]
            odims = [odims[0]] + extra_dims + [odims[-1]]
        self.set_input_ein(0, idims)
        self.set_input_ein(1, wdims)
        if self.inputs(2) is not None:
            self.set_input_ein(2, bdims)
        self.set_output_ein(0, odims)

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        linear = Linear(self.signature, inputs, self.name)
        for idx, output in enumerate(outputs):
            linear.set_output(idx, output)
        return linear


class BatchLinear(IREinops):
    """
    b m k, b k n -> b m n
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

    def make_expression(self):
        expr = 'b m k, b k n -> b m n'
        input_dims, output_dims = self.parse(expr)
        for idx, input_dim in enumerate(input_dims):
            self.set_input_ein(idx, input_dim)
        for idx, output_dim in enumerate(output_dims):
            self.set_output_ein(idx, output_dim)

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        bmm = BatchLinear(self.signature, inputs, self.name)
        for idx, output in enumerate(outputs):
            bmm.set_output(idx, output)
        return bmm


class ElementWise(IREinops):
    """
    *, _ -> *
    """

    def __init__(self, signature, inputs, name='elementwise', **kwargs):
        if len(inputs) != 2:
            raise TypeError(f"Expected 2 inputs but got {inputs}")
        super().__init__(
            name, signature,
            input_length=2,
            output_length=1
        )
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)

    def make_expression(self):
        """
        """
        dims = string.ascii_lowercase
        i1, i2 = self.inputs()
        if isinstance(i1, IRTensor) and isinstance(i2, IRTensor):
            shape1 = [EinDim(dims[d]) for d in range(len(i1.shape))]
            shape2 = [EinDim(dims[d]) for d in range(len(i2.shape))]
            if len(i1.shape) == len(i2.shape):
                for idx, (dim1, dim2) in enumerate(zip(i1.shape, i2.shape)):
                    if dim1 != dim2:
                        shape1[idx] = EinDim(str(dim1), EinDim.ReduceType.Stay)
                        shape2[idx] = EinDim(str(dim2), EinDim.ReduceType.Stay)
            else:
                if len(i1.shape) == 1:
                    shape1[0].name = str(i1.shape[0])
                elif len(i2.shape) == 1:
                    shape2[0].name = str(i2.shape[0])
            out_shape = shape1 if i1.nele() > i2.nele() else shape2
            self.set_input_ein(0, shape1)
            self.set_input_ein(1, shape2)
            self.set_output_ein(0, out_shape)
        else:
            if isinstance(i1, IRTensor):
                shape1 = [EinDim(dims[d]) for d in range(len(i1.shape))]
                self.set_input_ein(0, shape1)
                self.set_output_ein(0, shape1)
            elif isinstance(i2, IRTensor):
                shape2 = [EinDim(dims[d]) for d in range(len(i2.shape))]
                self.set_input_ein(1, shape2)
                self.set_output_ein(0, shape2)
            else:
                raise RuntimeError("both inputs {i1} and {i2} are not IRTensor")

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        elew = ElementWise(self.signature, inputs, self.name)
        for idx, output in enumerate(outputs):
            elew.set_output(idx, output)
        return elew


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

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        inputs = inputs = self.kwags['alpha']
        add = Add(self.signature, inputs, self.name)
        for idx, output in enumerate(outputs):
            add.set_output(idx, output)
        return add


class Sub(ElementWise):
    """
    torch.add
    """
    def __init__(self, signature, inputs, name='sub', **kwargs):
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

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        inputs = inputs = self.kwags['alpha']
        add = Sub(self.signature, inputs, self.name)
        for idx, output in enumerate(outputs):
            add.set_output(idx, output)
        return add


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


class Activation(IREinops):
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

    def make_expression(self):
        """
        * -> *
        """
        dims = string.ascii_lowercase
        dim1 = [EinDim(dims[d]) for d in range(len(self.inputs(0).shape))]
        self.set_input_ein(0, dim1)
        self.set_output_ein(0, dim1)

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        op = Activation(self.signature, inputs, self.name)
        for idx, output in enumerate(outputs):
            op.set_output(idx, output)
        return op


class GELU(Activation):
    """
    torch.nn.functional.gelu(input, approximate: bool = False)

    Note `approximate` argument is new at pytorch version v1.11
    """
    def __init__(self, signature, inputs, name='gelu', **kwargs):

        super().__init__(signature, [inputs[0]], name)
        if len(inputs) == 2:
            self.kwargs['approximate'] = inputs[1]
        self.set_input(0, inputs[0])

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        if 'approximate' in self.kwargs:
            inputs.append(self.kwargs['approximate'])
        op = GELU(self.signature, inputs, self.name)
        op.set_output(0, outputs[0])
        return op


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

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        inputs = inputs + [self.kwargs['p'], self.kwargs['training'], self.kwargs['inplace']]
        op = Dropout(self.signature, inputs, self.name)
        for idx, output in enumerate(outputs):
            op.set_output(idx, output)
        return op


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
    
    def make_expression(self):
        super().make_expression()
        dim = self.kwargs['dim']
        self._ieins[0][dim].reduce = EinDim.ReduceType.Stay

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        inputs = inputs + [self.kwargs['dim'], self.kwargs['_stacklevel'], self.kwargs['dtype']]
        op = Dropout(self.signature, inputs, self.name)
        for idx, output in enumerate(outputs):
            op.set_output(idx, output)
        return op

# ===================== Loss Computation (Reduce) =========================

class Sum(IREinops):
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

    def make_expression(self):
        """
        * -> 1         (no extra kwarg)
        a b c -> a c   (dim b)
        a b c -> a 1 c (dim b and keepdim)
        """
        reducedim = None if 'dim' not in self.kwargs else self.kwargs['dim']
        keepdim = False if 'keepdim' not in self.kwargs else self.kwargs['keepdim']
        input = self.inputs(0)
        dims = string.ascii_lowercase
        in_dim = [
            EinDim(dims[d]) for d in range(len(input.shape))]
        ou_dim = copy.copy(in_dim)
        if reducedim is not None:
            in_dim[reducedim].reduce = EinDim.ReduceType.Sum
            if keepdim:
                ou_dim[reducedim] = EinDim('1')
            else:
                ou_dim.pop(reducedim)
        else:
            for dim in in_dim:
                dim.reduce = EinDim.ReduceType.Sum
            ou_dim = [EinDim('1')]
        self.set_input_ein(0, in_dim)
        self.set_output_ein(0, ou_dim)

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        reducedim = None if 'dim' not in self.kwargs else self.kwargs['dim']
        keepdim = False if 'keepdim' not in self.kwargs else self.kwargs['keepdim']
        inputs += [reducedim]
        if reducedim is not None:
            if keepdim:
                inputs += [keepdim]
        sum_op = Sum(self.signature, inputs, self.name)
        for idx, output in enumerate(outputs):
            sum_op.set_output(idx, output)
        return sum_op

# ========================= Memory Operation ==========================

class Transpose(IREinops):
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

    def make_expression(self):
        """
        similar like a b c -> a c b
        """
        dims = string.ascii_lowercase
        dim0 = self.kwargs['dim0']
        dim1 = self.kwargs['dim1']
        input = self.inputs(0)
        in_dim = [EinDim(dims[d]) for d in range(len(input.shape))]
        ou_dim = copy.copy(in_dim)
        ou_dim[dim0], ou_dim[dim1] = in_dim[dim1], in_dim[dim0]
        self.set_input_ein(0, in_dim)
        self.set_output_ein(0, ou_dim)

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        dim0 = self.kwargs['dim0']
        dim1 = self.kwargs['dim1']
        inputs += [dim0, dim1]
        op = Transpose(self.signature, inputs, self.name)
        for idx, output in enumerate(outputs):
            op.set_output(idx, output)
        return op


class Conv2D(IREinops):
    """
    torch.conv2d(input, weight, bias, stride, padding, dialation, groups)
    https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html?highlight=torch%20conv2d#torch.nn.functional.conv2d
    """
    def __init__(self, signature, inputs, name='conv2d', **kwargs):
        if len(inputs) != 7:
            raise RuntimeError(f"expected 7 operators for conv2d but got {len(inputs)}")
        super().__init__(
            name, signature,
            input_length=3,
            output_length=1
        )
        for idx, input in enumerate(inputs[:3]):
            self.set_input(idx, input)
        self.kwargs['stride'] = inputs[3]
        self.kwargs['padding'] = inputs[4]
        self.kwargs['dilation'] = inputs[5]
        self.kwargs['groups'] = inputs[6]

    def make_expression(self):
        input = 'N I {iH} {iW}'
        weight = 'O {group_channel} {kH} {kW}'
        bias = 'O'
        output = 'N O {oH} {oW}'
        # parameters
        groups = self.kwargs['groups']
        stride = self.kwargs['stride']
        padding = self.kwargs['padding']
        dilation = self.kwargs['dilation']
        kH = self.inputs(1).shape[2]
        kW = self.inputs(1).shape[3]

        iH, iW = self.inputs(0).shape[2:4]
        group_channel = self.inputs(0).shape[2] // groups
        oH = (iH + 2 * padding[0] - dilation[0] * (kH - 1) - 1) // stride[0] + 1
        oW = (iH + 2 * padding[1] - dilation[1] * (kW - 1) - 1) // stride[1] + 1

        input = input.format(iH=iH, iW=iW)
        weight = weight.format(group_channel=group_channel, kH=kH, kW=kW)
        output = output.format(oH=oH, oW=oW)
        
        expr = f'{input}, {weight}, {bias} -> {output}'
        [idims, wdims, bdims], [odims] = self.parse(expr)
        self.set_input_ein(0, idims)
        self.set_input_ein(1, wdims)
        if self.inputs(2) is not None:
            self.set_input_ein(2, bdims)
        self.set_output_ein(0, odims)

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        groups = self.kwargs['groups']
        stride = self.kwargs['stride']
        padding = self.kwargs['padding']
        dilation = self.kwargs['dilation']
        inputs += [groups, stride, padding, dilation]
        op = Conv2D(self.signature, inputs, self.name)
        op.set_output(0, outputs[0])
        return op


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
            raise RuntimeError(f"Expected 6 inputs but got {inputs}")
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


class CubeComplexEmbedding(IRFwOperation):
    """
    Embedding
    """
    def __init__(self, signature, inputs, name='embedding', **kwargs):
        if len(inputs) != 4:
            raise RuntimeError(f"Expected 4 inputs but got {inputs}")
        input, weight = inputs[0], inputs[1]
        start, stop = inputs[2], inputs[3]
        super().__init__(
            name, signature,
            input_length = 2,
            output_length = 1
        )
        self.set_input(0, input)
        self.set_input(1, weight)
        self.kwargs['start'] = start
        self.kwargs['stop'] = stop

    def infer_shape(self):
        if self.inputs(0).shape is None or self.inputs(1).shape is None:
            return False
        self.outputs(0).shape = self.inputs(0).shape + [self.inputs(1).shape[1]]
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
