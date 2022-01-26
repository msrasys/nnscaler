from cube.graph.operator.function.einops import EinDim, EinopAnno, IREinops
from cube.graph.operator.function.conv import IRConv2D


def Linear(signature, inputs):
    annos = [
        'b * k+, n k+ -> b * n',   # no bias
        'b * k+, n k+, n -> b * n' # have bias
    ]
    return IREinops(signature, annos, inputs, 'linear')


def BatchLinear(signature, inputs):
    annos = [
        'b m k, b k n -> b m n'
    ]
    return IREinops(signature, annos, inputs, 'bmm')


def Add(signature, inputs):
    assert len(inputs) == 3
    inputs, alpha = inputs[0:2], inputs[2]
    # TODO: support broadcast
    annos = [
        '*, 1 -> *',
        '1, * -> *',
        '*, * -> *',
    ]
    return IREinops(signature, annos, inputs, 'add', alpha=alpha)


def Sub(signature, inputs):
    assert len(inputs) == 3
    inputs, alpha = inputs[0:2], inputs[2]
    # TODO: support broadcast
    annos = [
        '*, 1 -> *',
        '1, * -> *',
        '*, * -> *',
    ]
    return IREinops(signature, annos, inputs, 'sub', alpha=alpha)


def Mul(signature, inputs):
    annos = [
        '*, 1 -> *',
        '1, * -> *',
        '*, * -> *',
    ]
    return IREinops(signature, annos, inputs, 'mul')


def Div(signature, inputs):
    annos = [
        '*, 1 -> *',
        '1, * -> *',
        '*, * -> *',
    ]
    return IREinops(signature, annos, inputs, 'div')


def GeLU(signature, inputs):
    annos = ['* -> *']
    tensor = inputs[0:1]
    if len(inputs) == 2:
        # adapt for newest pytorch version
        approximate = inputs[1]
        return IREinops(signature, annos, tensor, 'gelu',
                        approximate=approximate)
    else:
        return IREinops(signature, annos, tensor, 'gelu')


def Softmax(signature, inputs):
    annos = ['* -> *']
    tensor = inputs[0:1]
    dim, _stacklevel, dtype = inputs[1], inputs[2], inputs[3]
    return IREinops(signature, annos, tensor, 'softmax',
                    dim=dim, _stacklevel=_stacklevel, dtype=dtype)


def Dropout(signature, inputs):
    annos = [
        '* -> *'
    ]
    tensor = inputs[0:1]
    p, training, inplace = inputs[1], inputs[2], inputs[3]
    return IREinops(signature, annos, tensor, 'dropout',
                    p=p, traning=training, inplace=inplace)


def Sum(signature, inputs):
    # TODO: support dim reduction
    annos = [
        '* -> 1',
    ]
    tensor = inputs[0:1]
    dim = inputs[1]
    if dim is not None:
        keepdim = inputs[2] if len(inputs) > 2 else False
        return IREinops(signature, annos, tensor, 'sum',
                        dim=dim, keepdim=keepdim)
    else:
        return IREinops(signature, annos, tensor, 'sum')


def Transpose(signature, inputs):
    def adapt(anno: EinopAnno, node: IREinops) -> EinopAnno:
        dim0, dim1 = node.kwargs[0], node.kwargs[1]
        anno.outputs[0][dim0], anno.outputs[0][dim1] = \
            anno.inputs[0][dim1], anno.inputs[0][dim0]
        return anno
    annos = [('* -> *', adapt),]
    inputs, dim0, dim1 = inputs[0:1], inputs[1], inputs[2]
    return IREinops(signature, annos, inputs, 'transpose',
                    dim0=dim0, dim1=dim1)


# def Conv2D(signature, inputs):
#     """
#     torch.conv2d(input, weight, bias, stride, padding, dialation, groups)
#     https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html?highlight=torch%20conv2d#torch.nn.functional.conv2d
#     """
#     def adapt(anno: EinopAnno, node: IREinops) -> EinopAnno:
#         iH, iW = node.inputs(0).shape[2:4]
#         stride = node.kwargs['stride']
#         padding = node.kwargs['padding']
#         dilation = node.kwargs['dilation']
#         dH = node.inputs(1).shape[2]
#         dW = node.inputs(1).shape[3]
#         oH = (iH + 2 * padding[0] - dilation[0] * (dH - 1) - 1) // stride[0] + 1
#         oW = (iW + 2 * padding[1] - dilation[1] * (dW - 1) - 1) // stride[1] + 1
#         anno.outputs[0][2] = EinDim([str(oH)])
#         anno.outputs[0][3] = EinDim([str(oW)])
#         return anno
#     annos = [
#         ('N iC+ H^ W^, oC iC+ dH^ dW^, oC -> N oC oH^ oW^', adapt),
#         ('N iC+ H^ W^, oC iC+ dH^ dW^ -> N oC oH^ oW^', adapt),
#     ]
#     tensors = inputs[0:3]
#     if tensors[-1] is None:
#         tensors = inputs[0:2]
#     stride, padding, dilation, groups = inputs[3:]
#     return IREinops(signature, annos, tensors, 'conv2d',
#                     stride=stride, padding=padding, dilation=dilation, groups=groups)


def Conv2D(signature, inputs):
    """
    torch.conv2d(input, weight, bias, stride, padding, dialation, groups)
    https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html?highlight=torch%20conv2d#torch.nn.functional.conv2d
    """
    assert len(inputs) == 7, f"Expected 7 inputs but only got {len(inputs)}"
    tensors = inputs[0:3]
    stride, padding, dilation, groups = inputs[3:]
    if isinstance(padding, int):
        padding = [padding] * 4
    elif len(padding) == 2:
        padH, padW = padding
        padding = [padH, padH, padW, padW]
    return IRConv2D(signature, tensors, 'conv2d',
                    stride=stride, padding=padding, dilation=dilation, groups=groups)
