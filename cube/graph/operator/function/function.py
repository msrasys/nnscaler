from typing import Iterable, List, Optional, Union, Dict
import string
import copy

from cube.ir.cten import IRTensor
from cube.graph.operator.function.einops import EinDim, IREinops
from cube.graph.operator.function.conv import IRConv2D
from cube.graph.operator.function.conv import IRConv3D
from cube.graph.operator.function.pad import IRPad
from cube.graph.operator.function.scripteinops import IRScriptEinOps
from cube.graph.operator.function.customops import IRCustomOps


def _create_eshape(shape: List[int], iterator: Optional[Iterable] = None,
                  reduce: EinDim.ReduceType = EinDim.ReduceType.Spatial) -> List[str]:
    """
    Create dimension annotation given the shape and 
    letter iterator
    """
    if iterator is None:
        iterator = iter(string.ascii_lowercase)
    return [next(iterator) + reduce.value for _ in range(len(shape))]


def _create_anno(ins: List[List[Union[str, List[str]]]],
                 ous: List[List[Union[str, List[str]]]]) -> str:
    """
    Create annotation string
    e.g., 
        ins = [ ['a', 'b', 'c+'], ['c+', ['d', 'e']] ]
        ous = [ ['a', 'b', 'd', 'e'] ]
    =>
        'a b c+, c+ (d e) -> a b d e'
    """
    in_annos = list()
    ou_annos = list()
    for shape in ins:
        flatten = list()
        for edim in shape:
            if isinstance(edim, str):
                flatten.append(edim)
            # List
            elif len(edim) == 1:
                flatten.append(edim[0])
            else:
                flatten.append('(' + ' '.join(edim) + ')')
        in_annos.append(' '.join(flatten))
    for shape in ous:
        flatten = list()
        for edim in shape:
            if isinstance(edim, str):
                flatten.append(edim)
            # List
            elif len(edim) == 1:
                flatten.append(edim[0])
            else:
                flatten.append('(' + ' '.join(edim) + ')')
        ou_annos.append(' '.join(flatten))
    return ', '.join(in_annos) + ' -> ' + ', '.join(ou_annos)


def Linear(signature, inputs):
    if signature == 'torch.linear':
        import warnings
        warnings.warn(f'signature {signature} replaced into torch.nn.functional.linear')
        signature = 'torch.nn.functional.linear'

    annos = [
        'b * k+, n k+ -> b * n',   # no bias
        'b * k+, n k+, n -> b * n' # have bias
    ]
    if inputs[2] is None:
        inputs = inputs[0:2]
    return IREinops(signature, annos, inputs, 'linear')


def BatchLinear(signature, inputs):
    annos = [
        'b m k, b k n -> b m n'
    ]
    return IREinops(signature, annos, inputs, 'bmm')


def Add(signature, inputs):
    assert len(inputs) == 3
    inputs, alpha = inputs[0:2], inputs[2]
    annos = [
        '*, 1 -> *',
        '1, * -> *',
        '*, * -> *',
    ]
    # broadcast
    lhs, rhs = inputs
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor) and \
       len(lhs.shape) == len(rhs.shape):
        if not all([l == r for l, r in zip(lhs.shape, rhs.shape)]):
            # TODO: support spatial partitioning on broadcast dim
            lshape = _create_eshape(lhs.shape)
            rshape = copy.copy(lshape)
            oshape = copy.copy(lshape)
            for dim in range(len(lhs.shape)):
                if lhs.shape[dim] < rhs.shape[dim]:
                    oshape[dim] = rshape[dim]
                    lshape[dim] = str(lhs.shape[dim])
                elif lhs.shape[dim] > rhs.shape[dim]:
                    oshape[dim] = lshape[dim]
                    rshape[dim] = str(rhs.shape[dim])
            annos = [_create_anno([lshape, rshape], [oshape])]
    return IREinops(signature, annos, inputs, 'add', alpha=alpha)


def Sub(signature, inputs):
    assert len(inputs) == 3
    inputs, alpha = inputs[0:2], inputs[2]
    annos = [
        '*, 1 -> *',
        '1, * -> *',
        '*, * -> *',
    ]
    # broadcast
    lhs, rhs = inputs
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor) and \
       len(lhs.shape) == len(rhs.shape):
        if not all([l == r for l, r in zip(lhs.shape, rhs.shape)]):
            # TODO: support spatial partitioning on broadcast dim
            lshape = _create_eshape(lhs.shape)
            rshape = copy.copy(lshape)
            oshape = copy.copy(lshape)
            for dim in range(len(lhs.shape)):
                if lhs.shape[dim] < rhs.shape[dim]:
                    oshape[dim] = rshape[dim]
                    lshape[dim] = str(lhs.shape[dim])
                elif lhs.shape[dim] > rhs.shape[dim]:
                    oshape[dim] = lshape[dim]
                    rshape[dim] = str(rhs.shape[dim])
            annos = [_create_anno([lshape, rshape], [oshape])]
    return IREinops(signature, annos, inputs, 'sub', alpha=alpha)


def Mul(signature, inputs):
    annos = [
        '*, 1 -> *',
        '1, * -> *',
        '*, * -> *',
    ]
    # broadcast
    lhs, rhs = inputs
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor) and \
       len(lhs.shape) == len(rhs.shape):
        if not all([l == r for l, r in zip(lhs.shape, rhs.shape)]):
            # TODO: support spatial partitioning on broadcast dim
            lshape = _create_eshape(lhs.shape)
            rshape = copy.copy(lshape)
            oshape = copy.copy(lshape)
            for dim in range(len(lhs.shape)):
                if lhs.shape[dim] < rhs.shape[dim]:
                    oshape[dim] = rshape[dim]
                    lshape[dim] = str(lhs.shape[dim])
                elif lhs.shape[dim] > rhs.shape[dim]:
                    oshape[dim] = lshape[dim]
                    rshape[dim] = str(rhs.shape[dim])
            annos = [_create_anno([lshape, rshape], [oshape])]
    return IREinops(signature, annos, inputs, 'mul')


def Div(signature, inputs):
    annos = [
        '*, 1 -> *',
        '1, * -> *',
        '*, * -> *',
    ]
    # broadcast
    lhs, rhs = inputs
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor) and \
       len(lhs.shape) == len(rhs.shape):
        if not all([l == r for l, r in zip(lhs.shape, rhs.shape)]):
            # TODO: support spatial partitioning on broadcast dim
            lshape = _create_eshape(lhs.shape)
            rshape = copy.copy(lshape)
            oshape = copy.copy(lshape)
            for dim in range(len(lhs.shape)):
                if lhs.shape[dim] < rhs.shape[dim]:
                    oshape[dim] = rshape[dim]
                    lshape[dim] = str(lhs.shape[dim])
                elif lhs.shape[dim] > rhs.shape[dim]:
                    oshape[dim] = lshape[dim]
                    rshape[dim] = str(rhs.shape[dim])
            annos = [_create_anno([lshape, rshape], [oshape])]
    return IREinops(signature, annos, inputs, 'div')

def Neg(signature, inputs):
    annos = ['* -> *']
    tensor = inputs[0:1]
    if len(inputs) == 2:
        # adapt for newest pytorch version
        approximate = inputs[1]
        return IREinops(signature, annos, tensor, 'neg',
                        approximate=approximate)
    else:
        return IREinops(signature, annos, tensor, 'neg')

def Sin(signature, inputs):
    annos = ['* -> *']
    tensor = inputs[0:1]
    if len(inputs) == 2:
        # adapt for newest pytorch version
        approximate = inputs[1]
        return IREinops(signature, annos, tensor, 'sin',
                        approximate=approximate)
    else:
        return IREinops(signature, annos, tensor, 'sin')


def Cos(signature, inputs):
    annos = ['* -> *']
    tensor = inputs[0:1]
    if len(inputs) == 2:
        # adapt for newest pytorch version
        approximate = inputs[1]
        return IREinops(signature, annos, tensor, 'cos',
                        approximate=approximate)
    else:
        return IREinops(signature, annos, tensor, 'cos')


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
                    p=p, training=training, inplace=inplace)


def LayerNorm(signature, inputs):
    input, normalized_shape, weight, bias, eps = inputs
    if len(normalized_shape) != 1:
        raise NotImplementedError("Only support normalized_shape to be int")
    annos = [
        f'N *, 1, {normalized_shape[0]}, {normalized_shape[0]} -> N *',
        f'N *, 1, 1, 1 -> N *'
    ]
    return IREinops(signature, annos, [input, normalized_shape, weight, bias],
                    'layernorm', eps=eps)


def Sum(signature, inputs):
    # TODO: support dim reduction
    annos = [
        '*+ -> 1',
    ]
    tensor = inputs[0:1]
    dim = inputs[1]
    if dim is not None:
        keepdim = inputs[2] if len(inputs) > 2 else False
        dim_len = len(tensor[0].shape)
        anno = "".join([f'b{i} ' for i in range(dim_len)]) + " -> " + "".join([f'b{i} ' if i not in dim else "" for i in range(dim_len)])
        annos.append(anno)
        return IREinops(signature, annos, tensor, 'sum',
                        dim=dim, keepdim=keepdim)
    else:
        return IREinops(signature, annos, tensor, 'sum')


def Transpose(signature, inputs):
    """
    out = torch.transpose(tensor, dim0, dim1)
    """
    assert len(inputs) == 3
    input, dim0, dim1 = inputs

    edim_in = _create_eshape(input.shape)
    edim_ou = copy.copy(edim_in)
    edim_ou[dim0], edim_ou[dim1] = edim_ou[dim1], edim_ou[dim0]
    anno = _create_anno([edim_in], [edim_ou])

    return IREinops(signature, [anno], [input], 'transpose',
                    dim0=dim0, dim1=dim1)


def View(signature, inputs):
    """
    out = torch.Tensor.view(tensor: torch.Tensor, shape: List[int])
    """
    assert len(inputs) == 2
    input, shape = inputs
    if not all([isinstance(dim, int) for dim in shape]):
        raise TypeError("Expected tensor.view has static int shape")
    in_shape, ou_shape = list(input.shape), shape

    # shape check
    def nele(shape, nele=1):
        for dimlen in shape: nele *= dimlen
        return nele
    # handle '-1' in shape
    cnt = nele(in_shape)
    if -1 in ou_shape:
        idx = ou_shape.index(-1)
        ou_shape[idx] = cnt // (-nele(ou_shape))
    assert nele(in_shape) == nele(ou_shape), "shape mismatch"
    # generate annotation
    shape_map: Dict[str, int] = dict()
    letters = iter(string.ascii_lowercase)
    in_anno, ou_anno = [], []
    in_dim, ou_dim = 0, 0
    in_remain, ou_remain = in_shape[in_dim], ou_shape[ou_dim]
    in_bracket, ou_bracket = [], []
    in_dimlen, ou_dimlen = 1, 1
    while True:
        letter = next(letters)
        dimlen = min(in_remain, ou_remain)
        in_dimlen, ou_dimlen = in_dimlen * dimlen, ou_dimlen * dimlen
        in_remain, ou_remain = in_remain // dimlen, ou_remain // dimlen
        in_bracket.append(letter)
        ou_bracket.append(letter)
        shape_map[letter] = dimlen
        if in_remain == 1:
            in_anno.append(in_bracket)
            in_bracket, in_dimlen = [], 1
            in_dim += 1
            if in_dim < len(in_shape):
                in_remain = in_shape[in_dim]
        if ou_remain == 1:
            ou_anno.append(ou_bracket)
            ou_bracket, ou_dimlen = [], 1
            ou_dim += 1
            if ou_dim < len(ou_shape):
                ou_remain = ou_shape[ou_dim]
        if in_dim == len(in_shape) and ou_dim == len(ou_shape):
            break
    # setup reduction: only first dimension can be spatially partitioned
    spatial_in = set()
    spatial_ou = set()
    for in_bracket in in_anno:
        spatial_in.add(in_bracket[0])
    for ou_bracket in ou_anno:
        spatial_ou.add(ou_bracket[0])
    spatial = spatial_in.intersection(spatial_ou)
    for bracket in in_anno + ou_anno:
        for subdim, edim in enumerate(bracket):
            if edim not in spatial:
                bracket[subdim] = str(shape_map[edim])
                # bracket[subdim] = edim + '^'
    anno = _create_anno([in_anno], [ou_anno])
    return IREinops(signature, [anno], [input], 'view', shape=shape)


def Reshape(signature, inputs):
    return View(signature, inputs)


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


def Conv3D(signature, inputs):
    """
    conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) → Tensor
    https://pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html?highlight=conv3d#torch.nn.functional.conv3d
    """
    assert len(inputs) == 7, f"Expected 7 inputs but only got {len(inputs)}"
    tensors = inputs[0:3]
    stride, padding, dilation, groups = inputs[3:]
    if isinstance(padding, int):
        padding = [padding] * 4
    elif len(padding) == 2:
        padH, padW = padding
        padding = [padH, padH, padW, padW]
    return IRConv3D(signature, tensors, 'conv3d',
                    stride=stride, padding=padding, dilation=dilation, groups=groups)

def Pad(signature, inputs):
    """
    torch.nn.functional.pad(input, pad, mode='constant', value=0.0)
    https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad
    :param signature:
    :param inputs:
    :return:
    """
    # print("#Pad::inputs.len: {}".format(len(inputs)))
    # idx = 0
    # for input in inputs:
    #     if idx >= 0:
    #         print("#Pad::input[{}]: {}".format(idx, input))
    #     idx += 1
    tensors = inputs[0:1]
    pad, mode, value = inputs[1:]
    return IRPad(signature, tensors, 'pad', pad=pad, mode=mode, value=value)

def ScriptEinOps(signature, inputs):
    """
    apply_for_scriptable_torch(recipe: TransformRecipe, tensor: torch.Tensor, reduction_type: str) -> torch.Tensor:
    https://github.com/arogozhnikov/einops/blob/master/einops/_torch_specific.py
    :param signature:
    :param inputs:
    :return:
    """
    recipe = inputs[0]
    tensors = inputs[1:2]
    reduction_type = inputs[2]
    import pickle
    recipe_str = pickle.dumps(recipe)
    return IRScriptEinOps(signature, tensors, 'scripteinops', recipe_str=recipe_str, reduction_type=reduction_type)


def CustomOps(signature, inputs):
    if signature == 'examples.custom_ops.strip_2_borders':
        tensors = inputs[0:1]
        print(f'CustomOps:tensors[0] = {tensors[0]}')
        return IRCustomOps(signature, tensors, 'custom_ops')
    elif signature == 'example.custom_ops.update_diag':
        tensors = inputs[0:9]
        dz = inputs[9]
        dt = inputs[10]
        return IRCustomOps(signature, tensors, 'custom_ops', dz=dz, dt=dt)
    else:
        import warnings
        warnings.warn(f"ERROR Unknown custom op, signature{signature}")