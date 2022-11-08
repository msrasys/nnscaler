from typing import Any, Callable, List, Optional, Tuple, Dict, Union
import string
import copy
import torch
import warnings
import operator

from cube.ir.cten import IRTensor
from cube.ir.tensor import IRSubTensor
from cube.ir.dtype import IRDType
from cube.graph.function.dimops import DimopSplit, ShapeAnno, OpAnno, IRDimops, TransformRule
from cube.graph.function.conv import IRConv2D
from cube.graph.function.conv import IRConv3D
from cube.graph.function.pad import IRPad
from cube.graph.function.scripteinops import IRScriptEinOps
from cube.graph.function.creators import IROnes, IRToTensor, IRZeros, IRRand, IRNewTensor
from cube.graph.function.select import IRSelect, IRSlice
from cube.graph.function.scatter import IRSelectScatter
from cube.graph.function.repeat import IRRepeat
from cube.graph.function.anchor import IRGraphAnchor


ErasedDevice = 'str'


def Identity(signature, inputs: List[IRTensor]):
    signature = 'cube.runtime.function.identity'
    eshape = ShapeAnno.create_shape_str(inputs[0].shape)
    anno = OpAnno.create_op_str([eshape], [eshape])
    return IRDimops(Identity, 'identity', signature, [anno], inputs)


def Linear(signature, inputs):
    assert len(inputs) == 3
    signature = 'torch.nn.functional.linear'
    if inputs[2] is None:
        annos = ['b * k+, n k+ -> b * n']
        return IRDimops(Linear, 'linear', signature, annos, inputs[:2], bias=None)
    else:
        annos = ['b * k+, n k+, n -> b * n']
        rules = [TransformRule(
            [DimopSplit.D(-1), DimopSplit.D(1), DimopSplit.V()], [DimopSplit.V()]
        )]
        return IRDimops(Linear, 'linear', signature, annos, inputs, rules)


def BatchLinear(signature, inputs):
    annos = [
        'b m k+, b k+ n -> b m n'
    ]
    return IRDimops(BatchLinear, 'bmm', signature, annos, inputs)


def Zeros(signature,
          inputs: Tuple[ List[int], Optional[int], Optional[Any], ErasedDevice, Optional[bool] ]):
    # zeros(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
    #
    # REMARK: in the PyTorch-internal operator definition expression, an asterisk ("*") is merely a marker of
    #         the beginning of the sublist of _keyword arguments_, and does not result in an actual argument.

    from cube.graph.parser.mapping import DType2IRDType, TorchScalarTypeEnumMap

    size, dtype_underlying, layout, _erased_device, pin_memory = inputs

    # TODO parameters to support, currently they are all None
    assert layout is None
    assert pin_memory is None

    if dtype_underlying is not None:
        # If some torch.dtype is specified at the frontend, in TorchScript it becomes an int,
        # which is the underlying type of PyTorch C++ enum 'ScalarType'.
        dtype = TorchScalarTypeEnumMap.map(dtype_underlying)
    else:
        dtype = torch.get_default_dtype()

    ir_dtype : IRDType = DType2IRDType.map(dtype)

    for dim, i in enumerate(size):
        if not isinstance(dim, int) and not dim >= 0:
            raise RuntimeWarning(f"The {i}-th component of the size must be non-negative integer")
    return IRZeros(signature, size, 'zeros', ir_dtype)

def Ones(signature,
         inputs: Tuple[ List[int], Optional[int], Optional[Any], ErasedDevice, Optional[bool] ]):
    # ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    size, dtype_underlying, layout, _erased_device, pin_memory = inputs

    # TODO parameters to support, currently they are all None
    assert layout is None
    assert pin_memory is None
    from cube.graph.parser.mapping import DType2IRDType, TorchScalarTypeEnumMap

    if dtype_underlying is not None:
        # If some torch.dtype is specified at the frontend, in TorchScript it becomes an int,
        # which is the underlying type of PyTorch C++ enum 'ScalarType'.
        dtype = TorchScalarTypeEnumMap.map(dtype_underlying)
    else:
        dtype = torch.get_default_dtype()

    ir_dtype : IRDType = DType2IRDType.map(dtype)

    for dim, i in enumerate(size):
        if not isinstance(dim, int) and not dim >= 0:
            raise RuntimeWarning(f"The {i}-th component of the size must be non-negative integer")
    return IROnes(signature, size, 'ones', ir_dtype)

def Rand(signature,
         inputs: Tuple[ List[int], Optional[int], Optional[Any], ErasedDevice, Optional[bool] ]):
    # ones(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor

    size, dtype_underlying, layout, _erased_device, pin_memory = inputs

    # TODO parameters to support, currently they are all None
    assert layout is None
    assert pin_memory is None
    from cube.graph.parser.mapping import DType2IRDType, TorchScalarTypeEnumMap

    if dtype_underlying is not None:
        # If some torch.dtype is specified at the frontend, in TorchScript it becomes an int,
        # which is the underlying type of PyTorch C++ enum 'ScalarType'.
        dtype = TorchScalarTypeEnumMap.map(dtype_underlying)
    else:
        dtype = torch.get_default_dtype()

    ir_dtype : IRDType = DType2IRDType.map(dtype)

    for dim, i in enumerate(size):
        if not isinstance(dim, int) and not dim >= 0:
            raise RuntimeWarning(f"The {i}-th component of the size must be non-negative integer")
    return IRRand(signature, size, 'rand', ir_dtype)

def NewTensor(signature,
              inputs: Tuple[ list, Optional[int], ErasedDevice, bool ]):
    # aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor
    #
    # REMARK: in the PyTorch-internal operator definition expression, an asterisk ("*") is merely a marker of
    #         the beginning of the sublist of _keyword arguments_, and does not result in an actual argument.

    data, dtype_underlying, _erased_device, requires_grad = inputs

    # TODO parameters to support, currently they are all None
    assert requires_grad == False
    from cube.graph.parser.mapping import DType2IRDType, TorchScalarTypeEnumMap

    if dtype_underlying is not None:
        # If some torch.dtype is specified at the frontend, in TorchScript it becomes an int,
        # which is the underlying type of PyTorch C++ enum 'ScalarType'.
        dtype = TorchScalarTypeEnumMap.map(dtype_underlying)
    else:
        dtype = torch.get_default_dtype()

    ir_dtype : IRDType = DType2IRDType.map(dtype)

    # if 'data' is not:
    # 1) ints or floats of any precision, e.g. i8, i64, f16, f32
    # 2) non-ragged
    # ... then this call will throw.
    arr = torch.tensor(data, dtype=dtype)

    # TODO temporarily fake creation with Zeros
    # and remark that originally aten::tensor should be able to infer the dtype from the specified 'data',
    # but since we have omitted the 'data', we must do type inferrence ourselves,
    # only in this way we get correct dtype e.g. ints or bools.
    return IRNewTensor(signature, data, 'tensor', ir_dtype=ir_dtype)

def ToTensor(signature,
             inputs: Tuple[ IRTensor, ... ]):
    """
    'aten::to' has many overloadings that need resolution,
    they differ by both the arity and the type of the argument (possibly at the same position):

    ```
    aten::to.device(Tensor self, Device device, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor):
    aten::to.dtype(Tensor self, int dtype, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor):
    aten::to.dtype_layout(Tensor self, *, int dtype, int layout, Device device, bool pin_memory=False, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor):
    aten::to.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, int? memory_format=None) -> (Tensor):
    aten::to.prim_Device(Tensor(a) self, Device? device, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor(b|a)):
    aten::to.prim_dtype(Tensor(a) self, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor(b|a)):
    aten::to.prim_other(Tensor(a) self, bool non_blocking=False, bool copy=False) -> (Tensor(b|a)):
    ```
    ... where the 'int? dtype' is the underlying type for the enum 'ScalarType'.
    """

    # in our case we only care the overloading 'to.dtype' (arity=5)
    assert len(inputs) == 5
    tensor : IRTensor
    dtype_underlying : int
    non_blocking : bool
    copy : bool
    opt_memory_format : Optional[int]
    tensor, dtype_underlying, non_blocking, copy, opt_memory_format = inputs

    from cube.graph.parser.mapping import DType2IRDType, TorchScalarTypeEnumMap
    dtype : torch.dtype = TorchScalarTypeEnumMap.map(dtype_underlying)
    ir_dtype : IRDType = DType2IRDType.map(dtype)

    signature = 'torch.Tensor.to'
    return IRToTensor(signature, [tensor], 'to', ir_dtype=ir_dtype)


def _handle_broadcast(lhs: IRTensor, rhs: IRTensor) -> Tuple[List[str]]:
    """!
    Create shape annotations for element wise operator following broadcastable rules:
    https://pytorch.org/docs/stable/notes/broadcasting.html

    @param lhs IRTensor: the lhs input tensor
    @param rhs IRTensor: the rhs input tensor

    @return lhs_shape, rhs_shape, out_shape: the lhs, rhs and output shape annotation
    """
    lndims, rndims = len(lhs.shape), len(rhs.shape)
    # init lhs_shape and rhs_shape annotation string
    shape_anno = ShapeAnno.create_shape_str(lhs.shape if lndims > rndims else rhs.shape)
    lhs_shape = shape_anno[0-lndims:]
    rhs_shape = shape_anno[0-rndims:]
    # expand dimensions for empty dimensions
    lofst = max(lndims, rndims) - lndims
    lshape = [1] * lofst + list(lhs.shape)
    rofst = max(lndims, rndims) - rndims
    rshape = [1] * rofst + list(rhs.shape)
    # init out_shape
    out_shape = []
    for dim in range(len(lshape)):
        ldim_anno = None if dim - lofst < 0 else lhs_shape[dim-lofst]
        rdim_anno = None if dim - rofst < 0 else rhs_shape[dim-rofst]
        if lshape[dim] == rshape[dim]:
            assert rdim_anno is not None or ldim_anno is not None
            out_shape.append(rdim_anno if rdim_anno is not None else ldim_anno)
        elif lshape[dim] == 1:
            assert rdim_anno is not None
            out_shape.append(rdim_anno)
            if ldim_anno is not None:
                lhs_shape[dim-lofst] = '1'
        elif rshape[dim] == 1:
            assert ldim_anno is not None
            out_shape.append(ldim_anno)
            if rdim_anno is not None:
                rhs_shape[dim-rofst] = '1'
        else:
            raise ValueError(f"cannot broadcast lhs: {lhs.shape} and rhs: {rhs.shape}")
    # print(lhs.shape, rhs.shape, lhs_shape, rhs_shape, out_shape)
    return lhs_shape, rhs_shape, out_shape


def Clone(signature, inputs):
    """
    torch.clone(input, *, memory_format=torch.preserve_format)
    """
    assert len(inputs) == 2, f"inputs: {inputs}"
    tensor, memory_format = inputs
    annos = ['* -> *']
    tensor = inputs[0]
    assert memory_format is None, f"Not supported for a specific memory format"
    return IRDimops(Clone, 'clone', signature, annos, [tensor])


def Add(signature, inputs):
    if len(inputs) == 2:
        kwargs = {}
    elif len(inputs) == 3:
        alpha = inputs[2]
        kwargs = {'alpha': alpha}
        inputs = inputs[0:2]
    else:
        raise RuntimeError("The number of inputs must be 2 or 3")

    lhs, rhs = inputs

    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        # In this case there won't be an 'alpha' parameter.
        assert not('alpha' in kwargs)
        return lhs + rhs

    annos = [
        '*, ? -> *',
        '?, * -> *',
    ]
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(lhs, rhs)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Add, 'add', signature, annos, inputs, **kwargs)


def Sub(signature, inputs):
    if len(inputs) == 2:
        alpha = 1
        kwargs = {}
    elif len(inputs) == 3:
        alpha = inputs[2]
        kwargs = {'alpha': alpha}
        inputs = inputs[0:2]
    else:
        raise RuntimeError("The number of inputs must be 2 or 3")

    lhs, rhs = inputs

    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        # In this case there won't be an 'alpha' parameter.
        assert not('alpha' in kwargs)
        return lhs - rhs

    annos = [
        '*, ? -> *',
        '?, * -> *',
    ]
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(lhs, rhs)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Sub, 'sub', signature, annos, inputs, **kwargs)


def Mul(signature, inputs):
    lhs, rhs = inputs

    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return lhs * rhs

    annos = [
        '*, ? -> *',
        '?, * -> *',
    ]
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(lhs, rhs)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Mul, 'mul', signature, annos, inputs)


def Div(signature, inputs):
    lhs, rhs = inputs
    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        # For `aten::div` we always do floating division, even operands are both ints.
        # TorchScript would dispatch frontend `a // b` to another op `aten::floordiv`.
        return lhs / rhs

    annos = [
        '*, ? -> *',
        '?, * -> *',
    ]
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(lhs, rhs)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Div, 'div', signature, annos, inputs)


def FloorDiv(signature, inputs):
    lhs, rhs = inputs

    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return lhs // rhs

    annos = [
        '*, ? -> *',
        '?, * -> *',
    ]
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(lhs, rhs)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(FloorDiv, 'floordiv', signature, annos, inputs)


def Pow(signature, inputs):
    lhs, rhs = inputs

    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return lhs ** rhs

    annos = [
        '*, ? -> *',
        '?, * -> *',
    ]
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(lhs, rhs)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Pow, 'pow', signature, annos, inputs)


def Neg(signature, inputs):
    assert len(inputs) == 1 or len(inputs) == 2
    kwargs = {} if len(inputs) == 1 else {'approximate': inputs[1]}
    tensors = inputs[0:1]

    if isinstance(tensors[0], (int, float)):
        assert not('approximate' in kwargs)
        return -tensors[0]

    annos = ['* -> *']
    return IRDimops(Neg, 'neg', signature, annos, inputs, **kwargs)


def Sin(signature, inputs):
    annos = ['* -> *']
    tensor = inputs[0:1]
    if len(inputs) == 2:
        # adapt for newest pytorch version
        approximate = inputs[1]
        return IRDimops(Sin, 'sin', signature, annos, tensor,
                        approximate=approximate)
    else:
        return IRDimops(Sin, 'sin', signature, annos, tensor)


def Cos(signature, inputs):
    annos = ['* -> *']
    tensor = inputs[0:1]
    if len(inputs) == 2:
        # adapt for newest pytorch version
        approximate = inputs[1]
        return IRDimops(Cos, 'cos', signature, annos, tensor,
                        approximate=approximate)
    else:
        return IRDimops(Cos, 'cos', signature, annos, tensor)


def Tanh(signature, inputs):
    """
    torch.tanh(input, *, out=None)
    """
    assert len(inputs) == 1, f"inputs: {inputs}"
    annos = ['* -> *']
    tensor = inputs[0:1]
    return IRDimops(Tanh, 'tanh', signature, annos, tensor)


def GeLU(signature, inputs):
    annos = ['* -> *']
    signature = 'torch.nn.functional.gelu'
    tensor = inputs[0:1]
    if len(inputs) == 2:
        # adapt for newest pytorch version
        approximate = inputs[1]
        return IRDimops(GeLU, 'gelu', signature, annos, tensor,
                        approximate=approximate)
    else:
        return IRDimops(GeLU, 'gelu', signature, annos, tensor)


def SiLU(signature, inputs):
    assert len(inputs) == 1
    annos = ['* -> *']
    signature = 'torch.nn.functional.silu'
    tensor = inputs[0:1]
    return IRDimops(SiLU, 'silu', signature, annos, tensor)


def Softmax(signature, inputs):
    assert len(inputs) == 4
    annos = ['* -> *']
    tensor = inputs[0:1]
    dim, _stacklevel, dtype = inputs[1], inputs[2], inputs[3]
    return IRDimops(Softmax, 'softmax', signature, annos, tensor,
                    dim=dim, _stacklevel=_stacklevel, dtype=dtype)


def Dropout(signature, inputs):
    assert len(inputs) == 4
    annos = ['* -> *']
    tensor = inputs[0:1]
    p, training, inplace = inputs[1], inputs[2], inputs[3]
    return IRDimops(Dropout, 'dropout', signature, annos, tensor,
                    p=p, training=training, inplace=inplace)


def LayerNorm(signature, inputs):
    """
    torch.nn.functional.layer_norm(input, normliazed_shape, weight=None, bias=None, eps)
    cube.runtime.function.layer_norm(input, weight, bias, normliazed_shape, eps)
    """
    if 'torch.' in signature:
        tensor, normalized_shape, weight, bias, eps = inputs
        assert isinstance(normalized_shape, list), f"normalized_shape for layer_norm can only be List[int]"
    else:
        tensor, weight, bias, normalized_shape, eps = inputs
    letters = iter(string.ascii_lowercase)
    einput = ShapeAnno.create_shape_str(tensor.shape, iterator=letters)
    eoutput = copy.copy(einput)
    ndims = len(tensor.shape)
    for dim in range(len(normalized_shape)):
        einput[ndims-1-dim] += '^'
        eoutput[ndims-1-dim] += '^'
    assert not (bias is None is weight is not None), f"Not support for None of weight and parameter of bias"
    einputs, inputs = [einput], [tensor]
    kwargs = {}
    if weight is not None:
        eweight = ShapeAnno.create_shape_str(weight.shape, reduction='^', iterator=letters)
        einputs.append(eweight)
        inputs.append(weight)
    else:
        kwargs['weight'] = weight
    if bias is not None:
        ebias = ShapeAnno.create_shape_str(bias.shape, reduction='^', iterator=letters)
        einputs.append(ebias)
        inputs.append(bias)
    else:
        kwargs['bias'] = bias
    anno = OpAnno.create_op_str(einputs, [eoutput])
    kwargs['normalized_shape'] = normalized_shape
    kwargs['eps'] = eps
    signature = 'cube.runtime.function.layer_norm'
    return IRDimops(LayerNorm, 'layernorm', signature, [anno], inputs, **kwargs)


def Sum(signature, inputs):
    """
    torch.sum(input, *, dtype=None) -> Tensor
    torch.sum(input, dim, keepdim=False, *, dtype=None) -> Tensor
    """
    assert len(inputs) == 2 or len(inputs) == 4, f"{inputs}"
    tensor = inputs[0]
    einput = ShapeAnno.create_shape_str(tensor.shape)
    eoutput = copy.copy(einput)
    if len(inputs) == 2:
        dtype = inputs[1]
        assert dtype is None, "Currently Sum only support dtype=None"
        # torch.sum(input)
        inputs = [tensor]
        eoutput = ['1']
        # every dimension can be reduced
        einput = [edim + '+' for edim in einput]
        anno = OpAnno.create_op_str([einput], [eoutput])
        return IRDimops(Sum, 'sum', signature, [anno], [tensor], dtype=dtype)
    else:
        # torch.sum(input, dim, keepdim, *, dtype)
        dim, keepdim, dtype = inputs[1:4]
        assert dtype is None, "Currently Sum only support dtype=None"
        assert isinstance(dim, list), f"Expect dim to be list but got: {dim}"
        for dimidx in dim:
            einput[dimidx] += '+'
        if keepdim:
            for dimidx in dim:
                eoutput[dimidx] = '1'
        else:
            sort_dim = list(dim)
            sort_dim.sort()
            for dimidx in sort_dim[::-1]:
                eoutput.pop(dimidx)
        anno = OpAnno.create_op_str([einput], [eoutput])
        return IRDimops(Sum, 'sum', signature, [anno], [tensor], dim=dim, keepdim=keepdim, dtype=dtype)


def Mean(signature, inputs):
    if len(inputs) >= 2:
        tensor, dim = inputs[:2]
    elif len(inputs) == 1:
        tensor = inputs[0]
        dim = None
    einput = ShapeAnno.create_shape_str(tensor.shape)
    eoutput = copy.copy(einput)
    if dim is not None:
        keepdim = inputs[2]
        sort_dim = list(dim)
        sort_dim.sort()
        for dimidx in sort_dim[::-1]:
            eoutput.pop(dimidx)
            einput[dimidx] = einput[dimidx] + '+'
    else:
        eoutput = ['1']
        # every dimension is reduced
        einput = [edim + '+' for edim in einput]
    anno = OpAnno.create_op_str([einput], [eoutput])
    if dim is not None:
        return IRDimops(Mean, 'mean', signature, [anno], [tensor], dim=dim, keepdim=keepdim)
    else:
        return IRDimops(Mean, 'mean', signature, [anno], [tensor])


def Transpose(signature, inputs):
    """
    out = torch.transpose(tensor, dim0, dim1)
    """
    assert len(inputs) == 3
    input, dim0, dim1 = inputs

    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = copy.copy(edim_in)
    edim_ou[dim0], edim_ou[dim1] = edim_ou[dim1], edim_ou[dim0]
    anno = OpAnno.create_op_str([edim_in], [edim_ou])

    return IRDimops(Transpose, 'transpose', signature, [anno], [input],
                    dim0=dim0, dim1=dim1)


def View(signature, inputs):
    """
    out = torch.Tensor.view(tensor: torch.Tensor, size: List[int])
    """
    assert len(inputs) == 2
    input, shape = inputs
    if not all([isinstance(dim, int) for dim in shape]):
        raise TypeError("Expected tensor.view has static int shape")
    in_shape, ou_shape = list(input.shape), shape

    # infer -1
    def nele(shape, nele=1):
        for dimlen in shape: nele *= dimlen
        return nele

    cnt = nele(in_shape)
    if -1 in ou_shape:
        idx = ou_shape.index(-1)
        ou_shape[idx] = cnt // (-nele(ou_shape))
    assert nele(in_shape) == nele(ou_shape), "shape mismatch"

    # generate annotation
    rest_inshape = [dimlen for dimlen in in_shape]
    rest_oushape = [dimlen for dimlen in ou_shape]
    chain = []
    can_bucket = True
    while len(rest_inshape) != 0 or len(rest_oushape) != 0:
        if len(rest_inshape) == 0:
            chain = chain + rest_oushape
            rest_oushape = []
        elif len(rest_oushape) == 0:
            chain = chain + rest_inshape
            rest_inshape = []
        else:
            dimlen = min(rest_inshape[0], rest_oushape[0])
            if max(rest_inshape[0], rest_oushape[0]) % dimlen == 0:
                chain.append(dimlen)
                if dimlen == rest_inshape[0]:
                    rest_inshape.pop(0)
                else:
                    rest_inshape[0] = rest_inshape[0] // dimlen
                if dimlen == rest_oushape[0]:
                    rest_oushape.pop(0)
                else:
                    rest_oushape[0] = rest_oushape[0] // dimlen
            else:
                can_bucket = False
                break

    letters = iter(string.ascii_lowercase)
    if can_bucket:
        inchain = ouchain = chain
        inedims = ouedims = edims = [next(letters) for _ in chain]
    else:
        inchain, ouchain = in_shape, ou_shape
        inedims = [str(dimlen) for dimlen in in_shape]
        ouedims = [str(dimlen) for dimlen in ou_shape]
        chain = inchain + ouchain
        edims = inedims + ouedims
    shape_map: Dict[str, int] = {edim: eshape for (edim, eshape) in zip(edims, chain)}

    # generate input and output shape annotations
    def buckets(shape: List[int], chain: List[int], edims: List[int]) -> List[List[str]]:
        anno = []
        dimidx = 0
        for idx, dimlen in enumerate(shape):
            elements, bracket = 1, []
            maxele = len(chain) - dimidx - (len(shape) - 1 - idx)
            while True:
                if len(bracket) == maxele:
                    assert elements == dimlen, f"internal match error1: {bracket}"
                    break
                if dimidx >= len(chain) or elements * chain[dimidx] > dimlen:
                    assert elements == dimlen, f"internal match error2: {bracket}"
                    break
                else:
                    elements *= chain[dimidx]
                    bracket.append(edims[dimidx])
                    dimidx += 1
            anno.append(bracket)
        return anno

    in_anno = buckets(in_shape, inchain, inedims)
    ou_anno = buckets(ou_shape, ouchain, ouedims)

    # postprocess on dimlen == 1
    shape_map['1'] = 1
    for bracket in in_anno + ou_anno:
        for subdim, edim in enumerate(bracket):
            if shape_map[edim] == 1:
                bracket[subdim] = str(shape_map[edim])

    # find out the axis that can be partitioned
    ispatial = set()
    ifirst = []
    for bracket in in_anno:
        for hdim in range(len(bracket)):
            if bracket[hdim] == '1':
                continue
            ispatial.add(bracket[hdim])
            ifirst.append(bracket[hdim])
            break
    ospatial = set()
    ofirst = []
    for bracket in ou_anno:
        for hdim in range(len(bracket)):
            if bracket[hdim] == '1':
                continue
            ospatial.add(bracket[hdim])
            ofirst.append(bracket[hdim])
            break
    spatial = ispatial.intersection(ospatial)

    # set dimension cannot be partitioned
    for bracket in in_anno + ou_anno:
        for hdim in range(len(bracket)):
            if bracket[hdim] not in spatial:
                bracket[hdim] = str(shape_map[bracket[hdim]])

    # TODO: strange behaviour if every identitifer creates own
    # modifier, seems all previous modifiers will be overrided by
    # the last one.
    def view_modifier(kwargs: Dict, idx, dim, num: int) -> Dict:
        kwargs = dict(**kwargs)
        ofirst = [bracket[0] for bracket in ou_anno]
        identifier = in_anno[idx][0]
        oidx = ofirst.index(identifier)
        size = list(kwargs['size'])
        size[oidx] = size[oidx] // num
        kwargs['size'] = tuple(size)
        return kwargs

    # special rules: to change output size argument
    rules = []
    for identifier in spatial:
        iidx = ifirst.index(identifier)
        oidx = ofirst.index(identifier)
        rules.append(
            TransformRule([DimopSplit.D(iidx)], [DimopSplit.D(oidx)], view_modifier)
        )

    anno = OpAnno.create_op_str([in_anno], [ou_anno])
    signature = 'torch.Tensor.view'
    return IRDimops(View, 'view', signature, [anno], [input], rules, size=tuple(shape))


def Reshape(signature, inputs):
    """
    torch.reshape(Tensor self, int[] shape) -> Tensor
    """

    warnings.warn("""
    'torch.reshape' is currently dispatched to 'torch.Tensor.view',
    but 'reshape' has keyword parameter 'shape' while 'view' has 'size'.
    ArgumentMissing error may be raised during codegen.""")

    return View(signature, inputs)


# def Conv2D(signature, inputs):
#     """
#     torch.conv2d(input, weight, bias, stride, padding, dialation, groups)
#     https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html?highlight=torch%20conv2d#torch.nn.functional.conv2d
#     """
#     def adapt(anno: OpAnno, node: IRDimops) -> OpAnno:
#         iH, iW = node.input(0).shape[2:4]
#         stride = node.kwargs['stride']
#         padding = node.kwargs['padding']
#         dilation = node.kwargs['dilation']
#         dH = node.input(1).shape[2]
#         dW = node.input(1).shape[3]
#         oH = (iH + 2 * padding[0] - dilation[0] * (dH - 1) - 1) // stride[0] + 1
#         oW = (iW + 2 * padding[1] - dilation[1] * (dW - 1) - 1) // stride[1] + 1
#         anno.outputs[0][2] = DimAnno([str(oH)])
#         anno.outputs[0][3] = DimAnno([str(oW)])
#         return anno
#     annos = [
#         ('N iC+ H^ W^, oC iC+ dH^ dW^, oC -> N oC oH^ oW^', adapt),
#         ('N iC+ H^ W^, oC iC+ dH^ dW^ -> N oC oH^ oW^', adapt),
#     ]
#     tensors = inputs[0:3]
#     if tensors[-1] is None:
#         tensors = inputs[0:2]
#     stride, padding, dilation, groups = inputs[3:]
#     return IRDimops(signature, annos, tensors, 'conv2d',
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


def Accum(signature, inputs: Tuple[IRTensor]):
    """
    tensor = cube.runtime.function.accum(tensors)
    """
    assert all(isinstance(t, IRTensor) for t in inputs)
    signature = 'cube.runtime.function.accum'
    iannos = [ShapeAnno.create_shape_str(t.shape) for t in inputs]
    oannos = [copy.copy(iannos[0])]
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(Cat, 'accum', signature, [anno], inputs)


def Cat(signature, inputs: Tuple[List[IRTensor], int]):
    """
    torch.cat(inputs: List[Tensor], dim: int) -> Tensor
    torch.cat(tensor1: Tensor, tensor2: Tensor, ..., dim: int)

    e.g. cat(tensor([2,3]), tensor([2,3])).shape == [4,3]
    """
    assert len(inputs) >= 2
    if len(inputs) == 2:
        tensors, dim = inputs
    else:
        tensors, dim = inputs[:-1], inputs[-1]
    assert all(isinstance(tensor, IRTensor) for tensor in tensors)
    iannos = [ShapeAnno.create_shape_str(t.shape) for t in tensors]
    dimlens = [t.shape[dim] for t in tensors]
    for ashape, dimlen in zip(iannos, dimlens):
        ashape[dim] = str(dimlen)
    oannos = [copy.copy(iannos[-1])]
    oannos[0][dim] = str(sum(dimlens))
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(Cat, 'cat', signature, [anno], tensors, dim=dim)


def Stack(signature, inputs: Tuple[List[IRTensor], int]):
    """
    torch.stack(inputs: List[Tensor], dim: int) -> Tensor
    torch.stack(tensor1: Tensor, tensor2: Tensor, ..., dim: int) -> Tensor

    inputs:
        tensors: List[Tensor]: all tensors need to have same size
        dim: the new inserted dim

    e.g. stack(tensor([2,3]), tensor([2,3])).shape == [2,2,3]
    """
    assert len(inputs) >= 2
    if len(inputs) == 2:
        tensors, dim = inputs
    else:
        tensors, dim = inputs[:-1], inputs[-1]
    assert all(isinstance(tensor, IRTensor) for tensor in tensors)
    iannos = [ShapeAnno.create_shape_str(t.shape) for t in tensors]
    oannos = [copy.copy(iannos[-1])]
    oannos[0].insert(dim, str(len(tensors)))
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(Stack, 'stack', signature, [anno], tensors, dim=dim)


def Chunk(signature, inputs: Tuple[IRTensor, int, int]):
    """
    torch.chunk(input, chunks, dim=0)
    """
    assert len(inputs) == 3
    tensor, chunks, dim = inputs
    assert tensor.shape[dim] % chunks == 0
    iannos = [ShapeAnno.create_shape_str(tensor.shape)]
    oannos = [copy.copy(iannos[0]) for _ in range(chunks)]
    iannos[0][dim] = str(tensor.shape[dim])
    for oanno in oannos:
        oanno[dim] = str(tensor.shape[dim] // chunks)
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(Chunk, 'chunk', signature, [anno], [tensor], chunks=chunks, dim=dim)


def Select(signature, inputs: Tuple[IRTensor, int, int]):
    """
    torch.select(self:Tensor, dim:int, index:int) -> Tensor
    """
    tensor, dim, index = inputs
    return IRSelect(signature, [tensor], 'select', dim, index)

def Slice(signature, inputs: Tuple[IRTensor, int, Optional[int], Optional[int], int]):
    """
    aten::slice(input:Tensor, dim:int, start:Optional[int], end:Optional[int], step:int) -> Tensor
    """
    tensor, dim, start, end, step = inputs
    return IRSlice(signature, [tensor], 'slice', dim, start, end, step)

def SelectScatter(signature, inputs:Tuple[IRTensor, IRTensor, int, int]):
    """
    torch.select_scatter(self:Tensor, input:Tensor, dim:int, index:int) -> Tensor
    """
    self, input, dim, index = inputs
    return IRSelectScatter(signature, [self, input], 'scatter_select', dim, index)


def Repeat(signature, inputs:Tuple[IRTensor, List[int]]):
    """
    torch.repeat(tensor:Tensor, repeats: List[int]) -> Tensor
    """
    tensor, repeats = inputs

    assert signature == 'torch.repeat' # this is the API in TorchScript
    signature = 'torch.Tensor.repeat'  # this is the API in Python frontend and is not a Tensor member method

    return IRRepeat(signature, [tensor], 'repeat', repeats)


def Embedding(signature, inputs: List):
    """
    torch.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
    """
    signature = 'cube.runtime.function.embedding'
    itensor, weight = inputs[:2]
    padding_idx = inputs[2]
    if isinstance(weight, IRSubTensor):
        start, stop = weight.indmap[0]
    else:
        start, stop = 0, weight.shape[0]
    annos = ['*, n+ e -> * e']
    return IRDimops(Embedding, 'embedding', signature, annos, [itensor, weight],
                    padding_idx=padding_idx, start=start, stop=stop)


def Flatten(signature, inputs: List):
    tensor: IRTensor = inputs[0]
    start_dim, end_dim = inputs[1:]
    end_dim = len(tensor.shape) + end_dim if end_dim < 0 else end_dim
    ishape = ShapeAnno.create_shape_str(tensor.shape)
    for dim in range(start_dim, end_dim+1):
        ishape[dim] += '^'
    oshape = ishape[:start_dim]
    oshape.append(ishape[start_dim:end_dim+1])
    anno = OpAnno.create_op_str([ishape], [oshape])
    return IRDimops(Flatten, 'flatten', signature, [anno], [tensor], start_dim=start_dim, end_dim=end_dim)


def Roll(signature, inputs: Tuple[IRTensor, Union[int, Tuple[int]], Union[int, Tuple[int]]]):
    tensor = inputs[0]
    shifts, dims = inputs[1:]
    ishape = ShapeAnno.create_shape_str(tensor.shape)
    for dim in range(len(ishape)):
        if dims is None or dim in dims:
            ishape[dim] += '^'
    anno = OpAnno.create_op_str([ishape], [ishape])
    return IRDimops(Roll, 'roll', signature, [anno], [tensor], shifts=shifts, dims=dims)


def AdaptiveAvgPool1d(signature, inputs: Tuple[IRTensor, Tuple[int]]):
    tensor = inputs[0]
    out_size = inputs[1]
    ishape = ShapeAnno.create_shape_str(tensor.shape)
    ishape[-1] += '^'
    oshape = ishape[:-1] + [str(size) for size in out_size]
    anno = OpAnno.create_op_str([ishape], [oshape])
    return IRDimops(AdaptiveAvgPool1d, 'adaptive_avg_pool1d', signature, [anno], [tensor], output_size=out_size)


def CrossEntropy(signature, inputs):
    # FIXME: reduction is by default 'mean', in this way it cannot be partitioned
    # no N dimension.
    tensor, target, weight = inputs[0:3]
    assert weight is None, "weight not supported for cross entropy"
    size_average, ignore_index, reduce, reduction, label_smoothing = inputs[3:]
    annos = [
        'C^, N -> 1',
        'N+ C, N+ -> 1',
        'N+ C *, N+ * -> 1'
    ]
    return IRDimops(
        CrossEntropy, 'cross_entropy',
        signature, annos, [tensor, target],
        weight=weight, size_average=size_average, ignore_index=ignore_index,
        reduce=reduce, reduction=reduction, label_smoothing=label_smoothing
    )



def MultiRef(signature, inputs: List[IRTensor]):
    """
    cube.runtime.function.multiref(itensor: torch.Tensor, times: int) -> Tuple[torch.Tensor]
    """
    signature = 'cube.runtime.function.multiref'
    itensor, times = inputs
    assert isinstance(itensor, IRTensor), "require all inputs to be IRSubTensor"
    assert isinstance(times, int), "require int for second input"
    anno = '* -> ' + ', '.join('*' for _ in range(times))
    node = IRDimops(MultiRef, 'multiref', signature, [anno], [itensor], times=times)
    return node


def GraphAnchor(signature, inputs: List[IRSubTensor]):
    """
    cube.runtime.function.anchor() -> None
    """
    name: str = inputs[0]
    node = IRGraphAnchor(signature, name)
    return node


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


def _comparison(creator: Callable, f: Callable, name: str, signature: str, inputs):
    """
    if both operands are scalars, returns bool.
    if one operand is a tensor, returns a broadcasted tensor with dtype being bool.
    
    @param creator Callable: the outside creation function
    @param f Callable: (Scalar, Scalar) -> bools
    """
    assert len(inputs) == 2
    lhs, rhs = inputs

    if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
        return f(lhs, rhs)

    annos = [
        '*, ? -> *',
        '?, * -> *',
    ]
    if isinstance(lhs, IRTensor) and isinstance(rhs, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(lhs, rhs)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(creator, name, signature, annos, inputs)


def CompareGT(signature, inputs):
    """
    torch.gt(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareGT, operator.gt, 'gt', signature, inputs)


def CompareLT(signature, inputs):
    """
    torch.lt(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareLT, operator.lt, 'lt', signature, inputs)


def CompareGE(signature, inputs):
    """
    torch.ge(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareGE, operator.ge, 'ge', signature, inputs)

def CompareLE(signature, inputs):
    """
    torch.gt(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareLE, operator.le, 'le', signature, inputs)
