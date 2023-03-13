from typing import Any, Callable, List, Optional, Tuple, Dict, Union, Iterable
import string
import copy
import torch
import warnings
import operator

from cube.ir.cten import IRTensor, IRObject
from cube.ir.tensor import IRSubTensor, IRFullTensor
from cube.ir.dtype import IRDType
from cube.graph.function.pyfunc import IRPyFunc
from cube.graph.function.dimops import DimopSplit, ShapeAnno, OpAnno, IRDimops, TransformRule
from cube.graph.function.conv import IRPad, IRConv2D, IRConv3D
from cube.graph.function.creators import IRArange, IREmpty, IROnes, IRToTensor, IRZeros, IRRand, IRNewTensor
from cube.graph.function.anchor import IRGraphAnchor


ErasedDevice = 'str'


def Identity(tensor: IRObject, signature = None):
    signature = 'cube.runtime.function.identity'
    eshape = ShapeAnno.create_shape_str(tensor.shape)
    anno = OpAnno.create_op_str([eshape], [eshape])
    return IRDimops(Identity, 'identity', signature, [anno], [tensor])


def MultiRef(tensor: IRTensor, times: int, signature = None):
    """
    cube.runtime.function.multiref(itensor: torch.Tensor, times: int) -> Tuple[torch.Tensor]
    """
    signature = 'cube.runtime.function.multiref'
    assert isinstance(tensor, IRTensor), "require all inputs to be IRSubTensor"
    assert isinstance(times, int), "require int for second input"
    anno = '* -> ' + ', '.join('*' for _ in range(times))
    node = IRDimops(MultiRef, 'multiref', signature, [anno], [tensor], times=times)
    return node


def Accum(*inputs, signature = None):
    """
    tensor = cube.runtime.function.accum(tensors)
    """
    assert all(isinstance(t, IRTensor) for t in inputs)
    signature = 'cube.runtime.function.accum'
    iannos = [ShapeAnno.create_shape_str(t.shape) for t in inputs]
    oannos = [copy.copy(iannos[0])]
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(Cat, 'accum', signature, [anno], inputs)


def Linear(input, weight, bias=None, signature = None):
    signature = 'torch.nn.functional.linear'
    if bias is None:
        annos = ['b * k+, n k+ -> b * n']
        return IRDimops(Linear, 'linear', signature, annos, [input, weight], bias=None)
    else:
        annos = ['b * k+, n k+, n -> b * n']
        rules = [TransformRule(
            [DimopSplit.D(-1), DimopSplit.D(1), DimopSplit.V()], [DimopSplit.V()]
        )]
        return IRDimops(Linear, 'linear', signature, annos, [input, weight, bias], rules)


def BatchLinear(input, mat2, *, out=None, signature = None):
    assert out is None
    annos = ['b m k+, b k+ n -> b m n']
    return IRDimops(BatchLinear, 'bmm', signature, annos, [input, mat2])


def BMMAdd(input, batch1, batch2, *, beta=1, alpha=1, out=None, signature = None):
    """
    torch.baddbmm(input, batch1, batch2, *, beta=1, alpha=1, out=None)
    """
    assert out is None
    in_dims = ['b', 'm', 'n']
    assert len(input.shape) == 3
    for i, size in enumerate(input.shape):
        if size == 1:
            in_dims[i] = '1'
    in_anno = ' '.join(in_dims)
    anno = f'{in_anno}, b m k^, b k^ n -> b m n'
    return IRDimops(BMMAdd, 'baddbmm', signature, [anno], [input, batch1, batch2], alpha=alpha, beta=beta)


def CubeEinSum(*operands, equation=None, signature = None):
    assert isinstance(equation, str)
    signature = 'cube.runtime.function.einsum'
    lhs, rhs = equation.split('->')
    assert ',' not in rhs
    lhs_dims = set(lhs.replace(',', ' ').split(' '))
    for dim in lhs_dims:
        if dim not in rhs:
            lhs = lhs.replace(dim, f'{dim}+')
    anno = f'{lhs} -> {rhs}'
    return IRDimops(CubeEinSum, 'einsum', signature, [anno], operands, equation=equation)

def EinSum(equation: str, *operands, signature = None):
    return CubeEinSum(*operands, equation=equation, signature=signature)


def Matmul(input, other, *, out=None, signature=None):
    assert out is None
    annos = [
        'm k+, k+ n -> m n',
        'k+, k+ n -> n',
        'm k+, k+ -> m',
        '* m k+, k+ n -> * m n',
        '* m k+, * k+ n -> * m n'  # TODO: broadcast
    ]
    if len(input.shape) > 2 and len(other.shape) > 2:
        assert tuple(input.shape[:-2]) == tuple(other.shape[:-2]), "broadcast of matmul (bmm) is not supported"
    return IRDimops(Matmul, 'matmul', signature, annos, [input, other])


def Arange(*args, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, signature=None):
    """
    torch.arange(start=0, end, step=1, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
    """
    if len(args) == 1:
        start, end, step = 0, args[0], 1
    elif len(args) == 2:
        start, end, step = args[0], args[1], 1
    elif len(args) == 3:
        start, end, step = args
    else:
        raise RuntimeError(f'Invalid number {len(args)} of args in Arange.')
    assert isinstance(start, int) and isinstance(end, int) and isinstance(step, int)
    from cube.graph.parser.mapping import DType2IRDType
    if dtype is None:
        dtype = torch.get_default_dtype()

    import math
    size = (math.ceil((end-start)/step),)
    kwargs = {'start': start, 'end': end, 'step': step, 'out': out, 'dtype': dtype,
              'layout': layout, 'requires_grad': requires_grad}
    return IRArange(signature, size, 'arange', **kwargs)


def Empty(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False,
          pin_memory=False, memory_format=torch.contiguous_format, signature=None):
    """
    torch.empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None,
    requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format) → Tensor
    """
    from cube.graph.parser.mapping import DType2IRDType
    if dtype is None:
        dtype = torch.get_default_dtype()
    ir_dtype : IRDType = DType2IRDType.map(dtype)
    # example size: ((17, 17),)
    assert isinstance(size, tuple) and isinstance(size[0], tuple)
    for dim, i in enumerate(size[0]):
        if not isinstance(dim, int) and not dim >= 0:
            raise RuntimeWarning(f"The {i}-th component of the size must be non-negative integer")
    kwargs = {'dtype': ir_dtype, 'layout': layout, 'device': device, 'requires_grad': requires_grad,
              'pin_memory': pin_memory, 'memory_format': memory_format}
    return IREmpty(signature, size[0], 'empty', **kwargs)


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

def NewTensor(data: Union[int, float, list], dtype=None, device=None, requires_grad=False, pin_memory=False, signature=None):
    # NOTE: not sure all the keys of torch.tensor
    assert requires_grad == False
    from cube.graph.parser.mapping import DType2IRDType
    if dtype is None:
        dtype = torch.get_default_dtype()
    ir_dtype : IRDType = DType2IRDType.map(dtype)
    kwargs = {'dtype': ir_dtype, 'device': device, 'requires_grad': requires_grad, 'pin_memory': pin_memory}
    return IRNewTensor(signature, data, 'tensor', **kwargs)


# def NewTensor(signature,
#               inputs: Tuple[ list, Optional[int], ErasedDevice, bool ]):
#     # aten::tensor(t[] data, *, ScalarType? dtype=None, Device? device=None, bool requires_grad=False) -> Tensor
#     #
#     # REMARK: in the PyTorch-internal operator definition expression, an asterisk ("*") is merely a marker of
#     #         the beginning of the sublist of _keyword arguments_, and does not result in an actual argument.

#     data, dtype_underlying, _erased_device, requires_grad = inputs

#     # TODO parameters to support, currently they are all None
#     assert requires_grad == False
#     from cube.graph.parser.mapping import DType2IRDType, TorchScalarTypeEnumMap

#     if dtype_underlying is not None:
#         # If some torch.dtype is specified at the frontend, in TorchScript it becomes an int,
#         # which is the underlying type of PyTorch C++ enum 'ScalarType'.
#         dtype = TorchScalarTypeEnumMap.map(dtype_underlying)
#     else:
#         dtype = torch.get_default_dtype()

#     ir_dtype : IRDType = DType2IRDType.map(dtype)

#     # if 'data' is not:
#     # 1) ints or floats of any precision, e.g. i8, i64, f16, f32
#     # 2) non-ragged
#     # ... then this call will throw.
#     arr = torch.tensor(data, dtype=dtype)

#     # TODO temporarily fake creation with Zeros
#     # and remark that originally aten::tensor should be able to infer the dtype from the specified 'data',
#     # but since we have omitted the 'data', we must do type inferrence ourselves,
#     # only in this way we get correct dtype e.g. ints or bools.
#     return IRNewTensor(signature, data, 'tensor', ir_dtype=ir_dtype)

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


def Expand(input, *sizes, signature = None):
    """
    torch.Tensor.expand(*sizes)
    """
    signature = 'cube.runtime.function.expand'
    edim_in = ShapeAnno.create_shape_str(input.shape)
    assert len(input.shape) == len(sizes)
    for idx, (dim, expand_dim) in enumerate(zip(input.shape, sizes)):
        if dim == 1 and dim != expand_dim:
            edim_in[idx] += '^'
    edim_ou = copy.copy(edim_in)
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Expand, 'expand', signature, [anno], [input], sizes=sizes)


def Clone(input, *, memory_format=None, signature = None):
    """
    torch.clone(input, *, memory_format=torch.preserve_format)
    """
    assert memory_format is None, f"Not supported for a specific memory format"
    annos = ['* -> *']
    return IRDimops(Clone, 'clone', signature, annos, [input])


def BitwiseOr(input, other, *, out=None, signature=None):
    """
    torch.bitwise_or(input, other, *, out=None) → Tensor
    """
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input | other
    assert isinstance(input, IRTensor) and isinstance(other, IRTensor)
    annos = ['*, * -> *']
    return IRDimops(BitwiseOr, 'bitwise_or', signature, annos, [input, other])


def BitwiseNot(input, *, out=None, signature=None):
    assert out is None
    if not isinstance(input, IRObject):
        return ~input
    assert isinstance(input, IRTensor)
    annos = ['* -> *']
    return IRDimops(BitwiseNot, 'bitwise_not', signature, annos, [input])


def Add(input, other, alpha=1, *, out=None, signature = None):
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input + alpha * other
    signature = 'torch.add'
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
        return IRDimops(Add, 'add', signature, annos, [input, other], alpha=alpha)
    else:
        annos = ['* -> *']
        if isinstance(input, IRTensor):
            return IRDimops(Add, 'add', signature, annos, [input], other=other, alpha=alpha)
        else:
            return IRDimops(Add, 'add', signature, annos, [other], other=input, alpha=alpha)


def CubeSub(input, other, alpha=1, *, out=None, signature = None):
    signature = 'cube.runtime.function.sub'
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
        return IRDimops(CubeSub, 'sub', signature, annos, [input, other], alpha=alpha, swap_operands=False)
    else:
        annos = ['* -> *']
        if isinstance(input, IRTensor):
            return IRDimops(CubeSub, 'sub', signature, annos, [input], other=other, alpha=alpha, swap_operands=False)
        else:
            return IRDimops(CubeSub, 'sub', signature, annos, [other], other=input, alpha=alpha, swap_operands=True)


def Sub(input, other, alpha=1, *, out=None, signature = None):
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input - alpha * other
    return CubeSub(input, other, alpha, out=out, signature=signature)


def Mul(input, other, *, out=None, signature = None):
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input * other
    signature = 'torch.mul'
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
        return IRDimops(Mul, 'mul', signature, annos, [input, other])
    else:
        annos = ['* -> *']
        if isinstance(input, IRTensor):
            return IRDimops(Mul, 'mul', signature, annos, [input], other=other)
        else:
            return IRDimops(Mul, 'mul', signature, annos, [other], other=input)


def Div(input, other, *, rounding_mode=None, out=None, signature = None):
    assert rounding_mode is None and out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input / other
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
        return IRDimops(Div, 'div', signature, annos, [input, other])
    else:
        # if not all tensors, the second must not be IRObject
        assert isinstance(input, IRTensor) and not isinstance(other, IRObject)
        annos = ['* -> *']
        return IRDimops(Div, 'div', signature, annos, [input], other=other)


def FloorDiv(input, other, *, out=None, signature = None):
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return input // other
    if (not isinstance(input, IRTensor)) and (not isinstance(other, IRTensor)):
        return IRPyFunc(signature, [input, other], [IRObject()])
    annos = ['*, ? -> *', '?, * -> *',]
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(FloorDiv, 'floordiv', signature, annos, [input, other])


def Pow(input, exponent, *, out=None, signature = None):
    assert out is None
    if (not isinstance(input, IRObject)) and (not isinstance(exponent, IRObject)):
        return input ** exponent
    annos = ['*, ? -> *', '?, * -> *',]
    if isinstance(input, IRTensor) and isinstance(exponent, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, exponent)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
    return IRDimops(Pow, 'pow', signature, annos, [input, exponent])


def Neg(input, *, out=None, signature = None):
    assert out is None
    if not isinstance(input, IRObject): return -1 * input
    annos = ['* -> *']
    return IRDimops(Neg, 'neg', signature, annos, [input])


def Sin(input, *, out=None, signature = None):
    assert out is None
    annos = ['* -> *']
    return IRDimops(Sin, 'sin', signature, annos, [input])


def Cos(input, *, out=None, signature = None):
    assert out is None
    annos = ['* -> *']
    return IRDimops(Cos, 'cos', signature, annos, [input])


def Tanh(input, *, out=None, signature = None):
    assert out is None
    annos = ['* -> *']
    return IRDimops(Tanh, 'tanh', signature, annos, [input])


def GeLU(input, approximate='none', signature = None):
    annos = ['* -> *']
    signature = 'torch.nn.functional.gelu'
    return IRDimops(GeLU, 'gelu', signature, annos, [input], approximate=approximate)


def SiLU(input, inplace=False, signature = None):
    annos = ['* -> *']
    signature = 'torch.nn.functional.silu'
    return IRDimops(SiLU, 'silu', signature, annos, [input], inplace=inplace)


def ReLU(input, inplace=False, signature = None):
    annos = ['* -> *']
    signature = 'torch.nn.functional.relu'
    return IRDimops(ReLU, 'relu', signature, annos, [input], inplace=inplace)


def Softmax(input, dim=None, _stacklevel=3, dtype=None, signature = None):
    """
    torch.nn.functional.softmax(input, dim=None, _stacklevel=3, dtype=None)
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = copy.copy(edim_in)
    if dim is not None:
        edim_in[dim] += '^'
        edim_ou[dim] += '^'
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Softmax, 'softmax', signature, [anno], [input],
                    dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def Dropout(input, p=0.5, training=True, inplace=False, signature = None):
    """
    torch.nn.functional.dropout(input, p=0.5, training=True, inplace=False)
    """
    annos = ['* -> *']
    return IRDimops(Dropout, 'dropout', signature, annos, [input],
                    p=p, training=training, inplace=inplace)


def Detach(input, signature = None):
    """
    torch.Tensor.detach(input)
    """
    annos = ['* -> *']
    return IRDimops(Detach, 'detach', signature, annos, [input])


def NanToNum(input, nan=0.0, posinf=None, neginf=None, *, out=None, signature = None):
    assert out is None
    annos = ['* -> *']
    return IRDimops(NanToNum, 'nan_to_num', signature, annos, [input], nan=nan, posinf=posinf, neginf=neginf)


def Long(input, memory_format=None, signature = None):
    """
    torch.Tensor.long(memory_format=torch.preserve_format)
    """
    assert memory_format is None
    annos = ['* -> *']
    return IRDimops(Long, 'long', signature, annos, [input])


def Fill(input, value, signature = None):
    """
    torch.Tensor.fill_(value)
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = copy.copy(edim_in)
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Fill, 'fill', signature, [anno], [input], value=value)


def MaskedFill(input, mask, value, signature = None):
    """
    torch.Tensor.masked_fill_(mask, value)
    """
    edim_in0 = ShapeAnno.create_shape_str(input.shape)
    edim_in1 = ShapeAnno.create_shape_str(mask.shape)
    edim_ou = copy.copy(edim_in0)
    #TODO: add broadcast rule
    for idx, (lhs, rhs) in enumerate(zip(input.shape, mask.shape)):
        if lhs != rhs and rhs == 1:
            edim_in1[idx] = '1'
    anno = OpAnno.create_op_str([edim_in0, edim_in1], [edim_ou])
    return IRDimops(MaskedFill, 'masked_fill', signature, [anno], [input, mask], value=value)


def CubeLayerNorm(input, weight=None, bias=None, normalized_shape=None, eps=1e-05, signature = None):
    """
    cube.runtime.function.layer_norm(input, weight, bias, normliazed_shape, eps)
    """
    signature = 'cube.runtime.function.layer_norm'
    assert not (weight is None and bias is not None), f"Not support for None of weight and parameter of bias"
    letters = iter(string.ascii_lowercase)
    einput = ShapeAnno.create_shape_str(input.shape, iterator=letters)
    eoutput = copy.copy(einput)
    ndims = len(input.shape)
    for dim in range(len(normalized_shape)):
        einput[ndims-1-dim] += '^'
        eoutput[ndims-1-dim] += '^'
    einputs, inputs = [einput], [input]
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
    return IRDimops(CubeLayerNorm, 'layernorm', signature, [anno], inputs, **kwargs)


def LayerNorm(input, normalized_shape, weight=None, bias=None, eps=1e-05, signature = None):
    """
    torch.nn.functional.layer_norm(input, normliazed_shape, weight=None, bias=None, eps)
    """
    return CubeLayerNorm(input, weight, bias, normalized_shape, eps, signature=signature)


def Sum(input, dim=None, keepdim=False, *, dtype=None, signature = None):
    """
    torch.sum(input, *, dtype=None) -> Tensor
    torch.sum(input, dim, keepdim=False, *, dtype=None) -> Tensor

    @note troch.sum is overrided by two signatures, which may lead mismatch in torch.jit.script:
        may get (input, dtype) as input
    """
    assert dtype is None, "Currently Sum only support dtype=None"
    einput = ShapeAnno.create_shape_str(input.shape)
    eoutput = copy.copy(einput)
    if dim is None:
        einput = [edim + '+' for edim in einput]
        anno = OpAnno.create_op_str([einput], ['1'])
        return IRDimops(Sum, 'sum', signature, [anno], [input])
    else:
        dim = (dim,) if isinstance(dim, int) else dim
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
        return IRDimops(Sum, 'sum', signature, [anno], [input], dim=dim, keepdim=keepdim)

def Mean(input, dim=None, keepdim=False, *, dtype=None, signature = None):
    """
    torch.mean(input, *, dtype=None) -> Tensor
    torch.mean(input, dim, keepdim=False, *, dtype=None) -> Tensor
    """
    assert dtype is None
    einput = ShapeAnno.create_shape_str(input.shape)
    eoutput = copy.copy(einput)
    dim = (dim,) if isinstance(dim, int) else dim
    if dim is not None:
        sort_dim = sorted(dim)
        for dimidx in sort_dim[::-1]:
            eoutput.pop(dimidx)
            einput[dimidx] = einput[dimidx] + '^'
    else:
        eoutput = ['1']
        einput = [edim + '^' for edim in einput]
    anno = OpAnno.create_op_str([einput], [eoutput])
    if dim is not None:
        return IRDimops(Mean, 'mean', signature, [anno], [input], dim=dim, keepdim=keepdim)
    else:
        return IRDimops(Mean, 'mean', signature, [anno], [input])


def Transpose(input, dim0, dim1, signature = None):
    """
    out = torch.transpose(tensor, dim0, dim1)
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = copy.copy(edim_in)
    edim_ou[dim0], edim_ou[dim1] = edim_ou[dim1], edim_ou[dim0]
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Transpose, 'transpose', signature, [anno], [input],
                    dim0=dim0, dim1=dim1)


def View(input, size: Tuple[int], *arg_size, signature = None):
    """
    out = torch.Tensor.view(tensor: torch.Tensor, *size)
    """
    size = (size,) if isinstance(size, int) else tuple(size)
    size = size + arg_size
    assert all([isinstance(dim, int) for dim in size]), \
        f"Expected tensor.view has static int shape but got: {size}"
    in_shape, ou_shape = list(input.shape), list(size)

    # infer -1
    def nele(shape, nele=1):
        for dimlen in shape: nele *= dimlen
        return nele

    cnt = nele(in_shape)
    if -1 in ou_shape:
        idx = ou_shape.index(-1)
        ou_shape[idx] = cnt // (-nele(ou_shape))
    assert nele(in_shape) == nele(ou_shape), f"shape mismatch: {in_shape}, {ou_shape}"

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
    ispatial, ifirst = set(), []
    for bracket in in_anno:
        sdim = None
        for hdim in range(len(bracket)):
            if bracket[hdim] == '1' or shape_map[bracket[hdim]] == 1: continue
            sdim = bracket[hdim]
            break
        if sdim is not None:
            ispatial.add(sdim)
        ifirst.append(sdim)

    ospatial, ofirst = set(), []
    for bracket in ou_anno:
        sdim = None
        for hdim in range(len(bracket)):
            if bracket[hdim] == '1' or shape_map[bracket[hdim]] == 1: continue
            sdim = bracket[hdim]
            break
        if sdim is not None:
            ospatial.add(sdim)
        ofirst.append(sdim)
    
    # intersection for spatial partitioned dimensions
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
        identifier = ifirst[dim]
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
    return IRDimops(View, 'view', signature, [anno], [input], rules, size=tuple(size))


def Reshape(input, shape: Tuple[int], *arg_shape, signature = None):
    """
    torch.reshape(Tensor self, int[] shape) -> Tensor
    """

    size = (shape,) if isinstance(shape, int) else tuple(shape)
    size = size + arg_shape
    assert all([isinstance(dim, int) for dim in size]), \
        f"Expected tensor.view has static int shape but got: {size}"
    in_shape, ou_shape = list(input.shape), list(size)

    # infer -1
    def nele(shape, nele=1):
        for dimlen in shape: nele *= dimlen
        return nele

    cnt = nele(in_shape)
    if -1 in ou_shape:
        idx = ou_shape.index(-1)
        ou_shape[idx] = cnt // (-nele(ou_shape))
    assert nele(in_shape) == nele(ou_shape), f"shape mismatch: {in_shape}, {ou_shape}"

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
    ispatial, ifirst = set(), []
    for bracket in in_anno:
        sdim = None
        for hdim in range(len(bracket)):
            if bracket[hdim] == '1' or shape_map[bracket[hdim]] == 1: continue
            sdim = bracket[hdim]
            break
        if sdim is not None:
            ispatial.add(sdim)
        ifirst.append(sdim)

    ospatial, ofirst = set(), []
    for bracket in ou_anno:
        sdim = None
        for hdim in range(len(bracket)):
            if bracket[hdim] == '1' or shape_map[bracket[hdim]] == 1: continue
            sdim = bracket[hdim]
            break
        if sdim is not None:
            ospatial.add(sdim)
        ofirst.append(sdim)

    # intersection for spatial partitioned dimensions
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
        identifier = ifirst[dim]
        oidx = ofirst.index(identifier)
        size = list(kwargs['shape'])
        size[oidx] = size[oidx] // num
        kwargs['shape'] = tuple(size)
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

    new_signature = 'torch.Tensor.reshape'
    return IRDimops(Reshape, 'shape', new_signature, [anno], [input], rules, shape=tuple(size))


def Permute(input, dims: Tuple[int], *arg_dims, signature = None):
    """
    torch.Tensor.permute(input, *dims)
    torch.permute(input, dims: Tuple[int])
    """
    dims = (dims,) if isinstance(dims, int) else tuple(dims)
    dims = dims + arg_dims
    assert all(isinstance(dim, int) for dim in dims), f"but got {dims}"
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = [copy.copy(edim_in[dim]) for dim in dims]
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Permute, 'permute', signature, [anno], [input], dims=dims)


def Squeeze(input, dim=None, signature = None):
    """
    out = torch.squeeze(tensor)
    """
    assert dim is None, "got dim: {dim} != None, which is not supported"
    edim_in = ShapeAnno.create_shape_str(input.shape)
    assert len(edim_in) == len(input.shape)
    edim_ou = []
    for dim_anno, dim_size in zip(edim_in, input.shape):
        if dim_size > 1:
            edim_ou.append(copy.copy(dim_anno))
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Squeeze, 'squeeze', signature, [anno], [input])


def Unsqueeze(input, dim, signature = None):
    """
    out = torch.unsqueeze(tensor, dim)
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = copy.copy(edim_in)
    edim_ou.insert(dim, '1')
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Unsqueeze, 'unsqueeze', signature, [anno], [input],dim=dim)


def TypeAs(input, tensor, signature = None):
    """
    out = torch.Tensor.type_as(tensor0, tensor1)
    """
    edim_in0 = ShapeAnno.create_shape_str(tensor.shape)
    anno = OpAnno.create_op_str(['*', edim_in0], ['*'])
    return IRDimops(TypeAs, 'type_as', signature, [anno], [input, tensor])


def Triu(input, diagonal=0, *, out=None, signature = None):
    """
    out = torch.triu(tensor, diagonal)
    """
    edim_in = ShapeAnno.create_shape_str(input.shape)
    assert len(edim_in) >= 2
    edim_in[-1] += '^'
    edim_in[-2] += '^'
    edim_ou = copy.copy(edim_in)
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(Triu, 'triu', signature, [anno], [input], diagonal=diagonal)


def CumSum(tensor, dim, signature = None):
    """
    out = torch.cumsum(tensor, dim)
    """
    edim_in = ShapeAnno.create_shape_str(tensor.shape)
    edim_in[dim] += '^'
    edim_ou = copy.copy(edim_in)
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(CumSum, 'cumsum', signature, [anno], [tensor], dim=dim)


# def Pad(signature, inputs):
#     """
#     torch.nn.functional.pad(input: torch.Tensor, pad: List[int], mode='constant', value=0.0)
#     """
#     signature = 'torch.nn.functional.pad'
#     tensor, pad, mode, value = inputs
#     ianno = ShapeAnno.create_shape_str(tensor.shape)
#     oanno = []
#     ndims = len(pad) // 2
#     for dim in range(ndims):
#         pad_left, pad_right = pad[2 * dim], pad[2 * dim + 1]
#         if pad_left == 0 and pad_right == 0:
#             oanno.insert(0, ianno[-1-dim])
#         else:
#             ianno[-1-dim] = str(tensor.shape[-1-dim])
#             oanno.insert(0, str(tensor.shape[-1-dim] + pad_left + pad_right))
#     oanno = copy.copy(ianno[:len(tensor.shape) - ndims]) + oanno
#     anno = OpAnno.create_op_str([ianno], [oanno])
#     return IRDimops(Pad, 'pad', signature, [anno], [tensor], pad=pad, mode=mode, value=value)


def Pad(input, pad, mode='constant', value=0.0, signature = None):
    """
    torch.nn.functional.pad(input, pad, mode='constant', value=0.0)
    """
    return IRPad(signature, [input], 'pad', pad=pad, mode=mode, value=value)


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


def Conv2D(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, signature = None):
    """
    torch.nn.functional.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    """
    if isinstance(padding, int):
        padding = [padding] * 4
    elif len(padding) == 2:
        padH, padW = padding
        padding = [padH, padH, padW, padW]
    return IRConv2D(signature, [input, weight, bias], 'conv2d',
                    stride=stride, padding=padding, dilation=dilation, groups=groups)


def Conv3D(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, signature = None):
    """
    torch.nn.functional.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    """
    if isinstance(padding, int):
        padding = [padding] * 4
    elif len(padding) == 2:
        padH, padW = padding
        padding = [padH, padH, padW, padW]
    return IRConv3D(signature, [input, weight, bias], 'conv3d',
                    stride=stride, padding=padding, dilation=dilation, groups=groups)


def CubeCat(*tensors, dim: int, signature = None):
    """
    torch.cat(tensors, dim=0, *, out=None)
    """
    # REMARK: IRFwOperation doesn't support taking a list of IRTensors.
    # Therefore, the argument interface is adapted to take unpacked tensors
    # with dimension. dim=None is for the support of kwarg inputs from torchfx
    assert all(isinstance(tensor, IRTensor) for tensor in tensors)
    assert isinstance(dim, int)
    iannos = [ShapeAnno.create_shape_str(t.shape) for t in tensors]
    dimlens = [t.shape[dim] for t in tensors]
    for ashape, dimlen in zip(iannos, dimlens):
        ashape[dim] = str(dimlen)
    oannos = [copy.copy(iannos[-1])]
    oannos[0][dim] = str(sum(dimlens))
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(CubeCat, 'cat', signature, [anno], tensors, dim=dim)


def Cat(*tensors_and_dim, dim=0, out=None, signature=None):
    """
    torch.cat(tensors, dim=0, *, out=None)
    """
    assert out is None
    if len(tensors_and_dim) == 2:
        tensors, dim = tensors_and_dim[0], tensors_and_dim[1]
    else:
        tensors = tensors_and_dim[0]
    return CubeCat(*tensors, dim=dim, signature=signature)


def CubeStack(*tensors, dim=0, signature=None):
    # REMARK: IRFwOperation doesn't support taking a list of IRTensors.
    # Therefore, the argument interface is adapted to take unpacked tensors
    # with dimension.
    assert all(isinstance(tensor, IRTensor) for tensor in tensors), f'but got {tensors}'
    assert isinstance(dim, int), f"but not {dim}"
    signature = 'cube.runtime.function.stack'
    iannos = [ShapeAnno.create_shape_str(t.shape) for t in tensors]
    oannos = [copy.copy(iannos[-1])]
    oannos[0].insert(dim, str(len(tensors)))
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(CubeStack, 'stack', signature, [anno], tensors, dim=dim)


def Stack(tensors, dim=0, out=None, signature = None):
    """
    torch.stack(tensors, dim=0, *, out=None)
    """
    return CubeStack(*tensors, dim=dim, signature=signature)


def Chunk(input, chunks, dim=0, signature = None):
    """
    torch.chunk(input, chunks, dim=0)
    """
    assert input.shape[dim] % chunks == 0
    iannos = [ShapeAnno.create_shape_str(input.shape)]
    oannos = [copy.copy(iannos[0]) for _ in range(chunks)]
    iannos[0][dim] = str(input.shape[dim])
    for oanno in oannos:
        oanno[dim] = str(input.shape[dim] // chunks)
    anno = OpAnno.create_op_str(iannos, oannos)
    return IRDimops(Chunk, 'chunk', signature, [anno], [input], chunks=chunks, dim=dim)


def Select(input, dim, index, signature = None):
    """
    torch.select(self:Tensor, dim:int, index:int) -> Tensor
    """
    ianno = ShapeAnno.create_shape_str(input.shape)
    oanno = copy.copy(ianno)
    ianno[dim] += '^'
    oanno.pop(dim)
    anno = OpAnno.create_op_str([ianno], [oanno])
    return IRDimops(Select, 'select', signature, [anno], [input], dim=dim, index=index)


def CubeIndexSelect(input: torch.Tensor, index: torch.Tensor, dim: int, signature = None):
    signature = 'cube.runtime.function.index_select'
    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_in[dim] += '^'
    idx_anno = chr(ord(edim_in[-1]) + 1) + '^'
    edim_ou = copy.copy(edim_in)
    edim_ou[dim] = copy.copy(idx_anno)
    anno = OpAnno.create_op_str([edim_in, idx_anno], [edim_ou])
    # FIXME: runtime function support
    return IRDimops(CubeIndexSelect, 'index_select', signature, [anno], [input, index], dim=dim)


def IndexSelect(input: torch.Tensor, dim: int, index: torch.Tensor, *, out=None, signature = None):
    assert out is None
    return CubeIndexSelect(input, index, dim, signature=signature)


def FullSlice(tensor: IRTensor, slicers: Tuple[Union[None, slice]], signature=None):
    """
    subtensor = tensor[:,128:]
    subtensor = tensor[0,128:]
    subtensor = tensor[0]
    """
    signature = 'cube.runtime.function.fullslice'
    slicers = tuple(slicers) + (None,) * (len(tensor.shape) - len(slicers))
    edim_in = ShapeAnno.create_shape_str(tensor.shape)
    edim_ou = []
    for dim, slicer in enumerate(slicers):
        if slicer is None:
            if dim < len(edim_in):
                edim_ou.append(edim_in[dim])
            else:
                # expand the dimension
                edim_ou.append('1')
        else:
            edim_in[dim] += '^'
            if isinstance(slicer, slice):
                stop = tensor.shape[dim] if slicer.stop is None else slicer.stop
                start = 0 if slicer.start is None else slicer.start
                step = 1 if slicer.step is None else slicer.step
                dimlen = len(range(start, stop, step))
                edim_ou.append(str(dimlen))
            else:
                pass  # no shape for int
    # special case for loss = torch.Tensor([1,2,3])[0]
    if len(edim_ou) == 0:
        edim_ou = ['1^']
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(FullSlice, 'fullslice', signature, [anno], [tensor], slicers=slicers)


def Slice(tensor: torch.Tensor, dim, start, end, step, signature = None):
    """
    aten::slice(input:Tensor, dim:int, start:Optional[int], end:Optional[int], step:int) -> Tensor
    """
    signature = 'torch.ops.aten.slice'
    ianno = ShapeAnno.create_shape_str(tensor.shape)
    oanno = copy.copy(ianno)
    ianno[dim] = str(tensor.shape[dim])
    
    def clip(ofst):
        ofst = ofst + tensor.shape[dim] if ofst < 0 else ofst
        return min(tensor.shape[dim], max(0, ofst))

    # set start and end to possitive itegers
    start = 0 if start is None else start
    end = tensor.shape[dim] if end is None else end
    start, end = clip(start), clip(end)

    oanno[dim] = str(len(range(start, end, step)))
    anno = OpAnno.create_op_str([ianno], [oanno])
    return IRDimops(Slice, 'slice', signature, [anno], [tensor], dim=dim, start=start, end=end, step=step)


def SelectScatter(self: torch.Tensor, input: torch.Tensor, dim: int, index: int, signature = None):
    """
    torch.select_scatter(self:Tensor, input:Tensor, dim:int, index:int) -> Tensor
    """
    # 'torch.select_scatter' isn't supported by Torch2ONNX yet.
    signature = 'cube.runtime.function.select_scatter'
    # shape check
    self_shape, input_shape = self.shape, input.shape
    self_shape.pop(dim)
    assert tuple(self_shape) == tuple(input_shape)
    in1_anno = ShapeAnno.create_shape_str(self.shape)
    in2_anno = in1_anno.copy()
    in2_anno.pop(dim)
    in1_anno[dim] = str(self.shape[dim])
    out_anno = in1_anno.copy()
    anno = OpAnno.create_op_str([in1_anno, in2_anno], [out_anno])
    return IRDimops(SelectScatter, 'select_scatter', signature, 
                    [anno], [self, input], dim=dim, index=index)


def Repeat(tensor, repeats: Tuple[int], *arg_repeats, signature = None):
    """
    torch.Tensor.repeat(*sizes)
    """
    signature = 'torch.ops.aten.repeat'
    repeats = (repeats,) if isinstance(repeats, int) else tuple(repeats)
    repeats = repeats + arg_repeats
    in_shape = tensor.shape
    assert len(in_shape) <= len(repeats), "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor"
    expand = len(repeats) - len(tensor.shape)
    in_shape += [1] * expand
    ou_shape = [dimlen * repeat for dimlen, repeat in zip(in_shape, repeats)]
    ianno, oanno = ShapeAnno.create_shape_str(in_shape), []
    for dim, dimlen in enumerate(ou_shape):
        if dim < expand:
            oanno.append(str(dimlen))
        else:
            if repeats[dim] != 1:
                ianno[dim] += '^'
                dim_anno = [str(repeats[dim]), ianno[dim]]
            else:
                dim_anno = ianno[dim]
            oanno.append(dim_anno)
    anno = OpAnno.create_op_str([ianno[expand:]], [oanno])
    return IRDimops(Repeat, 'repeat', signature, [anno], [tensor], repeats=repeats)


def CubeEmbedding(input, weight, padding_idx, signature = None, **kwargs):
    """
    cube.runtime.function.embedding(input, weight, padding_idx, start, stop)
    """
    signature = 'cube.runtime.function.embedding'
    if isinstance(weight, IRSubTensor):
        start, stop = weight.indmap[0]
    else:
        start, stop = 0, weight.shape[0]
    annos = ['*, n+ e -> * e']
    return IRDimops(CubeEmbedding, 'embedding', signature, annos, [input, weight],
                    padding_idx=padding_idx, start=start, stop=stop)


def Embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0,
              scale_grad_by_freq=False, sparse=False, signature = None):
    """
    torch.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)
    """
    assert max_norm is None and norm_type == 2.0 and (not scale_grad_by_freq) and (not sparse)
    return CubeEmbedding(input, weight, padding_idx, signature=signature)


def Flatten(input, start_dim=0, end_dim=-1, signature = None):
    start_dim = len(input.shape) + start_dim if start_dim < 0 else start_dim
    end_dim = len(input.shape) + end_dim if end_dim < 0 else end_dim
    ishape = ShapeAnno.create_shape_str(input.shape)
    for dim in range(start_dim, end_dim+1):
        ishape[dim] += '^'
    oshape = ishape[:start_dim]
    oshape.append(ishape[start_dim:end_dim+1])
    anno = OpAnno.create_op_str([ishape], [oshape])
    return IRDimops(Flatten, 'flatten', signature, [anno], [input],
                    start_dim=start_dim, end_dim=end_dim)


def Roll(input, shifts: Union[int, Tuple[int]], dims=None, signature = None):
    shifts = (shifts,) if isinstance(shifts, int) else shifts
    ishape = ShapeAnno.create_shape_str(input.shape)
    for dim in range(len(ishape)):
        if dims is None or dim in dims:
            ishape[dim] += '^'
    anno = OpAnno.create_op_str([ishape], [ishape])
    return IRDimops(Roll, 'roll', signature, [anno], [input], shifts=shifts, dims=dims)


def Inverse(input, *, out=None, signature=None):
    """
    torch.inverse(input, *, out=None) → Tensor
    """
    ishape = ShapeAnno.create_shape_str(input.shape)
    ishape = [i + '^' for i in ishape]
    oshape = copy.copy(ishape)
    anno = OpAnno.create_op_str([ishape], [oshape])
    return IRDimops(Inverse, 'inverse', signature, [anno], [input])


def AdaptiveAvgPool1d(input, output_size, signature = None):
    """
    torch.nn.functional.adaptive_avg_pool2d(input, output_size)
    """
    ishape = ShapeAnno.create_shape_str(input.shape)
    ishape[-1] += '^'
    oshape = ishape[:-1] + [str(size) for size in output_size]
    anno = OpAnno.create_op_str([ishape], [oshape])
    return IRDimops(AdaptiveAvgPool1d, 'adaptive_avg_pool1d', signature, [anno], [input], output_size=output_size)


def CrossEntropy(input, target, weight=None, 
                 size_average=None, ignore_index=- 100, reduce=None,
                 reduction='mean', label_smoothing=0.0, signature = None):
    """
    torch.nn.functional.cross_entropy(
        input, target, weight=None, 
        size_average=None, ignore_index=- 100, reduce=None,
        reduction='mean', label_smoothing=0.0)
    """
    # FIXME: reduction is by default 'mean', in this way it cannot be partitioned
    # no N dimension.
    annos = [
        'C^, N -> 1',
        'N+ C, N+ -> 1',
        'N+ C *, N+ * -> 1'
    ]
    return IRDimops(
        CrossEntropy, 'cross_entropy',
        signature, annos, [input, target],
        weight=weight, size_average=size_average, ignore_index=ignore_index,
        reduce=reduce, reduction=reduction, label_smoothing=label_smoothing
    )


def GraphAnchor(name: str, signature = None):
    """
    cube.runtime.function.anchor() -> None
    """
    node = IRGraphAnchor(signature, name)
    return node


def _comparison(creator: Callable, f: Callable, name: str, signature: str, 
                input, other):
    """
    if both operands are scalars, returns bool.
    if one operand is a tensor, returns a broadcasted tensor with dtype being bool.
    
    @param creator Callable: the outside creation function
    @param f Callable: (Scalar, Scalar) -> bools
    """
    # case 0: return constant
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return f(input, other)
    # case1: torch.equal(tensor1, tensor2)
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
        return IRDimops(creator, name, signature, annos, [input, other])
    # case2: torch.equal(tensor1, obj2) / torch.equal(obj1, tensor2)
    if isinstance(input, IRTensor) or isinstance(other, IRTensor):
        annos = ['*, ? -> *', '?, * -> *',]
        return IRDimops(creator, name, signature, annos, [input, other])
    # case3: torch.equal(obj1, obj2)
    else:
        return IRPyFunc(signature, [input, other], [IRObject()])

def _comparison_hack(creator: Callable, f: Callable, name: str, signature: str, 
                input, other):
    """
    if both operands are scalars, returns bool.
    if one operand is a tensor, returns a broadcasted tensor with dtype being bool.
    
    @param creator Callable: the outside creation function
    @param f Callable: (Scalar, Scalar) -> bools
    """
    # case 0: return constant
    if (not isinstance(input, IRObject)) and (not isinstance(other, IRObject)):
        return f(input, other)
    # case1: torch.equal(tensor1, tensor2)
    if isinstance(input, IRTensor) and isinstance(other, IRTensor):
        lshape, rshape, oshape = _handle_broadcast(input, other)
        annos = [OpAnno.create_op_str([lshape, rshape], [oshape])]
        return IRDimops(creator, name, signature, annos, [input, other])
    # case2: torch.equal(tensor1, obj2) / torch.equal(obj1, tensor2)
    if isinstance(input, IRTensor) or isinstance(other, IRTensor):
        annos = ['* -> *']
        if isinstance(input, IRTensor):
            return IRDimops(creator, name, signature, annos, [input], other=other)
        else:
            return IRDimops(creator, name, signature, annos, [other], other=input)
    # case3: torch.equal(obj1, obj2)
    else:
        return IRPyFunc(signature, [input, other], [IRObject()])


def CompareGT(input, other, *, out=None, signature = None):
    """
    torch.gt(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareGT, operator.gt, 'gt', signature, input, other)


def CompareLT(input, other, *, out=None, signature = None):
    """
    torch.lt(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareLT, operator.lt, 'lt', signature, input, other)


def CompareGE(input, other, *, out=None, signature = None):
    """
    torch.ge(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareGE, operator.ge, 'ge', signature, input, other)


def CompareLE(input, other, *, out=None, signature = None):
    """
    torch.gt(input, other, *, out=None) -> Tensor
    """
    return _comparison(CompareLE, operator.le, 'le', signature, input, other)


def CompareEQ(input, other, *, out=None, signature = None):
    """
    torch.eq(input, other, *, out=None)
    """
    return _comparison_hack(CompareEQ, operator.eq, 'eq', signature, input, other)


def CompareNE(input, other, *, out=None, signature = None):
    """
    torch.ne(input, other, *, out=None)
    """
    return _comparison_hack(CompareNE, operator.eq, 'ne', signature, input, other)


def ShapeAsTensor(input: IRTensor, signature = None):
    """
    torch._shape_as_tensor
    """
    if isinstance(input.shape, list) and all(isinstance(dim, int) for dim in input.shape):
        return input.shape

    edim_in = ShapeAnno.create_shape_str(input.shape)
    edim_ou = [str(len(input.shape))]
    anno = OpAnno.create_op_str([edim_in], [edim_ou])
    return IRDimops(ShapeAsTensor, '_shape_as_tensor', signature, [anno], [input])



# ================== Non-autograd Function Space =================

def Size(tensor, dim=None, signature = None) -> Union[List[int], IRPyFunc]:
    """
    torch.Tensor.size(tensor, dim=None)
    """
    assert isinstance(tensor, IRTensor)
    # constant
    if all(isinstance(dimlen, int) for dimlen in tensor.shape) and not isinstance(dim, IRObject):
        return tensor.shape[dim] if isinstance(dim, int) else list(tensor.shape)
    return IRPyFunc(signature, [tensor, dim], [IRObject()])


def To(tensor: IRTensor, dtype_or_device, *, out=None, signature = None):
    """
    torch.Tensor.to(*args, **kwargs) → Tensor
    """
    assert out is None
    # FIXME: support full version of torch.Tensor.to
    # create "to" in cube runtime functions because dtype if not kwarg in torch.Tensor.to
    signature = 'cube.runtime.function.to'
    annos = ['* -> *']
    if isinstance(dtype_or_device, torch.device):
        return IRDimops(To, 'to', signature, annos, [tensor], dtype_or_device=dtype_or_device)
    elif isinstance(dtype_or_device, (IRDType, torch.dtype)):
        dtype = dtype_or_device if isinstance(dtype_or_device, torch.dtype) else eval('torch.'+dtype_or_device.value)
        return IRDimops(To, 'to', signature, annos, [tensor], dtype_or_device=dtype)
    elif isinstance(dtype_or_device, IRFullTensor):
        dtype = eval('torch.'+dtype_or_device.dtype.value)
        return IRDimops(To, 'to', signature, annos, [tensor], dtype_or_device=dtype)
    else:
        raise RuntimeError(f'function.To with unknown arg: {dtype_or_device}')



def GetItem(a, b, signature = None) -> Union[Any, IRPyFunc]:
    """
    _operator.getitem(obj, index: int)
    """
    obj, index = a, b
    if (not isinstance(obj, IRObject)) and isinstance(index, int):
        return obj[index]
    # case: subtensor = tensor[1,:2]
    if isinstance(obj, IRTensor):
        return FullSlice(obj, b)
    return IRPyFunc(signature, [obj, index], [IRObject()])


def GetAttr(instance: object, field: str, signature = None) -> Union[List[int], IRPyFunc]:
    """
    builtins.getattr(object, name[, default])
    NOTE: only deal with the attr "shape" of IRFullTensor, because other type of object may not
    have instantiated object or the attr is not simple value.
    """
    obj, name = instance, field
    if name in ('shape', 'dtype'):
        assert isinstance(obj, IRFullTensor), f"type {type(obj)} is not supported"
        assert hasattr(obj, name), f"attr {name} is not existed in {obj}"
        return getattr(obj, name)
    elif name == 'device':
        assert isinstance(obj, IRFullTensor), f"type {type(obj)} is not supported"
        # FIXME: this is hack, IRFullTensor does not have attribute "device"
        return torch.device('cpu')
    elif isinstance(obj, torch.finfo):
        return getattr(obj, name)
    else:
        # FIXME: is it right?
        return IRPyFunc(signature, [instance, field], [IRObject()])

def FInfo(dtype: IRDType, signature = None) -> torch.finfo:
    assert isinstance(dtype, IRDType)
    return torch.finfo(eval('torch.' + dtype.value))


def MakeTuple(inputs: Iterable, signature=None):
    return tuple(inputs)


def MakeList(inputs: Iterable, signature=None):
    return list(inputs)
