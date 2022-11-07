"""
Mapping of
    signature -> IROperator
    torch.dtype -> cube.ir.IRDType
    cube.ir.IRDType -> torch.dtype
"""
import torch

from typing import Callable, Dict, Union
from functools import partial

import cube.graph.function as function
import cube.ir as ir
from cube.ir.operator import IRFwOperation


class Sign2Op:

    @staticmethod
    def map(signature: str) -> Callable[..., Union[IRFwOperation, int, float]]:
        """
        Map the signature to GenericLogicalOp
        """
        if 'torch.' not in signature and 'cube.runtime.' not in signature:
            signature = signature.split('.')[-1]
        if signature in Sign2Op.kOpMap:
            return partial(Sign2Op.kOpMap[signature], signature=signature)
        else:
            raise KeyError(f"{signature} is not supported yet")
            # print(f'warning: {signature} is not recognized')
            # return partial(function.UnkownOperator, signature=signature)

    @staticmethod
    def register(signature: str, op: Callable[..., Union[IRFwOperation, int, float]], code):
        """
        Register an operator
        """
        if not isinstance(signature, str):
            raise TypeError(f"Expected signature to be str but got {type(signature)}")
        if signature in Sign2Op.kOpMap:
            raise KeyError(f"function {signature} is already registered")
        Sign2Op.kOpMap[signature] = op
        Sign2Op.kOpCodeDef[signature] = code

    # functional templates
    __ftemplate = lambda name: f'torch.nn.functional.{name}'

    # tensor template
    __ttemplate = lambda name: f'torch.{name}'

    # runtime template
    __rtemplate = lambda name: f'cube.runtime.function.function.{name}'

    # einops
    __einopsize = lambda name: f'einops._torch_specific.{name}'

    # custom ops
    __customops = lambda name: f'examples.custom_ops.{name}'

    kOpMap = {

        # torch nn functional

        __ftemplate('linear') : function.Linear,

        __ftemplate('softmax') : function.Softmax,

        __ftemplate('dropout') : function.Dropout,

        __ftemplate('gelu') : function.GeLU,
        __ttemplate('gelu') : function.GeLU,

        __ftemplate('silu') : function.SiLU,
        __ttemplate('silu') : function.SiLU,

        __ftemplate('_pad'): function.Pad,

        __ftemplate('layer_norm'): function.LayerNorm,

        __ftemplate('embedding'): function.Embedding,

        __ftemplate('cross_entropy'): function.CrossEntropy,

        # torch aten

        # creators
        __ttemplate('zeros'): function.Zeros,
        __ttemplate('ones'): function.Ones,
        __ttemplate('tensor'): function.NewTensor,
        __ttemplate('to'): function.ToTensor,
        __ttemplate('rand'): function.Rand,
        __ttemplate('clone'): function.Clone,

        __ttemplate('add') : function.Add,

        __ttemplate('sub') : function.Sub,

        __ttemplate('mul') : function.Mul,

        __ttemplate('div') : function.Div,

        __ttemplate('floordiv') : function.FloorDiv,

        __ttemplate('neg'): function.Neg,

        __ttemplate('gt'): function.CompareGT,
        __ttemplate('lt'): function.CompareLT,
        __ttemplate('ge'): function.CompareGE,
        __ttemplate('le'): function.CompareLE,

        __ttemplate('pow'): function.Pow,

        __ttemplate('sin'): function.Sin,

        __ttemplate('cos'): function.Cos,

        __ttemplate('tanh'): function.Tanh,

        __ttemplate('bmm') : function.BatchLinear,

        __ttemplate('sum') : function.Sum,
        __ttemplate('mean') : function.Mean,

        __ttemplate('transpose') : function.Transpose,

        __ttemplate('view'): function.View,

        __ttemplate('reshape'): function.Reshape,

        __ttemplate('conv2d'): function.Conv2D,

        __ttemplate('conv3d'): function.Conv3D,

        __ttemplate('pad'): function.Pad,

        __ttemplate('select'): function.Select,

        __ttemplate('slice'): function.Slice,

        #pytorch1.11
        __ttemplate('select_scatter'): function.SelectScatter,

        __ttemplate('repeat'): function.Repeat,

        #pytorch1.11
        __ttemplate('linear'): function.Linear,

        __ttemplate('cat'): function.Cat,

        __ttemplate('stack'): function.Stack,

        __ttemplate('flatten'): function.Flatten,

        __ttemplate('roll'): function.Roll,

        __ttemplate('adaptive_avg_pool1d'): function.AdaptiveAvgPool1d,

        # runtime functions
        __rtemplate('anchor'): function.GraphAnchor,

        __rtemplate('identity'): function.Identity,

        __rtemplate('multiref'): function.MultiRef,

        __rtemplate('accum'): function.Accum,

        #einops
        __einopsize('apply_for_scriptable_torch'): function.ScriptEinOps,

    }

    # customized operator code: signature -> code
    kOpCodeDef: Dict[str, str] = {}


class DType2IRDType:

    @staticmethod
    def map(dtype: torch.dtype):
        """
        Map the torch dtype to IRDType
        """
        return DType2IRDType.kDtypeMap[dtype]

    kDtypeMap = {
        torch.double:  ir.float64,
        torch.float64: ir.float64,
        torch.float32: ir.float32,
        torch.float  : ir.float32,
        torch.float16: ir.float16,
        torch.half   : ir.float16,
        torch.uint8  : ir.uint8,
        torch.int8   : ir.int8,
        torch.int16  : ir.int16,
        torch.short  : ir.int16,
        torch.int32  : ir.int32,
        torch.int    : ir.int32,
        torch.int64  : ir.int64,
        torch.long   : ir.int64,
        torch.bool   : ir.boolean
    }


class IRDType2TorchDType:

    @staticmethod
    def map(ir_dtype: ir.IRDType):
        """
        Map the IRDtype to torch dtype
        """
        return IRDType2TorchDType.kDtypeMap[ir_dtype]
    
    kDtypeMap = {val: key for key, val in DType2IRDType.kDtypeMap.items()}


# see https://github.com/pytorch/pytorch/blob/master/c10/core/ScalarType.h
#
# ScalarType enum is totally a PyTorch-internal object. Neither itself nor its underlying ints
# are accessible from its Python frontend.
class TorchScalarTypeEnumMap:

    @staticmethod
    def map(underlying: int) -> torch.dtype:
        
        assert isinstance(underlying, int), """
        This function is to convert an underlying 'int' for a Torch-internal 'at::ScalarType' enum
        to its corresponding Python-frontend 'torch.dtype' enum.
        """

        dtype = TorchScalarTypeEnumMap._fields[underlying]

        assert dtype is not None, f"""
        Referenced to an unsupported ScalarType with underlying int being {underlying}
        """
        
        return dtype

    # Less used dtypes are masked out because PyTorch keeps **exposing and hiding** them recently
    # from a view of Python frontend.
    _fields = [
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.half,
        torch.float32,
        torch.float64,
        None, #torch.complex32,    # complexHalf
        None, #torch.complex64,    # complexFloat
        None, #torch.complex128,   # complexDouble
        torch.bool,
        None, #torch.qint8,
        None, #torch.quint8,
        None, #torch.qint32,
        None, #torch.bfloat16,
        None, #torch.quint4x2,
        None, #torch.quint2x4,
    ]

    assert len(_fields) == 18, "Do not remove any item, mask it out with None"
