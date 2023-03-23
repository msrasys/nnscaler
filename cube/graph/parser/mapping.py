
import torch

from typing import Callable, Union
from functools import partial

import cube.graph.function as function
from cube.ir.operator import IRFwOperation
from cube.graph.parser.register import CustomizedOps


class Sign2Op:

    @staticmethod
    def map(signature: str) -> Callable[..., Union[IRFwOperation, int, float]]:
        """
        Map the signature to GenericLogicalOp
        """
        if signature in Sign2Op.kOpMap:
            return partial(Sign2Op.kOpMap[signature], signature=signature)
        if CustomizedOps.exist(signature):
            return CustomizedOps.map(signature)
        raise KeyError(f"{signature} is not supported yet")

    @staticmethod
    def exist(signature: str) -> bool:
        if signature in Sign2Op.kOpMap:
            return True
        if CustomizedOps.exist(signature):
            return True
        return False

    # functional templates
    __ftemplate = lambda name: f'torch.nn.functional.{name}'

    # tensor template
    __ttemplate = lambda name: f'torch.{name}'

    # runtime template
    __rtemplate = lambda name: f'cube.runtime.function.function.{name}'


    kOpMap = {

        # torch nn functional

        __ftemplate('linear') : function.Linear,

        __ttemplate('matmul'): function.Matmul,

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

        __ttemplate('chunk'): function.Chunk,

        __ttemplate('flatten'): function.Flatten,

        __ttemplate('roll'): function.Roll,

        __ttemplate('adaptive_avg_pool1d'): function.AdaptiveAvgPool1d,

        # runtime functions
        __rtemplate('anchor'): function.GraphAnchor,

        __rtemplate('identity'): function.Identity,

        __rtemplate('multiref'): function.MultiRef,

        __rtemplate('accum'): function.Accum,

    }


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
