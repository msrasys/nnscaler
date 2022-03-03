"""
Mapping of
    Signature -> IROperator
"""
import torch

from functools import partial

import cube.graph.operator.function as function
from cube.graph.operator.operator import IRFwOperation
import cube.ir as ir


class Sign2Op:

    @staticmethod
    def map(signature: str) -> IRFwOperation:
        """
        Map the signature to GenericLogicalOp
        """
        if signature in Sign2Op.kOpMap:
            return partial(Sign2Op.kOpMap[signature], signature=signature)
        else:
            raise KeyError(f"{signature} is not supported yet")
            # print(f'warning: {signature} is not recognized')
            # return partial(function.UnkownOperator, signature=signature)

    @staticmethod
    def register(signature: str, op: IRFwOperation):
        """
        Register an operator
        """
        if not isinstance(signature, str):
            raise TypeError(f"Expected signature to be str but got {type(signature)}")
        if signature in Sign2Op.kOpMap:
            raise KeyError(f"function {signature} is already registered")
        Sign2Op.kOpMap[signature] = op

    # functional templates
    __ftemplate = lambda name: f'torch.nn.functional.{name}'

    # tensor template
    __ttemplate = lambda name: f'torch.{name}'

    # customized
    __customize = lambda name: f'cube.runtime.function.complex.{name}'

    # einops
    __einopsize = lambda name: f'einops._torch_specific.{name}'

    kOpMap = {

        # torch nn functional

        __ftemplate('linear') : function.Linear,

        __ftemplate('softmax') : function.Softmax,

        __ftemplate('dropout') : function.Dropout,

        __ftemplate('gelu') : function.GeLU,

        __ftemplate('_pad'): function.Pad,

        # __ftemplate('layer_norm'): function.LayerNorm,

        # torch aten

        __ttemplate('add') : function.Add,

        __ttemplate('sub') : function.Sub,

        __ttemplate('mul') : function.Mul,

        __ttemplate('div') : function.Div,

        __ttemplate('bmm') : function.BatchLinear,

        __ttemplate('sum') : function.Sum,

        __ttemplate('transpose') : function.Transpose,

        __ttemplate('conv2d'): function.Conv2D,

        __ttemplate('conv3d'): function.Conv3D,

        #einops
        __einopsize('apply_for_scriptable_torch'): function.ScriptEinOps,


    }


class DType2IRDType:

    @staticmethod
    def map(dtype: torch.dtype):
        """
        Map the torch dtype to IRDType
        """
        return DType2IRDType.kDtypeMap[dtype]

    kDtypeMap = {
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
