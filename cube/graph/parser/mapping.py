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

    # functional templates
    __ftemplate = lambda name: f'torch.nn.functional.{name}'

    # tensor template
    __ttemplate = lambda name: f'torch.{name}'

    # customized
    __customize = lambda name: f'cube.runtime.function.complex.{name}'

    kOpMap = {

        # torch nn functional

        __ftemplate('linear') : function.Linear,

        __ftemplate('softmax') : function.Softmax,

        __ftemplate('dropout') : function.Dropout,

        __ftemplate('gelu') : partial(function.Activation, name='gelu'),

        __ftemplate('layer_norm'): function.LayerNorm,

        # torch aten

        __ttemplate('add') : function.Add,

        __ttemplate('mul') : partial(function.ElementWise, name='mul'),

        __ttemplate('bmm') : function.BatchLinear,

        __ttemplate('sum') : function.Sum,

        __ttemplate('transpose') : function.Transpose,

        # complex

        __customize('toqkv'): partial(function.CubeComplexToQKV, name='toqkv'),

        __customize('tril_mask'): function.CubeComplexTrilMask,

        __customize('attn_view'): function.CubeComplexAttnView,

        __customize('self_attn'): function.CubeComplexSelfAttention,
        
        __customize('feedforward'): function.CubeComplexFeedForward,

        __customize('embedding'): function.CubeComplexEmbedding,

    }


class DType2IRDType:

    @staticmethod
    def map(dtype: torch.dtype):
        """
        Map the torch dtype to IRDType
        """
        return DType2IRDType.kDtypeMap[dtype]

    kDtypeMap = {
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
