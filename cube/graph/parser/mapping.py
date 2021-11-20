"""
Mapping of
    Signature -> IROperator
"""
from functools import partial

import cube.graph.operator.function as function
from cube.graph.operator.operator import IRFwOperation


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

    }

