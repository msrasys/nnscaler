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

    kOpMap = {

        __ftemplate('linear') : function.Linear,

        __ftemplate('dropout') : partial(function.ElementWiseActivation, name='dropout'),

        __ftemplate('gelu') : partial(function.ElementWiseActivation, name='gelu'),

        __ttemplate('add') : partial(function.ElementWise, name='add'),

        __ttemplate('sum') : partial(function.Reduce, name='sum'),

    }

