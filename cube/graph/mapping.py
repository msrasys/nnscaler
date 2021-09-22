"""
Mapping of
    IROperation -> cube.operator.logic.generics.GenericLogicalOp
"""

import cube.operator.logic as logic

class IR2LogicOp:

    @staticmethod
    def map(signature: str) -> logic.GenericLogicalOp :
        """
        Map the signature to GenericLogicalOp
        """
        if signature in IR2LogicOp.kOpMap:
            return IR2LogicOp.kOpMap[signature]
        raise KeyError(f"{signature} is not supported yet")

    # functional templates
    __ftemplate = lambda name: f'torch.nn.functional.{name}'

    # tensor template
    __ttemplate = lambda name: f'torch.{name}'

    kOpMap = {

        __ftemplate('linear') : logic.Linear,

        __ftemplate('dropout') : logic.Dropout,

        __ftemplate('gelu') : logic.GeLU,

        __ttemplate('add') : logic.TensorAdd,

        # runtime collectives
        'cube.runtime.spatial.move': 'move',

    }

