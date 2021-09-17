from typing import List, Optional

from cube.operator.logic.generics import GenericLogicalOp
from cube.operator.logic.generics import ElementSameInputOp


class Linear(GenericLogicalOp):

    @staticmethod
    def candidates():
        raise NotImplementedError

    @staticmethod
    def shape_infer(input: List[int],
                    weight: List[int],
                    bias: Optional[List[int]] = None):
        """
        input:  [(D), M, K]
        weight: [N, K]
        bias:   [N,]
        """
        out_shape = list(input)
        out_shape[-1] = weight[0]
        return [out_shape]

    def translate(self, config):
        raise NotImplementedError


class GeLU(ElementSameInputOp):

    def __init__(self, signature: str):
        super().__init__(signature)

class Dropout(ElementSameInputOp):
        
    def __init__(self, signature: str):
        super().__init__(signature)


# ================== aten tensor op ========================

class TensorAdd(ElementSameInputOp):

    def __init__(self, signature: str):
        super().__init__(signature)
