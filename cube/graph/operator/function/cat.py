from copy import copy
import itertools
from typing import List

from cube.graph.operator.operator import IRFwOperation
from cube.ir.cten import IRTensor

class IRCat(IRFwOperation):
    def __init__(self, signature: str, inputs: List[IRTensor], name: str,
                 **kwargs):
        # torch.cat(inputs:List[Tensor], dim:int) -> Tensor
        # REMARK:   the input to 'cat' is a tensor list, so 'inputs' parameter directly reflects the singleton list containing that list,
        #           so the meaning of param 'inputs' is sligtly different from other IRXXXOp.
        assert len(inputs) > 0, "TODO handle zero inputs"
        assert len(kwargs) == 1, "Expected 1 kwargs: dim"

        super().__init__(name, signature, len(inputs), 1)
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        self.kwargs.update(kwargs)

    def infer_shape(self) -> bool:
        """
        Output shape inference given the input shapes
        """
        dim  = self.kwargs['dim']

        # validation
        # TODO how about zero inputs?
        tensors : List[IRTensor] = self.inputs(None) # None for all inputs

        # Shape without the dim-th component
        s0 : list = None
        for i, tensor in enumerate(tensors):
            s : list = copy(tensor.shape) # avoid mutating the original shape

            if len(s) == 0:
                # Any shape unknown
                return False

            s.pop(dim)
            if i == 0:
                s0 = s
            else:
                if s != s0:
                    # Inconsistent input shape
                    return False

        sumLen : int = sum(t.shape[dim] for t in tensors)
        s0.insert(dim, sumLen)
        self.outputs(0).shape = s0
        return True


class IRStack(IRFwOperation):
    def __init__(self, signature: str, inputs: List[IRTensor], name: str, dim: int):
        # torch.stack(inputs:List[Tensor], dim:int) -> Tensor
        assert len(inputs) > 0

        super().__init__(name, signature, len(inputs), 1)
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        self.kwargs.update({"dim": dim})

    def infer_shape(self) -> bool:
        dim  = self.kwargs['dim']
        tensors : List[IRTensor] = self.inputs(None) # None for all inputs
        
        # `stack` requires all input tensors to have the same shape
        if len(set(t.shape for t in tensors)) != 1:
            return False

        shp : list = tensors[0].shape.copy()
        shp.insert(dim, len(tensors))
        self.outputs(0).shape = shp
        return True

