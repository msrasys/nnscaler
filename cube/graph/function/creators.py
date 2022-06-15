from copy import copy
from typing import List, Optional
from cube.ir.dtype import IRDType

from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRTensor

class IRZeros(IRFwOperation):
    def __init__(self, signature: str, shape: List[int], name: str, ir_dtype:Optional[IRDType]=None):

        # The shape information must be statically known integer values
        assert all(isinstance(dim, int) for dim in shape)

        super().__init__(name, signature, input_length=0, output_length=1)

        # The positional argument to specify the shape is actually called 'size'.
        self.kwargs.update({"size": copy(shape), "dtype": ir_dtype})

    def infer_shape(self) -> bool:
        shape : list = copy(self.kwargs["size"])
        self.outputs(0).shape = shape
        return True


#class IRNewTensor(IRFwOperation):
#    def __init__(self, signature: str, data, name:str):
#        pass
#    def infer_shape(self) -> bool:
#        pass


# `aten::to` has several overloading, which one should be dispatched is determined by the argument types
# See
#   https://github.com/pytorch/pytorch/blob/483bb4f0cb273f42f655aa30eee6a1fbbaba69b0/torch/csrc/jit/runtime/register_prim_ops.cpp#L1057
#   https://github.com/pytorch/pytorch/blob/483bb4f0cb273f42f655aa30eee6a1fbbaba69b0/torch/csrc/jit/runtime/register_prim_ops.cpp#L2215
class IRToTensor(IRFwOperation):
    def __init__(self, signature: str, inputs, name:str, ir_dtype:IRDType):
        super().__init__(name, signature, input_length=1, output_length=1)
        self.set_input(0, inputs[0])
        self.kwargs.update({"dtype": ir_dtype})

    def infer_shape(self) -> bool:
        self.outputs(0).shape = self.inputs(0).shape
        return True


