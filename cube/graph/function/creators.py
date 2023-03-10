from copy import copy
from typing import List, Optional
from cube.ir.dtype import IRDType

from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRTensor

import numpy as np

class IRArange(IRFwOperation):
    def __init__(self, signature: str, shape: List[int], name: str, **kwargs):

        # The shape information must be statically known integer values
        assert all(isinstance(dim, int) for dim in shape)
        assert 'dtype' in kwargs
        assert isinstance(kwargs['dtype'], IRDType)

        super().__init__(name, signature, input_length=0, output_length=1)

        # Customize output's dtype only after 'super().__init__' and 'self.set_input',
        # otherwise it gets overwritten.
        self.output(0).dtype = kwargs['dtype']
        self.shape = shape
        self.kwargs = kwargs

    def infer_shape(self) -> bool:
        self.output(0).shape = copy(self.shape)
        return True

    def new(self, outputs: List[IRTensor]):
        op = IRArange(self.signature, outputs[0].shape, self.name, **self.kwargs)
        op.set_output(0, outputs[0])
        assert op.infer_shape(), "IRArange::new infer_shape failed"
        return op

class IREmpty(IRFwOperation):
    def __init__(self, signature: str, shape: List[int], name: str, **kwargs):

        # The shape information must be statically known integer values
        assert all(isinstance(dim, int) for dim in shape)
        assert 'dtype' in kwargs
        assert isinstance(kwargs['dtype'], IRDType)

        super().__init__(name, signature, input_length=0, output_length=1)

        # Customize output's dtype only after 'super().__init__' and 'self.set_input',
        # otherwise it gets overwritten.
        self.output(0).dtype = kwargs['dtype']

        # The positional argument to specify the shape is actually called 'size'.
        self.kwargs = kwargs
        self.kwargs.update({"size": copy(shape)})

    def infer_shape(self) -> bool:
        shape : list = copy(self.kwargs["size"])
        self.output(0).shape = shape
        return True

    def new(self, outputs: List[IRTensor]):
        op = IREmpty(self.signature, outputs[0].shape, self.name, **self.kwargs)
        op.set_output(0, outputs[0])
        assert op.infer_shape(), "IREmpty::new infer_shape failed"
        return op

class IRNewTensor(IRFwOperation):
    def __init__(self, signature: str, data: list, name: str, **kwargs):
        super().__init__(name, signature, input_length=0, output_length=1)
        assert 'dtype' in kwargs
        assert isinstance(kwargs['dtype'], IRDType)
        self.output(0).dtype = kwargs['dtype']
        self.data = data
        self.shape = np.array(data).shape
        self.kwargs = kwargs

    def infer_shape(self) -> bool:
        self.output(0).shape = copy(self.shape)
        return True

    def new(self, outputs: List[IRTensor]):
        op = IRNewTensor(self.signature, self.data, self.name, **self.kwargs)
        op.set_output(0, outputs[0])
        assert op.infer_shape(), "IRNewTensor::new infer_shape failed"
        return op

class IRZeros(IRFwOperation):
    def __init__(self, signature: str, shape: List[int], name: str, ir_dtype:IRDType):

        # The shape information must be statically known integer values
        assert all(isinstance(dim, int) for dim in shape)
        assert isinstance(ir_dtype, IRDType)

        super().__init__(name, signature, input_length=0, output_length=1)

        # Customize output's dtype only after 'super().__init__' and 'self.set_input',
        # otherwise it gets overwritten.
        self.output(0).dtype = ir_dtype

        # The positional argument to specify the shape is actually called 'size'.
        self.kwargs.update({"size": copy(shape), "dtype": ir_dtype})

    def infer_shape(self) -> bool:
        shape : list = copy(self.kwargs["size"])
        self.output(0).shape = shape
        return True

    def new(self, outputs: List[IRTensor]):
        op = IRZeros(self.signature, outputs[0].shape, self.name, self.kwargs['dtype'])
        op.set_output(0, outputs[0])
        assert op.infer_shape(), "IRZeros::new infer_shape failed"
        return op

class IROnes(IRFwOperation):
    def __init__(self, signature: str, shape: List[int], name: str, ir_dtype:IRDType):

        # The shape information must be statically known integer values
        assert all(isinstance(dim, int) for dim in shape)
        assert isinstance(ir_dtype, IRDType)

        super().__init__(name, signature, input_length=0, output_length=1)

        # Customize output's dtype only after 'super().__init__' and 'self.set_input',
        # otherwise it gets overwritten.
        self.output(0).dtype = ir_dtype

        # The positional argument to specify the shape is actually called 'size'.
        self.kwargs.update({"size": copy(shape), "dtype": ir_dtype})

    def infer_shape(self) -> bool:
        shape : list = copy(self.kwargs["size"])
        self.output(0).shape = shape
        return True
    
    def new(self, outputs: List[IRTensor]):
        op = IROnes(self.signature, outputs[0].shape, self.name, self.kwargs['dtype'])
        op.set_output(0, outputs[0])
        assert op.infer_shape(), "IROnes::new infer_shape failed"
        return op
    
class IRRand(IRFwOperation):
    def __init__(self, signature: str, shape: List[int], name: str, ir_dtype:IRDType):

        # The shape information must be statically known integer values
        assert all(isinstance(dim, int) for dim in shape)
        assert isinstance(ir_dtype, IRDType)

        super().__init__(name, signature, input_length=0, output_length=1)

        # Customize output's dtype only after 'super().__init__' and 'self.set_input',
        # otherwise it gets overwritten.
        self.output(0).dtype = ir_dtype

        # The positional argument to specify the shape is actually called 'size'.
        self.kwargs.update({"size": copy(shape), "dtype": ir_dtype})

    def infer_shape(self) -> bool:
        shape : list = copy(self.kwargs["size"])
        self.output(0).shape = shape
        return True
    
    def new(self, outputs: List[IRTensor]):
        op = IRRand(self.signature, outputs[0].shape, self.name, self.kwargs['dtype'])
        op.set_output(0, outputs[0])
        assert op.infer_shape(), "IRRand::new infer_shape failed"
        return op

# class IRNewTensor(IRFwOperation):
#    def __init__(self, signature: str, data: list, name: str, ir_dtype: IRDType):
#        super().__init__(name, signature, input_length=0, output_length=1)
#        self.output(0).dtype = ir_dtype
#        self.kwargs.update({'data': data, 'shape': np.array(data).shape, 'dtype': ir_dtype})
       
#    def infer_shape(self) -> bool:
#        shape : list = copy(self.kwargs['shape'])
#        self.output(0).shape = shape
#        return True
       


# `aten::to` has several overloading, which one should be dispatched is determined by the argument types
# See
#   https://github.com/pytorch/pytorch/blob/483bb4f0cb273f42f655aa30eee6a1fbbaba69b0/torch/csrc/jit/runtime/register_prim_ops.cpp#L1057
#   https://github.com/pytorch/pytorch/blob/483bb4f0cb273f42f655aa30eee6a1fbbaba69b0/torch/csrc/jit/runtime/register_prim_ops.cpp#L2215
class IRToTensor(IRFwOperation):
    def __init__(self, signature: str, inputs, name:str, ir_dtype:IRDType):
        
        assert isinstance(ir_dtype, IRDType)

        super().__init__(name, signature, input_length=1, output_length=1)
        self.set_input(0, inputs[0])

        # Customize output's dtype only after 'super().__init__' and 'self.set_input',
        # otherwise it gets overwritten.
        self.output(0).dtype = ir_dtype
        
        self.kwargs.update({"dtype": ir_dtype})

    def infer_shape(self) -> bool:
        self.output(0).shape = self.input(0).shape
        return True
    
    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        op = IRToTensor(self.signature, inputs, self.name, self.kwargs['dtype'])
        op.set_output(0, outputs[0])
        assert op.infer_shape(), "IRToTensor::new infer_shape failed"
        return op


