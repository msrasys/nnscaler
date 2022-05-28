from typing import List

from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRTensor

class IRCustomOps(IRFwOperation):
    def __init__(self, signature: str, inputs: List[IRTensor], name: str,
                 **kwargs):
        # torch.nn.functional.pad(input, pad, mode='constant', value=0.0)
        # pad: List[int]
        if signature == 'examples.custom_ops.strip_2_borders':
            signature = signature.replace('examples.custom_ops', 'cube.runtime.function')#'cube.runtime.function.strip_2_borders'
            assert len(inputs) == 1, "Expected only input, weight, bias as inputs"
            assert len(kwargs) == 0, "Expected 0 kwargs: "
            super().__init__(name, signature, 1, 1)
            for idx, input in enumerate(inputs):
                self.set_input(idx, input)
        elif signature == 'examples.custom_ops.update_diag_':
            signature = signature.replace('examples.custom_ops', 'cube.runtime.function')
            assert len(inputs) == 10, "Expected only input, weight, bias as inputs"
            assert len(kwargs) == 1, "Expected 0 kwargs: "
            super().__init__(name, signature, len(inputs), 1)
            for idx, input in enumerate(inputs):
                self.set_input(idx, input)
            self.kwargs.update(kwargs)
        elif signature == 'examples.custom_ops.update_geopotential_':
            signature = signature.replace('examples.custom_ops', 'cube.runtime.function')
            assert len(inputs) == 5, "Expected only input, weight, bias as inputs"
            assert len(kwargs) == 3, "Expected 0 kwargs: "
            super().__init__(name, signature, len(inputs), 1)
            for idx, input in enumerate(inputs):
                self.set_input(idx, input)
            self.kwargs.update(kwargs)
        else:
            raise RuntimeError(f'IRCustomOps::__init__ unknown signature: {self.signature}')


    def infer_shape(self) -> bool:
        """
        Output shape inference given the input shapes
        """
        if self.signature.endswith('strip_2_borders'):
            if len(self.inputs(0).shape) == 0:
                return False
            shape = self.inputs(0).shape
            shape[0] = shape[0]-2
            self.outputs(0).shape = shape
            return True
        elif self.signature.endswith('update_diag_'):
            shape = self.inputs(0).shape
            self.outputs(0).shape = shape
            return True
        elif self.signature.endswith('update_geopotential_'):
            shape = self.inputs(0).shape
            self.outputs(0).shape = shape
            return True
        else:
            raise RuntimeError(f'IRCustomOps::infer_shape unknown signature: {self.signature}')

    def new(self, inputs: List, outputs: List):
        """
        construct a new operator sharing same kwargs with new inputs
        and outputs
        """
        if self.signature.endswith('strip_2_borders'):
            op = IRCustomOps(self.signature, inputs, self.name,)
            assert len(outputs) == 1
            op.set_output(0, outputs[0])
            op.infer_shape()
            return op
        elif self.signature.endswith('update_diag_'):
            op = IRCustomOps(self.signature, inputs, self.name, self.kwargs)
            assert len(outputs) == 1
            op.set_output(0, outputs[0])
            op.infer_shape()
            return op
        elif self.signature.endswith('update_geopotential_'):
            op = IRCustomOps(self.signature, inputs, self.name, self.kwargs)
            assert len(outputs) == 1
            op.set_output(0, outputs[0])
            op.infer_shape()
            return op
        else:
            raise RuntimeError(f'IRCustomOps::new unknown signature: {self.signature}')

