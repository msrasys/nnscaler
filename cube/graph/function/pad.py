from typing import List

from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRTensor

class IRPad(IRFwOperation):
    def __init__(self, signature: str, inputs: List[IRTensor], name: str,
                 **kwargs):
        # torch.nn.functional.pad(input, pad, mode='constant', value=0.0)
        # pad: List[int]
        assert len(inputs) == 1, "Expected only input, weight, bias as inputs"
        assert len(kwargs) == 3, "Expected 2 kwargs: mode, value"
        super().__init__(name, signature, 1, 1)
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        self.kwargs.update(kwargs)

    def infer_shape(self) -> bool:
        """
        Output shape inference given the input shapes
        """
        if len(self.inputs(0).shape) == 0:
            return False

        N = self.inputs(0).shape[0]
        pad  = self.kwargs['pad']
        mode = self.kwargs['mode']
        value = self.kwargs['value']
        assert len(pad) % 2 == 0, "IRPad::infer_shape len(pad) % 2 == 0"

        shape = self.inputs(0).shape
        for pad_idx, pad_size in enumerate(pad):
            shape[-1 - (pad_idx // 2)] += pad_size

        self.outputs(0).shape = shape
        return True

    def new(self, inputs: List, outputs: List, pad = None):
        """
        construct a new operator sharing same kwargs with new inputs
        and outputs
        """
        if pad == None:
            pad = self.kwargs['pad']
        mode = self.kwargs['mode']
        value = self.kwargs['value']
        op = IRPad(self.signature, inputs, self.name,
                   pad=pad, mode=mode, value=value)
        assert len(outputs) == 1
        op.set_output(0, outputs[0])
        op.infer_shape()
        return op
