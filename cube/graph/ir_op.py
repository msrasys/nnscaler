from typing import List, Union

from cube.graph.ir_cten import IRTensor, IRCell
from cube.graph.mapping import IR2LogicOp


__call__ = ['IROperation']


class IROperation(IRCell):

    def __init__(self,
                 name: str, 
                 signature: str,
                 input_length: int,
                 output_length: int):
        """
        Create a node with name (variable name) and module type (module_name)

        Args:
            name (str): the op semantic name
            signature (str): the op signature, e.g., torch.functional.nn.linear
            input_length (int): the number of inputs for the op
            output_length (int): the number of outputs for the op
        """
        super().__init__(name, signature, input_length, output_length)
        self.semantic = IR2LogicOp.map(self.signature)

    def infer_shape(self):
        """
        Infer output value shape
        """
        shapes = list()
        for input in self.inputs():
            if isinstance(input, IRTensor):
                if input.shape is None:
                    return False
                shapes.append(input.shape)
            else:
                shapes.append([1,])
        shapes = tuple(shapes)
        out_shapes = self.semantic.shape_infer(*shapes)
        if len(out_shapes) != len(self._outputs):
            raise RuntimeError(
                "The logical op semantic doesn't match with parsed op"
            )
        for shape, val in zip(out_shapes, self._outputs):
            if isinstance(val, IRTensor):
                val.shape = shape
        return True

    def __repr__(self):
        inputs = list()
        for tensor in self.inputs():
            if isinstance(tensor, IRTensor):
                inputs.append(f't{tensor._id}-dev{tensor.device}')
            else:
                inputs.append(tensor)
        
        outputs = list()
        for tensor in self.outputs():
            if isinstance(tensor, IRTensor):
                outputs.append(f't{tensor._id}-dev{tensor.device}')
            else:
                outputs.append(tensor)

        dscp = f'Op(id={self._id}, signature={self.signature}, device={self.device}, inputs={inputs}, outputs={outputs})'
        return dscp
