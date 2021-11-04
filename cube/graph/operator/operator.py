from typing import Optional

from cube.ir.cten import IRTensor, IRCell
from cube.graph.tensor import IRFullTensor
from cube.algorithm.factory import DistAlgorithmFactory


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
        outputs = [IRFullTensor() for _ in range(output_length)]
        for idx, output in enumerate(outputs):
            self.set_output(idx, output)

    def infer_shape(self):
        """
        Infer output value shape
        """
        raise NotImplementedError

    def algorithms(self, tag: Optional[str] = None):
        """
        get algorithm from algorithm factory

        Args:
            tag: str or None. If None, return all 
        """
        factory = DistAlgorithmFactory()
        if tag is None:
            templates = list()
            if factory.exist(type(self)):
                templates = factory.algorithms(type(self))
            algos = list()
            for template in templates:
                algos.append(template(self))
            return algos
        else:
            if not factory.exist(type(self)):
                return None
            template = factory.algorithms(type(self), tag)
            return template(self)

    def __repr__(self):
        inputs = list()
        for tensor in self.inputs():
            if isinstance(tensor, IRTensor):
                inputs.append(f't{tensor._id}')
            else:
                inputs.append(tensor)
        
        outputs = list()
        for tensor in self.outputs():
            if isinstance(tensor, IRTensor):
                outputs.append(f't{tensor._id}')
            else:
                outputs.append(tensor)

        dscp = f'Op(id={self._id}, signature={self.signature}, device={self.device}, inputs={inputs}, outputs={outputs})'
        return dscp
