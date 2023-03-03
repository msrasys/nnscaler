from typing import Tuple

from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRObject


class IRPyFunc(IRFwOperation):
    """
    Python runtime function
    """

    def __init__(self, signature: str, 
                 inputs: Tuple[IRObject], outputs: Tuple[IRObject], **kwargs):
        name = signature.split('.')[-1]
        super().__init__(name, signature, len(inputs), len(outputs))
        for idx, t in enumerate(inputs):
            self.set_input(idx, t)
        for idx, t in enumerate(outputs):
            self.set_output(idx, t)
        self.kwargs.update(**kwargs)

    def infer_shape(self) -> bool:
        """
        Shape will not be inferred for python runtime
        """
        return True
    
    

