
from cube.ir.operator import IRFwOperation
from cube.ir.tensor import IRSubTensor


class IRGraphAnchor(IRFwOperation):
    """
    The anchor function for navigation inside the graph
    """
    def __init__(self, signature: str, name: str):
        super().__init__(name, signature, 0, 1)
        self.kwargs['name'] = name
        self.set_output(0, None)
    
    def infer_shape(self):
        return True

    def __repr__(self) -> str:
        sign = self.signature.split('.')[-1]
        ins = [t for t in self.inputs() if isinstance(t, IRSubTensor) and not t.is_attr()]
        dscp = (f"FwOp{self._id}(sign={sign}[{self.name}], "
                f"inputs={ins}, "
                f"outputs={self.outputs()})")
        return dscp
