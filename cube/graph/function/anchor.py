
from cube.ir.operator import IRFwOperation


class IRGraphAnchor(IRFwOperation):
    """
    The anchor function serves for
    1) navigation inside the graph
    2) staging boundary inside the graph

    This operator will eventually be removed from graph,
    user doesn't need to manipulate it.
    """
    def __init__(self, signature: str, name: str):
        super().__init__(name, signature, [], 1)
        self.kwargs['name'] = name
        self.set_output(0, None)
    
    def infer_dtype(self):
        return

    def infer_shape(self):
        return True

    def __repr__(self) -> str:
        return f"AnchorOp-{self.cid}(name={self.name})"
