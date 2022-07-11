from copy import copy
from typing import List, Optional, Tuple

from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRTensor

class IRSelectScatter(IRFwOperation):
    """
    torch.select_scatter(self:Tensor, input:Tensor, dim:int, index:int) -> Tensor

    identical to:
    ```
    x = self.copy()                 # Assume N-d tensor.
    view = x.select(dim, index)     # View and input are (N-1)-d tensors.
    view.copy_(input)               # See REMARK!
    return x
    ```

    REMARK:
        Unlike the `copy_` API in the identical code snippet above,
        `select_scatter` (as well as other scatter family APIs) are NOT broadcastable,
        namely it requires the `input` tensor to embed is an exactly (N-1)-dimensional tensor.
        
        But in-place Python code like
        ```
        self[index] = input
        ```
        involves broadcasting, so `input` can has any broadcastable shapes to `self.shape.pop(dim)`,
        including being scalars.
    """

    def __init__(self, signature: str, inputs:Tuple[IRTensor, IRTensor], name: str, dim:int, index:int):
        assert len(inputs) == 2

        signature = 'cube.runtime.function.select_scatter'
        super().__init__(name, signature, 2, 1)
        self.set_input(0, inputs[0])
        self.set_input(1, inputs[1])
        self.kwargs.update({"dim": dim, "index": index})

    def infer_shape(self) -> bool:
        shp_self : List[int] = self.inputs(0).shape
        if len(shp_self) == 0:
            return False

        shp_input = self.inputs(1).shape

        if len(shp_input) == 0:
            print("The 0-length input shape is ambiguous, may be uninferrable or just of a 0-d tensor")
        elif len(shp_input) > 0:
            dim: int = self.kwargs["dim"]
            copy_shp = shp_self.copy()
            copy_shp.pop(dim)
            if copy_shp != shp_input:
                raise RuntimeError(f"self shape {shp_self} and input shape {shp_input} with dim={dim} mismatch")

        s2 = copy(shp_self)
        self.outputs(0).shape = s2
        return True

