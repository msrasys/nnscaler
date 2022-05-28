from typing import List, Optional, Tuple
import itertools

from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRTensor

class IRRepeat(IRFwOperation):
    """
    torch.repeat(tensor:Tensor, repeats: List[int]) -> Tensor
    """

    def __init__(self, signature: str, inputs:Tuple[IRTensor], name: str, repeats:List[int]):
        assert len(inputs) == 1
        assert isinstance(repeats, list)
        assert all(isinstance(r, int) for r in repeats)

        super().__init__(name, signature, 1, 1)
        self.set_input(0, inputs[0])
        self.kwargs.update({"repeats": repeats})

    def infer_shape(self) -> bool:
        shp_self : List[int] = self.inputs(0).shape
        if len(shp_self) == 0:
            return False

        repeats : List[int] = self.kwargs["repeats"]

        # This API broadcasts the input tensor if the specified `repeats:list` is longer than the shape.
        s1 = shp_self.copy()
        s1.reverse()
        s2 = repeats.copy()
        s2.reverse()

        # Multiply from the end
        shp = [d1 * d2 for d1, d2 in itertools.zip_longest(s1, s2, fillvalue=1)]
        shp.reverse()

        self.outputs(0).shape = shp
        return True

