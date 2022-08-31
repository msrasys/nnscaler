from copy import copy
from typing import List, Optional, Tuple

from cube.ir.operator import IRFwOperation
from cube.ir.cten import IRTensor

class IRSelect(IRFwOperation):
    """
    torch.select(input:Tensor, dim:int, index:int) -> Tensor
    """
    def __init__(self, signature: str, inputs:Tuple[IRTensor], name: str, dim:int, index:int):
        assert len(inputs) == 1

        super().__init__(name, signature, 1, 1)
        self.set_input(0, inputs[0])
        self.kwargs.update({"dim": dim, "index": index})

    def infer_shape(self) -> bool:
        s : List[int] = self.input(0).shape
        if len(s) == 0:
            return False

        dim = self.kwargs["dim"]

        s2 = copy(s)
        s2.pop(dim)
        self.output(0).shape = s2

        return True

    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        op = IRSelect(self.signature, inputs, self.name, self.kwargs['dim'], self.kwargs['index'])
        assert len(outputs) == 1
        op.set_output(0, outputs[0])
        assert op.infer_shape(), "IRSelect::new infer_shape failed"
        return op

class IRSlice(IRFwOperation):
    """
    aten::slice(input:Tensor, dim:int=0, start:Optional[int]=None, end:Optional[int]=None, step:int=1) -> Tensor
    """

    def __init__(self, signature: str, inputs:Tuple[IRTensor], name: str, 
                 dim:int, start:Optional[int], end:Optional[int], step:int):
        assert len(inputs) == 1

        super().__init__(name, signature, 1, 1)
        self.set_input(0, inputs[0])
        self.kwargs.update({"dim": dim, "start": start, "end": end, "step": step})

    def infer_shape(self) -> bool:
        s : List[int] = self.input(0).shape
        if len(s) == 0:
            return False

        dim : int = self.kwargs["dim"]
        start : Optional[int] = self.kwargs["start"]
        end : Optional[int] = self.kwargs["end"]
        step : int = self.kwargs["step"]

        if start is None:
            start = 0
        if end is None:
            end = 2 ** 64

        dim_len = s[dim]

        def clip(offset):
            if offset < 0:
                offset += dim_len
            return min(dim_len, max(0, offset))

        start = clip(start)
        end = clip(end)

        sliced_dim_len = len(range(start, end, step))
        s2 = s.copy()
        s2[dim] = sliced_dim_len
        self.output(0).shape = s2

        return True
    
    def new(self, inputs: List[IRTensor], outputs: List[IRTensor]):
        assert len(inputs) == 1, "Slice: number of inputs not equal to 1"
        op = IRSlice(self.signature, inputs, self.name, self.kwargs['dim'], self.kwargs['start'], self.kwargs['end'], self.kwargs['step'])
        assert len(outputs) == 1
        op.set_output(0, outputs[0])
        assert op.infer_shape(), "IRSlice::new infer_shape failed"
        return op    


# torch.gather(input:Tensor, dim:int, index:LongTensor, *, sparse_grad=False, out=None) -> Tensor
