from typing import List, Tuple, Optional

from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.select import IRSelect, IRSlice
from cube.ir.tensor import IRSubTensor


class DimSplitSelect(GenericDistAlgo):
    """
    split Pad at dimension level

    """
    def __init__(self, node: IRSelect):
        if not isinstance(node, IRSelect):
            raise TypeError(f"Expect IRSelect")
        super().__init__(node)

    def satisfy(self, dim: int, num: int):
        """
        config = dict(idx=int, dim=int, num=num)

        """
        assert all(isinstance(t, int) for t in [dim, num]), "dim and num should be integer"
        node: IRSelect = self.node
        
        assert dim != node.kwargs['dim'], "Split dimension should not be equal to select dimension"
        assert dim < len(node.input(0).shape), "Split dimension should be smaller than tensor dimension"

        # split non-pad dim
        return node.input(0).shape[dim] >= num

    def instantiate(self, dim: int, num: int) -> Optional[List[IRSelect]]:

        node: IRSelect = self.node
        satisfy = self.satisfy(dim, num)
        if not satisfy:
            return None

        ins, ous = list(), list()
        for iidx, itensor in enumerate(node.inputs()):
            assert isinstance(itensor, IRSubTensor), "Input of select shoud be IRSubTensor"
            ins.append(itensor.split_dim(dim, num))
            
        odim = dim - int(node.kwargs['dim'] < dim)

        for oidx, otensor in enumerate(node.outputs()):
            assert isinstance(otensor, IRSubTensor), "Output of select should be IRSubTensor"
            ous.append(otensor.split_dim(odim, num))

        sub_nodes = list()
        for nid in range(num):
            inputs = [t[nid] for t in ins]
            outputs = [t[nid] for t in ous]
            sub_nodes.append(node.new(inputs, outputs))
        return sub_nodes
    
    
class DimSplitSlice(GenericDistAlgo):
    """
    split Pad at dimension level

    """
    def __init__(self, node: IRSlice):
        if not isinstance(node, IRSlice):
            raise TypeError(f"Expect IRSlice")
        super().__init__(node)

    def satisfy(self, dim: int, num: int):
        assert all(isinstance(t, int) for t in [dim, num]), "dim and num should be integer"
        node: IRSlice = self.node
        
        if dim == node.kwargs['dim']:
            return None
        assert dim < len(node.input(0).shape), "Split dimension should be smaller than tensor dimension"

        # split non-pad dim
        return node.input(0).shape[dim] >= num

    def instantiate(self, dim: int, num: int) -> Optional[List[IRSlice]]:
        
        node: IRSlice = self.node
        print(dim, node.kwargs['dim'])
        satisfy = self.satisfy(dim, num)
        if not satisfy:
            return None

        ins, ous = list(), list()
        for iidx, itensor in enumerate(node.inputs()):
            assert isinstance(itensor, IRSubTensor), "Input of select shoud be IRSubTensor"
            ins.append(itensor.split_dim(dim, num))

        for oidx, otensor in enumerate(node.outputs()):
            assert isinstance(otensor, IRSubTensor), "Output of select should be IRSubTensor"
            ous.append(otensor.split_dim(dim, num))

        sub_nodes = list()
        for nid in range(num):
            inputs = [t[nid] for t in ins]
            outputs = [t[nid] for t in ous]
            sub_nodes.append(node.new(inputs, outputs))
        return sub_nodes
        