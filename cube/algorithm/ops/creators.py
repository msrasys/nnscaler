from typing import List, Tuple, Optional

from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.creators import IRToTensor, IROnes, IRRand
from cube.ir.tensor import IRSubTensor


class DimSplitTo(GenericDistAlgo):
    """
    split Pad at dimension level

    """
    def __init__(self, node: IRToTensor):
        if not isinstance(node, IRToTensor):
            raise TypeError(f"Expect IRToTensor")
        super().__init__(node)

    def satisfy(self, dim: int, num: int):
        """
        config = dict(idx=int, dim=int, num=num)

        """
        assert all(isinstance(t, int) for t in [dim, num]), "dim and num should be integer"
        node: IRToTensor = self.node
        
        assert dim < len(node.input(0).shape), "Split dimension should be smaller than tensor dimension"

        # split non-pad dim
        return node.input(0).shape[dim] >= num

    def instantiate(self, dim: int, num: int) -> Optional[List[IRToTensor]]:

        node: IRToTensor = self.node
        satisfy = self.satisfy(dim, num)
        if not satisfy:
            return None

        ins, ous = list(), list()
        for iidx, itensor in enumerate(node.inputs()):
            assert isinstance(itensor, IRSubTensor), "Input of select shoud be IRSubTensor"
            ins.append(itensor.split_dim(dim, num))
            
        odim = dim

        for oidx, otensor in enumerate(node.outputs()):
            assert isinstance(otensor, IRSubTensor), "Output of select should be IRSubTensor"
            ous.append(otensor.split_dim(odim, num))

        sub_nodes = list()
        for nid in range(num):
            inputs = [t[nid] for t in ins]
            outputs = [t[nid] for t in ous]
            sub_nodes.append(node.new(inputs, outputs))
        return sub_nodes
    
    
class DimSplitOnes(GenericDistAlgo):
    def __init__(self, node: IROnes):
        if not isinstance(node, IROnes):
            raise TypeError(f"Expect IROnes")
        super().__init__(node)

    def satisfy(self, dim: int, num: int):
        """
        config = dict(idx=int, dim=int, num=num)

        """
        assert all(isinstance(t, int) for t in [dim, num]), "dim and num should be integer"
        node: IROnes = self.node
        
        assert dim < len(node.outputs(0).shape), "Split dimension should be smaller than tensor dimension"

        # split non-pad dim
        return node.outputs(0).shape[dim] >= num

    def instantiate(self, dim: int, num: int) -> Optional[List[IROnes]]:

        node: IROnes = self.node
        satisfy = self.satisfy(dim, num)
        if not satisfy:
            return None
        
        ous = list()
        for oidx, otensor in enumerate(node.outputs()):
            assert isinstance(otensor, IRSubTensor), "Output of select should be IRSubTensor"
            ous.append(otensor.split_dim(dim, num))

        sub_nodes = list()
        for nid in range(num):
            outputs = [t[nid] for t in ous]
            sub_nodes.append(node.new(outputs))
        return sub_nodes
    
class DimSplitRand(GenericDistAlgo):
    def __init__(self, node: IRRand):
        if not isinstance(node, IRRand):
            raise TypeError(f"Expect IRRand")
        super().__init__(node)

    def satisfy(self, dim: int, num: int):
        """
        config = dict(idx=int, dim=int, num=num)

        """
        assert all(isinstance(t, int) for t in [dim, num]), "dim and num should be integer"
        node: IRRand = self.node
        
        assert dim < len(node.outputs(0).shape), "Split dimension should be smaller than tensor dimension"

        # split non-pad dim
        return node.outputs(0).shape[dim] >= num

    def instantiate(self, dim: int, num: int) -> Optional[List[IRRand]]:

        node: IRRand = self.node
        satisfy = self.satisfy(dim, num)
        if not satisfy:
            return None
        
        ous = list()
        for oidx, otensor in enumerate(node.outputs()):
            assert isinstance(otensor, IRSubTensor), "Output of select should be IRSubTensor"
            ous.append(otensor.split_dim(dim, num))

        sub_nodes = list()
        for nid in range(num):
            outputs = [t[nid] for t in ous]
            sub_nodes.append(node.new(outputs))
        return sub_nodes