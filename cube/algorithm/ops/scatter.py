from typing import List, Tuple, Optional
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.scatter import IRSelectScatter
from cube.ir.tensor import IRSubTensor


class DimSplitScatter(GenericDistAlgo):
    """
    split Pad at dimension level

    """
    def __init__(self, node: IRSelectScatter):
        if not isinstance(node, IRSelectScatter):
            raise TypeError(f"Expect IRSelectScatter")
        super().__init__(node)

    def satisfy(self, diml: int, dimr: int, num: int):
        """
        config = dict(idx=int, dim=int, num=num)

        """
        assert all(isinstance(t, int) for t in [diml, dimr, num]), "dim and num should be integer"
        node: IRSelectScatter = self.node
        
        assert diml != node.kwargs['dim'], "Split dimension should not be equal to scatter dimension"
        assert diml < len(node.input(0).shape), "Split dimension should be smaller than tensor dimension"
        assert dimr < len(node.output(0).shape), "Split dimension should be smaller than tensor dimension"
        assert node.input(0).shape[diml] == node.input(1).shape[dimr], "Two split dimension should at least have equal size"
        
        return node.input(0).shape[diml] >= num

    def instantiate(self, diml: int, dimr: int, num: int) -> Optional[List[IRSelectScatter]]:

        node: IRSelectScatter = self.node
        satisfy = self.satisfy(diml, dimr, num)
        if not satisfy:
            return None
          
        assert len(node.inputs()) == 2, "Select_scatter do not has two inputs"
        assert len(node.outputs()) == 1, "Select_scatter do not has one outputs"

        ins, ous = list(), list()
        ins.append(node.input(0).split_dim(diml, num))
        ins.append(node.input(1).split_dim(dimr, num))

        ous.append(node.output(0).split_dim(diml, num))

        sub_nodes = list()
        for nid in range(num):
            inputs = tuple([t[nid] for t in ins])
            outputs = [t[nid] for t in ous]
            sub_nodes.append(node.new(inputs, outputs))
        return sub_nodes
        