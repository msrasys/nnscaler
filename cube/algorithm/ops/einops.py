from typing import List, Dict
import warnings

from cube.algorithm.utils import split_axis, split_value
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.operator.function import IREinops, EinDim


class DimSplitEinops(GenericDistAlgo):
    """
    split Einops at dimension level.

    The sum-reduce dimension and non-reduce dimension can be splitted.

    For sum-reduce dimension, the output keeps same shape but has partial-sum valmap result.
    For non-reduce dimension, the output keeps same valmap but has partial output shape.
    For stay-reduce dimension, this dimension is not allowed to be splitted.
    """

    def __init__(self, node: IREinops):
        if not isinstance(node, IREinops):
            raise TypeError(f"Expect IREinops")
        super().__init__(node)
    
    def satisfy(self, config: Dict):
        """
        config = dict(idx=int, dim=int)
        
        idx: int
            input index
        dim: int
            dimension of index-th input
        num: int
            number of chunks to partition
        """
        for attr in ['idx', 'dim', 'num']:
            if not attr in config:
                raise KeyError("Expected idx, dim, num in the config")
        node = self.node
        idx: int = config['idx']
        dim: int = config['dim']
        num: int = config['num']
        if not (isinstance(idx, int) and abs(idx) < len(node.inputs())):
            return False
        if node.inputs(idx).shape is None or abs(dim) >= len(node.inputs(idx).shape):
            return False
        if node.inputs(idx).shape[dim] % num != 0:
            return False
        return True

    def instantiate(self, config: Dict) -> List[IREinops]:
        if not self.satisfy(config):
            return False
        node: IREinops = self.node
        idx: int = config['idx']
        dim: int = config['dim']
        num: int = config['num']
        axis: EinDim = node._ieins[idx][dim]

        # print(f'splitting: {node.einexpr()}')

        ins, ous = list(), list()
        for iidx, input in enumerate(node.inputs()):
            if axis in node._ieins[iidx]:
                dim = node._ieins[iidx].index(axis)
                sub_tensors = split_axis(input, dim, num)
                ins.append(sub_tensors)
            else:
                if axis.is_reduce():
                    print(f'Warning: value split on one input tensor in node{node._id}:{node.name} as reduce axis {axis} not appeared.')
                    ins.append(split_value(input, num))
                else:
                    ins.append([input] * num)
        for oidx, output in enumerate(node.outputs()):
            # split on the non-reduce axis, the output value keeps same
            # but the output shape gets splitted
            if axis in node._oeins[oidx]:
                dim = node._oeins[oidx].index(axis)
                if axis.is_reduce():
                    raise RuntimeError(f"Reduced axis {dim} appeared in output")
                sub_tensors  = split_axis(output, dim, num)
                ous.append(sub_tensors)
            # split on the reduce axis, the output shape keeps same 
            # but the output value get splitted
            else:
                if not axis.is_reduce():
                    raise RuntimeError(f"Expect axis {axis} to be reduced axis")
                sub_tensors = split_value(output, num)
                ous.append(sub_tensors)

        sub_nodes = list()
        for nid in range(num):
            inputs = [t[nid] for t in ins]
            outputs = [t[nid] for t in ous]
            sub_node: IREinops = node.new(inputs, outputs)
            sub_node.make_expression()
            sub_nodes.append(sub_node)
        return sub_nodes
