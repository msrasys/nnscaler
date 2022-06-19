from typing import List, Dict, Optional

from cube.algorithm.utils import split_axis, split_value
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.einops import IREinops, DimAnno


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
        self._adim: str = None
        self._reduce: DimAnno.ReduceType = None
    
    def satisfy(self, idx: int, dim: int, num: int) -> bool:
        """
        Check whether the condition satisfies.
        
        @param idx int: input index
        @param dim int: input dimension
        @param num int: chunks to partition the dimension

        @return satisfy bool: true if can be partitioned, elsewise false.
        """
        assert all(isinstance(cond, int) for cond in [idx, dim, num]), "expect int condition"
        node: IREinops = self.node
        
        ninputs = len(node.inputs())
        if idx >= ninputs or idx < 0-ninputs:
            return False
        if node.inputs(idx).shape is None or abs(dim) >= len(node.inputs(idx).shape):
            return False
        # due to implementation limits, we only partition the first annotated dimension
        # for inner-dimension cases.
        self._adim: str = node.anno.inputs(idx).dims[dim].identifiers[0]
        self._reduce: DimAnno.ReduceType = node.anno.inputs(idx).dims[dim].reduces[0]
        dimlen = node.anno.getlen(self._adim)
        if self._reduce == DimAnno.ReduceType.Freeze:
            return False
        if dimlen % num != 0:
            return False
        return True

    def instantiate(self, idx: int, dim: int, num: int) -> Optional[List[IREinops]]:
        if not self.satisfy(idx, dim, num):
            return False
        node: IREinops = self.node
        print(f'{node.anno} | => partition: {self._adim} reduce: {self._reduce.value}')

        ins, ous = list(), list()
        for iidx, itensor in enumerate(node.inputs()):
            shape_anno = node.anno.inputs(iidx)
            split_dims = shape_anno.getdims(self._adim)
            assert len(split_dims) <= 1, f"find split dims ({self._adim}) more than 1: {shape_anno}"
            if len(split_dims) == 1:
                dim = split_dims[0]
                # split axis
                sub_tensors = split_axis(itensor, dim, num)
                ins.append(sub_tensors)
            else:
                # replicate if no split dimension of this tensor
                # ins.append([itensor] * num)
                # ad-hoc FIXME: for linear function Ax+b of splitting reduction dimension, b should
                # be splitted by value dimension.
                if self._reduce == DimAnno.ReduceType.Sum:
                    ins.append(split_value(itensor, num))
                else:
                    ins.append([itensor] * num)
        
        for oidx, otensor in enumerate(node.outputs()):
            shape_anno = node.anno.outputs(oidx)
            split_dims = shape_anno.getdims(self._adim)
            assert len(split_dims) <= 1, f"find split dims ({self._adim}) more than 1: {shape_anno}"
            # split axis
            if self._reduce == DimAnno.ReduceType.Dim:
                assert len(split_dims) == 1, f"expect only one spatial dimension in output tensor but got {len(split_dims)}"
                dim = split_dims[0]
                sub_tensors = split_axis(otensor, dim, num)
                ous.append(sub_tensors)
            # split numerical dimension
            else:
                assert len(split_dims) == 0, f"expect no numerical dimension in output tensor but got {len(split_dims)}"
                sub_tensors = split_value(otensor, num)
                ous.append(sub_tensors)

        sub_nodes = list()
        for nid in range(num):
            inputs = [t[nid] for t in ins]
            outputs = [t[nid] for t in ous]
            updated_kwargs = dict()
            if self._adim in node.kwargs and isinstance(node.kwargs[self._adim], int):
                updated_kwargs[self._adim] = node.kwargs[self._adim] // num
            sub_node: IREinops = node.new(inputs, outputs, **updated_kwargs)
            sub_node.infer_shape()
            sub_nodes.append(sub_node)
        return sub_nodes
