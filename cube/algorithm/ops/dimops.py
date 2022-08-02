from typing import List, Optional, Any
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.dimops import IRDimops, DimAnno
from cube.ir.tensor import IRSubTensor


class DimSplitEinops(GenericDistAlgo):
    """!
    Split Dimops at tensor dimension.

    Note: for dimensions of multiple identitifers, only the first identifier
    can be partitioned.

    Rules for identifier split:
        * Sum-reduce identifier ('+'):
            * For inputs/outputs that have the identifier, will be partitioned on its diemension uniformly..
            * For inputs that don't have the identifier, will be replicated
            * For outputs that don't have the identifier, will be partitioned on its value uniformly.

        * Spatial identifier (''):
            * For inputs/outputs that have the identifier, will be partitioned on its diemnsion uniformly.
            * For inputs/outputs that don't have the identifier, will be replicated

        * Frozen identifier ('^'):
            * Cannot be partitioned.
        
        Non-tensor will always be replicated.
    
    Note this rule will not correctly apply for some operators like linear: xw + b
    """

    def __init__(self, node: IRDimops):
        if not isinstance(node, IRDimops):
            raise TypeError(f"Expect IRDimops")
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
        node: IRDimops = self.node
        
        ninputs = len(node.inputs())
        idx = idx if idx >= 0 else idx + ninputs
        assert idx < ninputs, f"index out of boundary: {idx} >= {ninputs}"
        assert isinstance(node.input(idx), IRSubTensor), f"partitioning on a non-tensor input"
        dim = dim if dim >= 0 else dim + node.input(idx).ndims
        assert dim < node.input(idx).ndims, f"dimension output of boundary: {dim} >= {node.input(idx).ndims}"
        # we only partition the first annotated dimension for inner-dimension cases.
        self._adim: str = node.anno.input(idx).dims[dim].identifiers[0]
        self._reduce: DimAnno.ReduceType = node.anno.input(idx).dims[dim].reduces[0]
        dimlen = node.anno.getlen(self._adim)
        if self._reduce == DimAnno.ReduceType.Freeze:
            return False
        if dimlen < num:
            return False
        return True

    def instantiate(self, idx: int, dim: int, num: int) -> Optional[List[IRDimops]]:

        node: IRDimops = self.node
        satisfy = self.satisfy(idx, dim, num)
        print(f'partition {node.name}: {node.anno} | dim: {self._adim} reduce: {self._reduce.value}')
        if not satisfy:
            return None

        def transform(tensor: Any, split_dims: List[int], is_input: bool):
            # rule: non-tensor will always be replicated
            if not isinstance(tensor, IRSubTensor):
                return [tensor] * num
            assert len(split_dims) <= 1, "find split dims ({self._adim}) more than 1"
            # rule: spatial identifier ('')
            if self._reduce == DimAnno.ReduceType.Dim:
                return tensor.replicate(num) if len(split_dims) == 0 else tensor.split_dim(split_dims[0], num)
            # rule: reduce-sum identifier ('+')
            if self._reduce == DimAnno.ReduceType.Sum:
                if len(split_dims) == 0:
                    return tensor.replicate(num) if is_input else tensor.split_val(num)
                else:
                    return tensor.split_dim(split_dims[0], num)
            raise RuntimeError(f"no matching reduce type for transform: {self._reduce}")

        ins, ous = list(), list()
        for iidx, itensor in enumerate(node.inputs()):
            split_dims = node.anno.input(iidx).getdims(self._adim)
            ins.append(transform(itensor, split_dims, is_input=True))

        for oidx, otensor in enumerate(node.outputs()):
            split_dims = node.anno.output(oidx).getdims(self._adim)
            ous.append(transform(otensor, split_dims, is_input=False))

        sub_nodes = list()
        for nid in range(num):
            inputs = [t[nid] for t in ins]
            outputs = [t[nid] for t in ous]
            updated_kwargs = dict()
            if self._adim in node.kwargs and isinstance(node.kwargs[self._adim], int):
                updated_kwargs[self._adim] = node.kwargs[self._adim] // num
            sub_node: IRDimops = node.new(inputs, outputs, **updated_kwargs)
            sub_node.infer_shape()
            sub_nodes.append(sub_node)

        return sub_nodes


class SimpleViewSplitEinops(GenericDistAlgo):
    """
    split Einops at dimension level.

    The sum-reduce dimension and non-reduce dimension can be splitted.

    For sum-reduce dimension, the output keeps same shape but has partial-sum valmap result.
    For non-reduce dimension, the output keeps same valmap but has partial output shape.
    For stay-reduce dimension, this dimension is not allowed to be splitted.
    """

    def __init__(self, node: IRDimops):
        if not isinstance(node, IRDimops):
            raise TypeError(f"Expect IRDimops")
        super().__init__(node)
        self._adim: str = None
        self._reduce: DimAnno.ReduceType = None
    
    def satisfy(self, idx: int, dimi: int, dimo: int, num: int) -> bool:
        """
        Check whether the condition satisfies.
        
        @param idx int: input index
        @param dimi int: input dimension
        @param dimo int: corresponding output dimension
        @param num int: chunks to partition the dimension

        @return satisfy bool: true if can be partitioned, elsewise false.
        """
        # assert all(isinstance(cond, int) for cond in [idx, dim, num]), "expect int condition"
        node: IRDimops = self.node
        assert idx == 0, f"Index should be 0"
        assert len(node.inputs()) == 1, f"Inputs size should be 1"
        assert len(node.outputs()) == 1, f"Outputs size should be 1"
        dimi = dimi if dimi >= 0 else dimi + node.input(0).ndims
        dimo = dimo if dimo >= 0 else dimo + node.output(0).ndims
        assert dimi < node.input(0).ndims, f"dimension out of boundary: {dimi} >= {node.input(0).ndims}"
        assert dimo < node.output(0).ndims, f"dimension out of boundary"
        # # due to implementation limits, we only partition the first annotated dimension
        # # for inner-dimension cases.
        idi = 1 if dimi == 0 else 0
        ido = 1 if dimo == 0 else 0
        self._adimi: str = node.anno.input(0).dims[dimi].identifiers[idi]
        self._adimo: str = node.anno.output(0).dims[dimo].identifiers[ido]
        dimlen = node.anno.getlen(self._adimi)
        if dimlen < num:
            return False
        return True

    def instantiate(self, idx: int, dimi: int, dimo: int, num: int) -> Optional[List[IRDimops]]:

        node: IRDimops = self.node
        satisfy = self.satisfy(idx, dimi, dimo, num)
        if not satisfy:
            return None

        ins, ous = list(), list()
        for iidx, itensor in enumerate(node.inputs()):
            if not isinstance(itensor, IRSubTensor):
                assert 0, "should not happen"
            shape_anno = node.anno.input(iidx)
            split_dims = shape_anno.getdims(self._adimi)
            assert len(split_dims) <= 1, f"find split dims ({self._adimi}) more than 1: {shape_anno}"
            if len(split_dims) == 1:
                dim = split_dims[0]
                # split axis
                # print('dimi =', dim)
                ins.append(itensor.split_dim(dim, num))
            else:
                assert 0, "should not happen"

        for oidx, otensor in enumerate(node.outputs()):
            if not isinstance(otensor, IRSubTensor):
                assert 0, f"should not happen"
            shape_anno = node.anno.output(oidx)
            split_dims = shape_anno.getdims(self._adimo)
            assert len(split_dims) <= 1, f"find split dims ({self._adimo}) more than 1: {shape_anno}"
            # split axis
            if self._reduce != DimAnno.ReduceType.Dim:
                assert len(split_dims) == 1, f"expect only one spatial dimension in output tensor but got {len(split_dims)}"
                dim = split_dims[0]
                # print('dimo =', dim)
                ous.append(otensor.split_dim(dim, num))
            # split numerical dimension
            else:
                assert 0, f"not implemented"

        sub_nodes = list()
        for nid in range(num):
            inputs = [t[nid] for t in ins]
            outputs = [t[nid] for t in ous]
            updated_kwargs = dict()
            if self._adimi in node.kwargs and isinstance(node.kwargs[self._adimi], int):
                assert 0, "should not happen"
            if self._adimo in node.kwargs and isinstance(node.kwargs[self._adimo], int):
                assert 0, "should not happen"
            assert len(outputs) == 1, f"outputs len should be one"
            node.kwargs['size'] = outputs[0].shape
            sub_node: IRDimops = node.new(inputs, outputs, **updated_kwargs)
            sub_node.infer_shape()
            sub_nodes.append(sub_node)
        return sub_nodes