from typing import List, Optional
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.dimops import IRDimops, DimAnno
from cube.ir.tensor import IRSubTensor


class DimSplitEinops(GenericDistAlgo):
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
        assert isinstance(node.inputs(idx), IRSubTensor), f"partitioning on a non-tensor input"
        dim = dim if dim >= 0 else dim + node.inputs(idx).ndims
        assert dim < node.inputs(idx).ndims, f"dimension output of boundary: {dim} >= {node.inputs(idx).ndims}"
        # due to implementation limits, we only partition the first annotated dimension
        # for inner-dimension cases.
        self._adim: str = node.anno.inputs(idx).dims[dim].identifiers[0]
        self._reduce: DimAnno.ReduceType = node.anno.inputs(idx).dims[dim].reduces[0]
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

        ins, ous = list(), list()
        for iidx, itensor in enumerate(node.inputs()):
            if not isinstance(itensor, IRSubTensor):
                ins.append([itensor] * num)
                continue
            shape_anno = node.anno.inputs(iidx)
            split_dims = shape_anno.getdims(self._adim)
            assert len(split_dims) <= 1, f"find split dims ({self._adim}) more than 1: {shape_anno}"
            if len(split_dims) == 1:
                dim = split_dims[0]
                # split axis
                ins.append(itensor.split_dim(dim, num))
            else:
                # replicate if no split dimension of this tensor
                # ins.append([itensor] * num)
                # ad-hoc FIXME: for linear function Ax+b of splitting reduction dimension, b should
                # be splitted by value dimension.
                if self._reduce == DimAnno.ReduceType.Sum:
                    ins.append(itensor.split_val(num))
                else:
                    ins.append(itensor.replicate(num))

        for oidx, otensor in enumerate(node.outputs()):
            if not isinstance(otensor, IRSubTensor):
                ous.append([otensor] * num)
                continue
            shape_anno = node.anno.outputs(oidx)
            split_dims = shape_anno.getdims(self._adim)
            assert len(split_dims) <= 1, f"find split dims ({self._adim}) more than 1: {shape_anno}"
            # split axis
            if self._reduce == DimAnno.ReduceType.Dim:
                assert len(split_dims) == 1, f"expect only one spatial dimension in output tensor but got {len(split_dims)}"
                dim = split_dims[0]
                ous.append(otensor.split_dim(dim, num))
            # split numerical dimension
            else:
                assert len(split_dims) == 0, f"expect no numerical dimension in output tensor but got {len(split_dims)}"
                ous.append(otensor.split_val(num))

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
        dimi = dimi if dimi >= 0 else dimi + node.inputs(0).ndims
        dimo = dimo if dimo >= 0 else dimo + node.outputs(0).ndims
        assert dimi < node.inputs(0).ndims, f"dimension out of boundary: {dimi} >= {node.inputs(0).ndims}"
        assert dimo < node.outputs(0).ndims, f"dimension out of boundary"
        # # due to implementation limits, we only partition the first annotated dimension
        # # for inner-dimension cases.
        self._adimi: str = node.anno.inputs(0).dims[dimi].identifiers[0]
        self._adimo: str = node.anno.outputs(0).dims[dimo].identifiers[0]
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
            shape_anno = node.anno.inputs(iidx)
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
            shape_anno = node.anno.outputs(oidx)
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