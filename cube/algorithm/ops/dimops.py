from typing import List, Optional, Any, Dict, Union
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.dimops import IRDimops, DimAnno, DimopSplit, TransformRule
from cube.ir.tensor import IRSubTensor


class DimSplitEinops(GenericDistAlgo):
    """!
    Split Dimops at tensor dimension.

    Note: for dimensions of multiple identitifers, only the first identifier
    can be partitioned.

    Default rule for identifier split:
        * Sum-reduce identifier ('+'):
            * For inputs/outputs that have the identifier, will be partitioned on its diemension uniformly..
            * For inputs that don't have the identifier, will be replicated
            * For outputs that don't have the identifier, will be partitioned on its value uniformly.

        * Spatial identifier (''):
            * For inputs/outputs that have the identifier, will be partitioned on its diemnsion uniformly.
            * For inputs/outputs that don't have the identifier, will be replicated

        * Frozen identifier ('^'):
            * Cannot be partitioned.

        If the identifier appears as the same name in argument name, the
        argument will also be uniformly partitioned.
        
        Non-tensor will always be replicated.
    
    Note the default rule isn't always expressive for all possible partition algorithms.
    E.g., linear xw + b to partition on reduction dimension,
    whitch requires b to be value split but actually according to the default rule, will be replicated.
    Therefore we require special rules for such cases.
    """

    def __init__(self, node: IRDimops):
        if not isinstance(node, IRDimops):
            raise TypeError(f"Expect IRDimops")
        super().__init__(node)
    
    def satisfy(self, idx: int, dim: Union[int, str], num: int) -> bool:
        """
        Check whether the condition satisfies.
        
        @param idx int: input index
        @param dim Union[int, str]: input dimension or 'v', ie., partition at value dimension
        @param num int: chunks to partition the dimension

        @return satisfy bool: true if can be partitioned, elsewise false.
        """
        assert all(isinstance(cond, int) for cond in [idx, num]), "expect int condition"
        assert isinstance(dim, int) or dim == 'v', f"expect dim to be int or 'v'"
        node: IRDimops = self.node
        
        assert isinstance(node.input(idx), IRSubTensor), f"partitioning on a non-tensor input"
        ninputs = len(node.inputs())
        idx = idx if idx >= 0 else idx + ninputs
        assert idx < ninputs, f"index out of boundary: {idx} >= {ninputs}"

        if isinstance(dim, int):
            dim = dim if dim >= 0 else dim + node.input(idx).ndims
            assert dim < node.input(idx).ndims, f"dimension output of boundary: {dim} >= {node.input(idx).ndims}"
        
        # try split at tensor spatial dimension
        if isinstance(dim, int):
            for adim in node.anno.input(idx).dims[dim].identifiers:
                if adim == '1^': continue
                break
            dimlen = node.anno.getlen(adim)
            # first check node special rules first
            for rule in node.transform_rules:
                if rule.input(idx) == DimopSplit.D(dim):
                    return dimlen >= num
            # then check default rules
            reduce = node.anno.input(idx).dims[dim].reduces[0]
            if reduce == DimAnno.ReduceType.Freeze:
                return False
            return dimlen >= num
        else:
            for rule in node.transform_rules:
                if rule.input(idx).isV():
                    return True
            return False
        

    def instantiate(self, idx: int, dim: Union[int, str], num: int) -> Optional[List[IRDimops]]:

        node: IRDimops = self.node
        satisfy = self.satisfy(idx, dim, num)

        if isinstance(dim, int):
            for adim in node.anno.input(idx).dims[dim].identifiers:
                if adim == '1^': continue
                break
            reduce: DimAnno.ReduceType = node.anno.input(idx).dims[dim].reduces[0]
        else:
            adim, reduce = 'Value', None
        color, default = '\033[32m' if satisfy else '\033[31m', '\033[0m'
        print(f"try split {node.name}: {node.anno} | dim: {adim} reduce: {reduce} ... {color}{'Success' if satisfy else 'Failed!'}{default}")
        if not satisfy: return None

        rule: TransformRule = self.infer(idx, dim, num)
    
        # transform
        def transform(tensor: Any, split: DimopSplit) -> List[Any]:
            if not isinstance(tensor, IRSubTensor):
                return [tensor] * num
            if split.isD():
                return tensor.split_dim(split.dim, num)
            if split.isR():
                return tensor.replicate(num)
            if split.isV():
                return tensor.split_val(num)
            assert False, f"got unknown split: {split}"

        ins = list()
        for split, itensor in zip(rule.inputs(), node.inputs()):
            ins.append(transform(itensor, split))
        ous = list()
        for split, otensor in zip(rule.outputs(), node.outputs()):
            ous.append(transform(otensor, split))
        kwargs = rule.modifier()(node.kwargs, idx, dim, num)

        sub_nodes = list()
        for nid in range(num):
            inputs = [t[nid] for t in ins]
            outputs = [t[nid] for t in ous]
            sub_node: IRDimops = node.new(inputs, outputs, **kwargs)
            sub_node.infer_shape()
            sub_nodes.append(sub_node)

        return sub_nodes

    def infer(self, idx: int, dim: Union[int, str], num: int) -> Optional[TransformRule]:
        """
        Given the partition choice on `dim` dimension of idx-th input,
        return the partitioning of the output tensor.

        @param idx int: the input index
        @param dim int: the dimension to partition

        @return rule TransformRule: the transformation rule
        """
        node: IRDimops = self.node
        assert isinstance(dim, int) or dim == 'v', f"expect dim to be int or 'v'"
        # check node special rules first
        for r in node.transform_rules:
            if isinstance(dim, int):
                if r.input(idx) == DimopSplit.D(dim):
                    return r
            else:
                if r.input(idx).isV():
                    return r
        # otherwise use default rule
        assert isinstance(dim, int), f"Error: expect dim to be int for default rules"
        adim: str = node.anno.input(idx).dims[dim].identifiers[0]
        reduce: DimAnno.ReduceType = node.anno.input(idx).dims[dim].reduces[0]
        if reduce == DimAnno.ReduceType.Freeze:
            return None
        itransform, otransform = [], []
        # input
        for idx, idim in enumerate(node.anno.inputs()):
            dims = idim.getdims(adim)
            assert len(dims) <= 1, "Cannot split on multple same tensors"
            if len(dims) == 1:
                itransform.append(DimopSplit.D(dims[0]))
            else:
                itransform.append(DimopSplit.R())
        # output
        for idx, odim in enumerate(node.anno.outputs()):
            dims = odim.getdims(adim)
            if len(dims) == 1:
                otransform.append(DimopSplit.D(dims[0]))
            else:
                otransform.append(
                    DimopSplit.R() if reduce == DimAnno.ReduceType.Dim else DimopSplit.V()
                )
        # modifier
        def modify(kwargs: Dict, idx: int, dim: int, num: int):
            updated_kwargs = dict(**kwargs)
            if adim in updated_kwargs:
                assert updated_kwargs[adim] % num == 0, \
                    f"cannot set kwargs: {adim}: {updated_kwargs[adim]} % num ({num}) != 0"
                updated_kwargs[adim] = updated_kwargs[adim] // num
            return updated_kwargs

        return TransformRule(itransform, otransform, modify)


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
            updated_kwargs['size'] = outputs[0].shape
            sub_node: IRDimops = node.new(inputs, outputs, **updated_kwargs)
            sub_node.infer_shape()
            sub_nodes.append(sub_node)
        return sub_nodes