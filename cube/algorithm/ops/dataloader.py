from typing import List
import copy

from cube.algorithm.generics import GenericDistAlgo
from cube.ir.operator import IRDataOperation


class DPDataLoader(GenericDistAlgo):

    def __init__(self, node: IRDataOperation):

        if not isinstance(node, IRDataOperation):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)

    def satisfy(self, num: int):
        """
        Check whether the condition satisfies.

        @param num int: number of chunks to partition
        """

        node: IRDataOperation = self.node
        dims: List[int] = node.get_batch_dims()
        # check batch size
        all_batch_size = set([output.shape[dim] for dim, output in zip(dims, node.outputs())])
        # batch size not same -- indicate a scientific model
        if len(all_batch_size) != 1:
            return False        
        for dim, output in zip(dims, node.outputs()):
            if output.shape[dim] % num != 0:
                return False
        return True

    def instantiate(self, num: int):
        if not self.satisfy(num):
            return False
        node: IRDataOperation = self.node
        dims: List[int] = node.get_batch_dims()

        outputs = list()
        for dim, output in zip(dims, node.outputs()):
            output = output.split_dim(dim, num)
            outputs.append(output)

        nodes = list()
        for outs in zip(*outputs):
            node = IRDataOperation(
                data_num=len(outs), batch_dims=copy.copy(dims))
            for idx, out in enumerate(outs):
                node.set_output(idx, out)
            nodes.append(node)
        return nodes
