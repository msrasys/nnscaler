from typing import Dict
import copy

from cube.algorithm.utils import split_axis
from cube.algorithm.generics import GenericDistAlgo
from cube.graph.operator.operator import IRDataOperation


_kWaitDecision = None


class DPDataLoader(GenericDistAlgo):

    def __init__(self, node: IRDataOperation):

        if not isinstance(node, IRDataOperation):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.batch_dims = node.get_batch_dims()

        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        for bdim, shape in zip(self.batch_dims, self.output_shapes):
            if chunk_num > 0 and shape[bdim] % chunk_num != 0:
                return False
        return True

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        sub_outputs = list()
        for bdim, output in zip(self.batch_dims, node.outputs()):
            sub_output = split_axis(output, bdim, self.chunk_num)
            sub_outputs.append(sub_output)
        
        nodes = list()
        for sub_outs in zip(*sub_outputs):
            node = IRDataOperation(
                data_num = len(sub_outs), batch_dims = copy.copy(self.batch_dims))
            for idx, out in enumerate(sub_outs):
                node.set_output(idx, out)
            nodes.append(node)
        return nodes
