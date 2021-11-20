from typing import List, Dict, Type

from cube.algorithm.utils import split_axis
from cube.algorithm.generics import GenericDistAlgo
from cube.graph.operator.operator import IRDataOperation


_kWaitDecision = None


class DPDataLoader(GenericDistAlgo):

    def __init__(self, node: IRDataOperation):

        if not isinstance(node, IRDataOperation):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)

        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        for shape in self.output_shapes:
            if chunk_num > 0 and shape[0] % chunk_num != 0:
                return False
        return True

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        
        sub_outputs = list()
        for output in node.outputs():
            sub_output = split_axis(output, 0, self.chunk_num)
            sub_outputs.append(sub_output)
        
        nodes = list()
        for sub_outs in zip(*sub_outputs):
            node = IRDataOperation(data_num = len(sub_outs))
            for idx, out in enumerate(sub_outs):
                node.set_output(idx, out)
            nodes.append(node)
        return nodes
