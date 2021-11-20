from typing import Dict

from cube.algorithm.utils import split_axis
from cube.algorithm.generics import GenericDistAlgo
from cube.graph.operator.function import ElementWise
from cube.ir.cten import IRTensor


_kWaitDecision = None


class ElementWiseDataParallel(GenericDistAlgo):

    def __init__(self, node: ElementWise):
        if not isinstance(node, ElementWise):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        input_shape = self.input_shapes[0]
        if chunk_num > 0 and input_shape[0] % chunk_num != 0:
            return False
        return True

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])

        sub_inputs = list()
        for input in node.inputs():
            if isinstance(input, IRTensor):
                sub_input = split_axis(input, 0, self.chunk_num)
            else:
                sub_input = [input] * self.chunk_num
            sub_inputs.append(sub_input)

        sub_outputs = list()
        for output in node.outputs():
            if isinstance(output, IRTensor):
                sub_output = split_axis(output, 0, self.chunk_num)
            else:
                sub_output = [output] * self.chunk_num
            sub_outputs.append(sub_output)

        nodes = list()
        for idx, sub_input in enumerate(zip(*sub_inputs)):
            node = ElementWise(node.signature, inputs=sub_input, name=node.name)
            nodes.append(node)
        for idx, sub_output in enumerate(zip(*sub_outputs)):
            node = nodes[idx]
            for idx, output in enumerate(sub_output):
                node.set_output(idx, output)
        return nodes
