from typing import Dict

from cube.algorithm.utils import split_axis
from cube.algorithm.generics import GenericDistAlgo
from cube.ir.cten import IRTensor

from cube.graph.operator.function import ElementWise


_kWaitDecision = None


class ElementWiseDimParallel(GenericDistAlgo):

    def __init__(self, node: ElementWise, dim=None):
        if not isinstance(node, ElementWise):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.ndim = len(node.inputs(0).shape)
        self.chunk_num = _kWaitDecision
        self.dim = dim

    def satisfy(self, config: Dict):
        if 'dim' in config:
            dim = config['dim']
        else:
            if self.dim is None:
                raise RuntimeError("Expected dim in config")
            dim = self.dim
        chunk_num = int(config['chunk_num'])
        shape = self.input_shapes[0]
        if dim >= 0 and dim < self.ndim and shape[dim] % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        if 'dim' in config:
            self.dim = config['dim']

        sub_inputs = list()
        for input in node.inputs():
            if isinstance(input, IRTensor):
                sub_input = split_axis(input, self.dim, self.chunk_num)
            else:
                sub_input = [input] * self.chunk_num
            sub_inputs.append(sub_input)

        sub_outputs = list()
        for output in node.outputs():
            if isinstance(output, IRTensor):
                sub_output = split_axis(output, self.dim, self.chunk_num)
            else:
                sub_output = [output] * self.chunk_num
            sub_outputs.append(sub_output)

        nodes = list()
        for idx, sub_input in enumerate(zip(*sub_inputs)):
            sub_node = ElementWise(node.signature, inputs=sub_input, name=node.name)
            nodes.append(sub_node)
        for idx, sub_output in enumerate(zip(*sub_outputs)):
            sub_node = nodes[idx]
            for idx, output in enumerate(sub_output):
                sub_node.set_output(idx, output)
        return nodes
