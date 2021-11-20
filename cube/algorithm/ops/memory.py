from typing import Dict
import copy

from cube.algorithm.utils import split_axis
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.operator.function import Transpose


_kWaitDecision = None


class TransposeDimParallel(GenericDistAlgo):

    def __init__(self, node: Transpose, dim=None):
        if not isinstance(node, Transpose):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)

        self.dim0 = node.kwargs['dim0']
        self.dim1 = node.kwargs['dim1']
        self.ndim = len(node.inputs(0).shape)

        # config
        self.chunk_num = _kWaitDecision
        self.dim = dim

    def satisfy(self, config: Dict):
        if 'dim' in config:
            dim = config['dim']
            dim = self.ndim + dim if dim < 0 else dim
        else:
            if self.dim is None:
                raise RuntimeError("Expected dim in config")
            dim = self.dim
        chunk_num = int(config['chunk_num'])
        shape = self.input_shapes[0]
        if dim >= 0 and dim < self.ndim and shape[dim] % chunk_num == 0:
            return True
        return False

    def instantiate(self, node: Transpose, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        if 'dim' in config:
            self.dim = config['dim']

        input = node.inputs(0)
        sub_inputs = split_axis(input, self.dim, self.chunk_num)

        output = node.outputs(0)
        target_dim = self.dim
        if self.dim == self.dim0:
            target_dim = self.dim1
        if self.dim == self.dim1:
            target_dim = self.dim0
        sub_outputs = split_axis(output, target_dim, self.chunk_num)

        nodes = list()
        for input, output in zip(sub_inputs, sub_outputs):
            sub_node = type(node)(
                node.signature, inputs=[input, self.dim0, self.dim1], name=node.name
            )
            sub_node.kwargs = copy.copy(node.kwargs)
            sub_node.set_output(0, output)
            nodes.append(sub_node)
        return nodes
