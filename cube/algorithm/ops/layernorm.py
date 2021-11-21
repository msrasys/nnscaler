from typing import Dict
import copy

from cube.algorithm.utils import split_axis
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.operator.function import LayerNorm


_kWaitDecision = None

class LayerNormDimParallel(GenericDistAlgo):

    def __init__(self, node: LayerNorm, dim=None):
        if not isinstance(node, LayerNorm):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.ndim = len(node.inputs(0).shape)
        last_ndims = len(node.inputs(1))
        self.stay_dims = list()
        for dim in range(last_ndims):
            self.stay_dims.append(self.ndim - dim - 1)

        self.chunk_num = _kWaitDecision
        self.dim = dim

    def satisfy(self, config: Dict):
        if 'dim' in config:
            dim = config['dim']
        else:
            if self.dim is None:
                raise RuntimeError("Expected dim in config")
            dim = self.dim
        if dim < 0:
            dim = self.ndim + dim
        chunk_num = int(config['chunk_num'])
        if dim in self.stay_dims:
            return False
        shape = self.input_shapes[0]
        if dim >= 0 and dim < self.ndim and shape[dim] % chunk_num == 0:
            return True
        return False

    def instantiate(self, node: LayerNorm, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        if 'dim' in config:
            self.dim = config['dim']

        input = node.inputs(0)
        sub_inputs = split_axis(input, self.dim, self.chunk_num)
        
        output = node.outputs(0)
        sub_outputs = split_axis(output, self.dim, self.chunk_num)

        nodes = list()
        for sub_input, sub_output in zip(sub_inputs, sub_outputs):
            inputs = [sub_input] + node.inputs()[1:] + [node.kwargs['eps']]
            sub_node = LayerNorm(node.signature, inputs, node.name)
            sub_node.set_output(0, sub_output)
            nodes.append(sub_node)
        return nodes
