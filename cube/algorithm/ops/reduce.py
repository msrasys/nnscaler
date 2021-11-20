from typing import Dict
import copy

from cube.algorithm.utils import split_axis, split_value
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.operator.function import Sum


_kWaitDecision = None


class SumDimParallel(GenericDistAlgo):

    def __init__(self, node: Sum, dim=None):
        if not isinstance(node, Sum):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.ndim = len(node.inputs(0).shape)
        self.reduce_dims = list(range(self.ndim))
        self.keepdim = [False] * self.ndim
        if 'dim' in node.kwargs:
            self.reduce_dims = [node.kwargs['dim']]
        if 'keepdim' in node.kwargs:
            self.keepdim = [node.kwargs['keepdim']] * self.ndim

        self.chunk_num = _kWaitDecision
        if dim is not None:
            dim = self.ndim + dim if dim < 0 else dim
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
        shape = self.input_shapes[0]
        if dim >= 0 and dim < self.ndim and shape[dim] % chunk_num == 0:
            return True
        return False

    def instantiate(self, node: Sum, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        if 'dim' in config:
            self.dim = config['dim']
            self.dim = self.ndim + self.dim if self.dim < 0 else self.dim

        assert len(node.inputs()) == 1
        input = node.inputs(0)
        sub_inputs = split_axis(input, self.dim, self.chunk_num)

        assert len(node.outputs()) == 1
        output = node.outputs(0)
        print(self.reduce_dims)
        if self.dim not in self.reduce_dims:
            sub_outputs = split_axis(output, self.dim, self.chunk_num)
        else:
            sub_outputs = split_value(output, self.chunk_num)

        nodes = list()
        if 'dim' in node.kwargs:
            dim = node.kwargs['dim']
        else:
            dim = None
        for input, output in zip(sub_inputs, sub_outputs):
            sub_node = type(node)(node.signature, inputs=[input, dim], name=node.name)
            sub_node.kwargs = copy.copy(node.kwargs)
            sub_node.set_output(0, output)
            nodes.append(sub_node)
        return nodes
