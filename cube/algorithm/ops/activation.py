from typing import Any, Dict, List
import copy

from cube.algorithm.utils import split_axis
from cube.algorithm.generics import GenericDistAlgo
from cube.ir.cten import IRTensor

from cube.graph.operator.function import Activation
from cube.graph.operator.function import Dropout
from cube.graph.operator.function import Softmax


_kWaitDecision = None


class ActivationDimParallel(GenericDistAlgo):

    def __init__(self, node: Activation, dim=None):
        if not isinstance(node, Activation):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.ndim = len(node.inputs(0).shape)
        self.chunk_num = _kWaitDecision
        self.dim = dim
        # stay dim convert to positive dim
        self.stay_dims = list()
        for sdim in node.stay_dims:
            sdim = sdim if sdim >= 0 else self.ndim + sdim
            self.stay_dims.append(sdim)

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

    def get_extra_kwargs(self, node) -> List[Any]:
        """
        Get extra kwarg inputs for the activation

        Returns:
            value in List
        """
        return []

    def instantiate(self, node: Activation, config: Dict):
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
            extra_input = self.get_extra_kwargs(node)
            sub_input = list(sub_input) + extra_input
            sub_node = type(node)(node.signature, inputs=sub_input, name=node.name)
            sub_node.stay_dims = copy.copy(node.stay_dims)
            nodes.append(sub_node)
        for idx, sub_output in enumerate(zip(*sub_outputs)):
            sub_node = nodes[idx]
            for idx, output in enumerate(sub_output):
                sub_node.set_output(idx, output)
        return nodes


class DropoutDimParallel(ActivationDimParallel):

    def __init__(self, node: Activation, dim=None, execlude_dims=None):
        super().__init__(node, dim=dim, execlude_dims=execlude_dims)

    def get_extra_kwargs(self, node: Dropout) -> List[Any]:
        if not isinstance(node, Dropout):
            raise TypeError("Expected Dropout for DropoutDimParallel")
        kwargs = [node.kwargs['p'], node.kwargs['training'], node.kwargs['inplace']]
        return kwargs


class SoftmaxDimParallel(ActivationDimParallel):

    def __init__(self, node: Activation, dim=None, execlude_dims=None):
        super().__init__(node, dim=dim, execlude_dims=execlude_dims)

    def get_extra_kwargs(self, node) -> List[Any]:
        if not isinstance(node, Softmax):
            raise TypeError("Expected Softmax for SoftmaxDimParallel")
        kwargs = [node.kwargs['dim'], node.kwargs['_stacklevel'], node.kwargs['dtype']]
        return kwargs
