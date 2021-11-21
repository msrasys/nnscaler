from typing import Dict

from cube.algorithm.utils import split_axis, split_value
from cube.algorithm.generics import GenericDistAlgo
from cube.graph.operator.function import Linear


_kWaitDecision = None


class LinearDataParallel(GenericDistAlgo):
    """
    Input:
        input: [N, *, in_features]
        weight: [out_features, in_features]
        bias: [out_features,]
    
    Output:
        [N, *, in_features]
    """

    def __init__(self, node: Linear):

        if not isinstance(node, Linear):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)

        # input dimension
        self.ndim = len(node.inputs(0).shape)
        self.dim_choice = list(range(self.ndim - 1))

        self.chunk_num = _kWaitDecision
        if len(self.dim_choice) == 1:
            self.dim = 0
        else:
            self.dim = _kWaitDecision

    def satisfy(self, config: Dict):
        input_shape = self.input_shapes[0]
        if input_shape is None:
            return False
        chunk_num = int(config['chunk_num'])
        if 'dim' in config:
            dim = config['dim']
        else:
            if self.dim is None:
                raise RuntimeError("Expected dim in config")
            dim = self.dim
        if dim < 0:
            dim = self.ndim + dim
        input_shape = self.input_shapes[0]
        if chunk_num > 0 and input_shape[dim] % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        if 'dim' in config:
            self.dim = config['dim']
        input, weight, bias = node.inputs()
        output = node.outputs(0)

        ins = split_axis(input, self.dim, self.chunk_num)
        outs = split_axis(output, self.dim, self.chunk_num)

        nodes = list()
        for input_chunk, output_chunk in zip(ins, outs):
            node = Linear(
                signature='torch.nn.functional.linear',
                inputs=[input_chunk, weight, bias],
                name='linear'
            )
            node.set_output(0, output_chunk)
            nodes.append(node)
        return nodes


class LinearColumnWeight(GenericDistAlgo):

    def __init__(self, node: Linear):

        if not isinstance(node, Linear):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)

        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        weight_shape = self.input_shapes[1]
        if chunk_num > 0 and weight_shape[0] % chunk_num == 0:
            return True
        return False
    
    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        input, weight, bias = node.inputs()
        output = node.outputs(0)

        ws = split_axis(weight, 0, self.chunk_num)
        if bias is not None:
            bs = split_axis(bias, 0, self.chunk_num)
        else:
            bs = [None] * self.chunk_num
        os = split_axis(output, 1, self.chunk_num)

        nodes = list()
        for w, b, o in zip(ws, bs, os):
            node = Linear(
                signature='torch.nn.functional.linear',
                inputs=[input, w, b],
                name='linear'
            )
            node.set_output(0, o)
            nodes.append(node)
        return nodes


class LinearRowWeight(GenericDistAlgo):

    def __init__(self, node: Linear):

        if not isinstance(node, Linear):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)

        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        weight_shape = self.input_shapes[1]
        if chunk_num > 0 and weight_shape[1] % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        input, weight, bias = node.inputs()
        output = node.outputs(0)

        ins = split_axis(input, -1, self.chunk_num)
        ws = split_axis(weight, 1, self.chunk_num)
        if bias:
            bs = split_value(bias, self.chunk_num)
        else:
            bs = [None] * self.chunk_num
        os = split_value(output, self.chunk_num)

        nodes = list()
        for x, w, b, o in zip(ins, ws, bs, os):
            node = Linear(
                signature='torch.nn.functional.linear',
                inputs=[x, w, b],
                name='linear'
            )
            node.set_output(0, o)
            nodes.append(node)
        return nodes
