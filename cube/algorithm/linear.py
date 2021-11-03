from typing import List, Optional, Dict

from cube.algorithm.utils import split_axis, split_value
from cube.algorithm.generics import GenericDistAlgo
from cube.graph.operator.function import Linear


_kWaitDecision = None


class LinearDataParallel(GenericDistAlgo):

    def __init__(self, input_shapes: List[Optional[List[int]]], output_shapes: List[int]):

        super().__init__(input_shapes, output_shapes)

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
        input, weight, bias = node.inputs()
        output = node.outputs(0)

        ins = split_axis(input, 0, self.chunk_num)
        outs = split_axis(output, 0, self.chunk_num)

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

    def __init__(self, input_shapes: List[Optional[List[int]]], output_shapes: List[int]):

        super().__init__(input_shapes, output_shapes)

        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        weight_shape = self.input_shapes[1]
        if weight_shape[0] % chunk_num != 0:
            return False
        return True
    
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

    def __init__(self, input_shapes: List[Optional[List[int]]], output_shapes: List[int]):

        super().__init__(input_shapes, output_shapes)

        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        weight_shape = self.input_shapes[1]
        if weight_shape[1] % chunk_num != 0:
            return False
        return True

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        input, weight, bias = node.inputs()
        output = node.outputs(0)

        ins = split_axis(input, 1, self.chunk_num)
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
