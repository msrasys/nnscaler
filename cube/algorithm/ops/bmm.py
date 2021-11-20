from typing import Dict

from cube.algorithm.utils import split_axis, split_value
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.operator.function import BatchLinear


_kWaitDecision = None


class BatchLinearDataParallel(GenericDistAlgo):
    """
    Inputs:
        input1: [B, N, M]
        input2: [B, M, P]

    Outputs:
        output: [B, N, P]
    """

    def __init__(self, node: BatchLinear):

        if not isinstance(node, BatchLinear):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        input_shape = self.input_shapes[0]
        if chunk_num > 0 and input_shape[0] % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        input1, input2 = node.inputs()
        output = node.outputs(0)

        in1s = split_axis(input1, 0, self.chunk_num)
        in2s = split_axis(input2, 0, self.chunk_num)
        outs = split_axis(output, 0, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            node = BatchLinear(
                signature='torch.bmm',
                inputs=[in1s[idx], in2s[idx]],
                name='bmm'
            )
            node.set_output(0, outs[idx])
            nodes.append(node)
        return nodes


class BatchLinearNParallel(GenericDistAlgo):
    """
    Inputs:
        input1: [B, N, M]
        input2: [B, M, P]

    Outputs:
        output: [B, N, P]
    """

    def __init__(self, node: BatchLinear):

        if not isinstance(node, BatchLinear):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        input_shape = self.input_shapes[0]
        if chunk_num > 0 and input_shape[1] % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        input1, input2 = node.inputs()
        output = node.outputs(0)

        in1s = split_axis(input1, 1, self.chunk_num)
        outs = split_axis(output, 1, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            node = BatchLinear(
                signature='torch.bmm',
                inputs=[in1s[idx], input2],
                name='bmm'
            )
            node.set_output(0, outs[idx])
            nodes.append(node)
        return nodes


class BatchLinearMParallel(GenericDistAlgo):
    """
    Inputs:
        input1: [B, N, M]
        input2: [B, M, P]

    Outputs:
        output: [B, N, P]
    """

    def __init__(self, node: BatchLinear):

        if not isinstance(node, BatchLinear):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        input_shape = self.input_shapes[0]
        if chunk_num > 0 and input_shape[2] % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        input1, input2 = node.inputs()
        output = node.outputs(0)

        in1s = split_axis(input1, 2, self.chunk_num)
        in2s = split_axis(input2, 1, self.chunk_num)
        outs = split_value(output, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            node = BatchLinear(
                signature='torch.bmm',
                inputs=[in1s[idx], in2s[idx]],
                name='bmm'
            )
            node.set_output(0, outs[idx])
            nodes.append(node)
        return nodes


class BatchLinearPParallel(GenericDistAlgo):
    """
    Inputs:
        input1: [B, N, M]
        input2: [B, M, P]

    Outputs:
        output: [B, N, P]
    """

    def __init__(self, node: BatchLinear):

        if not isinstance(node, BatchLinear):
            raise TypeError(f"f{type(node)} can not be transformed to {type(self)}")
        super().__init__(node)
        self.chunk_num = _kWaitDecision

    def satisfy(self, config: Dict):
        chunk_num = int(config['chunk_num'])
        input_shape = self.input_shapes[1]
        if chunk_num > 0 and input_shape[2] % chunk_num == 0:
            return True
        return False

    def instantiate(self, node, config: Dict):
        if not self.satisfy(config):
            raise RuntimeError("Instantiate failed. Condition not satisfied.")
        self.chunk_num = int(config['chunk_num'])
        input1, input2 = node.inputs()
        output = node.outputs(0)

        in2s = split_axis(input2, 2, self.chunk_num)
        outs = split_axis(output, 2, self.chunk_num)

        nodes = list()
        for idx in range(self.chunk_num):
            node = BatchLinear(
                signature='torch.bmm',
                inputs=[input1, in2s[idx]],
                name='bmm'
            )
            node.set_output(0, outs[idx])
            nodes.append(node)
        return nodes
