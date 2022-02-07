from typing import Dict

from cube.algorithm.utils import split_axis, split_axis_custom, split_value
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.operator.function.conv import IRConv2D


class DimSplitConv2D(GenericDistAlgo):
    """
    split Conv2D at dimension level

    N iC H W, oC iC dH dW, oC -> N oC oH oW
    """

    def __init__(self, node: IRConv2D):
        if not isinstance(node, IRConv2D):
            raise TypeError(f"Expect IRConv2D")
        super().__init__(node)

    def satisfy(self, config: Dict):
        """
        config = dict(idx=int, dim=int, num=num)

        N iC H W, oC iC dH dW, oC -> N oC oH oW

        Splittable dimension: N, oC
        Reduce dimension: oC 
        """
        for attr in ['idx', 'dim', 'num']:
            if not attr in config:
                raise KeyError("Expected idx, dim, num in the config")
        node: IRConv2D = self.node
        idx: int = config['idx']
        dim: int = config['dim']
        num: int = config['num']
        groups = node.kwargs['groups']
        # split N:
        if (idx, dim) == (0, 0):
            return node.inputs(0).shape[0] % num == 0
        # split oC
        if (idx, dim) == (1, 0):
            return node.inputs(1).shape[0] % num == 0
        # split iC
        if (idx, dim) == (0, 1) or (idx, dim) == (1, 1):
            return groups == 1 and node.inputs(1).shape[0] % 0 == num

    def instantiate(self, config: Dict):
        if not self.satisfy(config):
            return False
        node: IRConv2D = self.node
        idx: int = config['idx']
        dim: int = config['dim']
        num: int = config['num']

        inputs, weights, bias = list(), list(), list()
        outputs = list()
        # split N
        if (idx, dim) == (0, 0):
            inputs = split_axis(node.inputs(0), axis=0, chunk_num=num)
            weights = [node.inputs(1)] * num
            bias = [node.inputs(2)] * num
            outputs = split_axis(node.outputs(0), axis=0, chunk_num=num)
        # split oC
        if (idx, dim) == (1, 0):
            inputs = [node.inputs(0)] * num
            weights = split_axis(node.inputs(1), axis=0, chunk_num=num)
            if node.inputs(2) is None:
                bias = [None] * num
            else:
                bias = split_axis(node.inputs(2), axis=0, chunk_num=num)
            outputs = split_axis(node.outputs(0), axis=1, chunk_num=num)
        # split iC
        if (idx, dim) == (0, 1) or (idx, dim) == (1, 1):
            inputs = split_axis(node.inputs(0), axis=1, chunk_num=num)
            weights = split_axis(node.inputs(1), axis=1, chunk_num=num)
            if node.inputs(2) is None:
                bias = [None] * num
            else:
                bias = split_value(node.inputs(2), chunk_num=num)
            outputs = split_value(node.outputs(0), chunk_num=num)
        subnodes = list()
        for i, w, b, o in zip(inputs, weights, bias, outputs):
            subnodes.append(node.new([i, w, b], [o]))
        return subnodes


class HaloSplitConv2D(GenericDistAlgo):
    """
    Halo-exchange split

    N iC H W, oC iC dH dW, oC -> N oC oH oW
    """

    def __init__(self, node: IRConv2D):
        if not isinstance(node, IRConv2D):
            raise TypeError(f"Expect IRConv2D")
        super().__init__(node)

    def satisfy(self, config: Dict):
        for attr in ['idx', 'dim', 'num']:
            if not attr in config:
                raise KeyError("Expected idx, dim, num in the config")
        node: IRConv2D = self.node
        oH, oW = node.outputs(0).shape[2:]
        idx: int = config['idx']
        dim: int = config['dim']
        num: int = config['num']
        stride = node.kwargs['stride']
        dilation = node.kwargs['dilation']
        # FIXME: stride
        if stride != [1, 1]:
            raise NotImplementedError("Splitting on stride != [1,1] is not supported")
        if dilation != [1, 1]:
            raise NotImplementedError("Splitting on dilation != [1,1] is not supported")
        # split H
        if (idx, dim) == (0, 2):
            return oH % num == 0
        # split W
        if (idx, dim) == (0, 3):
            return oW % num == 0

    def instantiate(self, config: Dict):
        if not self.satisfy(config):
            return None
        node: IRConv2D = self.node
        H, W = node.inputs(0).shape[2:]
        dH, dW = node.inputs(1).shape[2:]
        oH, oW = node.outputs(0).shape[2:]
        idx: int = config['idx']
        dim: int = config['dim']
        num: int = config['num']
        groups = node.kwargs['groups']
        stride = node.kwargs['stride']
        padding = node.kwargs['padding']
        dilation = node.kwargs['dilation']
        # split H
        if (idx, dim) == (0, 2):
            # input and padding
            slicers = list()
            pads = list()
            start = 0 - padding[0]
            for cid in range(num):
                # padding
                padl = padding[0] if cid == 0 else 0
                padr = padding[1] if cid == num - 1 else 0
                pads.append([padl, padr, padding[2], padding[3]])
                # input  -- FIXME: only work for stride=[1,1]
                chunkH = oH // num + dilation[0] * (dH - 1)
                stop = start + chunkH - padr
                slicers.append(slice(max(0, start), min(H, stop)))
                start = stop - dilation[0] * (dH - 1)
                # start = 0 if cid == 0 else 1023
                # stop = 1025 if cid == 0 else H
            inputs = split_axis_custom(node.inputs(0), axis=dim, chunks=slicers)
            # weight
            weights = [node.inputs(1)] * num
            # bias
            bias = [node.inputs(2)] * num
            # padding
            pads.append([padl, padr, padding[2], padding[3]])
            # outputs
            outputs = split_axis(node.outputs(0), axis=dim, chunk_num=num)
        # split W
        if (idx, dim) == (0, 1):
            raise NotImplementedError("Split on W is not supported yet")
        sub_nodes = list()
        for i, w, b, pad, o in zip(inputs, weights, bias, pads, outputs):
            conv = IRConv2D(node.signature, [i, w, b], node.name,
                stride=stride, padding=pad, dilation=dilation, groups=groups)
            conv.set_output(0, o)
            sub_nodes.append(conv)
        return sub_nodes
