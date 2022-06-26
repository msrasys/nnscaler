
from typing import List, Tuple

from cube.ir.tensor import IRSubTensor
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.conv import IRConv2D
from cube.graph.function.conv import IRConv3D


def _split_axis_custom(tensor: IRSubTensor, dim: int, chunks: List[Tuple[int, int]]):
    """
    Split tensor along an axis with customized selection
    """
    dim = len(tensor.shape) + dim if dim < 0 else dim
    assert dim < len(tensor.shape), f"dim should within ndims ({dim} >= {tensor.ndims})"
    chunk_num = len(chunks)
    indmap = list()
    for nele in tensor.shape:
        indmap.append((0, nele))
    sub_tensors = list()
    for cid in range(chunk_num):
        indmap[dim] = chunks[cid]
        sub_tensors.append(tensor.select(
            indmap=tuple(indmap), valmap=(0,1)
        ))
    return sub_tensors


class DimSplitConv2D(GenericDistAlgo):
    """
    split Conv2D at dimension level

    N iC H W, oC iC dH dW, oC -> N oC oH oW
    """

    def __init__(self, node: IRConv2D):
        if not isinstance(node, IRConv2D):
            raise TypeError(f"Expect IRConv2D")
        super().__init__(node)

    def satisfy(self, idx: int, dim: int, num: int):
        """!
        Dimension split on Conv2D operator:

        N iC H W, oC iC dH dW, oC -> N oC oH oW

        Splittable dimension: N, oC
        Reduce dimension: oC 
        """
        assert all(isinstance(t, int) for t in [idx, dim, num]), "idx, dim and num should be integer"
        node: IRConv2D = self.node
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

    def instantiate(self, idx: int, dim: int, num: int):
        if not self.satisfy(idx, dim, num):
            return False
        node: IRConv2D = self.node
        inputs, weights, bias = list(), list(), list()
        outputs = list()
        # split N
        if (idx, dim) == (0, 0):
            inputs = node.inputs(0).split_dim(dim, num)
            weights = [node.inputs(1)] * num
            bias = [node.inputs(2)] * num
            outputs = node.outputs(0).split_dim(dim, num)
        # split oC
        if (idx, dim) == (1, 0):
            inputs = [node.inputs(0)] * num
            weights = node.inputs(1).split_dim(dim, num)
            if node.inputs(2) is None:
                bias = [None] * num
            else:
                bias = node.inputs(2).split_dim(dim, num)
            outputs = node.outputs(0).split_dim(dim=1, num=num)
        # split iC
        if (idx, dim) == (0, 1) or (idx, dim) == (1, 1):
            inputs = node.inputs(0).split_dim(dim, num)
            weights = node.inputs(1).split_dim(dim, num)
            if node.inputs(2) is None:
                bias = [None] * num
            else:
                bias = node.inputs(2).split_val(num)
            outputs = node.outputs(0).split_val(num)
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

    def satisfy(self, idx: int, dim: int, num: int):
        assert all(isinstance(t, int) for t in [idx, dim, num]), "idx, dim and num should be integer"
        node: IRConv2D = self.node
        oH, oW = node.outputs(0).shape[2:]
        stride = node.kwargs['stride']
        dilation = node.kwargs['dilation']
        if dim not in [2, 3]:
            return False
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

    def instantiate(self, idx: int, dim: int, num: int):
        if not self.satisfy(idx, dim, num):
            return None
        node: IRConv2D = self.node
        H, W = node.inputs(0).shape[2:]
        dH, dW = node.inputs(1).shape[2:]
        oH, oW = node.outputs(0).shape[2:]
        groups = node.kwargs['groups']
        stride = node.kwargs['stride']
        padding = node.kwargs['padding']
        dilation = node.kwargs['dilation']
        # split H
        if (idx, dim) == (0, 2):
            # input and padding
            indmap = list()
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
                indmap.append((max(0, start), min(H, stop)))
                start = stop - dilation[0] * (dH - 1)
                # start = 0 if cid == 0 else 1023
                # stop = 1025 if cid == 0 else H
            inputs = _split_axis_custom(node.inputs(0), dim=dim, chunks=tuple(indmap))
            # weight
            weights = [node.inputs(1)] * num
            # bias
            bias = [node.inputs(2)] * num
            # outputs
            outputs = node.outputs(0).split_dim(dim, num)
        # split W
        if (idx, dim) == (0, 3):
            # input and padding
            indmap = list()
            pads = list()
            start = 0 - padding[2]
            for cid in range(num):
                # padding
                padt = padding[2] if cid == 0 else 0
                padb = padding[3] if cid == num - 1 else 0
                pads.append([padding[0], padding[1], padt, padb])
                # input  -- FIXME: only work for stride=[1,1]
                chunkH = oW // num + dilation[0] * (dH - 1)
                stop = start + chunkH - padb
                indmap.append((max(0, start), min(H, stop)))
                start = stop - dilation[0] * (dH - 1)
            inputs = _split_axis_custom(node.inputs(0), dim=dim, chunks=tuple(indmap))
            # weight
            weights = [node.inputs(1)] * num
            # bias
            bias = [node.inputs(2)] * num
            # outputs
            outputs = node.outputs(0).split_dim(dim, num)
        sub_nodes = list()
        for i, w, b, pad, o in zip(inputs, weights, bias, pads, outputs):
            conv = IRConv2D(node.signature, [i, w, b], node.name,
                stride=stride, padding=pad, dilation=dilation, groups=groups)
            conv.set_output(0, o)
            sub_nodes.append(conv)
        return sub_nodes


class HaloSplitConv3D(GenericDistAlgo):
    """
    Halo-exchange split

    N iC D H W, oC iC dH dW, oC -> N oC oD oH oW
    (dim-N is optional)
    """

    def __init__(self, node: IRConv3D):
        if not isinstance(node, IRConv3D):
            raise TypeError(f"Expect IRConv3D")
        super().__init__(node)

    def satisfy(self, idx: int, dim: int, num: int):
        assert all(isinstance(t, int) for t in [idx, dim, num]), "idx, dim and num should be integer"
        node: IRConv3D = self.node
        oD, oH, oW = node.outputs(0).shape[2:]
        stride = node.kwargs['stride']
        dilation = node.kwargs['dilation']
        if dim not in [2, 3]:
            return False
        # FIXME: stride
        if stride != [1, 1, 1]:
            raise NotImplementedError("Splitting on stride != [1,1] is not supported")
        if dilation != [1, 1, 1]:
            raise NotImplementedError("Splitting on dilation != [1,1] is not supported")
        # split H
        if (idx, dim) == (0, 2):
            return oH >= num
            # return oH % num == 0
        # split W
        if (idx, dim) == (0, 3):
            return oW >= num
            # return oW % num == 0

    def instantiate(self, idx: int, dim: int, num: int):
        if not self.satisfy(idx, dim, num):
            return None
        node: IRConv3D = self.node
        D, H, W = node.inputs(0).shape[2:]
        dD, dH, dW = node.inputs(1).shape[2:]
        oD, oH, oW = node.outputs(0).shape[2:]
        groups = node.kwargs['groups']
        stride = node.kwargs['stride']
        padding = node.kwargs['padding']
        dilation = node.kwargs['dilation']
        # split H
        if (idx, dim) == (0, 2):
            # input and padding
            indmap = list()
            pads = list()
            start = 0 - padding[0]
            addone_num = oH % num
            for cid in range(num):
                padl = padding[1] if cid == 0 else 0
                padr = padding[1] if cid == num - 1 else 0
                # padding  -- FIXME: padding here is not correct, only work for pad=[0,..,0]
                pads.append([padding[0], padl, padr, padding[2], padding[2]])
                # input  -- FIXME: only work for stride=[1,1]
                chunkH = oH // num + dilation[0] * (dH - 1)
                addone = int(cid < addone_num)
                stop = start + chunkH - padr + addone
                # stop = start + chunkH - padr
                indmap.append((max(0, start), min(H, stop)))
                start = stop - dilation[0] * (dH - 1)
                # start = 0 if cid == 0 else 1023
                # stop = 1025 if cid == 0 else H
            inputs = _split_axis_custom(node.inputs(0), dim=dim+1, chunks=indmap)
            # weight
            weights = [node.inputs(1)] * num
            # bias
            bias = [node.inputs(2)] * num
            # outputs
            outputs = node.outputs(0).split_dim(dim+1, num)
        # split W
        if (idx, dim) == (0, 3):
            # input and padding
            indmap = list()
            pads = list()
            start = 0 - padding[2]
            addone_num = oW % num
            for cid in range(num):
                # padding
                padt = padding[2] if cid == 0 else 0
                padb = padding[2] if cid == num - 1 else 0
                # padding  -- FIXME: padding here is not correct, only work for pad=[0,..,0]
                pads.append([padding[0], padding[1], padding[1], padt, padb])
                # input  -- FIXME: only work for stride=[1,1]
                chunkH = oW // num + dilation[0] * (dW - 1)
                addone = int(cid < addone_num)
                stop = start + chunkH - padb + addone
                indmap.append((max(0, start), min(W, stop)))
                start = stop - dilation[0] * (dW - 1)
            inputs = _split_axis_custom(node.inputs(0), dim=dim+1, chunks=indmap)
            # weight
            weights = [node.inputs(1)] * num
            # bias
            bias = [node.inputs(2)] * num
            # outputs
            outputs = node.outputs(0).split_dim(dim+1, num)
        sub_nodes = list()
        for i, w, b, pad, o in zip(inputs, weights, bias, pads, outputs):
            conv = IRConv3D(node.signature, [i, w, b], node.name,
                stride=stride, padding=pad, dilation=dilation, groups=groups)
            conv.set_output(0, o)
            sub_nodes.append(conv)
        return sub_nodes
