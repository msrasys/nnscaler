from typing import List, Tuple
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.pad import IRPad
from cube.ir.tensor import IRSubTensor


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


class DimSplitPad(GenericDistAlgo):
    """
    split Pad at dimension level

    """
    def __init__(self, node: IRPad):
        if not isinstance(node, IRPad):
            raise TypeError(f"Expect IRConv2D")
        super().__init__(node)

    def satisfy(self, dim: int, num: int):
        """
        config = dict(idx=int, dim=int, num=num)

        """
        assert all(isinstance(t, int) for t in [dim, num]), "dim and num should be integer"
        node: IRPad = self.node
        pad = node.kwargs['pad']
        mode = node.kwargs['mode']
        value = node.kwargs['value']
        assert len(pad) % 2 == 0
        pad_dim_count = len(pad) / 2

        # split non-pad dim
        if dim < len(node.input(0).shape) - pad_dim_count:
            return node.input(0).shape[dim] >= num
            # return node.input(0).shape[dim] % num == 0
        # split pad dim
        else:
            dim_in_pad = len(node.input(0).shape) - 1 - dim
            return (node.input(0).shape[dim] + pad[dim_in_pad * 2] + pad[dim_in_pad * 2 + 1]) >= num
            # return (node.input(0).shape[dim] + pad[dim_in_pad * 2] + pad[dim_in_pad * 2 + 1]) % num == 0

    def instantiate(self, dim: int, num: int):
        if not self.satisfy(dim, num):
            return None
        node: IRPad = self.node
        pad = node.kwargs['pad']
        mode = node.kwargs['mode']
        value = node.kwargs['value']
        pad_dim_count = len(pad) / 2

        inputs = list()
        outputs = list()
        subnodes = list()

        # split non-pad dim
        if dim < len(node.input(0).shape) - pad_dim_count:
            inputs = node.input(0).split_dim(dim, num)
            outputs = node.output(0).split_dim(dim, num)
            for i, o in zip(inputs, outputs):
                subnodes.append(node.new([i], [o]))
        else: # split pad dim
            inputs = node.input(0).split_dim(dim, num)
            slicers = list()
            pads = list()
            dim_in_pad = len(node.input(0).shape) - 1 - dim
            global_padl = pad[dim_in_pad * 2]
            global_padr = pad[dim_in_pad * 2 + 1]
            chunk_size = (node.output(0).shape[dim] - global_padl - global_padr) // num
            addone_num = (node.output(0).shape[dim] - global_padl - global_padr) % num
            start = 0
            for cid in range(num):
                padl = global_padl if cid == 0 else 0
                padr = global_padr if cid == num-1 else 0

                cur_pad = pad.copy()
                cur_pad[dim_in_pad * 2] = padl
                cur_pad[dim_in_pad * 2 + 1] = padr
                pads.append(cur_pad)

                addone = int(cid < addone_num)
                stop = start + padl + padr + chunk_size + addone
                slicers.append((max(0, start), min(node.output(0).shape[dim], stop)))
                start = stop

            outputs = _split_axis_custom(node.output(0), dim, tuple(slicers))

            for i, o, p in zip(inputs, outputs, pads):
                subnodes.append(node.new([i], [o], pad=p))

        return subnodes