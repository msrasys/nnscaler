from cube.algorithm.utils import split_axis, split_axis_custom
from cube.algorithm.generics import GenericDistAlgo

from cube.graph.function.pad import IRPad

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
        if dim < len(node.inputs(0).shape) - pad_dim_count:
            return node.inputs(0).shape[dim] % num == 0
        # split pad dim
        else:
            dim_in_pad = len(node.inputs(0).shape) - 1 - dim
            return (node.inputs(0).shape[dim] + pad[dim_in_pad * 2] + pad[dim_in_pad * 2 + 1]) % num == 0

    def instantiate(self, dim: int, num: int):
        if not self.satisfy(dim, num):
            return False
        node: IRPad = self.node
        pad = node.kwargs['pad']
        mode = node.kwargs['mode']
        value = node.kwargs['value']
        pad_dim_count = len(pad) / 2

        inputs = list()
        outputs = list()
        subnodes = list()

        # split non-pad dim
        if dim < len(node.inputs(0).shape) - pad_dim_count:
            inputs = split_axis(node.inputs(0), axis=dim, chunk_num=num)
            outputs = split_axis(node.outputs(0), axis=dim, chunk_num=num)
            for i, o in zip(inputs, outputs):
                subnodes.append(node.new([i], [o]))
        else: # split pad dim
            inputs = split_axis(node.inputs(0), axis=dim, chunk_num=num)
            slicers = list()
            pads = list()
            dim_in_pad = len(node.inputs(0).shape) - 1 - dim
            global_padl = pad[dim_in_pad * 2]
            global_padr = pad[dim_in_pad * 2 + 1]
            chunk_size = (node.outputs(0).shape[dim] - global_padl - global_padr) // num
            start = 0
            for cid in range(num):
                padl = global_padl if cid == 0 else 0
                padr = global_padr if cid == num-1 else 0

                cur_pad = pad.copy()
                cur_pad[dim_in_pad * 2] = padl
                cur_pad[dim_in_pad * 2 + 1] = padr
                pads.append(cur_pad)

                stop = start + padl + padr + chunk_size
                slicers.append(slice(max(0, start), min(node.outputs(0).shape[dim], stop)))
                start = stop

            outputs = split_axis_custom(node.outputs(0), axis=dim, chunks=slicers)

            for i, o, p in zip(inputs, outputs, pads):
                subnodes.append(node.new([i], [o], pad=p))

        return subnodes