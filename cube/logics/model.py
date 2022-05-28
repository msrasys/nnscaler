from typing import Any
import copy

from cube.graph.graph import IRGraph
from cube.ir.tensor import IRSubTensor
from cube.ir.operator import IRFwOperation

from cube.ir.cten import IRTensor


__all__ = ['forward']


class _TensorGener:

    def __init__(self):
        self.symbol = dict()

    def renew(self, val: Any, keep_param=True):
        self._check_is_sub_tensor(val)
        if not isinstance(val, IRTensor):
            return val
        if keep_param and val.is_param():
            return val
        if val.parent._id not in self.symbol:
            self.symbol[val.parent._id] = val.parent.like()
        new_val = self.symbol[val.parent._id].select(
            indmap=val.indmap,
            valmap=val.valmap,
            shape=val.shape
        )
        return new_val

    def set_map(self, origin: Any, new: Any):
        self._check_is_sub_tensor(origin)
        self._check_is_sub_tensor(new)
        if isinstance(origin, IRSubTensor):
            tid = origin.parent._id
            if isinstance(new, IRSubTensor):
                self.symbol[tid] = new.parent
                return
        self.symbol[tid] = new

    def _check_is_sub_tensor(self, tensor):
        if isinstance(tensor, IRTensor):
            if not isinstance(tensor, IRSubTensor):
                raise TypeError("Tensor only allows to be SubTensor")


def forward(graph, *args) -> IRGraph:
    """
    Forward the IRGraph, replacing all the intermediate tensors
    """
    if not isinstance(graph, IRGraph):
        raise TypeError("Forwarding requires IRGraph")
    
    gener = _TensorGener()

    for input, arg in zip(graph.inputs(), args):
        gener.set_map(input, arg)

    fnodes = list()

    # generate forward nodes
    for fnode in graph.nodes():
        fidx = graph.detach(fnode)
        inputs = fnode.inputs()
        outputs = fnode.outputs()
        # fnode = copy.copy(fnode)
        fnode : IRFwOperation = fnode
        fnode._inputs = inputs
        fnode._outputs = outputs
        # set forward inputs
        for idx, val in enumerate(inputs):
            fnode.set_input(idx, gener.renew(val))
        # set forward outputs
        for idx, val in enumerate(outputs):
            fnode.set_output(idx, gener.renew(val))
        graph.attach(fnode, fidx)
        fnodes.append(fnode)

    # reverse is only to make op id looks consecutive
    for fnode in graph.nodes()[::-1]:
        fnode.gen_backward()
    
    inputs = [gener.renew(input) for input in graph.inputs()]
    outputs = [gener.renew(output) for output in graph.outputs()]

    for idx, input in enumerate(inputs):
        graph.set_input(idx, input)
    for idx, output in enumerate(outputs):
        graph.set_output(idx, output)

    # fgraph = IRGraph(fnodes, inputs, outputs, graph.name)
    return graph
