from typing import Any
import copy

from cube.graph.graph import IRGraph
from cube.graph.tensor import IRSubTensor
from cube.graph.operator import IRBpOperation

from cube.ir.cten import IRTensor


__all__ = ['forward', 'backward']


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
            indices=val.indices,
            val_map=val.val_map,
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
    bnodes = list()
    for node in graph.nodes():
        inputs = node.inputs()
        outputs = node.outputs()

        # forwrd node
        fnode = copy.copy(node)
        # set forward inputs
        for idx, val in enumerate(inputs):
            fnode.set_input(idx, gener.renew(val))
        # set forward outputs
        for idx, val in enumerate(outputs):
            fnode.set_output(idx, gener.renew(val))
        
        # backward node
        bnode = IRBpOperation(data_num=len(inputs), grad_num=len(outputs))
        # set backward grad
        for idx, val in enumerate(fnode.inputs()):
            # set input
            bnode.set_data(idx, val)
            # set gradient output
            val = val if isinstance(val, IRTensor) else None
            grad = gener.renew(val, keep_param=False)
            grad = grad.as_grad() if isinstance(grad, IRTensor) else grad
            if isinstance(val, IRTensor) and val.requires_grad:
                val.grad = grad
            bnode.set_output(idx, grad)
        for idx, val in enumerate(fnode.outputs()):
            # set gradient input
            grad = gener.renew(val, keep_param=False)
            grad = grad.as_grad() if isinstance(grad, IRTensor) else grad
            if isinstance(val, IRTensor) and val.requires_grad:
                val.grad = grad
            bnode.set_grad(idx, grad)

        fnode.device = node.device
        bnode.device = node.device
        
        # mirror node for forward / backward
        fnode.mirror = bnode
        bnode.mirror = fnode

        fnodes.append(fnode)
        bnodes.append(bnode)
    
    inputs = [gener.renew(input) for input in graph.inputs()]
    outputs = [gener.renew(output) for output in graph.outputs()]

    fgraph = IRGraph(fnodes, inputs, outputs, graph.name)
    for output in fgraph.outputs():
        output.set_trace(fgraph.nodes())
    return fgraph

