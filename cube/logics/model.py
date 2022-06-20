from typing import Any, List
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


def forward(graph: IRGraph, *args) -> IRGraph:
    """
    Forward the IRGraph, replacing all the intermediate tensors
    """
    if not isinstance(graph, IRGraph):
        raise TypeError("Requires IRGraph for forward")
    # align graph with input tensors
    itensors: List[IRSubTensor] = graph.inputs()
    for idx, (itensor, arg) in enumerate(zip(itensors, args)):
        graph.set_input(idx, arg)
        for producer in copy.copy(itensor.parent.producers):
            pidx = graph.detach(producer)
            while itensor in producer.outputs():
                oidx = producer.outputs().index(itensor)
                producer.set_output(oidx, arg)
            graph.attach(producer, pidx)
        for consumer in copy.copy(itensor.parent.consumers):
            cidx = graph.detach(consumer)
            while itensor in consumer.inputs():
                iidx = consumer.inputs().index(itensor)
                consumer.set_input(iidx, arg)
            graph.attach(consumer, cidx)
        while itensor in graph.outputs():
            oidx = graph.outputs().index(itensor)
            graph.set_output(oidx, arg)
    # setup gradient accum
    for ftensor in graph.full_tensors():
        naccum = len(ftensor.ctensors)
        for idx, ctensor in enumerate(ftensor.ctensors):
            ctensor.grad_accum = (idx, naccum)
        # actually producer doesn't need to know accumulation
        naccum = len(ftensor.producers)
        for idx, ptensor in enumerate(ftensor.ptensors):
            ptensor.grad_accum = (idx, naccum)
    # generate backward reverse is only to make op id looks consecutive
    for fnode in [n for n in graph.nodes() if isinstance(n, IRFwOperation)][::-1]:
        fnode.gen_backward()
    return graph
