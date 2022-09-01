from typing import Tuple, List
import copy

from cube.graph.graph import IRGraph
from cube.ir.dtype import IRDType, DTypeInferRule
from cube.ir.tensor import IRSubTensor
from cube.ir.operator import IRFwOperation


def forward(graph: IRGraph, *args) -> IRGraph:
    """
    Forward the IRGraph, replacing all the intermediate tensors
    """
    if not isinstance(graph, IRGraph):
        raise TypeError("Requires IRGraph for forward")

    # align graph with input tensors
    itensors: Tuple[IRSubTensor, ...] = graph.inputs()
    for idx, (itensor, arg) in enumerate(zip(itensors, args)):
        graph.set_input(idx, arg)
        for producer in copy.copy(itensor.parent.producers):
            with graph.update(producer):
                while itensor in producer.outputs():
                    oidx = producer.outputs().index(itensor)
                    producer.set_output(oidx, arg)
        for consumer in copy.copy(itensor.parent.consumers):
            with graph.update(consumer):
                while itensor in consumer.inputs():
                    iidx = consumer.inputs().index(itensor)
                    consumer.set_input(iidx, arg)
        while itensor in graph.outputs():
            oidx = graph.outputs().index(itensor)
            graph.set_output(oidx, arg)

    # dtype inference
    for node in graph.nodes():
        itensors = [t for t in node.inputs() if isinstance(t, IRSubTensor)]
        # setup gradient
        for itensor in itensors:
            if itensor.parent.grad is not None:
                itensor.parent.dtype = itensor.dtype
        if len(itensors) == 0:
            continue
        odtype = DTypeInferRule.infer(node, [t.dtype for t in itensors])
        assert odtype != IRDType.unknown, f"{node} : {[t.dtype for t in itensors]}"
        otensors = [t for t in node.outputs() if isinstance(t, IRSubTensor)]
        for tensor in otensors:
            tensor.dtype = odtype
            # setup graidient
            if tensor.parent.grad is not None:
                tensor.parent.grad.dtype = odtype

    # generate backward reverse is only to make op id looks consecutive
    for fnode in [n for n in graph.nodes() if isinstance(n, IRFwOperation)][::-1]:
        fnode.gen_backward()
    return graph
