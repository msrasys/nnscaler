from typing import Any, List
import copy

from cube.graph.graph import IRGraph
from cube.ir.tensor import IRSubTensor
from cube.ir.operator import IRFwOperation

from cube.ir.cten import IRTensor


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
    # generate backward reverse is only to make op id looks consecutive
    for fnode in [n for n in graph.nodes() if isinstance(n, IRFwOperation)][::-1]:
        fnode.gen_backward()
    return graph
