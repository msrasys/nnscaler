from typing import Tuple, List
import copy

from cube.graph.graph import IRGraph
from cube.ir.dtype import IRDType
from cube.ir.tensor import IRSubTensor
from cube.ir.operator import IRFwOperation


class DTypeInferRule:
    """
    According to promotion doc:
    https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc

    complex > floating > integral > boolean
    """
    @staticmethod
    def infer(node, dtypes: List[IRDType]) -> IRDType:
        dtypes = [dtype for dtype in dtypes if dtype != IRDType.unknown]
        if IRDType.unknown in dtypes:
            raise RuntimeError(f"Find an unkown dtype")
        if IRDType.float32 in dtypes and IRDType.float16 in dtypes:
            raise RuntimeError(f"Find node has both fp32 and fp16 inputs {node}")
        # in priority: fp32 > fp16 > bool > int64 > int16 >
        priority = [
            IRDType.float64, IRDType.float32, IRDType.float16,
            IRDType.int64, IRDType.int32, IRDType.int16, IRDType.int8,
            IRDType.boolean
        ]
        for dtype in priority:
            if dtype in dtypes:
                return dtype
        return IRDType.unknown


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
    for itensor in itensors:
        del graph._full_tensors[itensor.parent.tid]

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
