"""
Utilities for gradient modification
"""
from typing import Dict, List
from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.ir.operator import IRFwOperation
from cube.ir.tensor import IRFullTensor, IRSubTensor, ValueMap



def convert_add_to_valmap(graph: IRGraph, add_node: IRFwOperation):
    """
    Remove add node by replacing with tensor valmap

    @param graph IRGraph: the program
    @param add_node IRFwOperation: the add forward operation
    """
    assert add_node.name == 'add'
    ptensors, producers = [], []
    for itensor in add_node.inputs():
        iptensors = graph.ptensors(itensor.parent)
        assert len(set(t.valmap for t in iptensors)) == len(iptensors)
        ptensors += iptensors
        producers += graph.producers(itensor.parent)
    ftensor = add_node.output(0).parent
    for idx, (ptensor, producer) in enumerate(zip(ptensors, producers)):
        fidx = producer.outputs().index(ptensor)
        bidx = producer.mirror.inputs().index(ptensor.grad)
        ptensor = ftensor.select(ptensor.indmap, (idx, len(producers)))
        ptensor.grad = ftensor.grad.select(ptensor.indmap, (0,1))
        with graph.update(producer):
            producer.set_output(fidx, ptensor)
        with graph.mirror.update(producer.mirror) as bnode:
            bnode.set_input(bidx, ptensor.grad)
    graph.remove(add_node)
    graph.mirror.remove(add_node.mirror)


def flatten_grad(graph: IRSegment, ftensor: IRFullTensor):
    """
    Reset gradient for consumers that are different (no replica)
    Gradient valuemap will be flatten iter-devices, e.g.,(0,3), (1,3), (2,3)
    Gradient valuemap will be exponent intra-devices, e.g., (0,2), (2,4), (3,4)

    @param graph IRGraph: the graph
    @param ftensor IRFullTensor: the fulltensor

    @return None: this is an inplacement update.
    """
    if not isinstance(ftensor.grad, IRFullTensor): return
    
    grads = [t.grad for t in graph.ctensors(ftensor)]
    # require each consumer is a different operator (no replica)
    if len(set(grads)) != len(grads): return
    
    # group consumers by same tensor and same device
    devtensors : Dict[IRSubTensor, Dict[int, List[IRFwOperation]]] = dict()
    for ctensor in graph.ctensors(ftensor):
        devtensors[ctensor] = dict()
    for ctensor in graph.ctensors(ftensor):
        if len(ctensor.device) > 1: return
        devtensors[ctensor][ctensor.device[0]] = []
    for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
        devid = ctensor.device[0]
        devtensors[ctensor][devid].append(consumer)
    
    # setup gradient
    for ctensor in devtensors:
        nchunks = len(devtensors[ctensor])
        for vid, consumers in enumerate(devtensors[ctensor].values()):
            curr_valmap = ValueMap((vid, nchunks))
            for cidx, consumer in enumerate(consumers):
                valmap = curr_valmap.map((0, 2)) if cidx != len(consumers) - 1 else curr_valmap
                grad = ftensor.grad.select(ctensor.indmap, valmap)
                # update consumer and its mirror node
                fidx = consumer.inputs().index(ctensor)
                assert consumer.mirror is not None, consumer
                bidx = consumer.mirror.outputs().index(consumer.input(fidx).grad)
                consumer.input(fidx).grad = grad
                with graph.mirror.update(consumer.mirror) as bnode:
                    bnode.set_output(bidx, grad)
                # update current valmap
                curr_valmap = curr_valmap.map((1, 2)) if cidx != len(consumers) - 1 else curr_valmap
