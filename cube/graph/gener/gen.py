import itertools
from typing import Dict, List, Optional, Tuple, Set
import copy

from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.gener.concurrent import ConcurrentGener

from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.ir.cten import IRCell
from cube.ir.tensor import IRFullTensor, IRSubTensor

from cube.ir.operator import IRBpOperation, IRFwOperation

from cube.ir.adapter import IRAdapter, IRWeightReducer
from cube.graph.function.function import Add, Cat, Identity, MultiRef


def to_device(tensor: IRSubTensor, device: int) -> IRFwOperation:
    """
    This is used for changing tensor device
    """
    fwop = IRFwOperation('dummy', 'dummpy', 1, 0)
    fwop.set_input(0, tensor)
    fwop.device = device
    return fwop.input(0)


class IRAdapterGener:

    @staticmethod
    def gen(graph: IRGraph) -> IRGraph:
        """
        Generate tensor adapter for both activations and weights
        Note weight reducers are always append to the last.

        @param graph IRGraph: the graph without adapter
        @return graph IRGraph: the graph with adapter inserted
        """
        # update the gradient before generate adapter
        for node in graph.nodes():
            if isinstance(node, IRBpOperation):
                graph.update_bwop(node)
        # generate adapters for activation
        graph = IRAdapterGener.gen_activation(graph)
        # generate weight reducer
        graph = IRAdapterGener.gen_weight(graph)
        # remove anchor node
        IRAdapterGener.remove_anchor(graph)
        print(graph.extra_repr())
        return graph

    @staticmethod
    def remove_anchor(graph: IRSegment):
        for anchor in graph.nodes():
            if isinstance(anchor, IRGraphAnchor):
                graph.remove(anchor)
                if anchor.mirror is not None:
                    graph.mirror.remove(anchor.mirror)
            if isinstance(anchor, IRSegment):
                IRAdapterGener.remove_anchor(anchor)

    @staticmethod
    def gen_weight(graph: IRGraph) -> IRGraph:
        weights = dict()
        for fnode in graph.nodes(flatten=True):
            if not isinstance(fnode, IRFwOperation): continue
            assert len(fnode.device) == 1
            for wtensor in fnode.inputs():
                if isinstance(wtensor, IRSubTensor) and wtensor.is_param():
                    if wtensor.grad is None: continue
                    if wtensor.parent not in weights:
                        weights[wtensor.parent] = dict()
                    if wtensor not in weights[wtensor.parent]:
                        weights[wtensor.parent][wtensor] = set()
                    weights[wtensor.parent][wtensor].add(wtensor.device[0])

        reducers: Dict[Tuple[int], List[IRSubTensor]] = dict()
        for ftensor, subtensors in weights.items():
            # TODO: check no overlapping (not same) weights on a device
            for subw in subtensors:
                if len(subtensors[subw]) == 1:
                    continue
                devices = list(subtensors[subw])
                devices.sort()
                devices = tuple(devices)
                if devices not in reducers:
                    reducers[devices] = []
                reducers[devices].append(subw)
        # generate reducer for each rank
        for devices in reducers:
            weights = reducers[devices]
            opt_op = IRWeightReducer(weights)
            opt_op.device = list(devices)
            graph.insert(opt_op, graph.nnodes)
        return graph

    @staticmethod
    def gen_activation(graph: IRSegment) -> IRSegment:
        """!
        Generate adapter for activation tensors.
        The forward/backward adapter is inserted before the first consumers of its full tensor.

        @param graph IRGraph: the graph the requires for adapter.

        @return graph IRGraph: the (inplace) modified graph with activation adapters. 
        """
        def skip(ptensors: List[IRSubTensor], ctensors: List[IRSubTensor]) -> bool:
            # e.g., loss or parameter/buffer
            if len(ptensors) == 0 or len(ctensors) == 0:
                return True
            # direct connection
            if len(ptensors) == 1 and len(ctensors) == 1 and \
                set(ptensors[0].device) == set(ctensors[0].device):
               return True
            return False
        
        devices = graph.device

        # generate adapter for inter-segments
        # FIXME: assume producers and consumers can run in parallel
        for ftensor in graph.full_tensors():
            # backward will gen in forward
            if ftensor.is_param() or ftensor.is_grad():
                continue

            # optimization: local fusion / multiref on producer / consumer
            # ftensor = IRAdapterGener.local_producer_fusion(graph, ftensor)
            # IRAdapterGener.local_consumer_multiref(graph, ftensor)

            # producers can be operators and graph inputs
            fproducers, bproducers, ptensors = [], [], []
            # operators
            for ptensor, producer in zip(graph.ptensors(ftensor), graph.producers(ftensor)):
                for devid in producer.device:
                    ptensors.append(to_device(ptensor, devid))
                fproducers.append(graph.index(producer)[0])
                if ptensor.requires_grad:
                    bproducers.append(graph.mirror.index(producer.mirror)[0])
            # graph inputs
            for ptensor in graph.inputs():
                if isinstance(ptensor, IRSubTensor) and ptensor.parent == ftensor:
                    # TODO: mapping back forawrd / backward
                    ptensor = ftensor.select(ptensor.indmap, (0, 1))
                    for devid in devices:
                        ptensors.append(to_device(ptensor, devid))
                    fproducers.append(0)
                    if ptensor.requires_grad:
                        bproducers.append(graph.mirror.nnodes)

            # consumers can be operators and graph outputs
            fconsumers, bconsumers, ctensors = [], [], []
            # operators
            for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
                for devid in consumer.device:
                    ctensors.append(to_device(ctensor, devid))
                fconsumers.append(graph.index(consumer)[0])
                if ctensor.requires_grad:
                    bconsumers.append(graph.mirror.index(consumer.mirror)[0])
            # graph outputs
            for ctensor in graph.outputs():
                if isinstance(ctensor, IRSubTensor) and ctensor.parent == ftensor:
                    # TODO: mapping back forward / backward
                    ctensor = ftensor.select(ctensor.indmap, (0, 1))
                    for devid in devices:
                        ctensors.append(to_device(ctensor, devid))
                    fconsumers.append(graph.nnodes)
                    if ctensor.requires_grad:
                        bconsumers.append(0)

            if skip(ptensors, ctensors): continue
            
            fadapter = ConcurrentGener.gen(ptensors, ctensors)
            if fadapter is None:
                continue

            # insert forward adapter
            # graph.insert(fadapter, max(producers))
            graph.insert(fadapter, min(fconsumers))

            # insert backward adapter
            if len(bproducers) > 0:
                assert isinstance(fadapter.mirror, IRAdapter)
                assert isinstance(graph.mirror, IRSegment)
                bidx = max(bproducers)
                graph.mirror.insert(fadapter.mirror, bidx)

        # generate adapter for each segment
        segments = [seg for seg in graph.nodes() if isinstance(seg, IRSegment)]
        for segment in segments:
            IRAdapterGener.gen_activation(segment)

        print(graph.extra_repr())
        return graph


    @staticmethod
    def local_producer_fusion(graph: IRGraph, ftensor: IRFullTensor) -> IRFullTensor:
        """!
        Fuse the producer tensors using concat and add.
        This will add a new full tensor by chaging from:
            producer --(ftensor)--> consumer
        to:
            producer --(ftensor)--> fused nodes --(new ftensor)--> consumer

        Recompute policy: if all the producers are recomputed in a same
        recompute group, then the additional generated cat/add are also
        apllied with same recompute region. Otherwise no recompute.

        @param tensors List[IRSubTensor]: tensors to be fused in local device
        
        @return new_ftensor IRFullTensor: the new full tensor.
                                          If cannot fuse, the original ftensor.
        """

        def like(tensor: IRSubTensor, share: Optional[IRFullTensor] = None) -> IRSubTensor:
            parent = tensor.parent.like() if share is None else share
            return parent.select(tensor.indmap, tensor.valmap)

        # collect device tensors
        devtensors: Dict[int, List[IRSubTensor]] = dict()
        # devid: old tensor -> [nodes,]
        fuse_tensors: Dict[int, Dict[IRSubTensor, List[IRSubTensor]]] = dict()
        tensor_map: Dict[int, Dict[IRSubTensor, IRSubTensor]] = dict()

        for tensor in graph.ptensors(ftensor):
            assert len(tensor.device) == 1
            devid = tensor.device[0]
            if devid not in devtensors:
                devtensors[devid] = []
                fuse_tensors[devid] = dict()
                tensor_map[devid] = dict()
            devtensors[devid].append(tensor)
            fuse_tensors[devid][tensor] = [tensor]
            tensor_map[devid][tensor] = tensor

        nodes: List[IRFwOperation] = []
        for devid, tensors in devtensors.items():
            if len(tensors) == 1:
                continue
            
            # repeatly search for combinable tensors
            while True:
                can_merge = False
                out = None
                node = None
                for t1, t2 in itertools.combinations(tensors, 2):
                    catdim = t1.catdim(t2)
                    if catdim is not None:
                        t1, t2 = [t1, t2] if t1.indmap[catdim][0] < t2.indmap[catdim][0] else [t2, t1]
                        out = t1.concat(t2, dim=catdim)
                        node = Cat(
                            'torch.cat',
                            ([tensor_map[devid][t1], tensor_map[devid][t2]], catdim)
                        )
                        can_merge = True
                        break
                    elif t1.accumable(t2):
                        out = t1.accum(t2)
                        node = Add(
                            'torch.add',
                            [tensor_map[devid][t1], tensor_map[devid][t2]]
                        )
                        can_merge = True
                        break
                # each time when creats a merge node, the output will be
                # updated with a new full tensor. The corresponding input
                # will be set according to the previous node output
                if can_merge:
                    tensor_map[devid][out] = like(out)
                    node.set_output(0, tensor_map[devid][out])  # update output to a new full tensor
                    tensors.remove(t1)
                    tensors.remove(t2)
                    tensors.append(out)
                    nodes.append(node)
                    node.device = devid
                    fuse_tensors[devid][out] = fuse_tensors[devid][t1] + fuse_tensors[devid][t2]
                    del fuse_tensors[devid][t1]
                    del fuse_tensors[devid][t2]
                else:
                    break

        if len(nodes) == 0: return ftensor

        # recompute
        rcid = set(producer.recompute for producer in graph.producers(ftensor))
        rcid = list(rcid)[0] if len(rcid) == 1 else None
        for node in nodes:
            node.recompute = rcid

        new_ftensor = ftensor.like()

        # update consumer 
        assert len(graph.ctensors(ftensor)) == len(graph.consumers(ftensor))
        for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
            with graph.update(consumer) as consumer:
                consumer.set_input(
                    consumer.inputs().index(ctensor),
                    new_ftensor.select(ctensor.indmap, ctensor.valmap)
                )
        min_idx = min(graph.index(consumer) for consumer in graph.consumers(ftensor))

        # insert new producer
        for devid, tensors in fuse_tensors.items():
            for ptensor in tensors:
                new_tensor = like(ptensor, share=new_ftensor)
                if len(tensors[ptensor]) == 1:
                    node = Identity('', [ptensor])
                    node.device = devid
                    node.set_output(0, new_tensor)
                    nodes.append(node)
                else:
                    for node in nodes:
                        if node.output(0) == tensor_map[devid][ptensor]:
                            node.set_output(0, new_tensor)

        for node in nodes[::-1]:
            assert node not in graph.nodes()
            assert len(node.outputs()) == 1
            graph.insert(node, min_idx)

        # insert and update backward node
        bgraph: IRSegment = graph.mirror
        # update backward node
        for consumer in graph.consumers(new_ftensor):
            assert isinstance(consumer.mirror, IRBpOperation)
            bnode = consumer.mirror
            bgraph.update_bwop(bnode)
        # insert backward node
        bnodes = [graph.bwop(node) for node in nodes]
        bidx = min(bgraph.index(producer.mirror) for producer in bgraph.producers(ftensor))
        for bnode in bnodes:
            bnode.device = bnode.mirror.device
            bgraph.insert(bnode, bidx)

        return new_ftensor

    @staticmethod
    def local_consumer_multiref(graph: IRGraph, ftensor: IRFullTensor):
        """!
        If a device have a same sub-tensor to be consumed multiple times,
        then create a multiref forward node for it to make
        each sub-tensor to be consumed only once in each device.

        This is to adapt with pytorch autograd function.

        producer -> consumers[0,1]

        producer -> multiref -> consumer[0]
                        |-----> consumer[1]

        @param graph IRGraph
        @param ftensor IRFullTensor: the forward full tensor
        """
        # collect to consumer tensors of each device
        devtensors: Dict[int, Dict[IRSubTensor, List[IRCell]]] = dict()
        for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
            assert len(ctensor.device) == 1
            devid = ctensor.device[0]
            if devid not in devtensors:
                devtensors[devid] = dict()
            if ctensor not in devtensors[devid]:
                devtensors[devid][ctensor] = []
            devtensors[devid][ctensor].append(consumer)
        
        # restrict each device has same subtensor
        nl = '\n'
        for devid in devtensors:
            assert len(devtensors[devid]) <= 1, (
                "Detect that a full tensor is partitioned differently on a device.\n"
                "To achieve this, need manually add multiref operator in model description.\n"
                f"Full Tensor: {ftensor}\n"
                f"Producers:\n{nl.join(repr(node) for node in graph.producers(ftensor))}\n"
                f"Consumers:\n{nl.join(repr(node) for node in graph.consumers(ftensor))}"
            )

        # add multiref forward node
        multirefs: Dict[MultiRef, List[IRFwOperation]] = dict()
        for devid in devtensors:
            for ctensor in devtensors[devid]:
                consumers = devtensors[devid][ctensor]
                if len(consumers) == 1:
                    continue
                multiref = MultiRef(None, [ctensor, len(consumers)])
                multiref.infer_shape()
                multiref.device = devid
                ftensors = [ctensor.parent.like() for _ in range(len(consumers))]
                itensors = [ft.select(ctensor.indmap, ctensor.valmap) for ft in ftensors]
                for idx, itensor in enumerate(itensors):
                    multiref.set_output(idx, itensor)

                # update consumer
                min_fidx = len(graph.nodes())
                for itensor, consumer in zip(itensors, consumers):
                    with graph.update(consumer) as consumer:
                        idx = consumer.inputs().index(ctensor)
                        consumer.set_input(idx, itensor)
                min_fidx = min(graph.nodes().index(consumer) for consumer in consumers)
        
                # insert forward multiref
                graph.attach(multiref, min_fidx)
                multirefs[multiref] = consumers

        # insert / update backward
        if graph.train:
            for multiref, consumers in multirefs.items():
                # update consumer backward
                for consumer in consumers:
                    assert isinstance(consumer.mirror, IRBpOperation)
                    bnode: IRBpOperation = consumer.mirror
                    with graph.update(bnode) as bnode:
                        bnode.update()
                # insert backward
                bnode = multiref.gen_backward()
                bnode.device = multiref.device
                bidx = max(graph.nodes().index(consumer.mirror) for consumer in consumers)
                bsid = graph.stage_id(graph.node(bidx))
                graph.attach(bnode, bidx+1, stage_idx=bsid)
