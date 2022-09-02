import itertools
from typing import Dict, List, Optional, Tuple

from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.gener.concurrent import ConcurrentGener

from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.ir.cten import IRCell
from cube.ir.tensor import IRFullTensor, IRSubTensor

from cube.ir.operator import IRBpOperation, IRFwOperation

from cube.ir.adapter import IRAdapter, IRWeightReducer
from cube.graph.function.function import Add, Cat, Identity, MultiRef


class IRAdapterGener:

    @staticmethod
    def gen(graph: IRGraph) -> IRGraph:
        """
        Generate tensor adapter for both activations and weights
        Note weight reducers are always append to the last.

        @param graph IRGraph: the graph without adapter
        @return graph IRGraph: the graph with adapter inserted
        """
        # insert identity operator for graph output
        devs = set()
        for node in graph.nodes():
            devs.update(node.device)
        outputs = [otensor for otensor in graph.outputs() if isinstance(otensor, IRSubTensor)]
        all_identities = []
        for otensor in outputs:
            identity = Identity('', [otensor])
            identity.set_output(0, identity.output(0).tosub())
            graph.insert(identity, len(graph.nodes()))
            identites = graph.replicate(identity, times=len(devs))
            all_identities += identites
            for devid, identity in zip(devs, identites):
                graph.assign(identity, devid)
        # update the gradient before generate adapter
        for node in graph.nodes():
            if isinstance(node, IRBpOperation):
                with graph.update(node):
                    node.update()
        # generate adapters for activation
        graph = IRAdapterGener.gen_activation(graph)
        # generate weight reducer
        graph = IRAdapterGener.gen_weight(graph)
        # remove inserted identity
        for identity in all_identities:
            graph.remove(identity)
        # remove anchor node
        IRAdapterGener.remove_anchor(graph)
        print(graph.extra_repr())
        return graph

    @staticmethod
    def remove_anchor(graph: IRSegment):
        for node in graph.nodes():
            if isinstance(node, IRGraphAnchor):
                graph.remove(node)
                if node.mirror is not None:
                    graph.remove(node.mirror)
            if isinstance(node, IRSegment):
                for anchor in node.nodes():
                    if isinstance(anchor, IRGraphAnchor):
                        graph.remove(anchor)
                        if anchor.mirror is not None:
                            graph.remove(anchor.mirror)

    @staticmethod
    def gen_weight(graph: IRGraph) -> IRGraph:
        # step 1: get weight and gradient
        # weights: Dict[weight_id: int, IRSubTensor]
        # grads  : Dict[weight_id: int, Dict[device: int, List[grad: IRSubTensor]]]
        grads = dict()
        weights = dict()
        for fnode in graph.flatten():
            if not isinstance(fnode, IRFwOperation):
                continue
            devid = fnode.device[0]
            for wtensor in fnode.inputs():
                if isinstance(wtensor, IRSubTensor) and wtensor.is_param():
                    grad: Optional[IRSubTensor] = wtensor.grad
                    if grad is None: continue
                    # nothing to sync
                    if grad.valmap == (0, 1):
                        continue
                    if wtensor._id not in grads:
                        grads[wtensor._id] = dict()
                        weights[wtensor._id] = wtensor
                    if devid not in grads[wtensor._id]:
                        grads[wtensor._id][devid] = list()
                    if grad in grads[wtensor._id][devid]:
                        raise RuntimeError(
                            "Find two same gradient (not expected). "
                            "This is usually due to replicated node assigned to same device. "
                            f"\nCheck node:\n\t{fnode}"
                        )
                    grads[wtensor._id][devid].append(grad)
        # step 2: generate reducers.
        # reducers: tuple(ranks): List[weight]
        reducers: Dict[Tuple[int], List[IRSubTensor]] = dict()
        for wid in grads:
            ranks = list(grads[wid].keys())
            ranks.sort()
            ranks = tuple(ranks)  # ranks are used for group
            if len(ranks) == 1:
                continue
            if ranks not in reducers:
                reducers[ranks] = list()
            reducers[ranks].append(weights[wid])
        # generate reducer for each rank
        for ranks in reducers:
            weights = reducers[ranks]
            opt_op = IRWeightReducer(weights)
            opt_op.device = list(ranks)
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
        segments = [seg for seg in graph.nodes() if isinstance(seg, IRSegment)]

        def skip(ptensors: List[IRSubTensor], ctensors: List[IRSubTensor]) -> bool:
            # e.g., loss or parameter/buffer
            if len(ptensors) == 0 or len(ctensors) == 0:
                return True
            # direct connection
            if len(ptensors) == 1 and len(ctensors) == 1 and \
                set(ptensors[0].device) == set(ctensors[0].device):
               return True
            return False

        def filter(nodes: List[IRCell], tensors: List[IRSubTensor]) -> Tuple[IRCell, IRSubTensor]:
            assert len(nodes) == len(tensors)
            filter_nodes, filter_tensors = [], []
            for node, tensor in zip(nodes, tensors):
                if node in graph.nodes():
                    filter_nodes.append(node)
                    filter_tensors.append(tensor)
            return filter_nodes, filter_tensors

        # generate adapter for inter-segments
        # FIXME: assume producers and consumers can run in parallel
        for ftensor in graph.full_tensors():
            # backward will gen in forward
            if ftensor.is_param() or ftensor.is_grad():
                continue

            # optimization: local fusion / multiref on producer / consumer
            if isinstance(graph, IRGraph) and graph.train:
                ftensor = IRAdapterGener.local_producer_fusion(graph, ftensor)
                IRAdapterGener.local_consumer_multiref(graph, ftensor)

            # producers can be operators and graph inputs
            producers, ptensors = filter(ftensor.producers, ftensor.ptensors)
            for itensor in graph.inputs():
                if isinstance(itensor, IRSubTensor):
                    if itensor.parent == ftensor:
                        ptensors.append(itensor)
            # consumers can be operators and graph outputs
            consumers, ctensors = filter(ftensor.consumers, ftensor.ctensors)
            for otensor in graph.outputs():
                if isinstance(otensor, IRSubTensor):
                    if otensor.parent == ftensor:
                        ctensors.append(otensor)

            if skip(ptensors, ctensors): continue
            
            fadapter = ConcurrentGener.gen(ptensors, ctensors)
            if fadapter is None:
                continue

            # insert forward adapter
            # fidx = max(graph.index(prod).gidx for prod in producers)
            fidx = min(graph.index(cons) for cons in consumers)
            graph.insert(fadapter, fidx)

            # insert backward adapter
            if fadapter.mirror is not None:
                bsegment = graph if isinstance(graph, IRGraph) else graph.mirror
                # bidx = max(graph.index(cons.mirror) for cons in consumers if cons.mirror is not None)
                bidx = min(bsegment.index(prod.mirror) for prod in producers if prod.mirror is not None)
                bsegment.insert(fadapter.mirror, bidx)

        # generate adapter for each segment
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

        for tensor in ftensor.ptensors:
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
        rcid = set(producer.recompute for producer in ftensor.producers)
        rcid = list(rcid)[0] if len(rcid) == 1 else None
        for node in nodes:
            node.recompute = rcid

        new_ftensor = ftensor.like()

        # update consumer 
        assert len(ftensor.ctensors) == len(ftensor.consumers)
        for ctensor, consumer in zip(ftensor.ctensors, ftensor.consumers):
            # TODO: the change can happend inside segment
            with graph.update(consumer) as consumer:
                consumer.set_input(
                    consumer.inputs().index(ctensor),
                    new_ftensor.select(ctensor.indmap, ctensor.valmap)
                )
        min_idx = min(graph.nodes().index(consumer) for consumer in ftensor.consumers)

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

        fsid = max(graph.stage_id(prod) for prod in ftensor.producers)
        for node in nodes[::-1]:
            # print(node)
            assert node not in graph.nodes()
            assert len(node.outputs()) == 1
            graph.attach(node, min_idx, stage_idx=fsid)

        # insert and update backward node
        if graph.train:
            # update backward node
            for consumer in new_ftensor.consumers:
                assert isinstance(consumer.mirror, IRBpOperation)
                bnode = consumer.mirror
                with graph.update(bnode) as bnode:
                    bnode.update()
            # insert backward node
            bnodes = [node.gen_backward() for node in nodes]
            bidx = min(graph.nodes().index(producer.mirror) for producer in ftensor.producers)
            for bnode in bnodes:
                bnode.device = bnode.mirror.device
                graph.attach(bnode, bidx)

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
        for ctensor, consumer in zip(ftensor.ctensors, ftensor.consumers):
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
                f"Producers:\n{nl.join(repr(node) for node in ftensor.producers)}\n"
                f"Consumers:\n{nl.join(repr(node) for node in ftensor.consumers)}"
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
