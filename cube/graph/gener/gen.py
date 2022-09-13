import itertools
from typing import Dict, List, Optional, Tuple, Dict

from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.gener.concurrent import ConcurrentGener
from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment

from cube.ir.cten import IRCell
from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.ir.operator import IRFwOperation

from cube.ir.adapter import IRAdapter, IRWeightReducer
from cube.graph.function.function import Add, Cat, Identity, MultiRef


class DummyInputOuput(IRFwOperation):

    def __init__(self, tensor: IRSubTensor, device: int, 
                 is_input=False, is_output=False,
                 name='dummy'):
        super().__init__(name, name,
            1 if is_input else 0,
            1 if is_output else 0
        )
        assert (is_input and not is_output) or (is_output and not is_input)
        if is_input:
            self.set_input(0, tensor)
        if is_output:
            self.set_output(0, tensor)
        self.device = device


def create_dummy(segment: IRSegment) -> List[IRFwOperation]:
    """
    Create dummy operators that 
    1) produce segment input tensors
    2) consume segment output tensors

    @param segment IRSegment: the target segment
    
    @return nodes List[IRCell]: the generated operation
    """
    devices = segment.device
    fwops = []
    for devid in devices:
        for tensor in segment.inputs():
            if not isinstance(tensor, IRSubTensor): continue
            assert tensor.valmap == (0, 1), f"valmap != (0, 1):\n{segment.extra_repr()}"
            fwop = DummyInputOuput(tensor, devid, is_output=True)
            segment.insert(fwop, 0)
            fwops.append(fwop)
        for tensor in segment.outputs():
            if not isinstance(tensor, IRSubTensor): continue
            assert tensor.valmap == (0, 1), f"valmap != (0, 1):\n{segment.extra_repr()}"
            fwop = DummyInputOuput(tensor, devid, is_input=True)
            segment.insert(fwop, segment.nnodes)
            fwops.append(fwop)
    return fwops


def expand_devices(tensors: List[IRSubTensor], 
                   producer: bool = False, consumer: bool = False) -> List[IRSubTensor]:
    """
    Scatter a tensor if it is on multiple devices. It produces a tensor list where
    each tensor is attached to one device, with tensor itself is replicated.

    @param tensors List[IRSubTensor]: each tensor can be on multiple devices.
    @param producer bool: if the tensor is producer role
    @param consumer bool: if the tensor is consumer role

    @return dtensors List[IRSubTensor]: each tensor is on one device
    """
    dtensors = []
    for tensor in tensors:
        if len(tensor.device) == 1:
            dtensors.append(tensor)
            continue
        for devid in tensor.device:
            if producer:
                fwop = DummyInputOuput(tensor, devid, is_output=True, name=tensor.cell.name)
                dtensors.append(fwop.output(0))
            elif consumer:
                fwop = DummyInputOuput(tensor, devid, is_input=True, name=tensor.cell.name)
                dtensors.append(fwop.input(0))
            else:
                raise ValueError("At least one of producer or consumer")
    return dtensors


class IRAdapterGener:

    @staticmethod
    def gen(graph: IRGraph) -> IRGraph:
        """
        Generate tensor adapter for both activations and weights
        Note weight reducers are always append to the last.

        @param graph IRGraph: the graph without adapter
        @return graph IRGraph: the graph with adapter inserted
        """
        # remove anchor node
        graph = IRAdapterGener.remove_anchor(graph)
        # update the gradient before generate adapter
        graph = IRAdapterGener.update_grad(graph)
        # generate adapters for activation
        graph = IRAdapterGener.gen_activation(graph)
        # generate weight reducer
        graph = IRAdapterGener.gen_weight(graph)
        # fuse consecutive non-differentiable adapters into one
        graph = IRAdapterGener.fusion(graph)
        # print(graph.extra_repr())
        return graph

    @staticmethod
    def update_grad(graph: IRSegment):
        for ftensor in graph.full_tensors():
            if ftensor.is_grad():
                graph.update_ftensor_bw(ftensor)
        for node in graph.nodes():
            if isinstance(node, IRSegment) and node.isbw():
                IRAdapterGener.update_grad(node)
        return graph

    @staticmethod
    def remove_anchor(graph: IRSegment):
        for anchor in graph.nodes():
            if isinstance(anchor, IRGraphAnchor):
                graph.remove(anchor)
                if anchor.mirror is not None:
                    graph.mirror.remove(anchor.mirror)
            elif isinstance(anchor, IRSegment):
                IRAdapterGener.remove_anchor(anchor)
        return graph

    @staticmethod
    def gen_weight(graph: IRGraph) -> IRGraph:
        """
        Generate gradient accumulation
        
        Only suuport cases that:

        1) each sub-tensor weight is consumed by different node cids (no replica)
        2) If the sub-tensor weight is consumed by same replicated node:
             The consumers can be grouped by node cids and satisfy:
                1. same number of nodes per cid group
                2. same device set or no-overlapping device set per cid group
        """
        # collect subtensor and consumer
        fweights: Dict[IRFullTensor, List[IRSubTensor]] = dict()
        fgrads: Dict[IRFullTensor, List[IRSubTensor]] = dict()
        consumers: Dict[IRFullTensor, List[IRFwOperation]] = dict()
        for fnode in graph.nodes(flatten=True):
            if not isinstance(fnode, IRFwOperation): continue
            assert len(fnode.device) == 1
            for wtensor in fnode.inputs():
                if isinstance(wtensor, IRSubTensor) and wtensor.is_param():
                    if wtensor.grad is None: continue
                    fweight = wtensor.parent
                    if fweight not in fweights:
                        fweights[fweight] = []
                        fgrads[fweight] = []
                        consumers[fweight] = []
                    fweights[fweight].append(wtensor)
                    fgrads[fweight].append(wtensor.grad)
                    consumers[fweight].append(fnode)
        
        # bucketing
        weights: Dict[IRFullTensor, Dict[IRSubTensor, List[int]]] = dict()
        for fweight in fweights.keys():
            cids = set(fnode.cid for fnode in consumers[fweight])
            nl = '\n'
            # case 1: no replica
            if len(cids) == len(consumers[fweight]):
                weights[fweight] = dict()
                for wtensor, consumer in zip(fweights[fweight], consumers[fweight]):
                    if wtensor not in weights[fweight]:
                        weights[fweight][wtensor] = set()
                    weights[fweight][wtensor].add(consumer.device[0])
            # case 2: replica but has same number of replicas and same/no-overlapping devices
            else:
                cid_fnodes = {cid : [n for n in consumers[fweight] if n.cid == cid] for cid in cids}
                cid_nnodes = [len(ns) for ns in cid_fnodes.values()]
                # same replica# for each cid
                assert all(cid_nnodes[0] == ns for ns in cid_nnodes), (
                    f"If one of the weight consumers are replicated, "
                    f"other same-weight consumers should also replicated in same way."
                    f"FullTensor Weight: {fweight}\n"
                    f"Consumers:\n{nl.join([repr(node) for node in consumers[fweight]])}"
                )
                cid_devs = {cid: set(n.device[0] for n in consumers[fweight]) for cid in cids}
                # case 2.1: same device sharing
                first = list(cid_devs.keys())[0]
                if all(cid_devs[first] == devs for devs in cid_devs.values()):
                    #TODO: need to be more robust
                    continue
                # case 2.2: no-overlapping device sharing
                all_devs = set()
                for devs in cid_devs.values():
                    all_devs.update(devs)
                if sum(len(devs) for devs in cid_devs.values()) == len(all_devs):
                    raise NotImplementedError(
                        f"Weight is consumed by multiple different operators.\n"
                        f"Replicating different operators on no-overlapping device group is not supported yet.\n"
                        f"FullTensor Weight: {fweight}\n"
                        f"Consumers:\n{nl.join([repr(node) for node in consumers[fweight]])}"
                    )
                else:
                    raise NotImplementedError(
                        f"Weight is consumed by multiple different operators.\n"
                        f"Replicating different operators on partial-overlapping device group is not supported yet.\n"
                        f"FullTensor Weight: {fweight}\n"
                        f"Consumers:\n{nl.join([repr(node) for node in consumers[fweight]])}"
                    )

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
            for ctensor in ctensors:
                if not any(t == ctensor and set(ctensor.device).issubset(set(t.device)) for t in ptensors):
                    return False
            return True

        fdummies = create_dummy(graph)
        bgraph: Optional[IRSegment] = graph.mirror
        bdummies = create_dummy(bgraph) if isinstance(bgraph, IRSegment) else []

        skip_grads = [t.parent for t in graph.inputs() + graph.outputs() if isinstance(t, IRSubTensor)]
        # generate adapter for inter-segments
        # FIXME: assume producers and consumers can run in parallel
        for ftensor in graph.full_tensors():
            # backward will gen in forward
            if ftensor.is_param() or ftensor.is_grad():
                continue

            # optimization: local fusion / multiref on producer / consumer
            ftensor = IRAdapterGener.local_producer_fusion(graph, ftensor)
            IRAdapterGener.local_consumer_multiref(graph, ftensor)

            # print(graph.debug_tensor_map_str(ftensor))
            # print(graph.debug_tensor_map_str(ftensor.grad))

            # producers can be operators and graph inputs
            fproducers, fptensors = graph.producers(ftensor), graph.ptensors(ftensor)
            fptensors = expand_devices(fptensors, producer=True)
            assert all(len(ptensor.device) == 1 for ptensor in fptensors), "Not support for multi-device"
            fconsumers, fctensors = graph.consumers(ftensor), graph.ctensors(ftensor)
            fctensors = expand_devices(fctensors, consumer=True)
            assert all(len(ctensor.device) == 1 for ctensor in fctensors), "Not support for multi-device"

            bproducers, bptensors = [], []
            bconsumers, bctensors = [], []
            if (ftensor not in skip_grads) and isinstance(ftensor.grad, IRFullTensor):
                bproducers, bptensors = bgraph.producers(ftensor.grad), bgraph.ptensors(ftensor.grad)
                bptensors = expand_devices(bptensors, producer=True)
                assert all(len(ptensor.device) == 1 for ptensor in bptensors), (
                    f"Not support for multi-device:\n"
                    f"{[ptensor.device for ptensor in bptensors]}"
                    f"{[ptensor.cell for ptensor in bptensors]}"
                )
                bconsumers, bctensors = bgraph.consumers(ftensor.grad), bgraph.ctensors(ftensor.grad)    
                bctensors = expand_devices(bctensors, consumer=True)
                assert all(len(ctensor.device) == 1 for ctensor in bctensors), "Not support for multi-device"

            if skip(fptensors, fctensors) and skip(bptensors, bctensors):
                continue

            fadapter = ConcurrentGener.gen(fptensors, fctensors, bptensors, bctensors)
            if fadapter is None:
                continue

            if not isinstance(graph, IRGraph):
                if not fadapter.differentiable:
                    raise NotImplementedError(
                        "Require adapter to be differentiable for nested IRAdapter."
                        "Condition to be differentiable: prodcuers have same device set with consumers"
                        f"Failed FullTensor: {ftensor}"
                        f"{graph.debug_tensor_map_str(ftensor)}"
                        f"Failed FullTensor.grad: {ftensor.grad}"
                        f"{bgraph.debug_tensor_map_str(ftensor.grad) if ftensor.grad is not None else None}"
                    )

            badapter: Optional[IRAdapter] = fadapter.mirror

            if (badapter is not None and len(fadapter.prims) == 0 and len(badapter.prims) == 0) or \
               (badapter is None and len(fadapter.prims) == 0):
                continue

            # insert forward adapter
            # graph.insert(fadapter, max(producers) + 1)
            graph.insert(fadapter, min(graph.index(c) for c in fconsumers))

            # insert backward adapter
            if badapter is not None:
                assert isinstance(badapter, IRAdapter)
                assert isinstance(bgraph, IRSegment)
                bproducers = [
                    bgraph.index(consumer.mirror) + 1 for \
                        consumer in graph.consumers(ftensor)
                ]
                bidx = max(bproducers) if len(bproducers) > 0 else 0
                bgraph.insert(badapter, bidx)

        # remove dummy op
        for dummy_op in fdummies:
            graph.remove(dummy_op)
        for dummy_op in bdummies:
            bgraph.remove(dummy_op)

        # generate adapter for each segment
        segments = [seg for seg in graph.nodes() if isinstance(seg, IRSegment) and seg.isfw()]
        for segment in segments:
            IRAdapterGener.gen_activation(segment)

        return graph

    @staticmethod
    def local_producer_fusion(graph: IRSegment, ftensor: IRFullTensor) -> IRFullTensor:
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
            for devid in tensor.device:
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
        min_idx = min(graph.index(consumer) for consumer in graph.consumers(ftensor))
        assert len(graph.ctensors(ftensor)) == len(graph.consumers(ftensor))
        for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
            with graph.update(consumer) as consumer:
                consumer.set_input(
                    consumer.inputs().index(ctensor),
                    new_ftensor.select(ctensor.indmap, ctensor.valmap)
                )

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
            if graph.mirror is not None:
                graph.finsert(node, min_idx)
            else:
                graph.insert(node, min_idx)

        # update backward
        if isinstance(ftensor.grad, IRFullTensor):
            graph.update_ftensor_bw(new_ftensor)
            graph.update_ftensor_bw(ftensor)

        return new_ftensor

    @staticmethod
    def local_consumer_multiref(graph: IRSegment, ftensor: IRFullTensor):
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
            for devid in ctensor.device:
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
                min_fidx = graph.nnodes
                for itensor, consumer in zip(itensors, consumers):
                    with graph.update(consumer) as consumer:
                        idx = consumer.inputs().index(ctensor)
                        consumer.set_input(idx, itensor)

                # insert forward multiref
                min_fidx = min(graph.index(consumer) for consumer in consumers)
                if graph.mirror is not None:
                    graph.finsert(multiref, min_fidx)
                else:
                    graph.insert(multiref, min_fidx)
                multirefs[multiref] = consumers
        
        if len(multirefs) > 0 and isinstance(ftensor.grad, IRFullTensor):
            graph.update_ftensor_bw(ftensor)

    @staticmethod
    def fusion(graph: IRSegment) -> IRSegment:
        """
        Fuse consecutive adapters into one
        """
        fadapters, badapters = [], []
        for adapter in graph.nodes():
            if isinstance(adapter, IRAdapter) and adapter.forward and not adapter.differentiable:
                fadapters.append(adapter)
                if adapter.mirror is not None:
                    badapters.insert(0, adapter.mirror)
            else:
                if len(fadapters) > 1:
                    # insert fused fadapter
                    fused_fadapter = IRAdapter.merge(fadapters)
                    for adapter in fadapters:
                        idx = graph.remove(adapter)
                    graph.insert(fused_fadapter, idx)
                    # insert fused badapter
                    fused_badapter = IRAdapter.merge(badapters) if len(badapters) > 0 else None
                    for adapter in badapters:
                        idx = graph.remove(adapter)
                    if fused_badapter is not None:
                        graph.insert(fused_badapter, idx)
                    IRCell.make_pair(fused_fadapter, fused_badapter)
                fadapters, badapters = [], []

        for segment in graph.nodes():
            if isinstance(segment, IRSegment) and segment.isfw():
                IRAdapterGener.fusion(segment)

        return graph