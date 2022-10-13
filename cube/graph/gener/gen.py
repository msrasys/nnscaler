from typing import Dict, List, Optional, Tuple, Dict
import numpy as np
import itertools

from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.gener.concurrent import ConcurrentGener
import cube.graph.gener.utils as utils
from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment

from cube.ir.cten import IRCell
from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.ir.operator import IRFwOperation

from cube.ir.adapter import IRAdapter, IRWeightReducer
from cube.graph.function.function import Accum, Cat, MultiRef


DeviceID = int


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
    Create dummy operators segment inputs and outputs. 
    The backward operator is also inserted.

    1) produce segment input tensors
    2) consume segment output tensors

    @param segment IRSegment: the target segment
    
    @return nodes List[IRCell]: the generated operation
    """
    # devices = segment.device
    fwops = []

    # create inputs
    for tensor in segment.inputs():
        devices = [consumer.device for consumer in segment.consumers(tensor.parent)][::-1]
        if not isinstance(tensor, IRSubTensor): continue
        assert tensor.valmap == (0, 1), f"valmap != (0, 1):\n{segment.extra_repr()}"
        fwop = DummyInputOuput(tensor, 0, is_output=True)
        for devid in devices:
            fop = fwop.replicate()
            fop.device = devid
            if tensor.requires_grad:
                fop.output(0).grad = tensor.grad.select(tensor.indmap, (0, 1))
                segment.finsert(fop, 0)
            else:
                segment.insert(fop, 0)
            fwops.append(fop)
    
    # create outputs
    for tensor in segment.outputs():
        devices = [producer.device for producer in segment.producers(tensor.parent)]
        if not isinstance(tensor, IRSubTensor): continue
        assert tensor.valmap == (0, 1), f"valmap != (0, 1):\n{segment.extra_repr()}"
        fwop = DummyInputOuput(tensor, 0, is_input=True)
        for devid in devices:
            fop = fwop.replicate()
            fop.device = devid
            if tensor.requires_grad and segment.mirror != segment:
                fop.input(0).grad = tensor.grad.select(tensor.indmap, (0, 1))
                segment.finsert(fop, segment.nnodes)
            else:
                segment.insert(fop, segment.nnodes)
            fwops.append(fop)

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
        # generate adapters for activation
        graph = IRAdapterGener.gen_activation(graph)
        # generate weight reducer
        graph = IRAdapterGener.gen_weight(graph)
        # fuse consecutive non-differentiable adapters into one
        graph = IRAdapterGener.fusion(graph)
        # print(graph.extra_repr())
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
    def gen_activation(graph: IRSegment, allow_recompute: bool = True) -> IRSegment:
        """!
        Generate adapter for activation tensors.
        The forward/backward adapter is inserted before the first consumers of its full tensor.

        @param graph IRGraph: the graph the requires for adapter.
        @param allow_recompute bool: Allow adapter recomputes. If this enables, all adapters will be
            set to the same recompute group with its consumed node.

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
        bdummies = [fwop.mirror for fwop in fdummies if fwop.mirror is not None]
        bgraph: Optional[IRSegment] = graph.mirror

        # generate adapter for inter-segments
        # FIXME: assume producers and consumers can run in parallel
        for ftensor in graph.full_tensors():
            # backward will gen in forward
            if ftensor.is_param() or ftensor.is_grad():
                continue

            # flatten gradient
            utils.flatten_grad(graph, ftensor)

            # optimization: local fusion / multiref on producer / consumer
            ftensor = IRAdapterGener.local_producer_fusion(graph, ftensor)
            IRAdapterGener.local_consumer_multiref(graph, ftensor)

            # print(graph.debug_tensor_map_str(ftensor))
            # print(graph.mirror.debug_tensor_map_str(ftensor.grad))

            # producers can be operators and graph inputs
            fproducers, fptensors = graph.producers(ftensor), graph.ptensors(ftensor)
            fptensors = expand_devices(fptensors, producer=True)
            assert all(len(ptensor.device) == 1 for ptensor in fptensors), "Not support for multi-device"
            fconsumers, fctensors = graph.consumers(ftensor), graph.ctensors(ftensor)
            fctensors = expand_devices(fctensors, consumer=True)
            assert all(len(ctensor.device) == 1 for ctensor in fctensors), "Not support for multi-device"

            bproducers, bptensors = [], []
            bconsumers, bctensors = [], []
            if isinstance(ftensor.grad, IRFullTensor):
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
                if not (fadapter.differentiable or fadapter.mirror is None):
                    raise NotImplementedError(
                        "Require adapter to be differentiable for nested IRAdapter.\n"
                        "Condition to be differentiable: prodcuers have same device set with consumers\n"
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
            fidx = min(graph.index(c) for c in fconsumers)
            # setup recompute
            if fadapter.differentiable and allow_recompute:
                fadapter.recompute = graph.node(fidx).recompute
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
        if not ftensor.requires_grad: return ftensor

        devtensors: Dict[DeviceID, List[IRSubTensor]] = dict()
        devops: Dict[DeviceID, List[IRCell]] = dict()

        # collect producers for each device
        for ptensor, producer in zip(graph.ptensors(ftensor), graph.producers(ftensor)):
            for devid in ptensor.device:
                if devid not in devtensors:
                    devtensors[devid], devops[devid] = [], []
                devtensors[devid].append(ptensor)
                devops[devid].append(producer)

        require_fusion = any(len(set(ts)) > 1 for ts in devtensors.values())
        if not require_fusion: return ftensor

        new_ftensor = ftensor.like()

        # update consumer
        for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
            itensor = new_ftensor.select(ctensor.indmap, ctensor.valmap)
            igrad = new_ftensor.grad.select(ctensor.grad.indmap, ctensor.grad.valmap)
            with graph.update(consumer) as consumer:
                idx = consumer.inputs().index(ctensor)
                consumer.set_input(idx, itensor)
            with graph.mirror.update(consumer.mirror) as bconsumer:
                idx = bconsumer.outputs().index(ctensor.grad)
                bconsumer.set_output(idx, igrad)

        for devid in devtensors:
            indmaps = [t.indmap for t in devtensors[devid]]
            valmaps = [t.valmap for t in devtensors[devid]]
            split_dim = len(set(indmaps)) > 1
            split_val = len(set(valmaps)) > 1
            assert not (split_dim and split_val), (
                f"Not support for simutaneously partitioning tensor dimension and tensor value.\n"
                f"{graph.debug_tensor_map_str(ftensor)}"
            )

            node = None

            # split dimension case
            if split_dim:
                catdim: int = None
                for dim in range(len(ftensor.shape)):
                    dim_maps = [ind[dim] for ind in indmaps]
                    if set(len(dim_maps)) != 1:
                        assert catdim is None, (
                            f"Not support for multi-dim partitioning on local producers.\n"
                            f"{graph.debug_tensor_map_str(ftensor)}"
                        )
                        catdim = dim
                assert catdim is not None
                start_idx = np.array([ind[catdim][0] for ind in indmaps])
                indices = np.argsort(start_idx)
                ptensors = [devtensors[devid][idx] for idx in indices]
                try:
                    otensor = ptensors[0]
                    for t in ptensors[1:]:
                        otensor = otensor.concat(t, dim=catdim)
                except Exception as e:
                    raise RuntimeError(
                        f"Device {devid}: Fail to concat local produced tensors on dimension: {catdim}\n"
                        f"Users can try to adjust node ordering to meet with concat order.\n"
                        f"{graph.debug_tensor_map_str(ftensor)}"
                    )
                # set concat input / output
                node = Cat('torch.cat', (ptensors, catdim))
                node.set_output(0, new_ftensor.select(otensor.indmap, otensor.valmap))
                # set gradient
                for idx, ptensor in enumerate(ptensors):
                    node.input(idx).grad = ftensor.grad.select(ptensor.indmap, (0,1))
                node.output(0).grad = new_ftensor.grad.select(otensor.indmap, (0,1))

            # split value case
            if split_val:
                # reverse to meet with add order
                ptensors = devtensors[devid]
                try:
                    nchunks = [t.valmap[1] for t in ptensors]
                    if len(set(nchunks)) == 1:
                        otensor = ptensors[0].accum(ptensors[1:])
                    else:
                        # the add order is to adapt with ordering valmap ordering: (3/4) (2/4) (0/2)
                        ptensors = ptensors[::-1]
                        otensor = ptensors[0]
                        for t in ptensors[1:]:
                            otensor = otensor.accum(t)
                except Exception as e:
                    raise RuntimeError(
                        f"Device {devid}: Fail to accum local produced tensors\n"
                        f"Users can try to adjust node ordering to meet with accum order\n"
                        f"{graph.debug_tensor_map_str(ftensor)}"
                    )
                # set accum input / output
                node = Accum('cube.runtime.accum', ptensors)
                node.set_output(0, new_ftensor.select(otensor.indmap, otensor.valmap))
                # set gradient
                for idx, ptensor in enumerate(ptensors):
                    node.input(idx).grad = ftensor.grad.select(ptensor.indmap, (0,1))
                node.output(0).grad = new_ftensor.grad.select(otensor.indmap, (0,1))

            # no need for fusion, change the producer output to new tensor
            if node is None:
                for ptensor, producer in zip(devtensors[devid], devops[devid]):
                    otensor = new_ftensor.select(ptensor.indmap, ptensor.valmap)
                    ograd = new_ftensor.grad.select(otensor.grad.indmap, otensor.grad.valmap)
                    with graph.update(producer):
                        idx = producer.outputs().index(ptensor)
                        producer.set_input(idx, otensor)
                        producer.input(idx).grad = ograd
                    with graph.mirror.update(producer.mirror) as bproducer:
                        idx = bproducer.inputs().index(otensor.grad)
                        bproducer.set_input(idx, ograd)
            else:
                node.device = devid
                # set recompute
                rcid = set(producer.recompute for producer in devops[devid])
                rcid = list(rcid)[0] if len(rcid) == 1 else None
                node.recompute = rcid
                # insert
                max_fid = max(graph.index(producer) for producer in devops[devid])
                graph.finsert(node, max_fid + 1)

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

        @return None
        """
        if not ftensor.requires_grad: return

        devtensors : Dict[DeviceID, List[IRSubTensor]] = dict()
        devops : Dict[DeviceID, List[IRCell]] = dict()

        # collect consumer of each device
        for ctensor, consumer in zip(graph.ctensors(ftensor), graph.consumers(ftensor)):
            for devid in ctensor.device:
                if devid not in devtensors:
                    devtensors[devid], devops[devid] = [], []
                assert len(devtensors[devid]) == 0 or devtensors[devid][0] == ctensor, (
                    f"Detect that a full tensor is partitioned differently on a device.\n"
                    f"To achieve this, need manually add multiref operator in model description.\n"
                    f"{graph.debug_tensor_map_str(ftensor)}"
                )
                devtensors[devid].append(ctensor)
                devops[devid].append(consumer)

        require_multiref = any(len(ops) > 1 for ops in devops.values())
        if not require_multiref: return

        for devid in devtensors:
            grads: List[IRSubTensor] = [t.grad for t in devtensors[devid]][::-1]
            try:
                nchunks = [grad.valmap[1] for grad in grads]
                if len(set(nchunks)) == 1:
                    accum_grad = grads[0].accum(grads[1:])
                else:
                    # the add order is to adapt with ordering valmap ordering: (3/4) (2/4) (0/2)
                    accum_grad = grads[0]
                    for grad in grads[1:]:
                        accum_grad = accum_grad.accum(grad)
            except Exception as e:
                raise RuntimeError(
                    f"Device {devid}: Fail to accumulate local gradient: {ftensor.grad}\n"
                    f"Error information: {str(e)}\n"
                    f"Users can try:\n"
                    f"  1) Replicate all operators whose inputs have multi-consumed tensors\n"
                    f"  2) Partition all operators whose inputs have multi-consumed tensors\n"
                    f"  3) Mannually add cube.runtime.multiref in model description to divide replicated and partitioned groups\n"
                    f"{graph.debug_tensor_map_str(ftensor)}"
                    f"{graph.mirror.debug_tensor_map_str(ftensor.grad)}"
                )

            multiref = MultiRef(None, [devtensors[devid][0], len(grads)])
            # set input gradient
            multiref.input(0).grad = accum_grad
            # set output and its gradient
            for idx, ctensor in enumerate(devtensors[devid]):
                new_ftensor = ctensor.parent.like()
                otensor = new_ftensor.select(ctensor.indmap, (0,1))
                multiref.set_output(idx, otensor)
                multiref.output(idx).grad = new_ftensor.grad.select(ctensor.indmap, (0,1))
                # set corresponding consumer input and its backward
                consumer = devops[devid][idx]
                with graph.update(consumer):
                    while ctensor in consumer.inputs():
                        fidx = consumer.inputs().index(ctensor)
                        consumer.set_input(fidx, otensor)
                        consumer.input(fidx).grad = new_ftensor.grad.select(ctensor.indmap, (0,1))
                with graph.mirror.update(consumer.mirror) as bconsumer:
                    while ctensor.grad in bconsumer.outputs():
                        bidx = bconsumer.outputs().index(ctensor.grad)
                        bconsumer.set_output(bidx, new_ftensor.grad.select(ctensor.indmap, (0,1)))
            # insert multiref
            multiref.device = devid
            min_fidx = min(graph.index(consumer) for consumer in devops[devid])
            # set recompute id
            multiref.recompute = graph.node(min_fidx).recompute
            graph.finsert(multiref, min_fidx)

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

    @staticmethod
    def tensor_merge(tensors: List[IRSubTensor], target: Optional[IRSubTensor] = None) -> List[Tuple[str, List, IRSubTensor]]:
        """
        Merge sub-tensors into one tensor or stop right after gets target tensor.

        Merge primtiives:
            "sum: output = sum(inputs)"
            "cat: output = cat(inputs, dim: int)

        @param tensors List[IRSubTensor]: list of tensors
        @param target Optional[IRSubTensor]: the target tensor (default None).

        @return primitives List[Tuple[str, List, IRSubTensor]]:
            List primitives of in forms of (op, inputs, outputs)
        """
        prims = []
        tensors = [t for t in tensors]
        while len(tensors) > 1:
            out = None
            for t1, t2 in itertools.combinations(tensors, 2):
                # try concat
                catdim = t1.catdim(t2)
                if catdim is not None:
                    tensors = [t1, t2] if t1.indmap[catdim][0] < t2.indmap[catdim][0] else [t2, t1]
                    out = tensors[0].concat(tensors[1], dim=catdim)
                    prims.append(('cat', tensors + [catdim], out))
                    break
                # try summation
                if t1.accumable(t2):
                    out = t1.accum(t2)
                    prims.append(('sum', [t1, t2], out))
                    break
            if out is not None:
                tensors.remove(t1)
                tensors.remove(t2)
                tensors.append(out)
                if target is not None and out == target: break
            else:
                remain = '\n\t'.join(t.extra_repr() for t in tensors)
                sprims = '\n\t'.join(repr(p) for p in prims)
                raise RuntimeError(
                    f"Fail to merge tensors into one tensor or cannot match with target.\n"
                    f"Remain Tensor:\n\t{remain}\n"
                    f"Existing primitives:\n\t{sprims}\n"
                )
        return prims
