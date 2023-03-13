from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import itertools

from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.gener.concurrent import ConcurrentGener
import cube.graph.gener.utils as utils
from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.pyfunc import IRPyFunc

from cube.ir.cten import IRCell, IRObject
from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.ir.operator import IRFwOperation

from cube.ir.adapter import IRAdapter, IRWeightReducer
from cube.graph.function.function import Accum, Cat, MultiRef


DeviceID = int


def create_dummy(segment: IRSegment, inputs: bool = True, outputs: bool = True) -> List[IRFwOperation]:
    """
    Create dummy operators segment inputs and outputs. 
    The backward operator is also inserted.

    @param segment IRSegment: the target segment
    @param inputs bool: True for creating dummy operators to produce segement's inputs
    @param outputs bool: True for creating dummpy operators to consume segment's outputs
    
    @return nodes List[IRCell]: the generated operation
    """
    # devices = segment.device
    fwops = []

    # create inputs
    if inputs:
        input_objects = IRGraph.get_objects_from_complex(segment.inputs())
        for tensor in input_objects:
            devices = [consumer.device for consumer in segment.consumers(tensor.parent)][::-1]
            if not isinstance(tensor, IRSubTensor): continue
            assert tensor.valmap == (0, 1), f"valmap != (0, 1):\n{segment.extra_repr()}"
            fwop = utils.DummyInputOuput(tensor, 0, is_output=True, name=f'segment{segment.cid}_input')
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
    if outputs:
        output_objects = IRGraph.get_objects_from_complex(segment.outputs())
        for tensor in output_objects:
            devices = [producer.device for producer in segment.producers(tensor.parent)]
            if not isinstance(tensor, IRSubTensor): continue
            assert tensor.valmap == (0, 1), f"valmap != (0, 1):\n{segment.extra_repr()}"
            fwop = utils.DummyInputOuput(tensor, 0, is_input=True, name=f'segment{segment.cid}_output')
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
                fwop = utils.DummyInputOuput(tensor, devid, is_output=True, name=tensor.cell.name)
                dtensors.append(fwop.output(0))
            elif consumer:
                fwop = utils.DummyInputOuput(tensor, devid, is_input=True, name=tensor.cell.name)
                dtensors.append(fwop.input(0))
            else:
                raise ValueError("At least one of producer or consumer")
    return dtensors


class IRAdapterGener:

    @staticmethod
    def gen(graph: IRGraph, cost_fn: Optional[Callable] = None) -> IRGraph:
        """
        Generate tensor adapter for both activations and weights
        Note weight reducers are always append to the last.

        @param graph IRGraph: the graph without adapter
        @param cost_fn Optional[Callable]: takes an IRAdapterPrim and outputs a cost in float.
            default to be None, which will use communication volume.
    
        @return graph IRGraph: the graph with adapter inserted
        """
        # reorder producer and consumer ordering
        graph._reorder_producer_consumer()
        # remove anchor node
        graph = IRAdapterGener.remove_anchor(graph)
        # automatic replace pyfunc
        graph = IRAdapterGener.auto_pyfunc(graph)
        # automatic transform multiref
        graph = IRAdapterGener.autoref(graph)
        # generate adapters for activation
        graph = IRAdapterGener.gen_activation(graph, cost_fn=cost_fn)
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
    def auto_pyfunc(graph: IRSegment):
        """
        Make pyfunc to be local
        IRPyFunc will be replicated to devices with its producers output
        """
        for func in graph.select(ntype=IRPyFunc, flatten=False):
            assert func.mirror is None, "PyFunc is only supported by inference"
            # get devices it will lowered to
            devices = set()
            for t in func.inputs():
                if not isinstance(t, IRObject): continue
                if t.is_attr():
                    cells = graph.consumers(t.parent)
                else:
                    cells = graph.producers(t.parent)
                for cell in cells:
                    devices.update(cell.device)
            pyfuncs = []
            # lower to each device
            for devid in devices:
                inputs = []
                # automatic partition to align with consumer (attr) or producer (activation)
                for t in func.inputs():
                    sub_ts = set()
                    if not isinstance(t, IRSubTensor):
                        sub_ts.add(t)  # replica for non-tensor
                    elif t.is_attr():
                        # get local consumers except func itself
                        sub_ts = set(tensor for tensor in graph.ctensors(t.parent) \
                                     if devid in tensor.device and tensor.cell != func)
                    else:
                        # get local producers
                        sub_ts = set(tensor for tensor in graph.ptensors(t.parent) \
                                     if devid in tensor.device)
                    inputs.append(t if len(sub_ts) == 0 else list(sub_ts)[0])
                lower_func = IRPyFunc(func.signature, inputs, func.outputs(), **func.kwargs)
                lower_func.device = devid
                pyfuncs.append(lower_func)
            position = graph.remove(func)
            for pyfunc in pyfuncs:
                graph.insert(pyfunc, position)
        for segment in graph.select(ntype=IRSegment, flatten=False):
            IRAdapterGener.auto_pyfunc(segment)
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
        
        nl = '\n'
        weights: Dict[IRFullTensor, Dict[IRSubTensor, List[int]]] = dict()
        for fweight in fweights.keys():
            weights[fweight] = {}
            weight_grads: Dict[IRSubTensor, Dict[IRSubTensor, List[IRFwOperation]]] = {}
            for weight, grad, consumer in zip(fweights[fweight], fgrads[fweight], consumers[fweight]):
                if weight not in weight_grads:
                    weight_grads[weight] = {}
                if grad not in weight_grads[weight]:
                    weight_grads[weight][grad] = []
                weight_grads[weight][grad].append(consumer)
            
            # TODO: check sub_weight is no-overlapping

            # assert all(sw.valmap[1] == len(weight_grads) for sw in weight_grads.keys())
            for sub_weight in weight_grads:
                diff_grads = weight_grads[sub_weight]
                diff_grads_len = [len(diff_grads[grads]) for grads in diff_grads]
                assert all(n == diff_grads_len[0] for n in diff_grads_len), (
                    f"If one of the weight consumers are replicated, "
                    f"other same-weight consumers should also replicated in same way."
                    f"FullTensor Weight: {fweight}\n"
                    f"Consumers:\n{nl.join([repr(node) for node in consumers[fweight]])}"
                )
                # get devices
                devices = []
                for sub_grad in diff_grads:
                    sub_grad_devices = [node.device[0] for node in diff_grads[sub_grad]]
                    sub_grad_devices.sort()
                    devices.append(sub_grad_devices)
                devices = np.array(devices, dtype=int).transpose((1, 0))
                for group_devices in devices:
                    group_devices = set(int(devid) for devid in group_devices)
                    group_devices = list(group_devices)
                    group_devices.sort()
                    weights[fweight][sub_weight] = group_devices

        reducers: Dict[Tuple[int], List[IRSubTensor]] = dict()
        for subtensors in weights.values():
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
    def gen_activation(graph: IRSegment, allow_recompute: bool = True, cost_fn: Optional[Callable] = None) -> IRSegment:
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

        fdummies = create_dummy(graph, inputs=True, outputs=True)
        bdummies = [fwop.mirror for fwop in fdummies if fwop.mirror is not None]
        bgraph: Optional[IRSegment] = graph.mirror
    
        # local producer fusion and local consumer multiref
        ftensors = []
        for ftensor in graph.full_tensors():
            # backward will gen in forward
            if ftensor.is_param() or ftensor.is_grad():
                continue
             # flatten gradient
            utils.flatten_grad(graph, ftensor)
            # optimization: local fusion / multiref on producer / consumer
            ftensor = IRAdapterGener.local_producer_fusion(graph, ftensor)
            IRAdapterGener.local_consumer_multiref(graph, ftensor)
            ftensors.append(ftensor)
        
        # reorder again since inserted multiref could be mis-ordered
        graph._reorder_producer_consumer()

        # generate adapter for inter-segments
        # FIXME: assume producers and consumers can run in parallel
        for ftensor in ftensors:

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

            fadapter = ConcurrentGener.gen(fptensors, fctensors, bptensors, bctensors, cost_fn)
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
            graph.insert(fadapter, fidx)

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
            # get recomput group
            rcid = set(producer.recompute for producer in devops[devid])
            rcid = list(rcid)[0] if len(rcid) == 1 else None

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
                node = Cat(ptensors, dim=catdim)
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
                
                # === Optimization: quick accumulation to early release tensor
                lhs, rhs = ptensors[0], None
                for ptensor in ptensors[1:]:
                    rhs = ptensor
                    output = ftensor.like().select(ptensors[0].indmap, (0,1))
                    node = Accum(lhs, rhs)
                    node.set_output(0, output)
                    node.device = devid
                    node.recompute = rcid
                    graph.insert(node, graph.index(ptensor.cell) + 1)
                    lhs = output
                # remove last node for adaptation
                graph.remove(node)

                # === Orignal way to at alst release tensor
                # node = Accum(*ptensors)
                # # set gradient
                # for idx, ptensor in enumerate(ptensors):
                #     node.input(idx).grad = ftensor.grad.select(ptensor.indmap, (0,1))

                # set output
                node.set_output(0, new_ftensor.select(otensor.indmap, otensor.valmap))
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
                devtensors.setdefault(devid, []).append(ctensor)
                devops.setdefault(devid, []).append(consumer)
                assert devtensors[devid][0] == ctensor, (
                    f"Detect that a full tensor is partitioned differently on a device.\n"
                    f"To achieve this, need call graph.multiref before graph transformation.\n"
                    f"{graph.debug_tensor_map_str(ftensor)}"
                )

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
                    f"  3) Call graph.multiref to divide tensors with different partition strategies\n"
                    f"{graph.debug_tensor_map_str(ftensor)}"
                    f"{graph.mirror.debug_tensor_map_str(ftensor.grad)}"
                )

            multiref = MultiRef(devtensors[devid][0], len(grads))
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
    def autoref(graph: IRSegment) -> IRGraph:
        """
        Automatically transform inserted multiref.
        Multiref is transformed to align with the output tensors on each device.

        @param graph IRGraph

        @return None
        """
        for multiref in graph.select(name='multiref', flatten=False):
            ftensor: IRFullTensor = multiref.input(0).parent
            multirefs = []
            for otensor in graph.ptensors(ftensor):
                mr = MultiRef(otensor, len(multiref.outputs()))
                for idx in range(len(multiref.outputs())):
                    output = multiref.output(idx).parent.select(otensor.indmap, otensor.valmap)
                    if otensor.requires_grad:
                        output.grad = multiref.output(idx).grad.parent.select(otensor.indmap, (0,1))
                    mr.set_output(idx, output)
                mr.device = otensor.device
                mr.recompute = otensor.cell.recompute
                multirefs.append(mr)
            # remove original multiref
            fidx = graph.remove(multiref)
            if multiref.mirror is not None:
                graph.mirror.remove(multiref.mirror)
            # insert multirefs
            for ofst, multiref in enumerate(multirefs):
                if ftensor.requires_grad:
                    graph.finsert(multiref, fidx + ofst)
                else:
                    graph.insert(multiref, fidx + ofst)
        for segment in graph.select(ntype=IRSegment, flatten=False):
            if not segment.isfw(): continue
            IRAdapterGener.autoref(segment)
        return graph

    @staticmethod
    def fusion(graph: IRSegment) -> IRSegment:
        """
        Fuse consecutive adapters into one
        """
        fadapters, badapters = [], []
        for adapter in graph.nodes():
            if isinstance(adapter, IRAdapter) and adapter.isfw() and not adapter.differentiable:
                fadapters.append(adapter)
                if adapter.mirror is not None:
                    badapters.append(adapter.mirror)
                    # badapters.insert(0, adapter.mirror)
            else:
                if len(fadapters) > 1:
                    # reorder adapter to match output of segment. This is temporally
                    # necessary for pipeline scheduling with multiple output.
                    ftids = np.array([fadapter.input(0).parent.tid for fadapter in fadapters])
                    indices = np.argsort(ftids)
                    fadapters = [fadapters[idx] for idx in indices]
                    if len(badapters) > 0:
                        badapters = [badapters[idx] for idx in indices]
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
