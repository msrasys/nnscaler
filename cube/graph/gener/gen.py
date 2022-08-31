import itertools
from typing import Dict, List, Optional, Tuple
import copy

from cube.graph.graph import IRGraph
from cube.ir.cten import IRCell
from cube.ir.tensor import IRFullTensor, IRSubTensor, IndexMap, ValueMap

from cube.ir.operator import IRBpOperation, IRFwOperation

from cube.ir.adapter.prim import IRAdapterPrim
from cube.ir.adapter import IRAdapter, IRWeightReducer
from cube.graph.function.function import Add, Cat, Identity, MultiRef
from cube.ir.adapter.prim import SelectPrim, MovePrim, SumPrim, MergeDimPrim
from cube.graph.gener.layout import GridLayout


class IRAdapterGener:

    @staticmethod
    def gen(graph: IRGraph) -> IRGraph:
        """
        Generate tensor adapter for both activations and weights

        Args:
            graph: IRGraph.
            eager (Boolean):
                if True,
                    each adapter will be inserted right after it's ready to execute.
                if False (i.e., lazy),
                    each adatper will be inserted right before the tensor needs it.
            Note weight reducers are always append to last.
        Returns:
            graph (IRGraph)
        """
        # insert identity operator for graph output
        devs = set()
        for node in graph.nodes():
            devs.update(node.device)
        outputs = [otensor for otensor in graph.outputs() if isinstance(otensor, IRSubTensor)]
        all_identities = []
        for otensor in outputs:
            identity = Identity('', [otensor])
            graph.attach(identity, len(graph.nodes()))
            identites = graph.replicate(identity, times=len(devs))
            all_identities += identites
            for devid, identity in zip(devs, identites):
                graph.assign(identity, devid)
        # update the gradient before generate adapter
        for node in graph.nodes():
            if isinstance(node, IRBpOperation):
                idx = graph.detach(node)
                node.update()
                graph.attach(node, idx)
        graph = IRAdapterGener.gen_activation(graph)
        graph = IRAdapterGener.gen_weight(graph)
        # remove inserted identity
        for identity in all_identities:
            graph.detach(identity)
        return graph

    @staticmethod
    def gen_weight(graph: IRGraph) -> IRGraph:
        # step 1: get weight and gradient
        # weights: Dict[weight_id: int, IRSubTensor]
        # grads  : Dict[weight_id: int, Dict[device: int, List[grad: IRSubTensor]]]
        grads = dict()
        weights = dict()
        for fnode in graph.nodes():
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
            graph._nodes.append(opt_op)
        return graph

    @staticmethod
    def gen_activation(graph: IRGraph) -> IRGraph:
        """!
        Generate adapter for activation tensors.
        The forward/backward adapter is inserted before the first consumers of its full tensor.

        @param graph IRGraph: the graph the requires for adapter.

        @return graph IRGraph: the (inplace) modified graph with activation adapters. 
        """
        for ftensor in graph.full_tensors():
            # backward will gen in forward
            if ftensor.is_param() or ftensor.is_grad():
                continue
            # no consumer usually mean loss
            if len(ftensor.consumers) == 0:
                continue
            # graph attribute: buffer
            if len(ftensor.producers) == 0:
                continue
            # no require for communication
            if len(ftensor.consumers) == 1 and len(ftensor.producers) == 1 and \
               ftensor.consumers[0].device == ftensor.producers[0].device:
                continue

            # optimization: local fusion on producer
            if graph.train:
                ftensor = IRAdapterGener.local_producer_fusion(graph, ftensor)
                IRAdapterGener.local_consumer_multiref(graph, ftensor)

            ptensors, ctensors = ftensor.ptensors, ftensor.ctensors
            pdevs = tuple(ptensor.device[0] for ptensor in ptensors)
            cdevs = tuple(ctensor.device[0] for ctensor in ctensors)

            fadapter = None
            # Case 1: sharing device (in-shard)
            inshard = set(pdevs) == set(cdevs) and len(ptensors) == len(ctensors) and \
                      len(pdevs) == len(ptensors)
            if inshard and len(pdevs) > 1:
                try:
                    fadapter = IRAdapterGener.gen_in_shard(ftensor, allow_reorder=True)
                except Exception as e:
                    fadapter = None
                    print(
                        f"full tensor: {ftensor} cannot use grid generation.\n"
                        f"Reason: {str(e)}\n"
                        f"Switch to general P2P communication."
                    )

            # Case 2: sperating device (cross-shard)
            if len(set(pdevs).intersection(cdevs)) == 0:
                pass

            # Case 3: General cases
            # warnings.warn('The adapter is generated using
            if fadapter is None:
                fadapter = IRAdapterGener.gen_general(ftensor)

            badapter: Optional[IRAdapter] = fadapter.mirror
            
            if (badapter is not None and len(fadapter.prims) == 0 and len(badapter.prims) == 0) or \
               (badapter is None and len(fadapter.prims) == 0):
                continue

            # set differentiable for autograd generation
            if inshard and badapter is not None:
                fadapter.differentiable = True
                badapter.differentiable = True

            # insert forward adapter
            fidx = min([graph.nodes().index(consumer) for consumer in ftensor.consumers])
            graph._nodes.insert(fidx, fadapter)

            # insert backward
            if badapter is not None:
                bidx = min(graph.nodes().index(consumer) for consumer in ftensor.grad.consumers)
                graph._nodes.insert(bidx, badapter)

        # print(graph.extra_repr())
        return graph

    @staticmethod
    def gen_in_shard(ftensor: IRFullTensor, allow_reorder=False) -> Optional[IRAdapter]:
        """
        Generate communication for sharing devices (SPMD-like)
        
        @param ftensor: IRFullTensor
        @param ptensors: List[IRSubTensor]: produced subtensors
        @param ctensors: List[IRSubTensor]: consumed subtensors

        @return adapter Optional[IRAdapter]: generated adapter.
        """
        # producer grid layout
        ilayout = GridLayout.togrid(ftensor, ftensor.ptensors)
        # reorder ctensors to match with ptensors
        devs = [ptensor.device for ptensor in ilayout.mat.flatten()]
        ctensors = [None] * len(devs)
        for ctensor in ftensor.ctensors:
            idx = devs.index(ctensor.device)
            ctensors[idx] = ctensor
        assert all(t is not None for t in ctensors), f"empty device slot {ctensors}"
        # consumer grid layout
        olayout = GridLayout.togrid(ftensor, ctensors)
        # find path
        paths, fprims = ilayout.path(olayout)

        # re-assign the operator if miss-ordered
        names, from_dev, to_dev = [], [], []
        for itensor, otensor in zip(paths[-1].mat.flatten(), olayout.mat.flatten()):
            assert len(itensor.device) == 1 and len(otensor.device) == 1, \
                "Expect tensor only has one device. Report this as a bug"
            if itensor.device != otensor.device:
                inode, onode = itensor.cell, otensor.cell
                names.append(f'{onode.name}{onode.cid}')
                from_dev.append(onode.device[0])
                to_dev.append(inode.device[0])
                if allow_reorder:
                    onode.device = inode.device
                    if onode.mirror is not None:
                        onode.mirror.device = inode.device
                else:
                    raise RuntimeError("device mismatch. Try to enable reorder")
        if len(names) > 0:
            print(f'UserWarning: a better device placement is found and set for op {names}: {from_dev} -> {to_dev}')

        fadapter = IRAdapter(ftensor.ptensors, ftensor.ctensors)
        fadapter.prims = fprims

        # generate backward
        grad: IRFullTensor = ftensor.grad
        bprims = []
        if grad is not None and (len(grad.ptensors) != 0 or len(grad.ctensors) != 0):
            # reorder ptensors to match with forward
            ptensors = [None] * len(devs)
            for ptensor in grad.ptensors:
                idx = devs.index(ptensor.device)
                assert ptensors[idx] is None, "same device of different tensors"
                ptensors[idx] = ptensor
            ilayout = GridLayout.togrid(grad, ptensors)
            olayout = GridLayout.togrid(grad, grad.ctensors)
            paths, bprims = ilayout.path(olayout)
            # check the device order
            for itensor, otensor in zip(paths[-1].mat.flatten(), olayout.mat.flatten()):
                assert len(itensor.device) == len(otensor.device), "backward device not match"
            badapter = IRAdapter(grad.ptensors, grad.ctensors)
            badapter.prims = bprims
            IRAdapter.make_pair(fadapter, badapter)

        return fadapter

    @staticmethod
    def gen_cross_shard(ftensor: IRFullTensor, ptensors: List[IRSubTensor], ctensors: List[IRSubTensor]) -> Optional[IRAdapter]:
        pass

    @staticmethod
    def gen_general(ftensor: IRFullTensor) -> IRAdapter:
        fprims = []
        for ctensor in ftensor.ctensors:
            fprims += IRAdapterGener.gen_subtensor(ctensor, ftensor.ptensors)
        fadapter = IRAdapter(ftensor.ptensors, ftensor.ctensors)
        fadapter.prims = fprims
        if ftensor.grad is not None:
            bprims = []
            for cgrad in ftensor.grad.ctensors:
                bprims += IRAdapterGener.gen_subtensor(cgrad, ftensor.grad.ptensors)
            badapter = IRAdapter(ftensor.grad.ptensors, ftensor.grad.ctensors)
            badapter.prims = bprims
            IRAdapter.make_pair(fadapter, badapter)
        return fadapter

    @staticmethod
    def gen_subtensor(ctensor: IRSubTensor, ptensors: List[IRSubTensor]) -> List[IRAdapterPrim]:
        """
        Generate communiction primitives for ctensor
        
        @param ctensor IRSubTensor: the consumed tensor as destination
        @param ptensors List[IRSubTensor]: the produced tensors as source

        @return prims List[IRAdapterPrim]: the primitives for adapter
        """
        # category to local tensor and remote tensor
        local = [t for t in ptensors if t.device == ctensor.device]
        remote = [t for t in ptensors if t.device != ctensor.device]
        prims = []

        # ==== select ==== #
        intersections = []
        # check local
        for itensor in local+remote:
            if itensor.device == ctensor.device and itensor == ctensor:
                return []
            common: Optional[IRSubTensor] = itensor.common(ctensor)
            if common is None:
                continue
            common.cell = itensor.cell
            intersections.append(common)
            # create select primitive
            if common != itensor:
                indmap = []
                for dim in range(itensor.ndims):
                    (s1, e1), (s2, e2) = itensor.indmap[dim], common.indmap[dim]
                    start = s2 - s1
                    end = start + e2 - s2
                    indmap.append((start, end))
                indmap = IndexMap(tuple(indmap))
                assert itensor.valmap == common.valmap, "Value map not same"
                valmap = ValueMap((0, 1))
                select_prim = SelectPrim(itensor, indmap, valmap, common)
                prims.append(select_prim)
            if itensor.device == ctensor.device and common == ctensor:
                return [select_prim]
            # TODO: check union == subtensor
            if common == ctensor:
                break

        # print(intersections)
        # ====== move ===== #
        tmoved = []
        for tensor in intersections:
            assert len(tensor.device) == 1 and len(ctensor.device) == 1, "Expected only one device."
            mtensor = tensor
            if tensor.device != ctensor.device:
                mtensor = copy.copy(tensor)
                mtensor.cell = ctensor.cell
                prims.append(MovePrim(tensor, mtensor))
            tmoved.append(mtensor)

        # ===== merge ===== #
        remain_tensors: List[IRSubTensor] = copy.copy(tmoved)
        if ctensor in remain_tensors:
            return prims
        out = None
        while out != ctensor:
            out, merged = None, False
            for idx1 in range(len(remain_tensors) - 1):
                for idx2 in range(idx1+1, len(remain_tensors)):
                    t1, t2 = remain_tensors[idx1], remain_tensors[idx2]
                    catdim = t1.catdim(t2)
                    if catdim is not None:
                        tensors = [t1, t2] if t1.indmap[catdim][0] < t2.indmap[catdim][0] else [t2, t1]
                        out = tensors[0].concat(tensors[1], dim=catdim)
                        out.cell = ctensor.cell
                        prims.append(MergeDimPrim(tensors, out, catdim))
                        merged = True
                        break
                    # reduction
                    if t1.accumable(t2):
                        out = t1.accum(t2)
                        out.cell = ctensor.cell
                        prims.append(SumPrim([t1, t2], out))
                        merged = True
                        break
                if merged:
                    remain_tensors.remove(t1)
                    remain_tensors.remove(t2)
                    remain_tensors.append(out)
                    break
            if out is None:
                ptensors = '\n\t'.join(t.extra_repr() for t in ptensors)
                remain = '\n\t'.join(t.extra_repr() for t in remain_tensors)
                print(remain_tensors[0].extra_repr())
                print(remain_tensors[1].extra_repr())
                print('cadim:', remain_tensors[0].catdim(remain_tensors[1]))
                raise RuntimeError(
                    f"Fail to build adapter.\n"
                    f"FullTensor:{ctensor.parent}\n"
                    f"Producers:\n\t{ptensors}\n"
                    f"SubTensor:\n\t{ctensor.extra_repr()}\n"
                    f"Remain Tensor:\n\t{remain}"
                )
        return prims

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
        min_idx = len(graph.nodes())
        assert len(ftensor.ctensors) == len(ftensor.consumers)
        for ctensor, consumer in zip(ftensor.ctensors, ftensor.consumers):
            fidx = graph.detach(consumer)
            consumer.set_input(
                consumer.inputs().index(ctensor),
                new_ftensor.select(ctensor.indmap, ctensor.valmap)
            )
            graph.attach(consumer, fidx)
            min_idx = min(fidx, min_idx)

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
            # print(node)
            assert node not in graph.nodes()
            assert len(node.outputs()) == 1
            graph.attach(node, min_idx)

        # insert and update backward node
        if graph.train:
            # update backward node
            for consumer in new_ftensor.consumers:
                assert isinstance(consumer.mirror, IRBpOperation)
                bidx = graph.detach(consumer.mirror)
                consumer.mirror.update()
                graph.attach(consumer.mirror, bidx)
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

        # add multiref forward node
        multirefs: Dict[MultiRef, List[IRFwOperation]] = dict()
        for devid in devtensors:
            for ctensor in devtensors[devid]:
                consumers = devtensors[devid][ctensor]
                if len(consumers) == 1:
                    continue
                multiref = MultiRef(None, [ctensor, len(consumers)])
                multiref.device = devid
                ftensors = [ctensor.parent.like() for _ in range(len(consumers))]
                itensors = [ft.select(ctensor.indmap, ctensor.valmap) for ft in ftensors]
                for idx, itensor in enumerate(itensors):
                    multiref.set_output(idx, itensor)

                # update consumer
                min_fidx = len(graph.nodes())
                for itensor, consumer in zip(itensors, consumers):
                    fidx = graph.detach(consumer)
                    idx = consumer.inputs().index(ctensor)
                    consumer.set_input(idx, itensor)
                    graph.attach(consumer, fidx)
                    min_fidx = min(fidx, min_fidx)
    
                # insert forward multiref
                graph.attach(multiref, min_fidx)
                multirefs[multiref] = consumers

        # insert / update backward
        if graph.train:
            for multiref, consumers in multirefs.items():
                # update consumer backward
                for consumer in consumers:
                    assert isinstance(consumer.mirror, IRBpOperation)
                    bidx = graph.detach(consumer.mirror)
                    consumer.mirror.update()
                    graph.attach(consumer.mirror, bidx)
                # insert backward
                bnode = multiref.gen_backward()
                bnode.device = multiref.device
                bidx = max(graph.nodes().index(consumer.mirror) for consumer in consumers)
                graph.attach(bnode, bidx+1)
