from typing import Dict, List, Optional, Tuple
import warnings
import copy

from cube.graph.graph import IRGraph
from cube.graph.tensor import IRFullTensor, IRSubTensor, IndexMap, ValueMap
from cube.graph.adapter.adapter import IRAdapter, IRWeightReducer

from cube.graph.operator.operator import IRBpOperation, IRFwOperation
from cube.ir.cten import IRCell

from cube.graph.adapter.prim import IRAdapterPrim
from cube.graph.adapter.prim import SelectPrim, MovePrim, SumPrim, MergeDimPrim
from cube.graph.adapter.layout import GridLayout


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
        # update the gradient before generate adapter
        for node in graph.nodes():
            if isinstance(node, IRBpOperation):
                idx = graph.detach(node)
                node.update()
                graph.attach(node, idx)
        graph = IRAdapterGener.gen_activation(graph)
        graph = IRAdapterGener.gen_weight(graph)
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
            for input in fnode.inputs():
                if isinstance(input, IRSubTensor) and input.is_param():
                    grad = input.grad
                    if grad is None:
                        continue
                    # nothing to sync
                    if grad.valmap == ValueMap(0, 1):
                        continue
                    if input._id not in grads:
                        grads[input._id] = dict()
                        weights[input._id] = input
                    if devid not in grads[input._id]:
                        grads[input._id][devid] = list()
                    if grad in grads[input._id][devid]:
                        raise RuntimeError(
                            "Find two same gradient (not expected). "
                            "This is usually due to replicated node assigned to same device. "
                            f"\nCheck node:\n\t{fnode}"
                        )
                    grads[input._id][devid].append(grad)
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
        for ftensor in graph.full_tensors():
            # backward will gen in forward
            if ftensor.is_param() or ftensor.is_grad():
                continue
            adapters = IRAdapterGener.gen_fulltensor(ftensor)
            if len(adapters) == 0:
                continue
            for fadapter in adapters:
                # insert forward adapter
                idx = min([graph.nodes().index(c) for c in ftensor.consumers])
                graph._nodes.insert(idx, fadapter)
                # insert backward adapter
                grad: Optional[IRFullTensor] = ftensor.grad
                if grad is not None: 
                    badapter: IRAdapter = fadapter.mirror
                    idx = min([graph.nodes().index(c) for c in grad.consumers])
                    graph._nodes.insert(idx, badapter)
        return graph

    @staticmethod
    def gen_fulltensor(ftensor: IRFullTensor) -> List[IRAdapter]:
        # print(f'analyzing ftensor: {ftensor}')
        # print(f'ptensors: {ftensor.ptensors}')
        # print(f'ctensors: {ftensor.ctensors}')
        if len(ftensor.consumers) == 0:
            return []
        pdevs = set()
        for pnode in ftensor.producers:
            pdevs.update(pnode.device)
        cdevs = set()
        for cnode in ftensor.consumers:
            cdevs.update(cnode.device)
        # sharing devices
        if pdevs == cdevs:
            return IRAdapterGener.gen_gridlayout(ftensor)
        # no-sharing devices
        # elif len(pdevs.intersection(cdevs)) == 0:
        #     print(f'detect no intersection')
        #     return []
        # general cases
        warnings.warn('The adapter is generated using inefficient P2P send/recv')
        fprims, bprims = [], []
        for subtensor in ftensor.ctensors:
            fprims += IRAdapterGener.gen_subtensor(subtensor)
        fadapter = IRAdapter(ftensor.ptensors, ftensor.ctensors)
        fadapter.prims = fprims
        # print(fadapter.extra_repr())
        grad: IRFullTensor = ftensor.grad
        if grad is not None:
            for subtensor in grad.ctensors:
                bprims += IRAdapterGener.gen_subtensor(subtensor)
            badapter = IRAdapter(grad.ptensors, grad.ctensors)
            badapter.prims = bprims
            # print(badapter.extra_repr())
            IRCell.make_pair(fadapter, badapter)
        if len(fprims) == 0 and len(bprims) == 0:
            return []
        return [fadapter]

    @staticmethod
    def gen_gridlayout(ftensor: IRFullTensor) -> List[IRAdapter]:
        """
        Generate adapters for connecting producer with consumer with
        shared devices for forward and backward.

        ftensor: IRFullTensor: forward full tensor.
        """
        # producer grid layout
        ilayout = GridLayout.togrid(ftensor, ftensor.ptensors)
        # reorder ctensors to match with ptensors
        devs = [ptensor.device for ptensor in ilayout.mat.flatten()]
        ctensors = [None] * len(devs)
        for ctensor in ftensor.ctensors:
            idx = devs.index(ctensor.device)
            assert ctensors[idx] is None, "same device of different tensors"
            ctensors[idx] = ctensor
        # consumer grid layout
        olayout = GridLayout.togrid(ftensor, ctensors)
        # print(f'forward full tensor: {ftensor}\n producer: {ilayout}, consumer: {olayout}')
        # find path
        paths, fprims = ilayout.path(olayout, auto_replace=True)

        # re-assign the operator if miss-ordered
        names, from_dev, to_dev = [], [], []
        reorder : Dict[str, Tuple[int, int]] = dict()
        for itensor, otensor in zip(paths[-1].mat.flatten(), olayout.mat.flatten()):
            assert len(itensor.device) == 1 and len(otensor.device) == 1, \
                "Expect tensor only has one device. Report this as a bug"
            if itensor.device != otensor.device:
                inode, onode = itensor._cell, otensor._cell
                names.append(f'{onode.name}{onode._id}')
                from_dev.append(onode.device[0])
                to_dev.append(inode.device[0])
                onode.device = inode.device
                if onode.mirror is not None:
                    onode.mirror.device = inode.device
        if len(reorder) > 0:
            warnings.warn(f'UserWarning: a better device placement is found and set for op {names}: {from_dev} -> {to_dev}')

        # print('find path:')
        # for path in paths: print(path)
        # print('comm prims:')
        # for prim in fprims: print(prim)
        fadapter = IRAdapter(ftensor.ptensors, ftensor.ctensors)
        fadapter.prims = fprims

        # generate backward
        grad: IRFullTensor = ftensor.grad
        bprims = []
        if grad is not None:
            # reorder ptensors to match with forward
            ptensors = [None] * len(devs)
            for ptensor in grad.ptensors:
                idx = devs.index(ptensor.device)
                assert ptensors[idx] is None, "same device of different tensors"
                ptensors[idx] = ptensor
            ilayout = GridLayout.togrid(grad, ptensors)
            olayout = GridLayout.togrid(grad, grad.ctensors)
            # print(f'backward full tensor: {grad}\n producer: {ilayout}, consumer: {olayout}')
            paths, bprims = ilayout.path(olayout)
            # check the device order
            for itensor, otensor in zip(paths[-1].mat.flatten(), olayout.mat.flatten()):
                assert len(itensor.device) == len(otensor.device), "backward device not match"
            # print('find path:')
            # for path in paths: print(path)
            # print('comm prims')
            # for prim in bprims: print(prim)
            badapter = IRAdapter(grad.ptensors, grad.ctensors)
            badapter.prims = bprims
            IRCell.make_pair(fadapter, badapter)
        if len(fprims) == 0 and len(bprims) == 0:
            return []
        # print('=====')
        return [fadapter]

    @staticmethod
    def gen_subtensor(subtensor: IRSubTensor) -> List[IRAdapterPrim]:
        """
        Generate communication prims for a sub-tensor.
        The subtensor should be a IRSubTensor of consumer.
        
        The generation takes three stages: select, move, merge
        """
        ftensor = subtensor.parent
        # category to local tensor and remote tensor
        local = [t for t in ftensor.ptensors if t.device == subtensor.device]
        remote = [t for t in ftensor.ptensors if t.device != subtensor.device]
        prims = []

        # ==== select ==== #
        intersections = []
        # check local
        for tensor in local:
            common = tensor.common(subtensor)
            if tensor == subtensor:
                return prims
            elif common == subtensor:
                indmap = []
                for islicer, oslicer in zip(tensor.indmap.get(), common.indmap.get()):
                    start = oslicer.start - islicer.start
                    stop = start + oslicer.stop - oslicer.start
                    indmap.append(slice(start, stop, 1))
                valmap = ValueMap(0, 1)
                common.attach_cell(subtensor._cell)
                prims.append(SelectPrim(tensor, indmap, valmap, common))
                return prims
        # check local + remote
        if len(intersections) == 0:
            for itensor in local+remote:
                if not itensor.overlap(subtensor):
                    continue
                common = itensor.common(subtensor)
                common.attach_cell(itensor._cell)
                print(f'get common: {common.extra_repr()}')
                intersections.append(common)
                if common == itensor:
                    continue
                indmap = []
                for islicer, oslicer in zip(itensor.indmap.get(), common.indmap.get()):
                    start = oslicer.start - islicer.start
                    stop = start + oslicer.stop - oslicer.start
                    indmap.append(slice(start, stop, 1))
                assert itensor.valmap == common.valmap or itensor.valmap == ValueMap(0,1), \
                    f"Not supported value select: {itensor.valmap} -> {common.valmap}"
                valmap = ValueMap(0, 1)
                prims.append(SelectPrim(itensor, indmap, valmap, common))
                # TODO: check union == subtensor
                if common == subtensor:
                    break
        print(intersections)
        # ====== move ===== #
        tmoved = []
        for tensor in intersections:
            assert len(tensor.device) == 1 and len(subtensor.device) == 1, "Expected only one device."
            mtensor = tensor
            if tensor.device != subtensor.device:
                mtensor = copy.copy(tensor)
                mtensor.attach_cell(subtensor._cell)
                prims.append(MovePrim(tensor, mtensor))
            tmoved.append(mtensor)

        # ===== merge ===== #
        remain_tensors: List[IRSubTensor] = copy.copy(tmoved)
        if subtensor in remain_tensors:
            return prims
        out = None
        while out != subtensor:
            out, merged = None, False
            for idx1 in range(len(remain_tensors) - 1):
                for idx2 in range(idx1, len(remain_tensors)):
                    t1, t2 = remain_tensors[idx1], remain_tensors[idx2]
                    # check reducable
                    if t1.indmap == t2.indmap and t1.valmap.chunk_num == t2.valmap.chunk_num:
                        vid1, vid2 = t1.valmap.idx, t2.valmap.idx
                        # sum e.g., 0,1 but not 1,2
                        if min(vid1, vid2) % 2 == 0 and abs(vid1-vid2) == 1:
                            vid = min(vid1, vid2) // 2
                            valmap = ValueMap(vid, t1.valmap.chunk_num // 2)
                            out = subtensor.parent.select(t1.indmap, valmap, t1.shape)
                            out.attach_cell(subtensor._cell)
                            prims.append(SumPrim([t1, t2], out))
                            merged = True
                            break
                    # try merge dimension
                    elif t1.valmap == t2.valmap:
                        cat_dim: Dict[int, List[IRSubTensor]] = dict()
                        indmap = []
                        for dim, (s1, s2) in enumerate(zip(t1.indmap.get(), t2.indmap.get())):
                            if s1 != s2:
                                if min(s1.stop, s2.stop) == max(s1.start, s2.start):
                                    if s1.start < s2.start:
                                        cat_dim[dim] = [t1, t2]
                                    else:
                                        cat_dim[dim] = [t1, t2]
                                    indmap.append(slice(min(s1.start, s2.start), max(s1.stop, s2.stop), 1))
                                else:
                                    cat_dim[dim] = None
                                    indmap.append(None)
                            else:
                                indmap.append(s1)
                        if None in indmap:
                            continue
                        indmap = IndexMap(tuple(indmap))
                        valmap = t1.valmap
                        out = t1.parent.select(indmap, valmap, indmap.shape)
                        out.attach_cell(subtensor._cell)
                        cdim = list(cat_dim.keys())[0]
                        prims.append(MergeDimPrim(cat_dim[cdim], out, cdim))
                        merged = True
                        break
                if merged:
                    remain_tensors.remove(t1)
                    remain_tensors.remove(t2)
                    remain_tensors.append(out)
                    break
            if out is None:
                ptensors = '\n\t'.join(t.extra_repr() for t in ftensor.ptensors)
                raise RuntimeError(
                    f"Fail to build adapter.\n"
                    f"FullTensor:{ftensor}\n"
                    f"Producers:\n\t{ptensors}\n"
                    f"SubTensor:\n\t{subtensor.extra_repr()}"
                )
        return prims
                    