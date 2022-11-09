"""
Concurrent producer / consumer Adapter Generator
"""
from typing import List, Optional, Dict, Tuple
import copy
import numpy as np

from cube.ir.tensor import IRFullTensor, IRSubTensor, IndexMap, ValueMap
from cube.ir.adapter.prim import IRAdapterPrim
from cube.ir.adapter import IRAdapter
from cube.ir.adapter.prim import SelectPrim, MovePrim, SumPrim, MergeDimPrim
from cube.ir.adapter.prim import BroadcastPrim

from cube.graph.gener.layout import GridLayout, PathFinder


class ConcurrentGener:

    @staticmethod
    def gen(fptensors: List[IRSubTensor], fctensors: List[IRSubTensor], 
            bptensors: List[IRSubTensor], bctensors: List[IRSubTensor]) -> Optional[IRAdapter]:
        """
        Generate forward adapter and backward adapter

        @param fptensors List[IRSubTensor]: forward producer tensors
        @param fctensors List[IRSubTensor]: forward consumer tensors
        @param bptensors List[IRSubTensor]: backward producer tensors
        @param bctensors List[IRSubTensor]: backward consumer tensors

        @return fadapter Optional[IRAdapter]: forward adapter
            None indicate no adapter required.
        """
        pdevs = tuple(t.device[0] for t in fptensors)
        cdevs = tuple(t.device[0] for t in fctensors)

        fadapter: IRAdapter = None

        # case 1: sharing device (in-shard)
        inshard = (set(pdevs) == set(cdevs)) and (len(fptensors) == len(fctensors)) and (len(pdevs) == len(fptensors))
        if inshard and len(pdevs) > 1:
            # fadapter = ConcurrentGener.gen_in_shard(fptensors, fctensors, bptensors, bctensors, allow_reorder=True)
            try:
                fadapter = ConcurrentGener.gen_in_shard(fptensors, fctensors, bptensors, bctensors, allow_reorder=True)
            except Exception as e:
                fadapter = None
                print(
                    f"full tensor: {fptensors[0].parent} cannot use intra-transformation generation.\n"
                    f"Reason: {str(e)}\n"
                    f"Switch to general P2P communication."
                )

        # Case 2: sperating device (cross-shard)
        if len(set(pdevs).intersection(cdevs)) == 0:
            # fadapter = ConcurrentGener.gen_cross_shard(fptensors, fctensors, bptensors, bctensors)
            try:
                fadapter = ConcurrentGener.gen_cross_shard(fptensors, fctensors, bptensors, bctensors)
            except Exception as e:
                fadapter = None
                print(
                    f"full tensor: {fptensors[0].parent} cannot use inter-transformation generation.\n"
                    f"Reason: {str(e)}\n"
                    f"Switch to general P2P communication."
                )

        # Case 3: General cases
        # warnings.warn('The adapter is generated using P2P communication')
        if fadapter is None:
            fadapter = ConcurrentGener.gen_general(fptensors, fctensors, bptensors, bctensors)
        
        if set(pdevs) == set(cdevs) and fadapter.mirror is not None:
            fadapter.differentiable = True
            fadapter.mirror.differentiable = True

        return fadapter

    @staticmethod
    def gen_in_shard(fptensors: List[IRSubTensor], fctensors: List[IRSubTensor], 
                     bptensors: List[IRSubTensor], bctensors: List[IRSubTensor],
                     allow_reorder=False):
        ftensor = fptensors[0].parent
        # producer grid layout
        ilayout = GridLayout.togrid(ftensor, fptensors)
        # reorder ctensors to match with ptensors
        devs = [ptensor.device for ptensor in ilayout.mat.flatten()]
        ctensors = [None] * len(devs)
        for ctensor in fctensors:
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

        fadapter = IRAdapter(fptensors, fctensors)
        fadapter.prims = fprims

        # generate backward
        grad: IRFullTensor = ftensor.grad
        bprims = []
        if grad is not None and (len(bptensors) != 0 or len(bctensors) != 0):
            # reorder ptensors to match with forward
            ptensors = [None] * len(devs)
            for bptensor in bptensors:
                idx = devs.index(bptensor.device)
                assert ptensors[idx] is None, "same device of different tensors"
                ptensors[idx] = bptensor
            ilayout = GridLayout.togrid(grad, ptensors)
            olayout = GridLayout.togrid(grad, bctensors)
            paths, bprims = ilayout.path(olayout)
            # check the device order
            for itensor, otensor in zip(paths[-1].mat.flatten(), olayout.mat.flatten()):
                assert len(itensor.device) == len(otensor.device), "backward device not match"
            badapter = IRAdapter(bptensors, bctensors)
            badapter.prims = bprims
            IRAdapter.make_pair(fadapter, badapter)

        return fadapter

    @staticmethod
    def gen_cross_shard(fptensors: List[IRSubTensor], fctensors: List[IRSubTensor], 
                        bptensors: List[IRSubTensor], bctensors: List[IRSubTensor],) -> IRAdapter:
        """
        This assumes ptensors and ctensors can be represented by RVD layout.
        
        pdevices: devices of ptensors
        cdevices: devices of ctensors

        @param fptensors List[IRSubTensor]: produced tensors
        @param fctensors List[IRSubTensor]: consumed tensors
        @param bptensors List[IRSubTensor]: produced tensors
        @param bctensors List[IRSubTensor]: consumed tensors

        @return fadapter IRAdapter
        """
        ftensor = fptensors[0].parent
        ilayout = GridLayout.togrid(ftensor, fptensors)
        olayout = GridLayout.togrid(ftensor, fctensors)
        fpaths, fprims = PathFinder.inter_path(ftensor, ilayout, olayout)
        fadapter = IRAdapter(fptensors, fctensors)
        fadapter.prims = fprims

        grad: IRFullTensor = ftensor.grad
        if grad is not None and (len(bptensors) != 0 or len(bctensors) != 0):
            ilayout = GridLayout.togrid(grad, bptensors)
            olayout = GridLayout.togrid(grad, bctensors)
            bpaths, bprims = PathFinder.inter_path(grad, ilayout, olayout)
            badapter = IRAdapter(bptensors, bctensors)
            badapter.prims = bprims
            IRAdapter.make_pair(fadapter, badapter)
        return fadapter

    @staticmethod
    def gen_general(fptensors: List[IRSubTensor], fctensors: List[IRSubTensor],
                    bptensors: List[IRSubTensor], bctensors: List[IRSubTensor]) -> IRAdapter:
        """
        A general way to generate adapter.
        
        @param ftensor IRFullTensor
        @return adapter IRAdapter
        """
        fprims = []
        fpdevs = set(t.device[0] for t in fptensors)
        fcomm_workload = {t.device[0]: 0 for t in fptensors}
        # first try collectives
        ret, prims = ConcurrentGener.gen_subtensor_coll(fctensors, fptensors, fcomm_workload)
        if ret:
            fprims += prims
        # otherwise use general p2p send recv
        else:
            for ctensor in fctensors:
                fprims += ConcurrentGener.gen_subtensor(ctensor, fptensors, fcomm_workload)
        fadapter = IRAdapter(fptensors,fctensors)
        fadapter.prims = fprims
        # backward
        if len(bptensors) > 0 and len(bctensors) > 0:
            bprims = []
            bcomm_workload = {t.device[0]: 0 for t in bptensors}
            # first try collectives
            ret, prims = ConcurrentGener.gen_subtensor_coll(bctensors, bptensors, bcomm_workload)
            if ret:
                bprims += prims
            # otherwise use general p2p send recv
            else:
                for cgrad in bctensors:
                    bprims += ConcurrentGener.gen_subtensor(cgrad, bptensors, bcomm_workload)
            badapter = IRAdapter(bptensors, bctensors)
            badapter.prims = bprims
            IRAdapter.make_pair(fadapter, badapter)
        return fadapter

    @staticmethod
    def gen_subtensor_coll(ctensors: List[IRSubTensor], ptensors: List[IRSubTensor], workload: Dict[int, int]) -> Tuple[bool, List[IRAdapterPrim]]:
        """
        Generate communication primitives for a tensor using collectives of
        broadcast, [reduce, gather and scatter]. => [...] Not supported yet.

        @param ctensors List[IRSubTensor]: the consumed tensors as destination.
        @param ptensors List[IRSubTensor]: the produced tensors as source

        @return success bool: whether succeed in generate collective
        @return prims List[IRAdapterPrim]: the primitives for adapter
        """
        ret = False
        prims = []
        # broadcast
        if len(ptensors) == 1 and \
           len(set(ctensor.device[0] for ctensor in ctensors)) > 2 and \
           all(ptensors[0] == ctensor for ctensor in ctensors):
            dev_ctensors = []
            cdevs = set()
            for ctensor in ctensors:
                if ctensor.device[0] not in cdevs:
                    cdevs.add(ctensor.device[0])
                    dev_ctensors.append(ctensor)
            prims.append(BroadcastPrim(ptensors, dev_ctensors)) 
            ret = True
        return ret, prims

    @staticmethod
    def gen_subtensor(ctensor: IRSubTensor, ptensors: List[IRSubTensor], workload: Dict[int, int]) -> List[IRAdapterPrim]:
        """
        Generate communiction primitives for ctensor
        
        @param ctensor IRSubTensor: the consumed tensor as destination
        @param ptensors List[IRSubTensor]: the produced tensors as source

        @return prims List[IRAdapterPrim]: the primitives for adapter
        """
        # category to local tensor and remote tensor
        local = [t for t in ptensors if t.device == ctensor.device]
        # reorder remote devices: higher priority to use tensor with lower communication workload
        devices = np.array([devid for devid in workload.keys()], dtype=int)
        volume = np.array([workload[devid] for devid in workload.keys()])
        indices = np.argsort(volume)
        sorted_devices = devices[list(indices)]
        remote: List[IRSubTensor] = []
        for devid in sorted_devices:
            if devid == ctensor.device[0]: continue
            remote += [t for t in ptensors if t.device[0] == devid]

        prims = []

        # ==== select ==== #
        intersections: List[IRSubTensor] = []
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
                if itensor.valmap == common.valmap:
                    valmap = ValueMap((0, 1))
                else:
                    assert itensor.valmap == (0, 1)
                    valmap = common.valmap
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
                prims.append(MovePrim([tensor], [mtensor]))
                workload[tensor.device[0]] += tensor.nelement()
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
                raise RuntimeError(
                    f"Fail to build adapter.\n"
                    f"FullTensor:{ctensor.parent}\n"
                    f"Produced Tensors:\n\t{ptensors}\n"
                    f"Consumed Tensors:\n\t{ctensor.extra_repr()}\n"
                    f"Consumer:\n\t{ctensor.cell}\n"
                    f"Remain Tensor:\n\t{remain}"
                )
        return prims
