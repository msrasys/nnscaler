from typing import List, Dict
from cube.execplan import ExectuionPlan
from cube.ir.cten import IRTensor
from cube.schedule.su import SUType, ScheduleUnit
from cube.execplan.planpass.planpass import PlanPass

from cube.schedule.adapter.collectives import IRCollType, IRCollectives


class P2PFusion(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        # dict[pid][devid] = list of sub_tensors
        fous, fins = P2PFusion.collect_tensors(execplan, SUType.Forward)
        bous, bins = P2PFusion.collect_tensors(execplan, SUType.Backward)
        # debug
        print('=====> forward')
        for pid in fins:
            if pid not in fous:
                continue
            if P2PFusion.have_comm(fous[pid], fins[pid]):
                print(f'=> parent tensor id: {pid}')
                for devid in fous[pid]:
                    print(f'  ==> device: {devid}')
                    for val in fous[pid][devid]:
                        print(f'  o:', val)
                for devid in fins[pid]:
                    print(f'  ==> device: {devid}')
                    for val in fins[pid][devid]:
                        print(f'  i:', val)
        matchers = [
            P2PFusion.match_allreduce, P2PFusion.match_allgather,
            P2PFusion.match_reducescatter, P2PFusion.match_broadcast,
        ]
        for pid in fins:
            if pid not in fous:
                continue
            tous, tins = fous[pid], fins[pid]
            if P2PFusion.have_comm(tous, tins):
                colls : List[ScheduleUnit] = None
                for matcher in matchers:
                    colls = matcher(tous, tins)
                    if colls:
                        break
                if colls is not None:
                    for coll_su in colls:
                        P2PFusion.add_collective(execplan, coll_su)
        return execplan

    @staticmethod
    def collect_tensors(execplan: ExectuionPlan, stype: SUType):
        # dict[pid][devid] = list of sub_tensors
        ous = dict()
        ins = dict()
        for devid in execplan.devices():
            dev_seq = execplan.sequence(devid)
            for su in dev_seq:
                if su.stype == stype:
                    for val in su.inputs():
                        if isinstance(val, IRTensor):
                            pid = val.parent._id
                            if pid not in ins:
                                ins[pid] = dict()
                            if devid not in ins[pid]:
                                ins[pid][devid] = list()
                            # TODO: may have redundancy
                            ins[pid][devid].append(val)
                    for idx, val in enumerate(su.outputs()):
                        if isinstance(val, IRTensor):
                            pid = val.parent._id
                            if pid not in ous:
                                ous[pid] = dict()
                            if devid not in ous[pid]:
                                ous[pid][devid] = list()
                            select_su = su.select_adapters(idx)
                            if select_su:
                                for out in select_su.outputs():
                                    # TODO: may have redundancy
                                    ous[pid][devid].append(out)
                            else:
                                # TODO: may have redundancy
                                ous[pid][devid].append(val)
        return ous, ins

    @staticmethod
    def have_comm(tensor_ous, tensor_ins):
        """
        Check if they don't have communications
        """
        for devid in tensor_ins:
            if devid not in tensor_ous:
                return True
            # no transmission
            if input in tensor_ous[devid]:
                continue
            # have transmission
            else:
                return True
        return False

    @staticmethod
    def add_collective(execplan: ExectuionPlan, coll_su: ScheduleUnit):
        print(f'inserting Collective SU: {coll_su}')
        # find insert place: the first send
        devid = coll_su.device[0]
        ranks = coll_su.nodes(0).ranks
        for idx, su in enumerate(execplan.sequence(devid)):
            # send or recv
            if su.stype == SUType.P2P:
                sr_tensor = (su.inputs() + su.outputs())[0]
                if sr_tensor in coll_su.inputs() + coll_su.outputs():
                    execplan.at(devid)[idx] = coll_su
                    break

        else:
            raise RuntimeError("Cannot find a send P2P")
        # all the send, recv of the inputs will be removed in ranks
        for input in coll_su.inputs():
            for rank in ranks:
                for su in execplan.sequence(rank):
                    # remove send / recv
                    if su.stype == SUType.P2P and input in (su.inputs() + su.outputs()):
                        execplan.at(rank).remove(su)

    @staticmethod
    def transmission(tensor_ous, in_tensor) -> Dict[int, List[IRTensor]]:
        trans_tensors = dict()
        for devid in tensor_ous:
            for out in tensor_ous[devid]:
                if devid not in trans_tensors:
                    trans_tensors[devid] = list()
                if in_tensor.overlap(out):
                    trans_tensors[devid].append(out)
        return trans_tensors

    @staticmethod
    def match_allreduce(tous, tins):
        """
        Allreduce semantic:

        Each device holds a recvs same spatial tensor from all device and 
        sends to all device.
        The recved tensors are summed into one
        """
        return None

    @staticmethod
    def match_allgather(tous, tins):
        """
        Allgather semantic:

        Each device performs same transformation merge.

        !!Note: Each input in merge su can be paired with a <send, recv> pair, find
                them and remove!! Fuse merge, send, recv into one merge!!
        """
        allgather_sus = list()
        # {tensor_id: [device_id]}
        in_devices: Dict[int, List[int]] = dict()
        # {tensor_id: [tensors]
        in_tensors: Dict[int, List[IRTensor]] = dict()
        for devid in tins:
            for in_tensor in tins[devid]:
                tid = in_tensor._id
                if tid not in in_devices:
                    in_devices[tid] = list()
                    in_tensors[tid] = list()
                in_devices[tid].append(devid)
                in_tensors[tid].append(in_tensor)
        for tid in in_devices:
            # P2P transmission
            if len(in_devices[tid]) <= 1:
                continue
            in_tensor = in_tensors[tid][0]
            # {rank: [IRTensor]}}
            out_tensors = P2PFusion.transmission(tous, in_tensor)
            out_devices = set(out_tensors.keys())
            if out_devices == set(in_devices[tid]):
                # multiple transmission FIXME: remove redundancy
                if not all([len(out_tensors[odev]) == 1 for odev in out_devices]):
                    continue
                # check same value map and no overlap indices
                unique_valmaps = list()
                for odev in out_tensors:
                    valmap = out_tensors[odev][0].val_map
                    if valmap not in unique_valmaps:
                        unique_valmaps.append(valmap)
                if len(unique_valmaps) != 1:
                    continue
                # check no overlap indices
                all_indices = list()
                overlap = False
                for odev in out_tensors:
                    indices = out_tensors[odev][0].indices
                    for pre_indices in all_indices:
                        overlap = pre_indices.overlap(indices)
                    all_indices.append(indices)
                if overlap:
                    continue

                ranks = list(out_tensors.keys())
                inputs = [[out_tensors[rank][0]] for rank in ranks]

                for input, rank in zip(inputs, ranks):
                    outputs = [
                        input[0] for idx, input in enumerate(inputs) if idx != rank
                    ]
                    op = IRCollectives(input, outputs, ranks, IRCollType.AllGather)
                    su = ScheduleUnit([op], SUType.Coll, name='allgather')
                    su.device = rank
                    allgather_sus.append(su)

                print('>> find allgather pattern:')
                print(f'device group: {ranks}')
                for input in inputs:
                    print(f'src: {input}')
                for output in outputs:
                    print(f'dst: {output}')

        if len(allgather_sus) == 0:
            return None
        else:
            return allgather_sus

    @staticmethod
    def match_reducescatter(tous, tins):
        """
        ReduceScatter semantic:

        Each device performs same
        """
        rs_sus = list()
        # {tensor_id: [device_id]}
        in_devices: Dict[int, List[int]] = dict()
        # {tensor_id: [tensors]
        in_tensors: Dict[int, List[IRTensor]] = dict()
        for devid in tins:
            for in_tensor in tins[devid]:
                tid = in_tensor._id
                if tid not in in_devices:
                    in_devices[tid] = list()
                    in_tensors[tid] = list()
                in_devices[tid].append(devid)
                in_tensors[tid].append(in_tensor)

    @staticmethod
    def match_broadcast(tous, tins):
        """
        Broadcast semantic:

        The root device send the its tensor to all the devices
        """
        return None