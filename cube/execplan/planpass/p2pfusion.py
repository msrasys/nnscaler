from typing import List, Dict
from cube.execplan import ExectuionPlan
from cube.graph.tensor import ValueMap
from cube.ir.cten import IRTensor
from cube.schedule.su import SUType, ScheduleUnit
from cube.execplan.planpass.planpass import PlanPass

from cube.schedule.adapter.collectives import IRCollType, IRCollectives


class P2PFusion(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        # dict[pid][devid] = list of sub_tensors
        fous, fins = P2PFusion.collect_tensors(
            execplan, [SUType.Dataloader, SUType.Forward]
        )
        bous, bins = P2PFusion.collect_tensors(
            execplan, [SUType.Backward]
        )
        # debug
        # print('=====> forward')
        # for pid in fins:
        #     if pid not in fous:
        #         continue
        #     if P2PFusion.have_comm(fous[pid], fins[pid]):
        #         print(f'=> parent tensor id: {pid}')
        #         for devid in fous[pid]:
        #             print(f'  ==> device: {devid}')
        #             for val in fous[pid][devid]:
        #                 print(f'  o:', val)
        #         for devid in fins[pid]:
        #             print(f'  ==> device: {devid}')
        #             for val in fins[pid][devid]:
        #                 print(f'  i:', val)

        matchers = [
            P2PFusion.match_allreduce,
            P2PFusion.match_allgather,
            P2PFusion.match_reducescatter,
            P2PFusion.match_broadcast,
        ]
        for ous, ins in zip([fous, bous], [fins, bins]):
            for pid in ins:
                if pid not in ous:
                    continue
                tous, tins = ous[pid], ins[pid]
                # if they are on the single device, matching is skipped
                if len(tous) == 1 and set(tous.keys()) == set(tins.keys()):
                    continue
                if P2PFusion.have_comm(tous, tins):
                    colls : List[ScheduleUnit] = None
                    for matcher in matchers:
                        colls = matcher(tous, tins)
                        if colls:
                            break
                    if colls is not None:
                        P2PFusion.add_collectives(execplan, colls)
        return execplan

    @staticmethod
    def collect_tensors(execplan: ExectuionPlan, stypes: List[SUType]):
        # dict[pid][devid] = list of sub_tensors
        ous = dict()
        ins = dict()
        for devid in execplan.devices():
            dev_seq = execplan.sequence(devid)
            for su in dev_seq:
                if su.stype in stypes:
                    for val in su.inputs():
                        # FIXME: remove parameter constraints
                        if isinstance(val, IRTensor) and not val.is_param():
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
    def add_collectives(execplan: ExectuionPlan, coll_sus: List[ScheduleUnit]):
        for coll_su in coll_sus:
            print(f'inserting Collective SU: {coll_su.name}: {coll_su}')
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
                # merge
                if su.stype == SUType.Transform and len(su.inputs()) > 1:
                    merge_out = su.outputs(0)
                    if merge_out in coll_su.outputs():
                        assert len(coll_su.outputs()) == 1
                        execplan.at(devid)[idx] = coll_su
                        break
            else:
                raise RuntimeError("Cannot find a send P2P")
        # all the send, recv of the inputs will be removed in ranks
        for coll_su in coll_sus:
            ranks = coll_su.nodes(0).ranks
            for input in coll_su.inputs():
                for rank in ranks:
                    for su in execplan.sequence(rank):
                        # remove send / recv
                        if su.stype == SUType.P2P and input in (su.inputs() + su.outputs()):
                            execplan.at(rank).remove(su)
                        # remove merge if coll generate merge results
                        if su.stype == SUType.Transform and len(su.inputs()) > 1:
                            merge_out = su.outputs(0)
                            if merge_out in coll_su.outputs():
                                assert len(coll_su.outputs()) == 1
                                execplan.at(rank).remove(su)

    @staticmethod
    def transmission(tensor_ous, in_tensor) -> Dict[int, List[IRTensor]]:
        trans_tensors = dict()
        for devid in tensor_ous:
            for out in tensor_ous[devid]:
                if in_tensor.overlap(out):
                    if devid not in trans_tensors:
                        trans_tensors[devid] = list()
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
        allreduce_sus = list()
        # {tensor_id: [device_id]}
        in_devices: Dict[int, List[int]] = dict()
        # {tensor_id: [tensors]
        in_tensors: Dict[int, List[IRTensor]] = dict()
        for devid in tins:
            for in_tensor in tins[devid]:
                if in_tensor.val_map != ValueMap(0, 1):
                    continue
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
            # check out tensor and reduce in tensor devices are the same set
            if out_devices == set(in_devices[tid]):
                # multiple transmission FIXME: remove redundancy
                if not all([len(out_tensors[odev]) == 1 for odev in out_devices]):
                    continue
                # check same indice map and no overlap value map
                unique_indices = list()
                for odev in out_tensors:
                    indices = out_tensors[odev][0].indices
                    if indices not in unique_indices:
                        unique_indices.append(indices)
                if len(unique_indices) != 1:
                    continue
                # check no overlap valmaps
                all_valmaps = list()
                overlap = False
                for odev in out_tensors:
                    valmap = out_tensors[odev][0].val_map
                    for pre_valmp in all_valmaps:
                        overlap = pre_valmp.overlap(valmap)
                    all_valmaps.append(valmap)
                if overlap:
                    continue

                ranks = list(out_tensors.keys())
                inputs = [[out_tensors[rank][0]] for rank in ranks]

                for input, rank in zip(inputs, ranks):
                    for in_tensor in in_tensors[tid]:
                        if in_tensor.device[0] == rank:
                            outputs = [in_tensor]
                            break
                    else:
                        raise RuntimeError("Internal Error")
                    op = IRCollectives(input, outputs, ranks, IRCollType.AllReduce)
                    su = ScheduleUnit([op], SUType.Coll, name='allreduce')
                    su.device = rank
                    allreduce_sus.append(su)

                # print('>> find allreduce pattern:')
                # print(f'device group: {ranks}')
                # for input in inputs:
                #     print(f'src: {input}')
                # for output in outputs:
                #     print(f'dst: {output}')

        if len(allreduce_sus) == 0:
            return None
        else:
            return allreduce_sus

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
                inputs = [out_tensors[rank][0] for rank in ranks]

                for input, rank in zip(inputs, ranks):
                    outputs = [t for t in inputs if t != input]
                    op = IRCollectives([input], outputs, ranks, IRCollType.AllGather)
                    su = ScheduleUnit([op], SUType.Coll, name='allgather')
                    su.device = rank
                    allgather_sus.append(su)

                # print('>> find allgather pattern:')
                # print(f'device group: {ranks}')
                # for input in inputs:
                #     print(f'src: {input}')
                # for output in outputs:
                #     print(f'dst: {output}')

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
                if in_tensor.val_map != ValueMap(0, 1):
                    continue
                if tid not in in_devices:
                    in_devices[tid] = list()
                    in_tensors[tid] = list()
                in_devices[tid].append(devid)
                in_tensors[tid].append(in_tensor)
        # {in_tensor_id: [reduce_tensor device]}
        reduce_out_devices = dict()
        # {in_tensor_id: [reduce out tensors]}
        reduce_out_tensors = dict()
        for tid in in_devices:
            # P2P transmission
            if len(in_devices[tid]) != 1:
                continue
            in_tensor = in_tensors[tid][0]
            out_tensors = P2PFusion.transmission(tous, in_tensor)

            is_reduce = True
            for devid in out_tensors:
                # multiple transmission FIXME: remove redundancy
                if not all([len(out_tensors[odev]) == 1 for odev in out_tensors]):
                    continue
                if out_tensors[devid][0].val_map == ValueMap(0, 1):
                    is_reduce = False
                    break
                if out_tensors[devid][0].indices != in_tensor.indices:
                    is_reduce = False
                    break
            if is_reduce:
                reduce_out_devices[tid] = list()
                reduce_out_tensors[tid] = list()
                for devid in out_tensors:
                    reduce_out_devices[tid].append(devid)
                    reduce_out_tensors[tid].append(out_tensors[devid][0])
        # reverse reduce_devices {tuple(devices): [in_tensors]}
        reduce_tensors = dict()
        for tid in reduce_out_devices:
            devices = tuple(set(reduce_out_devices[tid]))
            if devices not in reduce_tensors:
                reduce_tensors[devices] = list()
            reduce_tensors[devices].append(in_tensors[tid][0])
        # check conditions
        for ranks in reduce_tensors:
            reduce_in_tensors = reduce_tensors[ranks]
            # reduce-scatter requires tensor num to be equal of num devs
            if len(reduce_in_tensors) != len(ranks):
                continue
            # reduce in tensors should place on different devices
            devices = [t.device[0] for t in reduce_in_tensors]
            if set(devices) != set(ranks):
                continue

            # satisfied! set up inputs, outputs and ranks
            ranks = list(ranks)
            ranks.sort()

            device_inputs = [None] * len(ranks)
            for in_tensor in reduce_in_tensors:
                out_tensors = reduce_out_tensors[in_tensor._id]
                out_devs = [t.device[0] for t in out_tensors]
                inputs = [
                    out_tensors[out_devs.index(odev)] for odev in ranks
                ]
                ridx = ranks.index(in_tensor.device[0])
                device_inputs[ridx] = inputs
            for in_tensor in reduce_in_tensors:
                rank = in_tensor.device[0]
                outputs = [in_tensor]
                inputs = [inputs[ranks.index(rank)] for inputs in device_inputs]
                op = IRCollectives(inputs, outputs, ranks, IRCollType.ReduceScatter)
                su = ScheduleUnit([op], SUType.Coll, name='reducescatter')
                su.device = rank
                rs_sus.append(su)

            # print('>> find reduce-scatter pattern:')
            # print(f'device group: {ranks}')
            # for output in reduce_in_tensors:
            #     tid = output._id
            #     for input in reduce_out_tensors[tid]:
            #         print(f'src: {input}')
            #     print(f'dst: {output}')

        if len(rs_sus) == 0:
            return None
        else:
            return rs_sus


    @staticmethod
    def match_broadcast(tous, tins):
        """
        Broadcast semantic:

        The root device send the its tensor to all the devices
        """
        broadcast_sus = list()
        # {tensor_id: [device_id]}
        in_devices: Dict[int, List[int]] = dict()
        # {tensor_id: [tensors]
        in_tensors: Dict[int, List[IRTensor]] = dict()
        for devid in tins:
            for in_tensor in tins[devid]:
                tid = in_tensor._id
                if in_tensor.val_map != ValueMap(0, 1):
                    continue
                if tid not in in_devices:
                    in_devices[tid] = list()
                    in_tensors[tid] = list()
                in_devices[tid].append(devid)
                in_tensors[tid].append(in_tensor)
        
        for tid in in_devices:
            # P2P transmission
            if len(in_devices[tid]) <= 2:
                continue
            in_tensor = in_tensors[tid][0]
            out_tensors = P2PFusion.transmission(tous, in_tensor)
            # multiple transmission FIXME: remove redundancy
            if len(out_tensors.keys()) != 1:
                continue
            # multiple transmission FIXME: remove redundancy
            if len(out_tensors[list(out_tensors.keys())[0]]) != 1:
                continue
            root_tensor = out_tensors[list(out_tensors.keys())[0]][0]
            is_equal = True
            for in_tensor in in_tensors[tid]:
                if in_tensor != root_tensor:
                    is_equal = False
                    break
            if not is_equal:
                continue
            ranks = [root_tensor.device[0]]
            inputs = [[root_tensor],]
            outputs = [[],]
            for output in in_tensors[tid]:
                devid = output.device[0]
                if devid in ranks:
                    continue
                ranks.append(devid)
                outputs.append([output])
                inputs.append([])
            for input, output, rank in zip(inputs, outputs, ranks):
                op = IRCollectives(input, output, ranks, IRCollType.Broadcast)
                su = ScheduleUnit([op], SUType.Coll, name='broadcast')
                su.device = rank
                broadcast_sus.append(su)

                print('>> find broadcast pattern:')
                print(f'device group: {ranks}')
                print(su)

        if len(broadcast_sus) == 0:
            return None


        else:
            return broadcast_sus
