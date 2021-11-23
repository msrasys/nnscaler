from typing import List, Optional, Union
import copy
from cube.graph.tensor import IRSubTensor

from cube.ir.cten import IRCell, IRTensor
from cube.graph.operator import IRBpOperation
from cube.graph.operator import IRDataOperation
from cube.graph.operator import IRFwOperation
from cube.schedule.su import SUType, ScheduleUnit
from cube.schedule.adapter.comm import IRCommunication
from cube.schedule.adapter.transform import IRTensorTransform


class SUGraph(IRCell):

    def __init__(self, sus: List[ScheduleUnit]):

        if not all([isinstance(su, ScheduleUnit) for su in sus]):
            raise TypeError(
                f"Expected a list of ScheduleUnits, but got {type(sus)}"
            )

        inputs = IRCell.get_inputs(sus)
        inputs = [input for input in inputs if not input.is_param()]
        outputs = IRCell.get_outputs(sus)
        outputs = [output for output in outputs if not output.is_param()]
        super().__init__(
            name = 'SU',
            signature = 'None',
            input_length = len(inputs),
            output_length = len(outputs)
        )
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        for idx, output in enumerate(outputs):
            self.set_output(idx, output)

        self.sequence = sus
        SUGraph.reset_dependency(self.sequence)

    @property
    def nnodes(self) -> int:
        """
        Get number of nodes (int)
        """
        return len(self.sequence)

    @staticmethod
    def reset_dependency(sus: List[ScheduleUnit]):
        """
        Reset the node dataflow dependency
        """
        if not all([isinstance(su, ScheduleUnit) for su in sus]):
            raise TypeError("Expected list of schedule unit")
        adapters = [SUType.P2P, SUType.Coll, SUType.Transform]
        for su in sus:
            su.clear_predecessor()
            su.clear_successor()
        for src_idx in range(len(sus)):
            src = sus[src_idx]
            for dst in sus[src_idx+1:]:
                # inter-adapter has no dependency
                if src.stype in adapters and \
                   dst.stype in adapters and \
                   src.stype == dst.stype:
                    continue
                for out_idx, out_tensor in enumerate(src.outputs()):
                    if not isinstance(out_tensor, IRTensor):
                        continue
                    # special dependency for communication adapter
                    if dst.stype == SUType.P2P:
                        for recv_tensor in dst.outputs():
                            if out_tensor.overlap(recv_tensor):
                                src.add_successor(out_idx, dst)
                                dst.add_predecessor(-1, src)
                    for in_idx, in_tensor in enumerate(dst.inputs()):
                        if out_tensor.overlap(in_tensor):
                            src.add_successor(out_idx, dst)
                            dst.add_predecessor(in_idx, src)

    @staticmethod
    def gen_comm_adapter(sus: List[ScheduleUnit]):
        """
        Generate communication adapter for each SU
        """
        pass

    @staticmethod
    def gen_trans_adapter(sus: List[ScheduleUnit]):
        """
        Generate transformation adapter for each SU
        """
        pass

    def __len__(self):
        return len(self.sequence)

    def sus(self, index: Optional[int] = None):
        """
        Return ScheduleUnit

        Args:
            
        """
        if isinstance(index, int):
            if index >= len(self.sequence):
                raise RuntimeError(
                    f"Get node out of range ({index} >= {len(self.sequence)})"
                )
            return self.sequence[index]
        elif index is None:
            return copy.copy(self.sequence)
        else:
            raise TypeError("Expected index to be None or int")

    def get_sus(self, stype: SUType) -> List[ScheduleUnit]:
        """
        Get SUs that are of stype
        """
        return [su for su in self.sequence if su.stype == stype]

    def fsus(self) -> List[ScheduleUnit]:
        """
        Get forward ScheduleUnits sequence.
        """
        return [su for su in self.sequence if su.stype == SUType.Forward]

    def happen_before(self, su1, su2, visited=None):
        """
        Check if the su1 -> (happened before) su2

        Returns:
            Boolean
        """
        # FIXME: there is still a strange bug may cause infinite loop
        if visited is None:
            visited = list()
        if su1 in visited:
            return False
        visited.append(su1)

        if not isinstance(su1, ScheduleUnit) or \
           not isinstance(su2, ScheduleUnit):
            raise TypeError("Expected su to be an ScheduleUnit")
        if su2 in su1.successors():
            return True
        else:
            for succ_su in su1.successors():
                # don't need to consider P2P comm dependency
                if succ_su.stype == SUType.P2P:
                    continue
                if self.happen_before(succ_su, su2, visited):
                    return True
            return False

    def merge(self, su1: ScheduleUnit, su2: ScheduleUnit) -> ScheduleUnit:
        """
        Merge two ScheduleUnit as well as their adapters. This requires
        
        1). all the nodes in one SU happens before / after
        all the nodes in another SU. (Guaranteed by default
        as all the operations on sequence are semantic-correct)

        2). all the nodes in both SU are on the same device,
            have same tags and they are not equal.

        3). Deadlock-free merge. Suppose
                SU1 (dev0) -> SU2 (dev1) -> SU3 (dev0)
            Then merge SU1 and SU3 to SU4 will cause
            deadlock on SU4 -> <- SU2

        Note due to PyTorch limitation,
        merging two forward ScheduleUnits will also cause
        the merge of corresponding two backward ScheduleUnits.

        Returns:
            if succeed: A merged ScheduleUnit.
            if fail: None
        """

        fsus = self.fsus()
        if su1 not in fsus:
            raise RuntimeError(f"SU1: {su1} not in forward SUs")
        if su2 not in fsus:
            raise RuntimeError(f"SU2: {su2} not in forward SUs")

        idx1, idx2 = self.sequence.index(su1), self.sequence.index(su2)
        su1, su2 = (su1, su2) if idx1 < idx2 else (su2, su1)

        # condition 1): same device
        if su1.device != su2.device:
            return None

        # condition 2): su2 input cannot be got from both su1 and other su
        start, stop = min(idx1, idx2), max(idx1, idx2)
        inter_sus = self.sequence[start+1:stop]
        inter_sus = [su for su in inter_sus if su.stype != SUType.P2P]
        for su in inter_sus:
            # FIXME: currently only allow other device su exists
            if self.happen_before(su1, su) or self.happen_before(su, su2):
                return None
        for idx in range(len(su2.inputs())):
            prev_sus = su2.predecessors(idx)
            prev_sus = [su for su in prev_sus if su.stype != SUType.P2P]
            if su2 in prev_sus and len(prev_sus) > 1:
                return None

        # start merging
        fnodes = su1.nodes() + su2.nodes()
        # TODO: fix multi-branch
        fsu = ScheduleUnit(fnodes, SUType.Forward, name='fsu')
        fsu.device = su1.device

        bnodes = [node.mirror for node in fnodes][::-1]
        skip_bp = all([bnode is None for bnode in bnodes])
        if not skip_bp:
            bnode = IRBpOperation(
                data_num=len(fsu.inputs()),
                grad_num=len(fsu.outputs())
            )
            for idx, fin in enumerate(fsu.inputs()):
                bnode.set_data(idx, fin)

            for idx, fout in enumerate(fsu.outputs()):
                bnode.set_grad(idx, fout.grad)

            for idx, fin in enumerate(fsu.inputs()):
                bnode.set_output(idx, fin.grad)
            bsu = ScheduleUnit([bnode], stype=SUType.Backward, name='bsu')
            bsu.device = su2.mirror.device
            IRCell.make_pair(fsu, bsu)

        def _set_adapters(su1: ScheduleUnit, su2: ScheduleUnit, msu: ScheduleUnit):
            # set adapter
            for idx, input in enumerate(msu.inputs()):
                if input in su1.inputs():
                    su1_idx = su1.inputs().index(input)
                    adapters = su1.in_adapters(su1_idx)
                    merge_adapter = su1.merge_adapters(su1_idx)
                elif input in su2.inputs():
                    su2_idx = su2.inputs().index(input)
                    adapters = su2.in_adapters(su2_idx)
                    merge_adapter = su2.merge_adapters(su2_idx)
                else:
                    print(f'> Error: msu: {msu}')
                    print(f'> Error: su1: {su1}')
                    print(f'> Error: su2: {su2}')
                    raise RuntimeError("Internal Error: not found input SU")
                msu._add_in_adapter(idx, *adapters)
                msu._set_merge_adapter(idx, merge_adapter)
            for idx, output in enumerate(msu.outputs()):
                if output in su1.outputs():
                    su1_idx = su1.outputs().index(output)
                    adapters = su1.out_adapters(su1_idx)
                    select_adapter = su1.select_adapters(su1_idx)
                elif output in su2.outputs():
                    su2_idx = su2.outputs().index(output)
                    adapters = su2.out_adapters(su2_idx)
                    select_adapter = su2.select_adapters(su2_idx)
                else:
                    raise RuntimeError("Internal Error: not found output SU")
                msu._add_out_adapter(idx, *adapters)
                msu._set_merge_adapter(idx, select_adapter)
            # remove adapters
            for idx, input in enumerate(su2.inputs()):
                if input not in msu.inputs():
                    sadapters, radapters = su2.in_adapters(idx)
                    for adapter in sadapters + radapters:
                        if adapter in self.sequence:
                            self.sequence.remove(adapter)

        _set_adapters(su1, su2, fsu)
        if not skip_bp:
            _set_adapters(su2.mirror, su1.mirror, bsu)

        # replace 
        self.sequence[self.sequence.index(su1)] = fsu
        self.sequence.remove(su2)
        if not skip_bp:
            self.sequence[self.sequence.index(su2.mirror)] = bsu
            self.sequence.remove(su1.mirror)

        # re-gen adapter
        SUGraph.reset_dependency(self.sequence)
        return fsu

    def add_flow(self, su1: ScheduleUnit, su2: ScheduleUnit):
        """
        Add control flow dependency su1 -> su2
        """
        if not isinstance(su1, ScheduleUnit) or not isinstance(su2, ScheduleUnit):
            raise TypeError("Expected both SU1 and SU2 are ScheduleUnit")
        if su1 not in self.sequence:
            raise ValueError(f"su1 {su1} not in SUGraph")
        if su2 not in self.sequence:
            raise ValueError(f"su1 {su2} not in SUGraph")
        if self.happen_before(su2, su1):
            return False
        su1.add_successor(-1, su2)
        su2.add_predecessor(-1, su1)
        return True

    def assign(self, su: ScheduleUnit, ranks: Union[int, List[int]]):
        """
        Assign SU to devices.

        The assignment will automatically set device of its Adapter SU.

        1) if ranks has multiple int, then the su is copied as the same
           SU will be happened redundantly on multiple devices.

        2) if the input tensor this su is decided to be generated on
           other devices, then Adapter SUs (send SU and recv SU) will
           be generated and inserted right before this SU.
        """
        if su not in self.sequence:
            raise ValueError(f"SU {su} is not in the SUGraph")
        if isinstance(ranks, int):
            ranks = [ranks]
        elif not all([isinstance(rank, int) for rank in ranks]):
            raise TypeError("Expected type ranks to be Union[int, List[int]]")

        if su.stype == SUType.P2P:
            return False

        if set(su.device) == set(ranks):
            return True

        if len(ranks) != 1:
            if su.stype == SUType.Dataloader:
                su.device = ranks
            else:
                raise NotImplementedError("Assign multiple ranks to one SU is not supported")
            # print('warning: Missing adapter copy!!')
            # sus = [copy.copy(su) for _ in range(len(ranks)-1)]
            # for su in sus:
            #     index = self.sus().index(su)
            #     self.sequence.insert(index, su)
            # SUGraph.reset_dependency(self.sequence)
            # for su, rank in zip(sus, ranks):
            #     self.assign(su, rank)

        # set device
        su.device = ranks

        # set adapter device for the input
        for idx in range(len(su.inputs())):
            send_adapters, recv_adapters = su.in_adapters(idx)
            merge_adapter = su.merge_adapters(idx)
            for send_adapter in send_adapters:
                send_adapter.nodes(0).send_ranks = [ranks[0],]
            for recv_adapter in recv_adapters:
                recv_adapter.device = ranks
            if merge_adapter is not None:
                merge_adapter.device = ranks

        # set adapter device for the output
        for idx in range(len(su.outputs())):
            send_adapters, recv_adapters = su.out_adapters(idx)
            select_adapter = su.select_adapters(idx)
            for send_adapter in send_adapters:
                send_adapter.device = ranks
            for recv_adapter in recv_adapters:
                recv_adapter.nodes(0).recv_ranks = [ranks[0],]
            if select_adapter is not None:
                select_adapter.device = ranks
        return True

    def set_order(self, seq: List[ScheduleUnit]):
        """
        set a topological order for SUGraph, which requires seq:

        1). The set of SUs in seq must be equal to set of SUGraph
        2). Staisfies topological order

        """
        if not all([isinstance(su, ScheduleUnit) for su in seq]):
            raise ValueError("Expected a list of SUs")
        if len(seq) != len(self.sequence):
            return False
        for su in seq:
            if su not in self.sequence:
                return False
        # correctness check
        if not SUGraph.is_topo_order(seq, integrity_check=True):
            return False
        self.sequence = seq
        return True

    def partial_set_order(self, seq: List[ScheduleUnit], lazy=False):
        """
        Set a order of the sequence using part of SUs.

        A random topological order will be set under
        the constraints of given `seq` order

        Args:
            seq: partial scheduling sequence
            lazy:
                if True, the remaining SU is inserted only when it is needed.
                if False, the remaining SU is inserted once it is ready.

        """
        if lazy:
            raise NotImplementedError("Not supported for Lazy")
        seq = copy.copy(seq)
        for su in seq:
            if su not in self.sequence:
                raise RuntimeError(f"SU {su} is not in SUGraph")
        if not SUGraph.is_topo_order(seq, integrity_check=False):
            return False
        remain_sus : ScheduleUnit = list()
        for su in self.sequence:
            if su not in seq:
                remain_sus.append(su)
        for rsu in remain_sus:
            happen_before_sus = rsu.predecessors()
            # A temporal fix for loss computation and backward
            # -- as they have no dependency in theory
            if rsu.stype == SUType.Backward:
                if rsu.mirror not in happen_before_sus:
                    happen_before_sus.append(rsu.mirror)
            # send / recv su pair should be colocated
            if rsu.stype == SUType.P2P:
                if rsu in seq:
                    continue
                if rsu.mirror in seq:
                    index = seq.index(rsu.mirror)
                    seq.insert(idx+1, rsu)
                    continue
            if rsu in seq:
                raise RuntimeError(f"Internal Error: should not appear SU: {rsu}")
            idx = 0
            while len(happen_before_sus) > 0:
                if idx == len(seq):
                    raise RuntimeError(
                        f"Internal Error: SU {rsu} cannot be inserted"
                    )
                su = seq[idx]
                if su in happen_before_sus:
                    happen_before_sus.remove(su)
                idx += 1
            seq.insert(idx, rsu)

        if not SUGraph.is_topo_order(seq, integrity_check=True):
            raise RuntimeError("Internal Error: topo is not guaranteed.")
        self.sequence = seq
        return True
        

    @staticmethod
    def gen_adapter(sus: List[ScheduleUnit]) -> List[ScheduleUnit]:
        """
        Each computation SU has adapters for its inputs.
        """
        sugraph = SUGraph(sus)

        # clear adapters
        for su in sugraph.sus():
            su._clear_adapters()

        for su in sugraph.sus():
            for in_idx, input in enumerate(su.inputs()):
                if not isinstance(input, IRTensor):
                    continue
                pre_sus = su.predecessors(in_idx)
                tensor_segments = list()
                for pre_su in pre_sus:
                    for out_idx, output in enumerate(pre_su.outputs()):
                        if output.overlap(input):
                            sub_tensor = input.common(output)
                            if sub_tensor != input and sub_tensor not in tensor_segments:
                                tensor_segments.append(sub_tensor)
                            send_op = IRCommunication(
                                send_tensors=[sub_tensor],
                                send_ranks = [-1]
                            )
                            recv_op = IRCommunication(
                                recv_tensors=[sub_tensor],
                                recv_ranks = [-1]
                            )
                            IRCell.make_pair(send_op, recv_op)
                            send_su = ScheduleUnit([send_op], SUType.P2P, name='send')
                            recv_su = ScheduleUnit([recv_op], SUType.P2P, name='recv')
                            su._add_in_adapter(in_idx, send_su, recv_su)
                            send_su.device = su.device
                            pre_su._add_out_adapter(out_idx, send_su, recv_su)
                            recv_su.device = su.device
                            IRCell.make_pair(send_su, recv_su)
                # add adapter for merge
                if len(tensor_segments) != 0:
                    try:
                        merge_op = IRTensorTransform(
                            src_tensors=tensor_segments, dst_tensors=[input]
                        )
                    except Exception:
                        raise RuntimeError(f"Merge Generation Error: {su}")
                    merge_su = ScheduleUnit([merge_op], SUType.Transform, name='merge')
                    su._set_merge_adapter(in_idx, merge_su)
                    merge_su.device = su.device

        # add adapter for select
        for su in sugraph.sus():
            for out_idx, output in enumerate(su.outputs()):
                if not isinstance(output, IRTensor):
                    continue
                select_tensors = list()
                send_adapters, recv_adapters = su.out_adapters(out_idx)
                for send_adapter in send_adapters:
                    for tensor in send_adapter.nodes(0).send_tensors:
                        if tensor != output and tensor not in select_tensors:
                            select_tensors.append(tensor)
                if len(select_tensors) != 0:
                    try:
                        select_op = IRTensorTransform(
                            src_tensors=[output], dst_tensors=select_tensors
                        )
                    except Exception:
                        raise RuntimeError(f"Select Generation Error: {su}")
                    select_su = ScheduleUnit(
                        [select_op], SUType.Transform, name='select'
                    )
                    su._set_select_adapter(out_idx, select_su)
                    select_su.device = su.device
    
        sus_with_adapter = list()
        for su in sus:
            # send + recv + merge
            for idx in range(len(su.inputs())):
                merge_su = su.merge_adapters(idx)
                send_adapters, recv_adapters = su.in_adapters(idx)
                # PyTorch implementation issue: forward + backward happened on same device
                if su.stype == SUType.Backward and not su.inputs(idx).is_grad():
                    continue
                for send_su, recv_su in zip(send_adapters, recv_adapters):
                    sus_with_adapter.append(send_su)
                    sus_with_adapter.append(recv_su)
                if merge_su:
                    sus_with_adapter.append(merge_su)
            # excute
            sus_with_adapter.append(su)
            # select
            for idx in range(len(su.outputs())):
                select_su = su.select_adapters(idx)
                if select_su:
                    sus_with_adapter.append(select_su)
        return sus_with_adapter


    @staticmethod
    def is_topo_order(seq: List[ScheduleUnit], integrity_check=False):
        """
        Check whether seq satisfies topological order.
        
        Args:
            seq: List of ScheduleUnit
            integrity_check:
                If true, performs additional integrity check that requires
                all the SUs in predecessor and successor of a SU should
                appear in the sequence.
        
        Returns:
            Boolean: True for satisfying topo order, otherwise False.
        """

        for index, su in enumerate(seq):
            for pre_su in su.predecessors():
                # find the pre-su not appear in sequence
                if integrity_check:
                    if pre_su not in seq:
                        return False
                if pre_su in seq:
                    pre_idx = seq.index(pre_su)
                    # violate topological order
                    if pre_idx >= index:
                        return False
        return True

    def __repr__(self):
        dscp = f'ScheduleSeq (len={len(self)}):\n'
        for node in self.sequence:
            succ_node_ids = [None] * len(node.outputs())
            for out_idx in range(len(node.outputs())):
                node_list = [snode._id for snode in node.successors(out_idx)]
                succ_node_ids[out_idx] = node_list
            # dscp += f"{node._id}: {node}\n"
            dscp += f"\n{node._id}: {node} -> su id {succ_node_ids}\n"
        return dscp


class SUGraphGener:

    @staticmethod
    def gen_sugraph(nodes) -> SUGraph:
        """
        Generate SUGraph from SchedulePool
        """
        sus = list()
        fnodes = list()
        fsus: List[ScheduleUnit] = list()
        for node in nodes:
            su = ScheduleUnit([node], stype=SUType.Empty, name='su')
            if isinstance(node, IRDataOperation):
                stype = SUType.Dataloader
            elif isinstance(node, IRFwOperation):
                stype = SUType.Forward
                fnodes.append(node)
                fsus.append(su)
            elif isinstance(node, IRBpOperation):
                stype = SUType.Backward
                # get the last one same node
                index = len(fnodes) - fnodes[::-1].index(node.mirror) - 1
                fsu = fsus[index]
                IRCell.make_pair(su, fsu)
                # remove fsu
                fnodes.pop(index)
                fsus.remove(fsu)
            else:
                raise NotImplementedError("Not implemented node type")
            su.stype = stype
            sus.append(su)
        sus_with_adapter = SUGraph.gen_adapter(sus)
        sugraph = SUGraph(sus_with_adapter)
        return sugraph


class SeqSpace:

    @staticmethod
    def space_size(seq, device_num=1):
        """
        Calculate legal 
        """

        def _comb(n, m):
            """
            Calcualte combination C(n,m): select n from m (n < m)
            """
            res = 1
            for j in range(0, min(n, m)):
                res *= (m-j) / (min(n, m) - j)
            return int(res)

        raise NotImplementedError
