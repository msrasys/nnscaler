import enum
from typing import List, Optional, Union
import copy
from cube import schedule

from cube.ir.cten import IRCell
from cube.schedule.su import SUType, ScheduleUnit


class SUGraph(IRCell):

    def __init__(self, sus: List[ScheduleUnit]):

        if not all([isinstance(su, ScheduleUnit) for su in sus]):
            raise TypeError(
                f"Expected a list of ScheduleUnits, but got {type(sus)}"
            )

        inputs = IRCell.get_inputs(sus)
        outputs = IRCell.get_outputs(sus)
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
        self.reset_dependency()

    @property
    def nnodes(self) -> int:
        """
        Get number of nodes (int)
        """
        return len(self.sequence)

    def reset_dependency(self):
        """
        Reset the node dataflow dependency
        """
        # set node predecessors and successors
        for src_idx in range(self.nnodes):
            src_su = self.sequence[src_idx]
            src_su._successors = [
                list() for _ in range(len(src_su.outputs()))
            ]
            for dst_su in self.sequence[src_idx+1:]:
                dst_su._predecessors = [
                    list() for _ in range(len(dst_su.inputs()))
                ]
                for out_idx, out_tensor in enumerate(src_su.outputs()):
                    for in_idx, in_tensor in enumerate(dst_su.inputs()):
                        if out_tensor.overlap(in_tensor):
                            src_su.add_successor(out_idx, dst_su)
                            dst_su.add_predecessor(in_idx, src_su)

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

    def happen_before(self, su1, su2):
        """
        Check if the su1 -> (happened before) su2

        Returns:
            Boolean
        """
        if not isinstance(su1, ScheduleUnit) or \
           not isinstance(su2, ScheduleUnit):
            raise TypeError("Expected su to be an ScheduleUnit")
        if su2 in su1.successors():
            return True
        else:
            for succ_su in su1.successors():
                if self.happen_before(succ_su, su2):
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

        def _adapter_merge(first_su: ScheduleUnit, second_su: ScheduleUnit, merged_su: ScheduleUnit):
            # move from first_su adapter
            # print(f' 1st SU: {first_su} \n 2nd SU: {second_su} \n merged SU: {merged_su}')
            for idx, input in enumerate(first_su.inputs()):
                send_adapters, recv_adapters = first_su.in_adapters(idx)
                merge_adapter = first_su.merge_adapters(idx)
                merge_idx = merged_su.inputs().index(input)
                for send_adapter, recv_adapter in zip(send_adapters, recv_adapters):
                    merged_su._add_in_adapter(merge_idx, send_adapter, recv_adapter)
                if merge_adapter in self.sequence:
                    merged_su._set_merge_adapter(merge_idx, merge_adapter)
            for idx, output in enumerate(first_su.outputs()):
                send_adapters, recv_adapters = first_su.out_adapters(idx)
                select_adapter = first_su.select_adapters(idx)
                if output in merged_su.outputs() and output not in second_su.outputs():
                    merge_idx = merged_su.outputs().index(output)
                    for send_adapter, recv_adapter in zip(send_adapters, recv_adapters):
                        merged_su._add_out_adapter(merge_idx, send_adapter, recv_adapter)
                    if select_adapter:
                        merged_su._set_select_adapter(merge_idx, select_adapter)
                else:
                    if merge_adapter in self.sequence:
                        self.sequence.remove(merge_adapter)
            # move from su2 adapter
            for idx, input in enumerate(second_su.inputs()):
                send_adapters, recv_adapters = second_su.in_adapters(idx)
                merge_adapter = second_su.merge_adapters(idx)
                if input in merged_su.inputs() and input not in first_su.inputs():
                    merge_idx = merged_su.inputs().index(input)
                    for send_adapter, recv_adapter in zip(send_adapters, recv_adapters):
                        merged_su._add_in_adapter(merge_idx, send_adapter, recv_adapter)
                    if merge_adapter:
                        merged_su._set_merge_adapter(merge_idx, merge_adapter)
                else:
                    for send_adapter, recv_adapter in zip(send_adapters, recv_adapters):
                        # print(f'removing: {send_adapter}')
                        # print(f'removing: {recv_adapter}')
                        if send_adapter in self.sequence:
                            self.sequence.remove(send_adapter)
                        if recv_adapter in self.sequence:
                            self.sequence.remove(recv_adapter)
                    if merge_adapter in self.sequence:
                        self.sequence.remove(merge_adapter)
            for idx, output in enumerate(second_su.outputs()):
                send_adapters, recv_adapters = second_su.out_adapters(idx)
                select_adapter = second_su.select_adapters(idx)
                if output in merged_su.outputs():
                    merge_idx = merged_su.outputs().index(output)
                    for send_adapter, recv_adapter in zip(send_adapters, recv_adapters):
                        merged_su._add_out_adapter(merge_idx, send_adapter, recv_adapter)
                    if select_adapter:
                        merged_su._set_select_adapter(merge_idx, select_adapter)
                else:
                    if select_adapter:
                        self.sequence.remove(select_adapter)

        if not isinstance(su1, ScheduleUnit) or \
           not isinstance(su2, ScheduleUnit):
            raise TypeError("Expected SU1 and SU2 are ScheduleUnit")
        if su1 not in self.sequence:
            raise ValueError(f"su1: {su1}  not in sequence")
        if su2 not in self.sequence:
            raise ValueError(f"su2: {su2}  not in sequence")
        
        # 2) all the nodes in both SU are on the same device
        if su1 == su2 or su1.stype != su2.stype:
            return None
        if su1.device != su2.device:
            return None

        if su1.stype == SUType.Adapter:
            raise NotImplementedError("Not supported for merging Adapter")

        index_su1 = self.sequence.index(su1)
        index_su2 = self.sequence.index(su2)
        su1, su2 = (su1, su2) if index_su1 < index_su2 else (su2, su1)
        # 3) deadlock-free merge
        index_su1, index_su2 = min(index_su1, index_su2), max(index_su1, index_su2)
        inter_sus = self.sequence[index_su1+1:index_su2]
        for su in inter_sus:
            # in theory the below condition satisfies merge, but it may
            # break the topo order
            # e.g., su1 -> adapter1 ,....., adapter2 -> su2
            # if self.happen_before(su1, su) and self.happen_before(su, su2):
            # to keep topo order:
            if self.happen_before(su, su2):
                return None

        # merge forward su
        sub_nodes = su1.nodes() + su2.nodes()
        merged_su = ScheduleUnit(sub_nodes, su1.stype)
        merged_su.device = su1.device
        _adapter_merge(su1, su2, merged_su)

        # merge mirrored su
        # mirror_su2 -> mirror_su1
        mirror_su1, mirror_su2 = su1.mirror, su2.mirror
        merged_mirror_su = None
        if mirror_su1 and mirror_su2:
            if mirror_su1.device == mirror_su2.device:
                sub_nodes = mirror_su2.nodes() + mirror_su1.nodes()
                merged_mirror_su = ScheduleUnit(sub_nodes, mirror_su1.stype)
                merged_mirror_su.device = mirror_su1.device
                _adapter_merge(mirror_su2, mirror_su1, merged_mirror_su)
                # set mirror
                merged_su.set_mirror(merged_mirror_su)
                merged_mirror_su.set_mirror(merged_su)
        elif mirror_su1 or mirror_su2:
            raise RuntimeError(
                "The merged su should be both have mirror or both not have."
            )

        # replace
        self.sequence[index_su1] = merged_su
        self.sequence.remove(su2)
        if merged_mirror_su:
            if mirror_su1 in self.sequence and mirror_su2 in self.sequence:
                index_mirror_su2 = self.sequence.index(mirror_su2)
                self.sequence[index_mirror_su2] = merged_mirror_su
                self.sequence.remove(mirror_su1)

        # TODO: optimize: reset dependency
        self.reset_dependency()
        return merged_su

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
        elif not all([isinstance(int, rank) for rank in ranks]):
            raise TypeError("Expected type ranks to be Union[int, List[int]]")

        if su.stype == SUType.Adapter:
            return False

        if set(su.device) == set(ranks):
            return True

        if len(ranks) != 1:
            # copy su
            # TODO: adatper copy
            print('warning: Missing adapter copy!!')
            sus = [copy.copy(su) for _ in range(len(ranks)-1)]
            for su in sus:
                index = self.sus().index(su)
                self.sequence.insert(index, su)
            self.reset_dependency()
            for su, rank in zip(sus, ranks):
                self.assign(su, rank)

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
                if integrity_check and not pre_su in seq:
                        return False
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
            dscp += f"\n{node._id}: {node} -> su id {succ_node_ids}\n"
        return dscp


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
