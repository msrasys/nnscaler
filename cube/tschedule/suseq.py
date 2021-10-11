from typing import List, Any, Optional
import copy

from cube.graph.ir_cten import IRCell, IRTensor
from cube.tschedule.su import ScheduleUnit


class SUSequence(IRCell):

    def __init__(self, sus: List[ScheduleUnit]):

        if not all([isinstance(su, ScheduleUnit) for su in sus]):
            raise TypeError(
                f"Expected a list of ScheduleUnits, but got {type(sus)}"
            )

        super().__init__(
            name = 'SU',
            signature = 'None',
            input_length = 0,
            output_length = 0
        )
        self.sequence = sus
        self.reset_dependency()

    def reset_dependency(self):
        """
        Reset the node dataflow dependency
        """
        # set node predecessors and successors
        for src_idx in range(len(self.sequence)):
            src_cell = self.sequence[src_idx]
            src_cell._successors = [
                list() for _ in range(len(src_cell.outputs()))
            ]
            for dst_idx in range(src_idx + 1, len(self.sequence)):
                dst_su = self.sequence[dst_idx]
                dst_su._predecessors = [
                    list() for _ in range(len(dst_su.inputs()))
                ]
                for tensor in src_cell.outputs():
                    if isinstance(tensor, IRTensor):
                        if tensor in dst_su.inputs():
                            src_output_idx = src_cell.outputs().index(tensor)
                            src_cell.add_successor(src_output_idx, dst_su)
                            dst_input_idx = dst_su.inputs().index(tensor)
                            dst_su.add_predecessor(dst_input_idx, src_cell)

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

    def happen_after(self, su1, su2):
        """
        Check if the su2 -> (happened before) su1

        Returns:
            Boolean
        """
        return self.happen_before(su2, su1)

    def merge(self, su1: ScheduleUnit, su2: ScheduleUnit) -> ScheduleUnit:
        """
        Merge two ScheduleUnit. This requires
        
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
        if set(su1.device) != set(su2.device):
            return None

        # 3) deadlock-free merge
        index_su1 = self.sequence.index(su1)
        index_su2 = self.sequence.index(su2)
        # make su1 happen before su2
        su1, su2 = (su1, su2) if index_su1 < index_su2 else (su2, su1)
        index_su1, index_su2 = min(index_su1, index_su2), max(index_su1, index_su2)
        inter_sus = self.sequence[index_su1+1:index_su2]
        for su in inter_sus:
            if su1.happen_after(su) and su.happen_before(su2):
                return None

        # merge forward su
        sub_nodes = su1.nodes() + su2.nodes()
        merged_su = ScheduleUnit(
            sub_nodes, su1.global_graph, su1.device, su1.stype
        )

        # merge mirrored su
        # mirror_su2 -> mirror_su1
        mirror_su1, mirror_su2 = su1.mirror, su2.mirror
        sub_nodes = mirror_su2.nodes() + mirror_su1.nodes()
        merged_mirror_su = ScheduleUnit(
            sub_nodes, mirror_su1.global_graph, mirror_su1.device, mirror_su1.stype
        )

        # set mirror
        merged_su.set_mirror(merged_mirror_su)
        merged_mirror_su.set_mirror(merged_su)

        # replace
        self.sequence[index_su1] = merged_su
        self.sequence.remove(su2)
        if mirror_su1 in self.sequence and mirror_su2 in self.sequence:
            index_mirror_su2 = self.sequence.index(mirror_su2)
            self.sequence[index_mirror_su2] = merged_mirror_su
            self.sequence.remove(mirror_su1)
        
        # TODO: optimize: reset dependency
        self.reset_dependency()
        return merged_su

    def add_flow(self, su1, su2):
        """
        Add control flow dependency su1 -> su2
        """
        if not isinstance(su1, ScheduleUnit) or not isinstance(su2, ScheduleUnit):
            raise TypeError("Expected both SU1 and SU2 are ScheduleUnit")
        su1.add_successors(-1, su2)
        su2.add_predecessors(-1, su1)

    def is_correct(self):
        """
        Check whether sequence 
        satisfies the sequential consistency model
        """

        for index, su in enumerate(self.sequence):
            for pre_su in su.predecessors():
                # find the pre-su not appear in sequence
                if not pre_su in self.sequence:
                    return False
                pre_idx = self.sequence.index(pre_su)
                # violate sequential consistency model
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
