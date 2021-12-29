from typing import List, Dict

from cube.execplan import ExectuionPlan
from cube.graph.adapter.adapter import IRAdapter
from cube.graph.tensor import IRSubTensor, ValueMap
from cube.execplan.planpass.planpass import PlanPass


class P2PFusion(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        adapters = list()
        for node in execplan.graph.nodes():
            if isinstance(node, IRAdapter):
                adapters.append(node)
        pass

    @staticmethod
    def allgather_matcher(execplan: ExectuionPlan):
        """
        Allgather semantic:

        Given a list of adapters:
        1). [Num] each adapter has same multiple inputs and same one output
        2). [Dev] inputs/outputs among adapters are from different device. device# = number of adapters.
        3). [Indmap] No-overlap index-map among inputs.
        4). [Valmap] each input value-map is same with output valuemap
        """
        pass

    @staticmethod
    def allreduce_matcher(execplan: ExectuionPlan):
        """
        Allreduce semantic:

        Given a list of adapters:
        1). [Num] each adapter has different one input and same one output
        2). [Dev] inputs/outputs among adapters are from different devices
        2). [Indmap] inputs among adapters has same index-map with output.
        3). [Valmap] inputs have parital value-map. Output has full value-map
        """
        pass

    @staticmethod
    def reducescatter_matcher(execplan: ExectuionPlan):
        """
        ReduceScatter semantic:

        Given a list of adapters:
        1). [Num] each adapter has different one input and different one output
        2). [Dev] inputs/outputs among adapters are from different devices
        3). [Indmap] inputs among adapters have same index-map
        4). [Indmap] outputs among adapters have different index-map
        5). [Valmap] inputs among adapters have different partial val-map.
        6). [Valmap] outputs among adapters have same Full val-map
        """
        pass

    @staticmethod
    def broadcast_matcher(execplan: ExectuionPlan):
        """
        Broadcast semantic:

        Given a list of adapters:
        1). [Num] each adapter has same input and output. input = output.
        2). [Dev] inputs among adapters are from a same device.
        3). [Dev] outputs among adapters are from different devices
        """
        pass

    # Utilities
    @staticmethod
    def group_by_output(adapters: List[IRAdapter]):
        """
        Group the adapters by same output tensor
        """
        tensors = dict()  # tensor_id -> tensor
        groups = dict()   # tensor_id -> List[IRAdapter]
        for adapter in adapters:
            if len(adapter.outputs()) != 1:
                raise RuntimeError("Expected only one output")
            tensor = adapter.outputs(0)
            tid = tensor._id
            if tid not in tensors:
                tensors[tid] = tensor
                groups[tid] = list()
            groups[tid].append(adapter)
        return tensors, groups

    @staticmethod
    def group_by_input(adapters: List[IRAdapter]):
        """
        Group the adapters by same input tensor(s)
        """
        tensors = dict()  # Tuple[tensor_id] -> tensor
        groups = dict()   # Tuple[tensor_id] -> List[IRAdapter]
        for adapter in adapters:
            tensors = adapter.inputs
            tids = [tensor._id for tensor in tensors]
            tids.sort()
            tids = tuple(tids)
            if tids not in tensors:
                tensors[tids] = tensors
                groups[tids] = list()
            groups[tids].append(adapter)
        return tensors, groups
