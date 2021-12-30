from typing import List, Dict

# debug only
# import sys
# if tid == tensor_id: print(f'out line: {sys._getframe().f_lineno}')

from cube.graph.tensor import IRSubTensor, ValueMap

from cube.graph.adapter.adapter import IRAdapter
from cube.graph.adapter.adapter import CollectivePrim, MergePrim

from cube.execplan import ExectuionPlan
from cube.execplan.planpass.planpass import PlanPass


class P2PFusion(PlanPass):

    @staticmethod
    def apply(execplan: ExectuionPlan) -> ExectuionPlan:
        adapters = list()
        for node in execplan.graph.nodes():
            if isinstance(node, IRAdapter):
                adapters.append(node)
        matchers = [
            P2PFusion.allreduce_matcher,
            P2PFusion.allgather_matcher,
            P2PFusion.reducescatter_matcher,
            P2PFusion.broadcast_matcher,
        ]
        for matcher in matchers:
            matcher(execplan, adapters)
        return execplan

    @staticmethod
    def allgather_matcher(execplan: ExectuionPlan, all_adapters: List[IRAdapter]):
        """
        Allgather semantic:

        Given a list of adapters:
        1). [Num] each adapter has same multiple inputs and same one output
        2). [Dev] inputs/outputs among adapters are from different device.
        3). [Dev] adapters have same device. adapters# is same to device set.
        4). [Indmap] inputs inside one adapter are not overlapped
        5). [Valmap] each input value-map is same with output valuemap
        """
        outputs, groups = P2PFusion.group_by_output(all_adapters)
        for tid in outputs:
            adapters: List[IRAdapter] = groups[tid]
            # condition 1)
            if not P2PFusion._check_multi_inputs(adapters):
                continue
            if not P2PFusion._check_same_inputs(adapters):
                continue
            # condition 2)
            if not P2PFusion._check_different_inputs_devices(adapters, among=False):
                continue
            if not P2PFusion._check_different_outputs_devices(adapters, among=True):
                continue
            # condition 3)
            cond = True
            for adapter in adapters:
                if len(adapters) != len(adapter.device):
                    cond = False
                    break
            if not cond:
                continue
            # condition 4)
            cond = True
            for adapter in adapters:
                if not P2PFusion._check_indmap_no_overlap(adapter.inputs()):
                    cond = False
                    break
            if not cond:
                continue
            # condition 5)
            cond = True
            for adapter in adapters:
                if not P2PFusion._check_valmap_same(adapter.inputs() + adapter.outputs()):
                    cond = False
                    break
            if not cond:
                continue
            # gen allgather
            print(f'generating allgather for tensor: {outputs[tid]} ...')
            for adapter in adapters:
                device = adapter.odevice(0)
                input_idx = adapter.idevice().index(device)
                inputs = [adapter.inputs(input_idx)]
                coll = CollectivePrim(
                    ctype = CollectivePrim.Type.AllGather,
                    device = device,
                    group = adapter.device,
                    inputs = inputs,
                    input_shapes = None,
                    input_dtypes = None,
                    outputs = adapter.inputs(),
                    output_shapes = None,
                    output_dtypes = None,
                )
                # merge prim still keeps, remove select and move prims
                prims = [coll] + adapter.prims(select=False, move=False, coll=False)
                adapter._prims = prims
            for adapter in adapters:
                all_adapters.remove(adapter)

    @staticmethod
    def allreduce_matcher(execplan: ExectuionPlan, all_adapters: List[IRAdapter]):
        """
        Allreduce semantic:

        Given a list of adapters:
        1). [Num] each adapter has different one input and same one output
        2). [Dev] inputs/outputs among adapters are from different devices
        3). [Indmap] inputs among adapters has same index-map with output.
        4). [Valmap] inputs have parital value-map. Output has full value-map
        """
        return
        outputs, groups = P2PFusion.group_by_output(all_adapters)
        for tid in outputs:
            adapters = groups[tid]
            # condition 1)
            if not P2PFusion._check_multi_inputs(adapters):
                continue
            if not P2PFusion._check_same_inputs(adapters):
                continue
            # condition 2)
            if not P2PFusion._check_different_inputs_devices(adapters, among=True):
                continue
            if not P2PFusion._check_different_outputs_devices(adapters, among=True):
                continue
            # condition 3)
            cond = True
            for adapter in adapters:
                if not P2PFusion._check_indmap_same(adapter.inputs() + adapter.outputs()):
                    cond = False
                    break
            if not cond:
                continue
            # condition 4)
            inputs = list()
            for adapter in adapters:
                inputs += adapter.inputs()
            if not P2PFusion._check_valmap_no_overlap(inputs):
                continue
            cond = True
            for adapter in adapters:
                if adapter.outputs(0).valmap != ValueMap(0, 1):
                    cond = False
                    break
            if not cond:
                continue
            # generate
            print(f'generating allreduce for tensor: {outputs[tid]} ...')
            for adapter in adapters:
                device = adapter.odevice(0)
                input_idx = adapter.idevice().index(device)
                inputs = [adapter.inputs(input_idx)]
                coll = CollectivePrim(
                    ctype = CollectivePrim.Type.AllReduce,
                    device = device,
                    group = adapter.device,
                    inputs = inputs,
                    outputs = adapter.outputs(),
                )
                adapter._prims = [coll]
            for adapter in adapters:
                all_adapters.remove(adapter)

    @staticmethod
    def reducescatter_matcher(execplan: ExectuionPlan, all_adapters: List[IRAdapter]):
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
    def broadcast_matcher(execplan: ExectuionPlan, all_adapters: List[IRAdapter]):
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

    @staticmethod
    def _check_same_inputs(adapters: List[IRAdapter]):
        """
        Check if the inputs are same among adapters
        """
        input_ids = list()
        for adapter in adapters:
            tids = [t._id for t in adapter.inputs()]
            tids.sort()
            input_ids.append(tids)
        ninputs = [len(tids) for tids in input_ids]
        # number of inputs not same
        if len(set(ninputs)) != 1:
            return False
        # input ids not same
        for tids in zip(*input_ids):
            if len(set(tids)) != 1:
                return False
        return True

    @staticmethod
    def _check_multi_inputs(adapters: List[IRAdapter]):
        for adapter in adapters:
            if len(adapter.inputs()) <= 1:
                return False
        return True

    @staticmethod
    def _check_single_inputs(adapters: List[IRAdapter]):
        for adapter in adapters:
            if len(adapter.inputs()) != 1:
                return False
        return True

    @staticmethod
    def _get_input_devices(adapter: IRAdapter) -> List[int]:
        """
        Return sorted device list for all inputs
        """
        device = set()
        for idevice in adapter.idevice():
            device.update(idevice)
        device = list(device)
        device.sort()
        return device

    @staticmethod
    def _get_output_devices(adapter: IRAdapter) -> List[int]:
        """
        Return sorted device list for all outputs
        """
        device = set()
        for odevice in adapter.odevice():
            device.update(odevice)
        device = list(device)
        device.sort()
        return device

    @staticmethod
    def _check_different_inputs_devices(adapters: List[IRAdapter], among: bool):
        if among:
            adapter_devices = list()
            for adapter in adapters:
                device = P2PFusion._get_input_devices(adapter)
                adapter_devices.append(tuple(device))
            if len(set(adapter_devices)) != len(adapters):
                return False
            return True
        else:
            for adapter in adapters:
                device = P2PFusion._get_input_devices(adapter)
                # assume each tensor is attached to one deivce
                if len(device) != len(adapter.inputs()):
                    return False
            return True

    @staticmethod
    def _check_different_outputs_devices(adapters: List[IRAdapter], among: bool):
        if among:
            adapter_devices = list()
            for adapter in adapters:
                device = set()
                for odevice in adapter.odevice():
                    device.update(odevice)
                device = list(device)
                device.sort()
                adapter_devices.append(tuple(device))
            if len(set(adapter_devices)) != len(adapters):
                return False
            return True
        else:
            for adapter in adapters:
                device = set()
                for odevice in adapter.odevice():
                    device.update(odevice)
                # assume each tensor is attached to one deivce
                if len(device) != len(adapter.outputs()):
                    return False
            return True

    @staticmethod
    def _check_indmap_same(tensors: List[IRSubTensor]):
        if len(tensors) == 0:
            return True
        indmap = tensors[0].indmap
        for tensor in tensors[1:]:
            if tensor.indmap != indmap:
                return False
        return True

    @staticmethod
    def _check_indmap_no_overlap(tensors: List[IRSubTensor]):
        if len(tensors) == 0:
            return True
        for idx1 in range(len(tensors) - 1):
            for idx2 in range(idx1 + 1, len(tensors)):
                t1 = tensors[idx1]
                t2 = tensors[idx2]
                if t1.indmap.overlap(t2.indmap):
                    return False
        return True

    @staticmethod
    def _check_valmap_same(tensors: List[IRSubTensor]):
        if len(tensors) == 0:
            return True
        valmap = tensors[0].valmap
        for tensor in tensors[1:]:
            if tensor.valmap != valmap:
                return False
        return True

    @staticmethod
    def _check_valmap_no_overlap(tensors: List[IRSubTensor]):
        if len(tensors) == 0:
            return True
        for idx1 in range(len(tensors) - 1):
            for idx2 in range(idx1 + 1, len(tensors)):
                t1 = tensors[idx1]
                t2 = tensors[idx2]
                if t1.valmap.overlap(t2.valmap):
                    return False
        return True
