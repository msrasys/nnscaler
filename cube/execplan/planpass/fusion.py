from typing import List
import copy

# debug only
# import sys
# if tid == tensor_id: print(f'out line: {sys._getframe().f_lineno}')

from cube.graph.tensor import IRSubTensor, ValueMap

from cube.graph.adapter.adapter import IRAdapter
from cube.graph.adapter.adapter import CollectivePrim

from cube.execplan import ExectuionPlan
from cube.execplan.planpass.planpass import PlanPass

# FIXME: all fusions don't consider input order!
# May get incorrect result in some cases.

# FIXME: all fusions don't check if the communication can be happened at
# the same time


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
        # update adapter devices
        for node in execplan.graph.nodes():
            if isinstance(node, IRAdapter):
                node.update_device()
        for devid in execplan.devices():
            for node in execplan.sequence(devid):
                if isinstance(node, IRAdapter):
                    if devid not in node.device:
                        execplan.at(devid).remove(node)
        return execplan

    @staticmethod
    def allreduce_matcher(execplan: ExectuionPlan, all_adapters: List[IRAdapter]):
        """
        Allreduce semantic:

        Given a list of adapters:
        1). [Num] each adapter has different one input and same one output
        2). [Dev] inputs/outputs among adapters are from different devices
        3). [Dev] adapters have same device. adapters# is same to device set.
        4). [Indmap] inputs among adapters has same index-map with output.
        5). [Valmap] inputs have parital value-map. Output has full value-map
        """
        outputs, groups = P2PFusion.group_by_output(all_adapters)
        for tid in outputs:
            cond = True
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
            for adapter in adapters:
                if len(adapters) != len(adapter.device):
                    cond = False
                    break
            if not cond: continue
            # condition 4)
            for adapter in adapters:
                if not P2PFusion._check_indmap_same(adapter.inputs() + adapter.outputs()):
                    cond = False
                    break
            if not cond: continue
            # condition 5)
            for adapter in adapters:
                if not P2PFusion._check_valmap_no_overlap(adapter.inputs()):
                    cond = False
                    break
            if not cond: continue
            for adapter in adapters:
                if adapter.outputs(0).valmap != ValueMap(0, 1):
                    cond = False
                    break
            if not cond: continue
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
            cond = True
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
            for adapter in adapters:
                if len(adapters) != len(adapter.device):
                    cond = False
                    break
            if not cond: continue
            # condition 4)
            for adapter in adapters:
                if not P2PFusion._check_indmap_no_overlap(adapter.inputs()):
                    cond = False
                    break
            if not cond: continue
            # condition 5)
            for adapter in adapters:
                if not P2PFusion._check_valmap_same(adapter.inputs() + adapter.outputs()):
                    cond = False
                    break
            if not cond: continue
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
    def reducescatter_matcher(execplan: ExectuionPlan, all_adapters: List[IRAdapter]):
        """
        ReduceScatter semantic:

        Given a list of adapters:
        1). [Num] each adapter has same multiple input and different one output
        2). [Dev] inputs/outputs among adapters are from different devices
        3). [Dev] adapters have same device. adapters# is same to device set
        4). [Indmap] inputs of each adapter have same index-map
        5). [Indmap] outputs among adapters have different index-map
        6). [Valmap] inputs of each adapter have different partial val-map
        7). [Valmap] outputs among adapters have same Full val-map
        """
        inputs, groups = P2PFusion.group_by_input(all_adapters)
        for tids in inputs:
            adapters: List[IRAdapter] = groups[tids]
            cond = True
            # cond 1)
            otids = [adapter.outputs(0)._id for adapter in adapters]
            if len(set(otids)) != len(adapters):
                continue
            # cond 2)
            if not P2PFusion._check_different_inputs_devices(adapters, among=False):
                continue
            if not P2PFusion._check_different_outputs_devices(adapters, among=True):
                continue
            # cond 3)
            for adapter in adapters:
                if len(adapters) != len(adapter.device):
                    cond = False
                    break
            if not cond: continue
            # cond 4)
            for adapter in adapters:
                if not P2PFusion._check_indmap_same(adapter.inputs()):
                    cond = False
                    break
            if not cond: continue
            # cond 5)
            outputs = [adapter.outputs(0) for adapter in adapters]
            if not P2PFusion._check_indmap_no_overlap(outputs):
                continue
            # cond 6)
            for adapter in adapters:
                if not P2PFusion._check_valmap_no_overlap(adapter.inputs()):
                    cond = False
                    break
            if not cond: continue
            # cond 7)
            for adapter in adapters:
                if adapter.outputs(0).valmap != ValueMap(0, 1):
                    cond = False
                    break
            if not cond: continue
            # gen reduce-scatter
            print(f'generating reduce-scatter for tensor: {tids} ...')
            all_select_prims = list()
            for adapter in adapters:
                all_select_prims += adapter.prims(move=False, merge=False, coll=False)
            for adapter in adapters:
                device = adapter.odevice(0)
                sprims = [prim for prim in all_select_prims if prim.device == device]
                if len(sprims) != len(adapters):
                    raise RuntimeError(f"got {len(sprims)} (!={len(adapters)}) select prims for reduce-scatter")
                inputs = [sprim.output for sprim in sprims]
                coll = CollectivePrim(
                    ctype = CollectivePrim.Type.ReduceScatter,
                    device = device,
                    group = adapter.device,
                    inputs = inputs,
                    outputs = adapter.outputs(),
                )
                prims = sprims + [coll]
                adapter._prims = prims
            for adapter in adapters:
                all_adapters.remove(adapter)

    @staticmethod
    def broadcast_matcher(execplan: ExectuionPlan, all_adapters: List[IRAdapter]):
        """
        Broadcast semantic:

        Given a list of adapters:
        1). [Num] each adapter has same one input and one output. input = output.
        2). [Dev] inputs among adapters are from a same device.
        3). [Dev] outputs among adapters are from different devices
        """
        outputs, groups = P2PFusion.group_by_output(all_adapters)
        for tid in outputs:
            adapters: List[IRAdapter] = groups[tid]
            cond = True
            # cond 1)
            if not P2PFusion._check_same_inputs(adapters):
                continue
            if not P2PFusion._check_single_inputs(adapters):
                continue
            for adapter in adapters:
                if adapter.inputs(0) != adapter.outputs(0):
                    cond = False
                    break
            if not cond: continue
            # cond 2)
            root_device = set()
            for adapter in adapters:
                root_device.update(P2PFusion._get_input_devices(adapter))
            if len(root_device) != 1:
                continue
            # cond 3)
            if not P2PFusion._check_different_outputs_devices(adapters, among=True):
                continue
            # gen broadcast
            print(f'generating broadcast for tensor: {outputs[tid]} ...')
            # put root rank to the first
            root = list(root_device)[0]
            group = set()
            for adapter in adapters:
                group.update(P2PFusion._get_output_devices(adapter))
            group = [root] + list(group)
            # input
            tensor = adapters[0].inputs(0)
            
            prims = list()
            for device in group:
                inputs = [tensor] if device == root else None
                output_shapes = [tensor.shape]
                output_dtypes = [tensor.dtype]
                coll = CollectivePrim(
                    ctype = CollectivePrim.Type.Broadcast,
                    device = [device],
                    group = group,
                    inputs = inputs,
                    outputs = [tensor],
                    output_shapes = output_shapes,
                    output_dtypes = output_dtypes
                )
                prims.append(coll)
            
            # add aditional adapter to root node
            root_adapter = IRAdapter(
                prims = [prims[0]],
                inputs=[tensor], idevices=[[root],],
                outputs=[tensor], odevices=[[root],]
            )
            # insert into graph and execution plan
            index = min([execplan.graph.nodes().index(n) for n in adapters])
            execplan.graph._nodes.insert(index, root_adapter)
            seq = [node for node in execplan.graph.nodes() if root in node.device]
            execplan.set(root, seq)

            for adapter in adapters:
                device = adapter.odevice(0)[0]
                prim = prims[group.index(device)]
                adapter._prims = [prim]
            
            for adapter in adapters:
                all_adapters.remove(adapter)

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
            tids = [tensor._id for tensor in adapter.inputs()]
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
