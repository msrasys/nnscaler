
from typing import List, Optional, Tuple
import copy
import numpy as np

from cube.graph.tensor import IRSubTensor, IndexMap, ValueMap
from cube.ir.cten import IRCell


class SelectPrim:

    def __init__(self, tensor: IRSubTensor, indmap: IndexMap, valmap: ValueMap,
                 shape: List[int], output: IRSubTensor):
        self.tensor = tensor
        self.indmap = indmap
        self.valmap = valmap
        self.shape = shape
        self.output = output
        self.device: List[int] = tensor.device

    def __repr__(self):
        dscp = f'{self.output} = select({self.tensor})'
        return dscp


class MovePrim:

    def __init__(self, tensor: IRSubTensor, from_rank: int, to_rank: int):
        self.tensor = tensor
        self.from_rank = from_rank
        self.to_rank = to_rank
        self.shape = tensor.shape
        self.dtype = tensor.dtype
        self.device: List[int] = [from_rank, to_rank]

    def __repr__(self):
        dscp = f'move({self.tensor}, from={self.from_rank}, to={self.to_rank})'
        return dscp


class MergePrim:
    def __init__(self, tensors: List[IRSubTensor],
                 output: IRSubTensor, device: List[int],
                 concat: Optional[int] = None, add: bool = False):
        if not ((concat is not None) ^ (add is True)):  # xor condition
            raise RuntimeError("Expected concat or add")
        self.tensors = tensors
        self.concat = concat
        self.add = add
        self.output = output
        # re-order tensor
        if isinstance(concat, int):
            slicers = [tensor.indmap.get()[concat] for tensor in tensors]
            starts = np.array([slicer.start for slicer in slicers], dtype=int)
            sorted_idx = np.argsort(starts)
            tensors = np.array(tensors)[sorted_idx]
            self.tensors = tensors.tolist()
        self.device: List[int] = device

    def set_output(self, output: IRSubTensor):
        self.output = output

    @staticmethod
    def concat(tensor1: IRSubTensor, tensor2: IRSubTensor) ->  Optional[Tuple[IRSubTensor, int]]:
        """
        Check if two tensor can be merged.
        If they can be merged, return the merge index
        """
        if not isinstance(tensor1, IRSubTensor) or not isinstance(tensor2, IRSubTensor):
            raise TypeError("Expected two tensors")
        if tensor1.overlap(tensor2):
            return None
        if tensor1.parent != tensor2.parent:
            return None
        if tensor1.valmap != tensor2.valmap:
            return None
        indices1 = tensor1.indmap.get()
        indices2 = tensor2.indmap.get()
        indmap = list()
        if len(indices1) != len(indices2):
            return None
        axis = None
        for dim, (slicer1, slicer2) in enumerate(zip(indices1, indices2)):
            if slicer1 != slicer2:
                start1, stop1, step1 = slicer1.start, slicer1.stop, slicer1.step
                start2, stop2, step2 = slicer2.start, slicer2.stop, slicer2.step
                if step1 != step2:
                    return None
                if axis is not None:
                    return None
                if start1 < start2 and stop1 == start2:
                    axis = dim
                    indmap.append(slice(start1, stop2, step1))
                elif start1 > start2 and start1 == stop2:
                    axis = dim
                    indmap.append(slice(start2, stop1, step1))
                else:
                    return None
            else:
                indmap.append(slicer1)
        shapes = list()
        for idx, (nele1, nele2) in enumerate(zip(tensor1.shape, tensor2.shape)):
            nele = nele1 if idx != axis else nele1 + nele2
            shapes.append(nele)
        mtensor = tensor1.parent.select(
            indmap = tuple(indmap),
            valmap = tensor1.valmap,
            shape = shapes
        )
        return mtensor, axis

    @staticmethod
    def add(tensor1: IRSubTensor, tensor2: IRSubTensor) -> Optional[IRSubTensor]:
        if not isinstance(tensor1, IRSubTensor) or not isinstance(tensor2, IRSubTensor):
            raise TypeError("Expected two tensors")
        if tensor1.overlap(tensor2):
            return None
        if tensor1.parent != tensor2.parent:
            return None
        if tensor1.indmap != tensor2.indmap:
            return None
        if tensor1.valmap.chunk_num != tensor2.valmap.chunk_num:
            return None
        chunk_num = tensor1.valmap.chunk_num
        idx1, idx2 = tensor1.valmap.idx, tensor2.valmap.idx
        if chunk_num % 2 != 0:
            return None
        chunk_num = int(chunk_num // 2)
        if int(idx1 // 2) != int(idx2 // 2):
            return None
        idx = int(idx1 // 2)
        mtensor = tensor1.parent.select(
            indmap = tensor1.indmap,
            valmap = (idx, chunk_num),
            shape = tensor1.shape
        )
        return mtensor

    def __repr__(self):
        dscp = f'{self.output} = merge({self.tensors}, axis={self.concat}, add={self.add})'
        return dscp


class IRAdapter(IRCell):
    """
    Tensor Adapter for each operator.

    A Tensor Adapter has three stages:
        * Select: select produced tensors
        * Move: transfer the produced tensors
        * Merge: merge the produced tensors
    """

    def __init__(self, prims,
                inputs: List[IRSubTensor], idevices: List[List[int]],
                outputs: List[IRSubTensor], odevices: List[List[int]]):
        
        self._prims = prims
        self._idevices = tuple(idevices)
        self._odevices = tuple(odevices)

        super().__init__(
            name='adapter', signature='adapter',
            input_length=len(inputs),
            output_length=len(outputs),
            init_outputs=False
        )
        for idx, tensor in enumerate(inputs):
            self.set_input(idx, tensor)
        for idx, tensor in enumerate(outputs):
            self.set_output(idx, tensor)

        # set up device
        device = set()
        for prim in self._prims:
            device.update(prim.device)
        self.device = list(device)

    def prims(self, select=True, move=True, merge=True):
        """
        Return prim list
        """
        prims = list()
        for prim in self._prims:
            if select and isinstance(prim, SelectPrim):
                prims.append(prim)
            if move and isinstance(prim, MovePrim):
                prims.append(prim)
            if merge and isinstance(prim, MergePrim):
                prims.append(prim)
        return prims

    def dispatch(self, rank: int):
        """
        Get Adapter for a specific rank

        Returns:
            IRAdapter
        """
        if not isinstance(rank, int):
            raise TypeError(f"Expected rank to be int but got {rank}")
        prims = list()
        for prim in self.prims():
            if rank in prim.device:
                prims.append(prim)
        inputs, idevs = list(), list()
        for input, devs in zip(self.inputs(), self._idevices):
            if rank in devs:
                inputs.append(input)
                idevs.append(devs)
        outputs, odevs = list(), list()
        for output, devs in zip(self.outputs(), self._odevices):
            if rank in devs:
                outputs.append(output)
                odevs.append(devs)
        adapter = IRAdapter(prims, inputs, idevs, outputs, odevs)
        adapter.name = self.name
        adapter._id = self._id
        adapter.device = rank
        return adapter

    def is_identity(self):
        """
        Check if the adapter does nothing

        Returns:
            Boolean
        """
        return len(self._prims) == 0

    def __repr__(self):
        dscp = f'Adapter{self._id}-{self.device}(inputs={self.inputs()}, outputs={self.outputs()})'
        return dscp

    def module_repr(self) -> str:
        return repr(self)

    @staticmethod
    def gen(dst_tensor: IRSubTensor):
        # print(f'generating adapter for: {dst_tensor}')
        if not isinstance(dst_tensor, IRSubTensor):
            raise RuntimeError("Expected IRSubTensor")
        inputs, intersections, select_prims = IRAdapter.gen_select(dst_tensor)
        move_prims = IRAdapter.gen_move(dst_tensor, intersections)
        merge_prims = IRAdapter.gen_merge(dst_tensor, intersections)
        prims = select_prims + move_prims + merge_prims
        idevs = [t.device for t in inputs]
        odevs = [dst_tensor.device]
        return IRAdapter(prims, inputs, idevs, [dst_tensor], odevs)

    @staticmethod
    def gen_select(dst_tensor):

        inputs = list()
        intersections = list()
        prims = list()

        otensor = dst_tensor
        odevice = otensor.device

        local, remote = list(), list()
        for ptensor in otensor.parent.ptensors:
            if ptensor.device == odevice:
                local.append(ptensor)
            else:
                remote.append(ptensor)

        # check local tensor
        if otensor in local:
            intersections.append(otensor)
            inputs.append(otensor)
            return inputs, intersections, prims

        # FIXME: multi producer may result overlapped region
        for itensor in otensor.parent.ptensors:
            if not itensor.overlap(otensor):
                continue

            # intersection
            common = otensor.common(itensor)
            common.attach_cell(itensor._cell)
            intersections.append(common)
            inputs.append(itensor)
            if common == itensor:
                continue

            islicers = itensor.indmap.get()
            oslicers = common.indmap.get()
            # index map
            indmap = list()
            for islicer, oslicer in zip(islicers, oslicers):
                istart, istop, istep = islicer.start, islicer.stop, islicer.step
                ostart, ostop, ostep = oslicer.start, oslicer.stop, oslicer.step
                if ostep % istep != 0:
                    raise RuntimeError("Step condition fails")
                # relative offset
                start = ostart - istart
                stop = start + ostop - ostart
                slicer = slice(start, stop, ostep)
                indmap.append(slicer)
            # value map
            if itensor.valmap == common.valmap:
                valmap = ValueMap(0, 1)
            elif itensor.valmap == ValueMap(0, 1):
                valmap = common.valmap
            else:
                print('from: ', itensor)
                print('to  : ', common)
                raise NotImplementedError(
                    f"Not supported value select: {input.valmap} -> {common.valmap}"
                )
            prim = SelectPrim(itensor, indmap, valmap, common.shape, common)
            prims.append(prim)
        
        return inputs, intersections, prims

    @staticmethod
    def gen_move(dst_tensor, intersections):
        prims = list()
        odevice = dst_tensor.device
        for tensor in intersections:
            if tensor.device != odevice:
                if len(tensor.device) != 1 or len(odevice) != 1:
                    raise RuntimeError(
                        f"Expected tensor on a single device but got {tensor.device} and {odevice}"
                    )
                prim = MovePrim(tensor, from_rank=tensor.device[0], to_rank=odevice[0])
                prims.append(prim)
        return prims

    @staticmethod
    def gen_merge(dst_tensor, intersections):
        prims = list()
        output = dst_tensor    
        remain_tensors = copy.copy(intersections)
        if output in remain_tensors:
            return prims
        out = None
        while out != output:
            out = None
            merged = False
            for idx1 in range(len(remain_tensors) - 1):
                for idx2 in range(idx1 + 1, len(remain_tensors)):
                    tensor1 = remain_tensors[idx1]
                    tensor2 = remain_tensors[idx2]
                    # try concat
                    out = MergePrim.concat(tensor1, tensor2)
                    if out is not None:
                        out, concat_dim = out
                        prim = MergePrim([tensor1, tensor2], out, output.device, concat_dim, False)
                        prims.append(prim)
                        merged = True
                        break
                    # try add
                    out = MergePrim.add(tensor1, tensor2)
                    if out is not None:
                        prim = MergePrim([tensor1, tensor2], out, output.device, None, True)
                        prims.append(prim)
                        merged = True
                        break
                if merged:
                    remain_tensors.remove(tensor1)
                    remain_tensors.remove(tensor2)
                    remain_tensors.append(out)
                    break
            # cannot merge or add
            if out is None:
                raise RuntimeError("Merge Plan not found")
        return prims

    def extra_repr(self):
        """
        Detailed information
        """
        dscp = repr(self) + ':\n'
        # select
        for prim in self._select_prims + self._move_prims + self._merge_prims:
            dscp += '\t' + repr(prim) + '\n'
        return dscp
