import copy
from typing import List, Optional
from enum import Enum
import numpy as np

from cube.ir.cten import IRCell
from cube.graph.tensor import IRSubTensor, IndexMap, ValueMap


class IRTransformType(Enum):

    Select = 'cube.runtime.adapter.select'
    Merge  = 'cube.runtime.adapter.merge'


class IRTensorTransform(IRCell):
    """
    Tensor transformation by convert source tensors
    to destination tensors

    Select:
        src_tensors is only one tensor, dst_tensors has (multiple) tensors.
        This will select the sub_tensor and generate what it need

    Merge:
        src_tensors has (multiple) tensors, dst_tensors is only one tensor.
        This will merge the sub_tensor and generate what it need
    """
    def __init__(self, src_tensors: List[IRSubTensor], dst_tensors: List[IRSubTensor]):

        if not all([isinstance(t, IRSubTensor) for t in src_tensors]):
            raise TypeError("Expected src tensors to be IRSubTensor")
        if not all([isinstance(t, IRSubTensor) for t in dst_tensors]):
            raise TypeError("Expected dst tensors to be IRSubTensor")
        if not ((len(src_tensors) == 1) or (len(dst_tensors) == 1)):
            raise ValueError("Expected at least one of tensors has length 1")

        self.ttype = None
        self._trace = list()

        # select
        if len(src_tensors) == 1:
            self.ttype = IRTransformType.Select
            self._trace = SelectPlan.gen(src_tensors[0], dst_tensors)

        # merge
        elif len(dst_tensors) == 1:
            self.ttype = IRTransformType.Merge
            self._trace = MergePlan.gen(src_tensors, dst_tensors[0])

        else:
            raise NotImplementedError

        super().__init__(
            name = 'transformation',
            signature = self.ttype.value,
            input_length = len(src_tensors),
            output_length = len(dst_tensors)
        )
        for idx, input in enumerate(src_tensors):
            self.set_input(idx, input)
        for idx, output in enumerate(dst_tensors):
            self.set_output(idx, output)

    def trace(self):
        """
        Get trace of transformation
        """
        return copy.copy(self._trace)

    def is_identity(self):
        """
        Check if this transformation is a non-op
        """
        return len(self._trace) == 0


class SelectPrim:

    def __init__(self, tensor: IRSubTensor, indmap: IndexMap, valmap: ValueMap, shape: List[int]):
        self.tensor = tensor
        self.indmap = indmap
        self.valmap = valmap
        self.shape = shape
        self.output = None
    
    def set_output(self, output: IRSubTensor):
        self.output = output

    def __repr__(self):
        dscp = f't{self.output._id} = select(t{self.tensor._id}, {self.indmap}, {self.valmap}, {self.shape})'
        return dscp


class SelectPlan:

    @staticmethod
    def gen(input: IRSubTensor, outputs: List[IRSubTensor]) -> List[SelectPrim]:
        trace: List[SelectPrim] = list()
        islicers: List[slice] = input.indmap.get()
        for output in outputs:
            if output == input:
                continue
            oslicers: List[slice] = output.indmap.get()
            # indmap
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
            indmap = IndexMap(tuple(indmap))
            # value map
            if output.valmap == input.valmap:
                valmap = ValueMap(0, 1)
            elif input.valmap == ValueMap(0, 1):
                valmap = output.valmap
            else:
                print('from: ', input)
                print('to  : ', output)
                raise NotImplementedError(
                    f"Not supported value select: {input.valmap} -> {output.valmap}"
                )
            prim = SelectPrim(input, indmap, valmap, output.shape)
            prim.set_output(output)
            trace.append(prim)
        return trace


class MergePrim:
    def __init__(self,
                 tensors: List[IRSubTensor],
                 concat: Optional[int] = None,
                 add: bool = False):
        if not ((concat is not None) ^ (add is True)):  # xor condition
            raise RuntimeError("Expected concat or add")
        self.tensors = tensors
        self.concat = concat
        self.add = add
        self.output = None
        # re-order tensor
        if isinstance(concat, int):
            slicers = [tensor.indmap.get()[concat] for tensor in tensors]
            starts = np.array([slicer.start for slicer in slicers], dtype=int)
            sorted_idx = np.argsort(starts)
            tensors = np.array(tensors)[sorted_idx]
            self.tensors = tensors.tolist()

    def set_output(self, output: IRSubTensor):
        self.output = output


    def __repr__(self):
        tensors = [f't{t._id}' for t in self.tensors]
        tensors = '[' + ', '.join(tensors) + ']'
        dscp = f't{self.output._id} = merge({tensors}, axis={self.concat}, add={self.add})'
        return dscp


class MergePlan:

    @staticmethod
    def gen(inputs: List[IRSubTensor], output: IRSubTensor) -> List[MergePrim]:
        """
        Generate merge plan from input tensors to the output.
        """
        if not all([isinstance(t, IRSubTensor) for t in inputs]):
            raise TypeError("Expected inputs: List[IRSubTensor]")
        if not isinstance(output, IRSubTensor):
            raise TypeError("Expected inputs: List[IRSubTensor]")

        trace : List[MergePrim] = list()
        remain_tensors = copy.copy(inputs)
        dst_tensor = output
        if dst_tensor in remain_tensors:
            return trace
        out = None
        while out != dst_tensor:
            # concat or merge
            out = None
            merge = False
            for idx1 in range(len(remain_tensors) - 1):
                for idx2 in range(idx1 + 1, len(remain_tensors)):
                    tensor1 = remain_tensors[idx1]
                    tensor2 = remain_tensors[idx2]
                    out = MergePlan.concat(tensor1, tensor2)
                    if out is not None:
                        out_tensor, concat_dim = out
                        out = out_tensor
                        prim = MergePrim([tensor1, tensor2], concat_dim, False)
                        prim.set_output(out_tensor)
                        trace.append(prim)
                        merge = True
                        break
                    out = MergePlan.add(tensor1, tensor2)
                    if out is not None:
                        prim = MergePrim([tensor1, tensor2], None, True)
                        prim.set_output(out)
                        trace.append(prim)
                        merge = True
                        break
                if merge:
                    remain_tensors.remove(tensor1)
                    remain_tensors.remove(tensor2)
                    remain_tensors.append(out)
                    break
            # cannot merge or add
            if out is None:
                raise RuntimeError("Merge Plan not found")
        return trace


    @staticmethod
    def concat(tensor1: IRSubTensor, tensor2: IRSubTensor) -> int:
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
    def add(tensor1: IRSubTensor, tensor2: IRSubTensor) -> int:
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
