from typing import List, Optional
from enum import Enum
import numpy as np

from cube.ir.cten import IRCell
from cube.graph.tensor import IRSubTensor, IndexMap


class IRReshapeType(Enum):

    Select = 'cube.runtime.adapter.select'
    Merge  = 'cube.runtime.adapter.merge'


class IRTensorReshape(IRCell):
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

        if len(src_tensors) != 1 and len(dst_tensors) != 1:
            raise ValueError("Expected at least one of tensors has length 1")

        self.ttype = None

        self._select_indices: List[IndexMap] = list()
        self._merge_axis = None

        if len(src_tensors) == 1:
            self.ttype = IRReshapeType.Select
            src_tensor = src_tensors[0]
            if not isinstance(src_tensor, IRSubTensor):
                raise TypeError(f"Expected IRSubTensor but got {type(src_tensor)}")
            # select
            for tensor in dst_tensors:
                indices = tensor.indices & src_tensor.indices
                self._select_indices.append(indices)
        
        elif len(dst_tensors) == 1:
            self.ttype = IRReshapeType.Merge
            dst_tensor = dst_tensors[0]
            # find dims to concat
            ndims = len(dst_tensor.shape)
            indices = [set() for _ in range(ndims)]
            for src_tensor in src_tensors:
                if isinstance(src_tensor, IRSubTensor):
                    for ndim, slicer in enumerate(src_tensor.indices.get()):
                        indices[ndim].add((slicer.start, slicer.stop, slicer.step))
                else:
                    raise RuntimeError(
                        f"Expected SubTensor but got {type(src_tensor)}"
                    ) 
            # check if only one dim set has multiple slicer
            for dim, dim_indices in enumerate(indices):
                if len(dim_indices) != 1:
                    if self._merge_axis is not None:
                        print("src: ", src_tensors)
                        print("dst: ", dst_tensors)
                        raise NotImplementedError("Only support merge on one axis")
                    self._merge_axis = dim
            if self._merge_axis is None:
                # check the coverage
                if src_tensors[0].indices != dst_tensor.indices:
                    raise RuntimeError("Not cover all the indices to merge.")
            # get merge axis
            if self._merge_axis is not None:
                dim_indices = indices[self._merge_axis]
                # check if they are overlapped
                starts = np.array([slicer[0] for slicer in dim_indices])
                stops = np.array([slicer[1] for slicer in dim_indices])
                steps = np.array([slicer[2] for slicer in dim_indices])
                sorted_idx = np.argsort(starts)
                sorted_starts = starts[sorted_idx]
                sorted_stops = stops[sorted_idx]
                sorted_steps = steps[sorted_idx]
                for last_stop, begin_start in zip(sorted_stops[:-1], sorted_starts[1:]):
                    if last_stop != begin_start:
                        raise NotImplementedError(f"Concatenation fails due to axis {last_stop} != {begin_start}")
                for step in sorted_steps:
                    if step != 1:
                        raise NotImplementedError(f"Found a SubTensor step {step} != 1")
                # re-order
                src_tensors = np.array(src_tensors)[sorted_idx]

        else:
            raise RuntimeError("Internal Error")

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

    @property
    def select_indices(self) -> List[IndexMap]:
        return self._select_indices

    @property
    def merge_axis(self) -> Optional[int]:
        return self._merge_axis

    def is_identity(self):
        """
        Check if this transformation is a non-op
        """
        if self.ttype == IRReshapeType.Select:
            src_tensor = self.inputs(0)
            for dst_tensor in self.outputs():
                if dst_tensor != src_tensor:
                    return False
            return True
        if self.ttype == IRReshapeType.Merge:
            if self.merge_axis is None:
                return True
            return False
        return False
