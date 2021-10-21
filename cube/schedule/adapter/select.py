from typing import List
from enum import Enum
import numpy as np

from cube.ir.cten import IRCell, IRTensor
from cube.graph.tensor import IRSubTensor, IRFullTensor


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
    def __init__(self, src_tensors: List[IRTensor], dst_tensors: List[IRTensor]):

        if len(src_tensors) != 1 and len(dst_tensors) != 1:
            raise ValueError("Expected at least one of tensors has length 1")
        self._src_tensors = src_tensors
        self._dst_tensors = dst_tensors

        self.ttype = None

        self.select_indices = list()
        self.merge_axis = None

        if len(src_tensors) == 1:
            self.ttype = IRReshapeType.Select
            src_tensor = src_tensors[0]
            # select
            for tensor in dst_tensors:
                indices = tensor.common(src_tensor)
                self.select_indices.append(indices)
        
        elif len(dst_tensors) == 1:
            self.ttype = IRReshapeType.Merge
            dst_tensor = dst_tensors[0]
            # find dims to concat
            ndims = len(dst_tensor.shape)
            indices = [set() for _ in range(ndims)]
            for src_tensor in src_tensors:
                if isinstance(src_tensor, IRSubTensor):
                    for ndim, slicer in enumerate(src_tensor.indices.get()):
                        indices[ndim].add(slicer)
                elif isinstance(dst_tensor, IRFullTensor):
                    for ndim, dim_len in enumerate(src_tensor.shape):
                        slicer = slice(0, dim_len, 1)
                        indices[ndim].add(slicer)
            # check if only one dim set has multiple slicer
            for dim, dim_indices in enumerate(indices):
                if len(dim_indices) != 1:
                    if self.merge_axis is not None:
                        raise NotImplementedError("Only support merge on one axis")
                    self.merge_axis = dim
            dim_indices = indices[self.merge_axis]
            # check if they are overlapped
            starts = np.array([slicer.start for slicer in dim_indices])
            stops = np.array([slicer.stop for slicer in dim_indices])
            steps = np.array([slicer.step for slicer in dim_indices])
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
            dst_tensors = dst_tensors[sorted_idx]

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
