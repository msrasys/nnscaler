from typing import List, Optional, Callable
import copy

from cube.ir.cten import IRTensor


__all__ = ['IRFullTensor', 'IRSubTensor']


class IRFullTensor(IRTensor):

    def __init__(self, shape=None, name=None):

        super().__init__(shape, name)

        self._segments = list()
        # indices: List[IndexMap] for each segment
        self._indices: List = list()
        # value op
        self._val_ops: List = list()

    def segments(self, index: Optional[int] = None):
        """
        Get the SubTensors at index position
        """
        if index is None:
            return copy.copy(self._segments)
        else:
            return self._segments[index]

    def indices(self, index: Optional[int] = None):
        """
        Get the SubTensors mapping indices
        """
        if index is None:
            return copy.copy(self._indices)
        else:
            return self._indices[index]

    def val_ops(self, index: Optional[int] = None):
        """
        Get the SubTensors val_op
        """
        if index is None:
            return copy.copy(self._val_ops)
        else:
            return self._val_ops[index]

    def select(self, indices, val_op: Optional[Callable], shape: List[int]):
        """
        Select a SubTensor from FullTensor.

        Note due to implementation issue, one value in the full tensor
        cannot be splitted by different val_op

        Args:
            indices: the index of this tensor's index

            val_op: how the tensor is merged with the other
                    sub_tensor at same location

            shape: the sub_tensor shape.

        Returns:
            IRSubTensor
        """
        sub_tensor = IRSubTensor(self, indices, val_op, shape)
        self._segments.append(sub_tensor)
        self._indices.append(indices)
        self._val_ops.append(val_op)
        return sub_tensor


class IRSubTensor:

    def __init__(self, full_tensor: IRTensor, indices, val_op=None, shape=None):
        """
        Create an IRSubTensor.

        Args:
            full_tensor: the full tensor
            indices: index list
            val_op: the value operation to merge SubTensors into one
        """
        super.__init__(shape=shape, name=full_tensor.name)

        # the full tensor
        self._full_tensor = full_tensor

        # the index from full_tensor
        self._index_map = indices

        # val merge op
        self.val_merge_op = val_op

    @property
    def parent(self) -> IRFullTensor:
        """
        Return the full tensor of this sub tensor
        """
        return self._full_tensor

    def index_map(self):
        """
        Return indices list mapped to the full tensor
        """
        return copy.copy(self._index_map)

    @property
    def val_op(self):
        return self.val_merge_op

    def select(self, indices, val_op, shape=None):
        """
        Select an IRSubTensor

        Args:
            indices: the index of this tensor's index

            val_op: the value operation to merge 
                    co-located indices of SubTensors into one

            shape: the sub_tensor shape

        Returns:
            IRSubTensor
        """
        index_map = self.index_map[indices]
        sub_tensor = self.full_tensor.select(index_map, val_op, shape)
        return sub_tensor
