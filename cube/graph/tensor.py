from typing import List, Optional, Callable
import copy

from cube.ir.cten import IRTensor


__all__ = ['IndexMap', 'IRFullTensor', 'IRSubTensor']


class IndexMap:

    def __init__(self, indices):

        if not isinstance(indices, tuple):
            raise TypeError("Expected indices to be a tuple")

        if not all([isinstance(s, slice) for s in indices]):
            raise NotImplementedError(
                "Only support for sliced index mapping"
            )
        self._indices = indices

    def get(self):
        """
        Get indices
        """
        return self._indices

    @property
    def ndims(self) -> int:
        """
        Number of dimensions of the index map
        """
        return len(self._indices)

    @property
    def neles(self) -> int:
        """
        Number of elements of the index map
        """
        nelements = 1
        for slicer in self._indices:
            count = slicer.stop - slicer.start
            if slicer.step:
                count = int(count // slicer.step)
            nelements *= count
        return nelements

    @property
    def shape(self) -> List[int]:
        """
        Get the shape of the slice
        """
        shape = list()
        for slicer in self._indices:
            count = slicer.stop - slicer.start
            if slicer.step:
                count = int(count // slicer.step)
            shape.append(count)
        return shape

    def map(self, submap):
        """
        Map from the current indices by sub_indices.

        Args:
            sub_indices: IndexMap

        Returns:
            sub_indices: IndexMap

        """
        if not isinstance(submap, IndexMap):
            raise TypeError("Expected IndexMap")
        if self.ndims != submap.ndims:
            raise ValueError("Expected same length of sub_indices")

        # e.g., (slice(0, M), slice(0, int(K // 2))
        sub = list()
        for dim_indices, dim_sub_indices in zip(self.get(), submap.get()):
            start, stop = dim_indices.start, dim_indices.stop
            step = dim_indices.step if dim_indices.step else 1

            sub_start, sub_stop = dim_sub_indices.start, dim_sub_indices.stop
            sub_step = dim_sub_indices.step if dim_sub_indices.step else 1
    
            new_start = start + sub_start
            new_stop = new_start + sub_stop - sub_start
            new_step = step * sub_step
            if new_stop > stop:
                raise ValueError("Trying to map a index out of range")
            sub.append(slice(new_start, new_stop, new_step))
        return IndexMap(tuple(sub))

    def overlap(self, other):
        """
        Check if this indices overlapped with the other

        Args:
            other: IndexMap

        Returns:
            Boolean: True has overlap, otherwise False
        """
        if not isinstance(other, IndexMap):
            raise TypeError("Expected IndexMap")
        
        if other.ndims != self.ndims:
            raise TypeError("Expected same dimension")

        for slicer1, slicer2 in zip(self.get(), other.get()):
            start1, stop1 = slicer1.start, slicer1.stop
            step1 = slicer1.step if slicer1.step else 1
        
            start2, stop2 = slicer2.start, slicer2.stop
            step2 = slicer2.step if slicer2.step else 1
        
            if step1 == step2:
                if min(stop1, stop2) <= max(start1, start2):
                    return False
                elif start1 % step1 != start2 % step2:
                    return False
            else:
                raise NotImplementedError(f"not supported for differnt steps")
        return True

    def __and__(self, other):
        """
        Get the common part

        Args:
            other: IndexMap
        
        Returns:
            IndexMap for the common part
        """
        if not self.overlap(other):
            return None
        slices = list()
        for slicer1, slicer2 in zip(self.get(), other.get()):
            start1, stop1 = slicer1.start, slicer1.stop
            step1 = slicer1.step if slicer1.step else 1
        
            start2, stop2 = slicer2.start, slicer2.stop
            step2 = slicer2.step if slicer2.step else 1

            if step1 == step2:
                start = max(start1, start2)
                stop = min(stop1, stop2)
                slices.append(slice(start, stop, step1))
            else:
                raise NotImplementedError(f"not supported for differnt steps")
        return IndexMap(tuple(slices))


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

    def indices(self, index: Optional[int] = None) -> IndexMap:
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
        self._indices.append(IndexMap(indices))
        self._val_ops.append(val_op)
        return sub_tensor

    def overlap(self, other):
        """
        Check if the two tensor is overlapped.

        Returns:
            True if they are sharing co-located position in
            the full tensor, otherwise False
        """
        if not isinstance(other, IRTensor):
            raise TypeError("Expected Tensor")
        if isinstance(other, IRFullTensor):
            return self == other
        elif isinstance(other, IRSubTensor):
            return other.parent == self
        else:
            raise TypeError("Customized IRTensor not support")

    def common(self, other) -> Optional[IRTensor]:
        """
        Get the common sub-tensor

        Args:
            IRTensor

        Returns:
            None for not overlap,
            else IRSubTensor or IRFullTensor
        """
        return other if self.overlap(other) else None


class IRSubTensor(IRTensor):

    def __init__(self, full_tensor: IRTensor, indices, val_op=None, shape=None):
        """
        Create an IRSubTensor.

        Args:
            full_tensor: the full tensor
            indices: index list
            val_op: the value operation to merge SubTensors into one
        """
        super().__init__(shape=shape, name=full_tensor.name)

        # the full tensor
        self._full_tensor = full_tensor

        # the index from full_tensor
        self._index_map = IndexMap(indices)

        # val merge op
        self.val_merge_op = val_op

    @property
    def parent(self) -> IRFullTensor:
        """
        Return the full tensor of this sub tensor
        """
        return self._full_tensor

    @property
    def indices(self) -> IndexMap:
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
        sub_map = IndexMap(indices)
        index_map = self.indices.map(sub_map)
        sub_tensor = self.parent.select(index_map.get(), val_op, shape)
        return sub_tensor

    def overlap(self, other):
        """
        Check if the two tensor is overlapped.

        Returns:
            True if they are sharing co-located position in
            the full tensor, otherwise False
        """
        if not isinstance(other, IRTensor):
            return False
        if isinstance(other, IRFullTensor):
            return self.parent == other
        elif isinstance(other, IRSubTensor):
            return self.indices.overlap(other.indices)
        else:
            raise TypeError("Customized IRTensor not support")

    def common(self, other):
        """
        Get the common sub-tensor

        Args:
            IRTensor

        Returns:
            None for not overlap,
            else IRSubTensor or IRFullTensor
        """
        if self.overlap(other):
            if isinstance(other, IRFullTensor):
                return self
            elif isinstance(other, IRSubTensor):
                indices = self.indices & other.indices
                sub_tensor = self.parent.select(
                    indices = indices.get(),
                    val_op = self.val_op,
                    shape = indices.shape
                )
                return sub_tensor
            else:
                raise NotImplementedError("Customized IRTensor not support")
        return None
