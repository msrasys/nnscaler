r"""
SubTensor Gradient rule:

SubTensor's logical grad = SubTensor.parent.grad.select(
    indices = SubTensor.indices, 
    val_map = SubTensor.val_map,
    shape   = SubTensor.shape
)

FwOperation -> BpOperation rule:

1). for (FwOp) input tensors, gradient SubTensor is:
    indices = input.indices;
    val is splitted by referencing times on the indices

2). for (FwOp) output tensors, gradient SubTensor is:
    indices = output.indices;
    val follows same value splitting rules with output
"""


from typing import List, Optional, Union, Tuple
import copy

from cube.ir.cten import IRCell, IRTensor


__all__ = ['IndexMap', 'ValueMap', 'IRFullTensor', 'IRSubTensor']


class IndexMap:

    def __init__(self, indices):

        if not isinstance(indices, tuple):
            raise TypeError("Expected indices to be a tuple")

        if not all([isinstance(s, slice) for s in indices]):
            raise NotImplementedError(
                "Only support for sliced index mapping"
            )
        self._indices = indices

    def __eq__(self, other):
        if isinstance(other, IndexMap):
            if self.ndims != self.ndims:
                return False
            for myslicer, oslicer in zip(self.get(), other.get()):
                mstart, mstop = myslicer.start, myslicer.stop
                mstep = myslicer.step if myslicer.step is not None else 1
                ostart, ostop = oslicer.start, oslicer.stop
                ostep = oslicer.step if oslicer.step is not None else 1
                if mstart != ostart or mstop != ostop or mstep != ostep:
                    return False
            return True
        return False

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

    def __repr__(self):
        dscp = repr(self._indices)
        return dscp


class ValueMap:
    r"""
    Represent the value split.

    Value is represented as a summation of several variables

        value = \sigma_{i=1}^{chunk_num} a_i

    two tensors consider as same value mapping:
        they have same chunk num and share the same a_i (idx)

    Note we regard these mapping as same:
        1.0 = 0.9 (a1) + 0.1 (a2)
        1.0 = 0.4 (a1) + 0.6 (a2)

    The mapping doesn't consider what a1 really contains, but only
    consider the variable (a) itself and number of variable.
    """

    def __init__(self, idx: int, chunk_num: int):
        if idx >= chunk_num or idx < 0:
            raise ValueError(f"Expected idx {idx} in [0, {chunk_num})")
        self._idx = idx
        self._chunk_num = chunk_num

    @property
    def idx(self):
        return self._idx

    @property
    def chunk_num(self):
        return self._chunk_num

    def map(self, sub_map):
        if not isinstance(sub_map, ValueMap):
            raise TypeError("Expected sub_map to be ValueMap")
        idx = self.idx * sub_map.chunk_num + sub_map.idx
        chunk_num = self.chunk_num * sub_map.chunk_num
        return ValueMap(idx, chunk_num)

    def overlap(self, other):
        if not isinstance(other, ValueMap):
            raise TypeError("Expected ValueMap")
        if self.chunk_num == other.chunk_num:
            return self.idx == other.idx
        else:
            if self.chunk_num == 1 or other.chunk_num == 1:
                return True
            else:
                raise NotImplementedError("Not Implemented")

    def __eq__(self, other):
        if isinstance(other, ValueMap):
            if other.idx == self.idx and other.chunk_num == self.chunk_num:
                return True
        return False

    def __and__(self, other):
        """
        Find the common part
        """
        if not isinstance(other, ValueMap):
            raise TypeError("Expected ValueMap for & operator")
        if not self.overlap(other):
            return None
        if self.chunk_num == other.chunk_num:
            return ValueMap(self.idx, self.chunk_num)
        if self.chunk_num == 1:
            return ValueMap(other.idx, other.chunk_num)
        else:
            return ValueMap(self.idx, self.chunk_num)

    def __repr__(self):
        return f'({self.idx}/{self.chunk_num})'


def _to_index_map(indices: Union[Tuple, IndexMap]):
    if not isinstance(indices, tuple) and not isinstance(indices, IndexMap):
        raise TypeError("Expected indices to be tuple or IndexMap")
    if isinstance(indices, tuple):
        indices = IndexMap(indices)
    return indices


def _to_value_map(val_map: Union[Tuple, ValueMap, None]):
    if not isinstance(val_map, tuple) and \
       not isinstance(val_map, ValueMap) and \
       not val_map is None:
        raise TypeError("Expected val_map to be tuple, IndexMap or None")
    if val_map is None:
        val_map = ValueMap(0, 1)
    elif isinstance(val_map, tuple):
        if len(val_map) != 2:
            raise ValueError("Expected tuple to be (idx, chunk_num)")
        val_map = ValueMap(*val_map)
    return val_map


class IRFullTensor(IRTensor):

    def __init__(self, shape=None, name=None, requires_grad=True):

        super().__init__(shape, name)

        self._segments = list()
        # indices: List[IndexMap] for each segment
        self._indices: List = list()
        # value op
        self._val_maps: List = list()

        # track gradient
        self._forward_dst_cells = list()

        self.requires_grad = requires_grad
        if requires_grad:
            grad = IRFullTensor(shape, 'g' + self.name, False).as_grad()
            self.grad = grad

    def __copy__(self):
        """
        Full tensor should only exist one instance per id

        Returns:
            tensor
        """
        return self

    def _add_fdst_cell(self, cell: IRCell):
        if not isinstance(cell, IRCell):
            raise TypeError("Expect an IRCell")
        if cell not in self._forward_dst_cells:
            if None in self._forward_dst_cells:
                idx = self._forward_dst_cells.index(None)
                self._forward_dst_cells[idx] = cell
            else:
                self._forward_dst_cells.append(cell)

    def _rm_fdst_cell(self, cell: IRCell):
        if not isinstance(cell, IRCell):
            raise TypeError("Expect an IRCell")
        if cell in self._forward_dst_cells:
            # setting to None to keep value map order
            idx = self._forward_dst_cells.index(cell)
            self._forward_dst_cells[idx] = None

    def forward_dst_cells(self):
        return [cell for cell in self._forward_dst_cells if cell is not None]

    def as_param(self):
        """
        Set the tensor as trainable parameter
        """
        self.requires_grad = True
        self._is_param = True
        self._is_grad = False
        for sub_tensor in self._segments:
            sub_tensor.as_param()

    def as_grad(self):
        self._is_param = False
        self._is_grad = True
        for sub_tensor in self._segments:
            sub_tensor.as_grad()
        return self

    def like(self):
        """
        Create a new tensor with same name and shape,
        but with a different new id

        Returns:
            tensor
        """
        tensor = IRFullTensor(self._shape, self.name)
        for attr in IRFullTensor._attr:
            setattr(tensor, attr, getattr(self, attr))
        return tensor

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

    def val_maps(self, index: Optional[int] = None):
        """
        Get the SubTensors val_map
        """
        if index is None:
            return copy.copy(self._val_maps)
        else:
            return self._val_maps[index]

    def select(self, indices: Union[Tuple, IndexMap], val_map: Union[Tuple, ValueMap, None], shape: List[int]):
        """
        Select a SubTensor from FullTensor.

        Note due to implementation issue, one value in the full tensor
        cannot be splitted by different val_map

        Args:
            indices: the index of this tensor's index

            val_map: how the tensor mapped from original value

            shape: the sub_tensor shape.

        Returns:
            IRSubTensor
        """
        indices = _to_index_map(indices)
        val_map = _to_value_map(val_map)

        for idx in range(len(self._segments)):
            indmap = self._indices[idx]
            valmap = self._val_maps[idx]
            sub_tensor = self._segments[idx]
            if indmap == indices and valmap == val_map:
                return sub_tensor

        sub_tensor = IRSubTensor(self, indices, val_map, shape)
        for attr in IRFullTensor._attr:
            setattr(sub_tensor, attr, getattr(self, attr))
        sub_tensor.grad = None

        self._segments.append(sub_tensor)
        self._indices.append(indices)
        self._val_maps.append(val_map)
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

    def tosub(self):
        """
        Convert to SubTensor by selecting all indices
        """
        if self.shape is None:
            raise RuntimeError("Expected know shape")
        slicers = list()
        for dim_len in self.shape:
            slicers.append(slice(0, dim_len, 1))
        sub_tensor = self.select(
            indices=tuple(slicers),
            val_map=None,
            shape=self.shape
        )
        return sub_tensor

    def __repr__(self):
        dscp = f'FullTensor(id={self._id}, shape={self.shape}, device={self.device})'
        return dscp


class IRSubTensor(IRTensor):

    def __init__(self, full_tensor: IRTensor, indices, val_map: Optional[ValueMap] =None, shape=None):
        """
        Create an IRSubTensor.

        Args:
            full_tensor: the full tensor
            indices: index list
            val_map: the value operation to merge SubTensors into one
        """
        if not isinstance(full_tensor, IRFullTensor):
            raise TypeError(f"Expected IRFullTensor but got {full_tensor}")
        super().__init__(shape=shape, name=full_tensor.name)

        # the full tensor
        self._full_tensor = full_tensor

        # the index from full_tensor
        self._index_map = _to_index_map(indices)

        # val map
        self._val_map = _to_value_map(val_map)

    def __eq__(self, other):

        if isinstance(other, IRFullTensor):
            return self.parent == other and \
                   self.shape == other.shape and \
                   self.val_map == ValueMap(0, 1)
        if isinstance(other, IRSubTensor):
            return self.parent == other.parent and \
                   self.indices == other.indices and \
                   self.val_map == other.val_map and \
                   self.shape == other.shape
        return False

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
    def val_map(self):
        return copy.copy(self._val_map)

    def __copy__(self):
        """
        Copy the tensor that will have the exactly same id
        except the empty attached cell

        Returns:
            tensor
        """
        tensor = IRSubTensor(self.parent, self.indices, self.val_map, self._shape)
        for key in self.__dict__:
            setattr(tensor, key, getattr(self, key))
        # clear attached cells
        tensor._cell = list()
        return tensor

    def as_param(self):
        """
        Set the tensor as trainable parameter
        """
        if not self.parent.is_param():
            self.parent.as_param()
        self.requires_grad = True
        self._is_param = True
        self._is_grad = False
        return self

    def as_grad(self):
        if not self.parent.is_grad():
            self.parent.as_grad()
        self._is_grad = True
        self._is_param = False
        return self

    def get_grad(self, fcell: IRCell):
        """
        Get gradient of this tensor which is associated by a
        forward cell
        """
        if not self.requires_grad:
            raise RuntimeError("require a gradient for a non-grad tensor")
        full_grad = self.parent.grad
        if full_grad is None:
            return None
        if self in fcell.inputs():
            fdst_cells = self.parent.forward_dst_cells()
            ref_cells = list()
            for dst_cell in fdst_cells:
                for input in dst_cell.inputs():
                    if self.overlap(input):
                        ref_cells.append(dst_cell)
                        break
            ref_times = len(ref_cells)
            if ref_times == 0:
                raise RuntimeError("Internal Error: ref time is 0")
            idx = ref_cells.index(fcell)
            grad = full_grad.select(
                indices = self.indices,
                val_map = (idx, ref_times),
                shape = self.shape
            )
            return grad.as_grad()
        elif self in fcell.outputs():
            grad = full_grad.select(
                indices = self.indices,
                val_map = self.val_map,
                shape = self.shape
            )
            return grad.as_grad()
        else:
            raise RuntimeError(f"{self} not found in cell {fcell}")

    def select(self, indices: Union[Tuple, IndexMap], val_map: Union[Tuple, ValueMap, None], shape=None):
        """
        Select an IRSubTensor

        Args:
            indices: the index of this tensor's index

            val_map: the value operation to merge 
                    co-located indices of SubTensors into one

            shape: the sub_tensor shape

        Returns:
            IRSubTensor
        """
        sub_ind_map = _to_index_map(indices)
        sub_val_map = _to_value_map(val_map)

        # index mapping
        index_map = self.indices.map(sub_ind_map)
        # value mapping
        val_map = self.val_map.map(sub_val_map)

        sub_tensor = self.parent.select(index_map, val_map, shape)
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
            if self.parent != other.parent:
                return False
            return self.indices.overlap(other.indices) and \
                   self.val_map.overlap(other.val_map)
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
                val_map = self.val_map & other.val_map
                sub_tensor = self.parent.select(
                    indices = indices,
                    val_map = val_map,
                    shape = indices.shape
                )
                return sub_tensor
            else:
                raise NotImplementedError("Customized IRTensor not support")
        return None

    def __repr__(self):
        dscp = f'SubTensor(id={self._id}, shape={self.shape}, device={self.device}, ind={self.indices}, val={self.val_map})'
        return dscp