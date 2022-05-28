r"""
SubTensor Gradient rule:

SubTensor's logical grad = SubTensor.parent.grad.select(
    indmap = SubTensor.indmap, 
    valmap = SubTensor.valmap,
    shape   = SubTensor.shape
)

FwOperation -> BpOperation rule:

1). for (FwOp) input tensors, gradient SubTensor is:
    indmap = input.indmap;
    val is splitted by referencing times on the indmap

2). for (FwOp) output tensors, gradient SubTensor is:
    indmap = output.indmap;
    val is always (0/1)
"""


from typing import List, Optional, Union, Tuple
import copy
import math

from cube.ir.cten import IRCell, IRTensor
import cube.ir.dtype as irdtype 


class IndexMap:

    def __init__(self, indmap):

        if not isinstance(indmap, tuple):
            raise TypeError("Expected indmap to be a tuple")

        if not all([isinstance(s, slice) for s in indmap]):
            raise NotImplementedError(
                "Only support for sliced index mapping"
            )
        self._indices: List[slice] = indmap

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
        Get indmap
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
        Map from the current indmap by sub_indices.

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
        Check if this indmap overlapped with the other

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

    def __sub__(self, other) -> Optional[List]:
        """
        Get the remaining part.
        We reuqire other should completely inside this tensor
        and the remaining part should be only one tile, else
        will return None

        Args:
            other: IndexMap

        Returns:
            IndexMap for the remaining part
        """
        if not isinstance(other, IndexMap):
            raise TypeError("Expected IndexMap")
        if self.ndims != other.ndims:
            return None
        dim_common: List[List[slice]] = [list() for _ in range(self.ndims)]
        dim_differ: List[List[slice]] = [list() for _ in range(self.ndims)]
        for dim, (slicer1, slicer2) in enumerate(zip(self.get(), other.get())):
            # self indices
            start1, stop1 = slicer1.start, slicer1.stop
            step1 = slicer1.step if slicer1.step else 1
            # other indices
            start2, stop2 = slicer2.start, slicer2.stop
            step2 = slicer2.step if slicer2.step else 1
            if step1 != 1 or step2 != 1:
                return None
            # no intersection
            if min(stop1, stop2) <= max(start1, start2):
                return None
            # set common
            start = max(start1, start2)
            stop = min(stop1, stop2)
            dim_common[dim].append(slice(start, stop, step1))
            # set difference
            if start1 == start2:
                if stop2 < stop1:
                    dim_differ[dim].append(slice(stop2, stop1, step1))
            elif stop1 == stop2:
                if start1 < start2:
                    dim_differ.append(slice(start1, start2, step1))
            else:
                raise NotImplementedError("Multipe indexmap is not supported")
        indmaps = list()
        splitdim = set()
        slices = list()
        for dim in range(self.ndims):
            common = dim_common[dim]
            differ = dim_differ[dim]
            if len(common) + len(differ) != 1:
                raise NotImplementedError("Multipe indexmap is not supported")
            if len(differ) == 1:
                splitdim.add(dim)
                slices.append(differ[0])
            else:
                slices.append(common[0])
        indmaps.append(IndexMap(tuple(slices)))
        return indmaps

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
                chk1, chk2 = self.chunk_num, other.chunk_num
                time1 = int(chk2 / math.gcd(chk1, chk2))
                time2 = int(chk1 / math.gcd(chk1, chk2))
                span1 = (self.idx * time1, self.idx * time1 + time1)
                span2 = (other.idx * time2, other.idx * time2 + time2)
                if max(span1[0], span2[0]) < min(span1[1], span2[1]):
                    return True
                else:
                    return False

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


def _to_indmap(indmap: Union[Tuple, IndexMap]) -> IndexMap:
    if not isinstance(indmap, tuple) and not isinstance(indmap, IndexMap):
        raise TypeError("Expected indmap to be tuple or IndexMap")
    if isinstance(indmap, tuple):
        indmap = IndexMap(indmap)
    return indmap


def _to_value_map(valmap: Union[Tuple, ValueMap, None]) -> ValueMap:
    if not isinstance(valmap, tuple) and \
       not isinstance(valmap, ValueMap) and \
       not valmap is None:
        raise TypeError("Expected valmap to be tuple, IndexMap or None")
    if valmap is None:
        valmap = ValueMap(0, 1)
    elif isinstance(valmap, tuple):
        if len(valmap) != 2:
            raise ValueError("Expected tuple to be (idx, chunk_num)")
        valmap = ValueMap(*valmap)
    return valmap


class IRFullTensor(IRTensor):
    """
    Full (logic) Tensor intermeidate representation.

    It records its Sub (physical) Tensors with corresponding
    producer operators and consumer operators following
    the sequentail execution order by its graph.
    """

    def __init__(self, shape=None, name=None, requires_grad=True, dtype=irdtype.float32):

        super().__init__(shape, name, dtype)

        # producer cell and produced sub tensor
        self._producers: List[IRCell] = list()
        self._ptensors : List[IRSubTensor] = list()

        # consumer cell and consumed sub tensor
        self._consumers: List[IRCell] = list()
        self._ctensors : List[IRSubTensor] = list()

        # record all created sub_tensors
        self._segments : List[IRSubTensor] = list()

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

    @property
    def producers(self) -> List[IRCell]:
        """
        Producer IRCell list
        """
        return self._producers

    @property
    def ptensors(self):
        """
        Produced IRSubTensor list correspongding to producer IRCell
        """
        return self._ptensors

    @property
    def consumers(self) -> List[IRCell]:
        """
        Consumer IRCell list
        """
        return self._consumers

    @property
    def ctensors(self):
        """
        Consumed IRSubTensor list correspongding to consumer IRCell
        """
        return self._ctensors

    def add_producer(self, cell: IRCell, tensor: IRTensor, idx: int = 0):
        if not isinstance(cell, IRCell) or not isinstance(tensor, IRTensor):
            raise TypeError("Expect an IRCell and an IRTensor")
        assert cell not in self._producers, f"{cell} already exists as producer"
        self._producers.insert(idx, cell)
        self._ptensors.insert(idx, tensor)

    def add_consumer(self, cell: IRCell, tensor: IRTensor, idx: int = 0):
        if not isinstance(cell, IRCell) or not isinstance(tensor, IRTensor):
            raise TypeError("Expect an IRCell and an IRTensor")
        assert cell not in self._consumers, f"{cell} already exists as consumer"
        self._consumers.insert(idx, cell)
        self._ctensors.insert(idx, tensor)

    def rm_producer(self, cell: IRCell) -> int:
        if cell not in self.producers:
            raise KeyError(f"Cell {cell} not found in producer")
        idx = self.producers.index(cell)
        self.producers.pop(idx)
        self.ptensors.pop(idx)
        return idx

    def rm_consumer(self, cell: IRCell) -> int:
        if cell not in self.consumers:
            raise KeyError(f"Cell {cell} not found in producer")
        idx = self.consumers.index(cell)
        self.consumers.pop(idx)
        self.ctensors.pop(idx)
        return idx

    def clear_producer_consumer(self) -> int:
        self._producers = []
        self._ptensors = []
        self._consumers = []
        self._ctensors = []

    def subtensors(self):
        """
        Get created sub-tensors of this tensor.
        """
        return copy.copy(self._segments)

    def as_param(self):
        """
        Set the tensor as trainable parameter
        """
        self.requires_grad = True
        self._is_param = True
        self._is_grad = False
        # for sub_tensor in self.ptensors + self.ctensors:
        #     sub_tensor.as_param()

    def as_grad(self):
        self._is_param = False
        self._is_grad = True
        # for sub_tensor in self.ptensors + self.ctensors:
        #     sub_tensor.as_grad()
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

    def select(self, indmap: Union[Tuple, IndexMap], valmap: Union[Tuple, ValueMap, None], shape: List[int]):
        """
        Select a SubTensor from FullTensor.

        Note due to implementation issue, one value in the full tensor
        cannot be splitted by different valmap

        Args:
            indmap: the index of this tensor's index

            valmap: how the tensor mapped from original value

            shape: the sub_tensor shape.

        Returns:
            IRSubTensor
        """
        indmap = _to_indmap(indmap)
        valmap = _to_value_map(valmap)

        # return tensor to keep id same for same sub tensor
        for sub_tensor in self.subtensors():
            if sub_tensor.indmap == indmap and sub_tensor.valmap == valmap:
                sub_tensor = copy.copy(sub_tensor)
                return sub_tensor

        sub_tensor = IRSubTensor(self, indmap, valmap, shape)
        for attr in IRFullTensor._attr:
            setattr(sub_tensor, attr, getattr(self, attr))
        sub_tensor.grad = None
        self._segments.append(sub_tensor)
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
        Convert to SubTensor by selecting all indmap and full value
        """
        if self.shape is None:
            raise RuntimeError("Expected know shape")
        slicers = list()
        for dim_len in self.shape:
            slicers.append(slice(0, dim_len, 1))
        sub_tensor = self.select(
            indmap=tuple(slicers),
            valmap=None,
            shape=self.shape
        )
        return sub_tensor

    def __repr__(self):
        dscp = f'FullTensor(id={self._id}, shape={self.shape}, device={self.device})'
        return dscp


class IRSubTensor(IRTensor):

    def __init__(self, full_tensor: IRTensor,
                 indmap: List[Union[Tuple, IndexMap]],
                 valmap: Optional[ValueMap] = None, shape=None):
        """
        Create an IRSubTensor.

        Args:
            full_tensor: the full tensor
            indmap: index list
            valmap: the value operation to merge SubTensors into one
        """
        if not isinstance(full_tensor, IRFullTensor):
            raise TypeError(f"Expected IRFullTensor but got {full_tensor}")
        super().__init__(shape=shape, name=full_tensor.name)

        # the full tensor
        self._full_tensor = full_tensor

        # the index from full_tensor
        self._indmap = _to_indmap(indmap)

        # val map
        self._valmap = _to_value_map(valmap)

    def __eq__(self, other):

        if isinstance(other, IRFullTensor):
            return self.parent == other and \
                   self.shape == other.shape and \
                   self.valmap == ValueMap(0, 1)
        if isinstance(other, IRSubTensor):
            return self.parent == other.parent and \
                   self.indmap == other.indmap and \
                   self.valmap == other.valmap and \
                   self.shape == other.shape
        return False

    @property
    def parent(self) -> IRFullTensor:
        """
        Return the full tensor of this sub tensor
        """
        return self._full_tensor

    @property
    def indmap(self) -> IndexMap:
        """
        Return indmap list mapped to the full tensor
        """
        return copy.copy(self._indmap)

    @property
    def valmap(self):
        return copy.copy(self._valmap)

    def __copy__(self):
        """
        Copy the tensor that will have the exactly same id
        except the empty attached cell

        Returns:
            tensor
        """
        tensor = IRSubTensor(self.parent, self.indmap, self.valmap, self._shape)
        for key in self.__dict__:
            setattr(tensor, key, getattr(self, key))
        # clear attached cells
        tensor._cell = None
        return tensor

    def get_grad(self, fcell: IRCell):
        """
        Get gradient of this tensor which is associated by a
        forward cell
        """
        if not self.requires_grad:
            self.grad = None
            return None
        full_grad = self.parent.grad
        if full_grad is None:
            self.grad = None
            return None
        if self in fcell.inputs():
            ref_cell_ids = list()
            for dst_cell in self.parent.consumers:
                for input in dst_cell.inputs():
                    if self.overlap(input) and dst_cell._id not in ref_cell_ids:
                        ref_cell_ids.append(dst_cell._id)
                        break
            ref_times = len(ref_cell_ids)
            if ref_times == 0:
                raise RuntimeError("Internal Error: ref time is 0")
            idx = ref_cell_ids.index(fcell._id)
            grad = full_grad.select(
                indmap = self.indmap,
                valmap = (idx, ref_times),
                shape = self.shape
            )
            self.grad = grad
            return grad
        elif self in fcell.outputs():
            grad = full_grad.select(
                indmap = self.indmap,
                valmap = (0, 1),
                shape = self.shape
            )
            self.grad = grad
            return grad
        else:
            raise RuntimeError(f"{self} not found in cell {fcell}")

    def select(self, indmap: Union[Tuple, IndexMap], valmap: Union[Tuple, ValueMap, None], shape=None):
        """
        Select an IRSubTensor

        Args:
            indmap: the index of this tensor's index

            valmap: the value operation to merge 
                    co-located indmap of SubTensors into one

            shape: the sub_tensor shape

        Returns:
            IRSubTensor
        """
        sub_ind_map = _to_indmap(indmap)
        sub_valmap = _to_value_map(valmap)

        # index mapping
        index_map = self.indmap.map(sub_ind_map)
        # value mapping
        valmap = self.valmap.map(sub_valmap)

        sub_tensor = self.parent.select(index_map, valmap, shape)
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
            return self.indmap.overlap(other.indmap) and \
                   self.valmap.overlap(other.valmap)
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
                indmap = self.indmap & other.indmap
                valmap = self.valmap & other.valmap
                sub_tensor = self.parent.select(
                    indmap = indmap,
                    valmap = valmap,
                    shape = indmap.shape
                )
                return sub_tensor
            else:
                raise NotImplementedError("Customized IRTensor not support")
        return None

    def difference(self, other):
        """
        Get differene part of sub-tensor

        Currently this requires tensor to be subset

        Args:
            other: IRSubTensor

        Returns:
            None for fail
        """
        pass

    def __repr__(self):
        anno = 't'
        if self.is_param():
            anno = 'w'
        if self.is_grad():
            anno = 'g'
        dscp = f'{anno}{self._id}(p{self.parent._id},{self.shape},{self.valmap})'
        return dscp

    def extra_repr(self):
        dscp = f'Tensor(id={self._id}, shape={self.shape}, device={self.device}, ind={self.indmap}, val={self.valmap})'
        return dscp
