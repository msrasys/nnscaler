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

from typing import List, Optional, Union, Tuple, NewType, Dict

from cube.ir.cten import IRCell, IRTensor
import cube.ir.dtype as irdtype

StartEnd = NewType('[start:end)', Tuple[int, int])
IdxChunk = NewType('(index, chunks)', Tuple[int, int])


class IndexMap:

    def __init__(self, indmap: Tuple[StartEnd]):
        """!
        Create an index map.

        @param indmap Union[Tuple[StartEnd], IndexMap]: index range [start, end) for each dimension
        
        @return indmap IndexMap: the created new instance of index map.
        """
        if isinstance(indmap, IndexMap):
            indmap = indmap.indices
        assert all(isinstance(dim, tuple) and len(dim) == 2 for dim in indmap), "expected Tuple[Tuple[int, int]]"
        self._indices: Tuple[StartEnd] = tuple(indmap)
        self._shape = tuple(end - start for (start, end) in self._indices)

    def __eq__(self, other):
        if isinstance(other, IndexMap):
            if self.ndims != self.ndims:
                return False
            for dim in range(self.ndims):
                if self.indices[dim] != other.indices[dim]:
                    return False
            return True
        return False

    def __hash__(self) -> int:
        return hash(tuple([self.ndims]+list(self._indices)))

    @property
    def indices(self) -> Tuple[StartEnd]:
        return self._indices

    @property
    def ndims(self) -> int:
        """
        Number of dimensions of the index map
        """
        return len(self._indices)

    @property
    def shape(self) -> List[int]:
        """
        Get the shape of the slice
        """
        return self._shape

    def map(self, submap: Tuple[StartEnd]):
        """!
        Map from the current indmap by sub_indices.

        @param submap Union[Tuple[StartEnd], IndexMap]: IndexMap of this IndexMap

        @return indmap IndexMap: the mapped IndexMap
        """
        submap: IndexMap = IndexMap(submap)
        assert self.ndims == submap.ndims, "Expected same dimensions of submap"
        sub = list()
        for dim in range(self.ndims):
            s1, e1 = self.indices[dim]
            s2, e2 = submap.indices[dim]
            start = s1 + s2
            end = start + e2 - s2
            assert end <= e1, f"select out of boundary at dim {dim}: ({self})[{submap}]"
            sub.append((start, end))
        return IndexMap(tuple(sub))

    def overlap(self, other) -> bool:
        """
        Check if this indmap overlapped with the other

        @param other IndexMap

        @return overlap bool: True has overlap, otherwise False
        """
        if not isinstance(other, IndexMap):
            raise TypeError("Expected IndexMap")
        
        if other.ndims != self.ndims:
            raise TypeError("Expected same dimension")

        for dim in range(self.ndims):
            start1, end1 = self.indices[dim]
            start2, end2 = other.indices[dim]
            if min(end1, end2) <= max(start1, start2):
                return False
        return True

    def __and__(self, other):
        """!
        Get the common part

        @param other IndexMap: the other one
        
        @return indexmap IndexMap: index map for the common part
        """
        if not self.overlap(other):
            return None
        tile = []
        for dim in range(self.ndims):
            start1, end1 = self.indices[dim]
            start2, end2 = other.indices[dim]
            start = max(start1, start2)
            end = min(end1, end2)
            tile.append((start, end))
        return IndexMap(tuple(tile))

    def __repr__(self) -> str:
        dscp = ','.join(f'{start}:{end}' for (start, end) in self.indices)
        return dscp


class ValueMap:
    r"""
    Represent the value split.

    replica: the replicated group:
        different replicated operator (no gradient accumulation) stands for different group

    weight: the partitioned but tensor replicated group:
        different replicated tensor (gradient accumulation) stands for different group
    """

    def __init__(self, weight: IdxChunk):
        """
        Create a value map.
        @param weight Union[IdxChunk, ValueMap]: the (idx, nchunks)

        @return valmap ValueMap: a new instance.
        """
        if isinstance(weight, ValueMap):
            weight = weight.weight
        assert len(weight) == 2 and all(isinstance(i, int) for i in weight), \
            "expected weight to be (idx, nchunks)"
        self._weight = weight

    @property
    def weight(self) -> IdxChunk:
        """!
        Get value partitioned chunks in tha accumulcated group
        """
        return self._weight
    
    def overlap(self, other) -> bool:
        """!
        Check on value overlapping.
        Note the overlap can only be within a same accumulation group and 
        a same replication group.
        """
        if not isinstance(other, ValueMap):
            raise TypeError("Expected ValueMap")
        idx1, nchunk1 = self.weight
        idx2, nchunk2 = self.weight
        span1 = (idx1 * nchunk2, idx1 * nchunk2 + nchunk2)
        span2 = (idx2 * nchunk1, idx2 * nchunk1 + nchunk1)
        if max(span1[0], span2[0]) < min(span1[1], span2[1]):
            return True
        return False

    def __eq__(self, other) -> bool:
        """!
        Check whether tensor is same to other tensor.
        Note we treat tensors in different replica region as different
        tensors, also they may have same data in reality.
        """
        if isinstance(other, ValueMap):
            return other.weight == self.weight
        return False

    def __hash__(self) -> int:
        return hash(self._weight)

    def map(self, submap: IdxChunk):
        """!
        Select the value chunk at position (idx, chunk) given the current view
        No change will make for the replica group.

        @param idnmap Union[ValueMap, IdxChunk]: the (index, chunk) for current view

        @return valmap ValueMap: the selected one
        """
        if isinstance(submap, ValueMap):
            submap: IdxChunk = submap.weight
        idx, chunk = self.weight
        sub_idx, sub_chunk = submap
        idx = idx * sub_chunk + sub_idx
        chunk = sub_chunk * chunk
        return ValueMap((idx, chunk))

    def __and__(self, other):
        """
        Find the common part

        @param other ValueMap

        @return Optional[None]
        """
        if not isinstance(other, ValueMap):
            raise TypeError("Expected ValueMap for & operator")
        if not self.overlap(other):
            return None
        if self.weight[1] == other.weight[1]:
            return ValueMap(self.weight)
        if self.weight[1] == 1:
            return ValueMap(other.weight)
        elif other.weight[1] == 1:
            return ValueMap(self.weight)
        else:
            raise ValueError(f"Not supported common value map: {self}, {other}")

    def __repr__(self):
        return f'({self.weight[0]}/{self.weight[1]})'


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
        self._segments : Dict[(ValueMap, IndexMap), int] = dict()

        self.requires_grad = requires_grad

    def __hash__(self) -> int:
        return self._id

    def __copy__(self):
        """
        Full tensor should only exist one instance per id

        Returns:
            tensor
        """
        return self

    def like(self):
        """!
        Create a IRFullTensor with same meta data but a different id.

        @return tensor IRFullTensor: the created tensor
        """
        tensor = IRFullTensor(self.shape, self.name, self.requires_grad, self.dtype)
        return tensor

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
        """!
        Add the tensor and its operator into consumer list.
        The tensor should be in cell.inputs()

        @param cell IRCell: node to be consumer
        @param tensor IRTensor: tensor to be consumed tensors
        @param idx int: the index to be inserted
        """
        assert tensor in cell.inputs(), f"tensor {tensor} not in node: {cell} inputs"
        if not isinstance(cell, IRCell) or not isinstance(tensor, IRTensor):
            raise TypeError("Expect an IRCell and an IRTensor")
        assert cell not in self._consumers, f"{cell} already exists as consumer"
        self._consumers.insert(idx, cell)
        self._ctensors.insert(idx, tensor)
        for t in self.ctensors:
            t._dirty_grad = True

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

    @property
    def grad(self) -> Optional[Union[IRTensor, float]]:
        return self._grad

    @grad.setter
    def grad(self, val: Optional[Union[IRTensor, float]]):
        """
        int indicates the tensor is the loss tensor.
        """
        assert isinstance(val, (IRFullTensor, float)) or val is None, f"grad can only be IRFullTensor or None, but got {val}"
        self._grad = val
        self._requires_grad = False if val is None else True
        if isinstance(val, IRFullTensor):
            assert val.shape == self.shape, f"IRFullTensor gradient shape mismatch."
        for tensor in self.ctensors + self.ptensors:
            tensor._dirty_grad = True

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._requires_grad = val
        if val and self.grad is None:
            self.grad = IRFullTensor(self.shape, 'g' + self.name, False).as_grad()
        elif not val and self.grad is not None:
            self.grad = None
        for tensor in self.ctensors + self.ptensors:
            tensor._dirty_grad = True

    def as_param(self):
        """
        Set the tensor as trainable parameter
        """
        self.requires_grad = True
        self._is_param = True
        self._is_grad = False

    def as_grad(self):
        self.requires_grad = False
        self._is_param = False
        self._is_grad = True
        return self

    def select(self, indmap: IndexMap, valmap: ValueMap):
        """!
        Select a SubTensor from FullTensor.

        @param indmap IndexMap: the index range of this tensor
        @param valmap ValueMap: the value range of this tensor

        @return subtensor IRSubTensor: the selected SubTensor
        """
        indmap, valmap = IndexMap(indmap), ValueMap(valmap)
        keys = (indmap, valmap)
        # print(f'key: {keys}, hash {hash(keys)}')
        # return tensor to keep id same for same sub tensor
        if keys in self._segments:
            tid = self._segments[keys]
            sub_tensor = IRSubTensor(self, indmap, valmap, tid=tid)
        else:
            sub_tensor = IRSubTensor(self, indmap, valmap)
            self._segments[keys] = sub_tensor.tid
        return sub_tensor

    def tosub(self):
        """!
        Convert to SubTensor by selecting all indmap and full value

        @return sub_tensor IRSubTensor: the sub-tensor
        """
        if self.shape is None:
            raise RuntimeError("Expected know shape")
        indmap = []
        for dimlen in self.shape:
            indmap.append((0, dimlen))
        sub_tensor = self.select(
            indmap=tuple(indmap),
            valmap=(0, 1),
        )
        return sub_tensor

    def __repr__(self):
        dscp = f'FullTensor(id={self._id}, shape={self.shape}, device={self.device})'
        return dscp


class IRSubTensor(IRTensor):

    def __init__(self, ftensor: IRFullTensor,
                 indmap: Union[Tuple[StartEnd], IndexMap],
                 valmap: Union[Tuple[StartEnd], ValueMap],
                 **kwargs):
        """
        Create an IRSubTensor.

        @param ftensor IRFullTensor: the full tensor
        @param indmap IndexMap: index map
        @param valmap ValueMap: value map
        """
        indmap, valmap = IndexMap(indmap), ValueMap(valmap)
        assert isinstance(ftensor, IRFullTensor), "Expcted ftensor to be IRFullTensor"
        super().__init__(shape=indmap.shape, name=ftensor.name, **kwargs)
        for attr in IRFullTensor._meta:
            setattr(self, attr, getattr(ftensor, attr))
        self.cell = None
        # the full tensor
        self._full_tensor = ftensor
        # the index from full_tensor
        self._indmap: IndexMap = indmap
        # val map
        self._valmap: ValueMap = valmap
        # grad flag
        self._dirty_grad = True

    def __eq__(self, other) -> bool:
        if isinstance(other, IRSubTensor):
            return self._id == other._id

    def __hash__(self) -> int:
        return self._id

    @property
    def parent(self) -> IRFullTensor:
        """
        Return the full tensor of this sub tensor
        """
        return self._full_tensor

    @property
    def indmap(self) -> Tuple[StartEnd]:
        """
        Get index range of each dimension of this tensor in its full tensor

        @return indices Tuple[StartEnd]: indices
        """
        return self._indmap.indices

    @property
    def valmap(self) -> IdxChunk:
        """
        Get value range of this tensor in tis full tensor

        @return idxchunk IdxChunk: (idx, nchunks)
        """
        return self._valmap.weight

    @property
    def ndims(self) -> int:
        return len(self.shape)

    def splitdims(self) -> Tuple[int]:
        """!
        Get partitioned dimensions

        @return dims int: the partitioned dimension.
        """
        return tuple(
            dim for dim in range(self.ndims) if self.shape[dim] != self.parent.shape[dim]
        )

    def catdim(self, other: IRTensor) -> Optional[int]:
        """!
        Get concatable dimensions with other IRSubTensor

        @parm other IRSubTensor
        @return dim int: the concatable dimension. None means no such dimension
        """
        assert isinstance(other, IRSubTensor), "expected IRSubTensor"
        if other.parent != self.parent or self.valmap != other.valmap:
            return None
        cat_dim: int = None
        for dim in range(self.ndims):
            if self.indmap[dim] != other.indmap[dim]:
                s1, e1 = self.indmap[dim]
                s2, e2 = other.indmap[dim]
                if min(e1, e2) == max(s1, s2):
                    if cat_dim is None:
                        cat_dim = dim
                    else:
                        return None
                else:
                    return None
        return cat_dim
    
    def concat(self, other: IRTensor, dim: int) -> IRTensor:
        """!
        concat dimension with other IRSubTensor. The concatenate
        order will follow the index map order.

        @param other IRSubTensor
        @param dim int: the concat dimension
        @return tensor IRSubTensor: the concatenated tensor
        """
        assert isinstance(other, IRSubTensor), "expected IRSubTensor"
        assert self.parent == other.parent and self.valmap == other.valmap
        indmap = []
        for cdim in range(self.ndims):
            if cdim == dim:
                (s1, e1), (s2, e2) = self.indmap[cdim], other.indmap[cdim]
                assert min(e1, e2) == max(s1, s2), f"fail to concat: {cdim} should be concatable"
                indmap.append((min(s1, s2), max(e1, e2)))
            else:
                assert self.indmap[cdim] == other.indmap[cdim], f"fail to concat: {cdim} should be same"
                indmap.append(self.indmap[cdim])
        valmap = self.valmap
        tensor = self.parent.select(tuple(indmap), valmap)
        return tensor

    def accumable(self, tensors: Union[IRTensor, List[IRTensor]]) -> bool:
        """!
        Check whether tensors are accumable with this tensor

        @param: tensors Union[IRTensor, List[IRTensor]]
        @return accumable bool: True if accumable
        """
        tensors: List[IRSubTensor] = [tensors,] if isinstance(tensors, IRSubTensor) else tensors
        assert all(isinstance(t, IRSubTensor) for t in tensors), "Expected IRSubTensor or List[IRSubTensor]"
        if any(t.parent != self.parent for t in tensors) or any(t.indmap != self.indmap for t in tensors):
            return False
        if any(t.indmap != self.indmap for t in tensors):
            return False
        if any(t.valmap[1] != self.valmap[1] for t in tensors):
            return False
        return self.valmap[1] % (len(tensors) + 1) == 0

    def accum(self, tensors: Union[IRTensor, List[IRTensor]]) -> IRTensor:
        """!
        Accumulate tensor on value dimension.
        The replica id will be 

        @param: tensors Union[IRTensor, List[IRTensor]]
        @return tensor IRSubTensor: accumulated tensor
        """
        tensors: List[IRSubTensor] = [tensors,] if isinstance(tensors, IRSubTensor) else tensors
        assert self.accumable(tensors), "Not accumable"
        nreduce = len(tensors) + 1
        assert self.valmap[1] % nreduce == 0
        # TODO: make accum more robust
        cid = min(t.valmap[0] for t in [self] + tensors) // nreduce
        valmap = (cid, self.valmap[1] // nreduce)
        indmap = self.indmap
        tensor = self.parent.select(indmap, valmap)
        return tensor

    def __copy__(self):
        """
        Copy the tensor that will have the exactly same id
        except the empty attached cell

        @return tensor IRSubTensor: the same tensor in a new instance
        """
        tensor = IRSubTensor(self.parent, self.indmap, self.valmap, tid=self.tid)
        for key in self.__dict__:
            setattr(tensor, key, getattr(self, key))
        # clear attached cells
        tensor._cell = None
        return tensor

    @property
    def grad(self) -> Optional[Union[IRTensor, float]]:
        """
        Get gradient of this tensor.

        Gradient can be:
         - None: the tensor doesn't require gradient
         - 1.0: the tensor is loss tensor (scalar)
         - IRSubTensor: the tensor requires gradient and is not the loss tensor (scalar)

        Gradient cannot be set and can only be inferred by its IRFullTensor.
        The gradient will be lazy updated when its IRFullTensor gets
        new consumed / produced tensors
        """
        if not self._dirty_grad:
            return self._grad

        assert isinstance(self.cell, IRCell), "No cell attached to this tensor."
        full_grad = self.parent.grad
        if full_grad is None or isinstance(full_grad, float):
            self._grad = full_grad
        # this tensor is consumed
        elif self in self.cell.inputs():
            # for backard, we assume in final distributed graph,
            # each tensor can be represented as nested <replica, value, dim>
            consumers = []
            for ctensor, consumer in zip(self.parent.ctensors, self.parent.consumers):
                if ctensor == self and consumer.cid not in consumers:
                    consumers.append(consumer.cid)
            valmap = (consumers.index(self.cell.cid), len(consumers))
            grad = full_grad.select(
                indmap = self.indmap,
                valmap = valmap,
            )
            self._grad = grad
            self._dirty_grad = False
            return grad
        # this tensor is produced
        elif self in self.cell.outputs():
            grad = full_grad.select(
                indmap = self.indmap,
                valmap = (0, 1),
            )
            self._grad = grad
        else:
            raise RuntimeError("Visit gradient of a tensor that is potentially generated by IRAdapter")
        self._dirty_grad = False
        self._requires_grad = False if full_grad is None else True
        return self._grad

    @property
    def requires_grad(self) -> bool:
        return self.parent._requires_grad

    # partition primitives

    def select(self,
               indmap: Union[Tuple[StartEnd], IndexMap],
               valmap: Union[IdxChunk, ValueMap]) -> IRTensor:
        """
        Select an IRSubTensor

        @param indmap IndexMap: the index map of this tensor's index

        @param valmap ValueMap: the value map of this tensor's value

        @return subtensor IRSubTensor: the selected tensor
        """
        indmap, valmap = IndexMap(indmap), ValueMap(valmap)
        # index mapping
        indmap = self._indmap.map(indmap)
        # value mapping
        valmap = self._valmap.map(valmap)
        sub_tensor = self.parent.select(indmap, valmap)
        return sub_tensor

    def replicate(self, num: int) -> List[IRTensor]:
        """!
        Partition primitive
            - replicate: replicate the tensor.

        @return tensor IRTensor: the copied tensor
        """
        tensors = []
        for _ in range(num):
            tensor = self.parent.select(
                indmap=self.indmap,
                valmap=self.valmap,
            )
            tensors.append(tensor)
        return tensors

    def split_dim(self, dim: int, num: int) -> List[IRTensor]:
        """
        Partition primitive:
            split_dim: uniformly split the tensor along a dimension.

        @param dim int: the dimension to get partitioned
        @param num int: the number of sub-tensor generated

        @return sub_tensors List[IRSubTensor]: the generated sub-tensors
        """
        dim = dim + self.ndims if dim < 0 else dim
        assert dim < self.ndims, f"Dim should within ndims but {dim} >= {self.ndims})"
        # assert self.shape[dim] % num == 0, f"Expected dimension can be split: {self.shape[dim]} % {num} != 0"
        chunk_size = self.shape[dim] // num
        addone_num = self.shape[dim] % num

        indmap = []
        for tdim, nele in enumerate(self.shape):
            if tdim != dim:
                indmap.append((0, nele))
            else:
                indmap.append(None)
        sub_tensors = list()
        for cid in range(num):
            num_prev_addone = addone_num if cid >= addone_num else cid
            addone = int(cid < addone_num)
            indmap[dim] = (chunk_size * cid + num_prev_addone, chunk_size * (cid+1) + addone + num_prev_addone)
            sub_tensor = self.select(
                indmap=tuple(indmap),
                valmap=(0,1),
            )
            sub_tensors.append(sub_tensor)
        return sub_tensors

    def split_val(self, num: int) -> List[IRTensor]:
        """!
        Partition primitive:
            split_val: uniformly split the tensor value.

        @param num int: the number of sub-tensor generated

        @return sub_tensors List[IRSubTensor]: the generated sub-tensors
        """
        # full shape
        indmap = []
        for nele in self.shape:
            indmap.append((0, nele))
        sub_tensors = list()
        for idx in range(num):
            valmap = self._valmap.map((idx, num))
            sub_tensor = self.select(
                indmap=tuple(indmap),
                valmap=valmap,
            )
            sub_tensors.append(sub_tensor)
        return sub_tensors

    def overlap(self, other) -> bool:
        """!
        Check whether the two subtensors are overlapped.

        @param other IRSubTensor

        @return overlapped bool: True if they are overlapped else False
        """
        if isinstance(other, IRSubTensor):
            if self.parent != other.parent:
                return False
            return self._indmap.overlap(other._indmap) and \
                   self._valmap.overlap(other._valmap)
        return False

    def common(self, other) -> Optional[IRTensor]:
        """!
        Get the common sub-tensor

        @param other IRSubTensor

        @return subtensor Optional[IRSubTensor]: the common sub-tensor.
            If not common region, return None
        """
        if self.overlap(other):
            indmap = self._indmap & other._indmap
            valmap = self._valmap & other._valmap
            sub_tensor = self.parent.select(
                indmap = indmap,
                valmap = valmap,
            )
            return sub_tensor
        return None

    def __repr__(self) -> str:
        anno = 't'
        if self.is_param():
            anno = 'w'
        if self.is_grad():
            anno = 'g'
        split_dims = self.splitdims()
        dscp = f'{anno}{self._id}(p{self.parent._id},{self.shape},d{split_dims},v{self._valmap})'
        return dscp

    def extra_repr(self) -> str:
        anno = 't'
        if self.is_param():
            anno = 'w'
        if self.is_grad():
            anno = 'g'
        dscp = f'{anno}{self._id}(id={self._id}, shape={self.shape}, dev={self.device}, ind=[{self._indmap}], val={self._valmap})'
        return dscp
