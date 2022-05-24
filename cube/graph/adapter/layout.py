from typing import Dict, List, Tuple, Union, Optional
import copy
from matplotlib.style import available
import numpy as np

from cube.graph.tensor import IRFullTensor, IRSubTensor, IndexMap, ValueMap


class GridLayout:
    """
    This class assumes a full-tensor can only be
    uniformly partitioned / replicated on dimensions and values.

    A partition plan N-dim tensor layout can be represented as
    <R, V, dim1, ...,dimN>: R (replica), V (value), dim_i (dimension)
    """

    def __init__(self, ftensor: IRFullTensor, subtensors: List[IRSubTensor], mats: np.ndarray):
        """
        ftensor: N-dim FullTensor
        subtensors: List[IRSubTensors]
        mats: Array[IRSubTensor]:
            (2+N)-dim matrix, with index respect to <R, V, dim1, ..., dimN>
        """
        self.ftensor = ftensor
        self.subtensors = subtensors
        self._tindex: Dict[int, List[int]] = dict()
        self._mats = mats

    @property
    def R(self) -> int:
        return self._mats.shape[0]

    @property
    def V(self) -> int:
        return self._mats.shape[1]

    @property
    def D(self) -> Tuple[int]:
        return tuple(self._mats.shape[2:])

    @property
    def vec(self) -> Tuple[int]:
        return tuple(self._mats.shape)

    @property
    def ndims(self):
        return len(self._mats.shape)

    # def index(self, subtensor: IRSubTensor) -> List[int]:
    #     """
    #     Get index of <R, V, dim1, ..., dimN> of the subtensor
    #     """
    #     assert id(subtensor) in self._tindex, f"tensor: {subtensor} not found"
    #     return copy.copy(self._tindex(id(subtensor)))

    def get(self, r: bool = False, v: bool = False, d: Union[bool, int]=False) -> List[IRSubTensor]:        
        if r:
            nchunks = self.R
            idx = 0
        elif v:
            nchunks = self.V
            idx = 1
        elif isinstance(d, int):
            nchunks = self.D[d]
            idx = 2 + d
        else:
            raise ValueError("r, v, d should set at least one")
        axes = list(range(idx)) + list(range(idx+1, self.ndims)) + [idx]
        mat = np.transpose(self._mats, axes).reshape((-1, nchunks))
        for i in mat.shape[0]:
            yield mat[i]

    # ====== primitives ===== #

    def d2r(self, dim: int, chunks: int):
        """
        dimension to replica: allgather
        """
        layout = list(self.vec)
        assert layout[2+dim] % chunks == 0, f"not dividable dim: {layout[2+dim]} // {chunks}"
        layout[0] = layout[0] * chunks
        layout[2+dim] = layout[2+dim] // chunks
        return grid(self.ftensor,
                    r=layout[0], v=layout[1], dims=layout[2:])

        # all_tensors = []
        # for tensors in self.get(d=dim):
        #     assert len(tensors) % chunks == 0, "not dividable dim and chunks"
        #     tensors = tensors.reshape((-1, chunks))
        #     for group_tensors in tensors:  # go through each row
        #         indmap = []
        #         for idim in range(self.ndims):
        #             if idim != dim:
        #                 indmap.append(group_tensors[0].valmap.get()[idim])
        #             else:
        #                 slicer = slice(
        #                     group_tensors[0].indmap.get()[idim].start,
        #                     group_tensors[-1].indmap.get()[idim].stop, 1
        #                 )
        #                 indmap.append(slicer)
        #         valmap = group_tensors[0].valmap
        #         for tensor in group_tensors:
        #             gtensor = self.ftensor.select(indmap, tuple(valmap))
        #             gtensor._cell = tensor._cell  # set device
        #             all_tensors.append(gtensor)
        # return GridLayout(self.ftensor, all_tensors)

    def d2d(self, from_dim: int, to_dim: int, chunks: int):
        """
        dimension to dimension: all-to-all
        """
        layout = list(self.vec)
        assert layout[2+from_dim] % chunks == 0, f"not dividable dim: {layout[2+from_dim]} // {chunks}"
        layout[2+from_dim] = layout[2+from_dim] // chunks
        layout[2+to_dim] = layout[2+to_dim] * chunks
        return grid(self.ftensor,
                    r=layout[0], v=layout[1], dims=layout[2:])

        # if from_dim == to_dim:
        #     return self
        # all_tensors = []
        # for tensors in self.get(d=from_dim):
        #     assert len(tensors) % chunks == 0, "not dividable dim and chunks"
        #     tensors = tensors.reshape((-1, chunks))
        #     for group_tensors in tensors:
        #         for cid, tensor in enumerate(group_tensors):
        #             indmap = []
        #             for dim in range(self.ndims):
        #                 # from_dim gets nchunks larger
        #                 if dim == from_dim:
        #                     slicer = slice(
        #                         group_tensors[0].indmap.get()[dim].start,
        #                         group_tensors[-1].indmap.get()[dim].stop, 1
        #                     )
        #                     indmap.append(slicer)
        #                 # to_dim gets nchunks smaller
        #                 elif dim == to_dim:
        #                     nele = tensor.shape[dim] // chunks
        #                     start = tensor.indmap.get()[dim].start
        #                     slicer = slice(
        #                         start + nele * cid,
        #                         start + nele * (cid + 1), 1
        #                     )
        #                     indmap.append(slicer)
        #                 # others keep unchanged
        #                 else:
        #                     indmap = tensor.indmap.get()[dim]
        #             valmap = tensor.valmap
        #             ttensor = self.ftensor.select(indmap, tuple(valmap))
        #             ttensor._cell = tensor
        #             all_tensors.append(ttensor)
        # return GridLayout(self.ftensor, all_tensors)

    def v2r(self, chunks: int):
        """
        value to replica: all-reduce
        """
        layout = list(self.vec)
        assert layout[1] % chunks == 0, f"not dividable value chunks: {layout[1]} // {chunks}"
        layout[1] = layout[1] // chunks
        layout[0] = layout[0] * chunks
        return grid(self.ftensor,
                    r=layout[0], v=layout[1], dims=layout[2:])
        

    def v2d(self, dim: int, chunks: int):
        """
        value to dimension: reduce-scatter 
        """
        layout = list(self.vec)
        assert layout[1] % chunks == 0, f"not dividable value chunks: {layout[0]} // {chunks}"
        layout[1] = layout[1] // chunks
        layout[2+dim] = layout[2+dim] * chunks
        return grid(self.ftensor,
                    r=layout[0], v=layout[1], dims=layout[2:])


    def r2d(self, dim: int, chunks: int):
        """
        replica to dimension: split
        """
        layout = list(self.vec)
        assert layout[0] % chunks == 0, f"not dividable replica: {layout[0]} // {chunks}"
        layout[0] = layout[0] // chunks
        layout[2+dim] = layout[2+dim] * chunks
        return grid(self.ftensor,
                    r=layout[0], v=layout[1], dims=layout[2:])

    # ================ solution ============= #

    def path(self, dst) -> List:
        """
        find path ways from this layout to the target layout

        order: R -> V -> S
        """
        def step(layout: GridLayout, dec_idx: int, inc_idx: int, chunks: int) -> GridLayout:
            if dec_idx >= 2 and inc_idx == 0:  # d2r
                return layout.d2r(dec_idx-2, chunks)
            if dec_idx >= 2 and inc_idx >= 2:  # d2d
                return layout.d2d(dec_idx-2, inc_idx-2, chunks)
            if dec_idx == 1 and inc_idx == 0:  # v2r
                return layout.v2r(chunks)
            if dec_idx == 1 and inc_idx >= 2:  # v2d
                return layout.v2d(inc_idx-2, chunks)
            if dec_idx == 0 and inc_idx >= 2:  # r2d
                return layout.r2d(inc_idx-2, chunks)
            raise RuntimeError("Cannot find primitive. Report as a bug")

        paths: List[GridLayout] = [self]
        dst: GridLayout = dst
        while paths[-1].vec != dst.vec:
            src: GridLayout = paths[-1]
            inc_idx, dec_idx = None, None
            for idx, (schunk, dchunk) in enumerate(zip(src.vec, dst.vec)):
                if schunk != dchunk:
                    print(f'src: {src.vec}, dst: {dst.vec}')
                    if schunk < dchunk:
                        inc_idx = idx  # src should increase chunks on idx-dim
                        need_chunks = dchunk // schunk if dchunk % schunk == 0 else dchunk
                        for dec_idx in range(inc_idx+1, self.ndims):
                            # print(f'{dec_idx}/{self.ndims}')
                            if src.vec[dec_idx] > dst.vec[dec_idx]:
                                if src.vec[dec_idx] % dst.vec[dec_idx] != 0:
                                    available_chunks = dst.vec[dec_idx]
                                else:
                                    available_chunks = src.vec[dec_idx] // dst.vec[dec_idx]
                                chunks = min(available_chunks, need_chunks)
                                break
                        else:
                            raise RuntimeError("Cannot find feassible dimension. Report this as a bug.")
                    else:
                        dec_idx = idx
                        need_chunks = schunk // dchunk if schunk % dchunk == 0 else schunk
                        for inc_idx in range(dec_idx+1, self.ndims):
                            if src.vec[inc_idx] < dst.vec[inc_idx]:
                                if dst.vec[inc_idx] % src.vec[inc_idx] != 0:
                                    available_chunks = dst.vec[inc_idx]
                                else:
                                    available_chunks = dst.vec[inc_idx] // src.vec[inc_idx]
                                chunks = min(available_chunks, need_chunks)
                                break
                        else:
                            raise RuntimeError("Cannot find feassible dimension. Report this as a bug.")
                    print(chunks, need_chunks)
                    layout = step(src, dec_idx, inc_idx, chunks)
                    paths.append(layout)
                    break
        return paths

    def __repr__(self):
        dscp = f'T{self.ftensor._id}<R({self.R}),V({self.V}),D({self.D})>'
        return dscp


def grid(ftensor: IRFullTensor, r: int, v: int, dims: Tuple[int]) -> np.ndarray:
    """
    partition a ftensor using grid layout of <r, v, *dims>
    """
    mats = np.empty([r, v] + dims, dtype=IRSubTensor)
    all_subtensors = []

    def iter_idx(dims: List[int]) -> Tuple[int]:
        if len(dims) == 0:
            yield ()
        else:
            for i in range(dims[0]):
                for indices in iter_idx(dims[1:]):
                    yield (i,) + indices
    # generate tensor for each index
    for indices in iter_idx([v,]+dims):
        valmap = ValueMap(indices[0], v)
        indmap = []
        shape = []
        for dim, (nchunk, index) in enumerate(zip(dims, indices[1:])):
            assert ftensor.shape[dim] % nchunk == 0, f"not dividable for {nchunk} chunks over dim {dim}"
            csize = ftensor.shape[dim] // nchunk
            start = csize * index
            indmap.append(slice(start, start+csize, 1))
            shape.append(csize)
        subtensor = ftensor.select(tuple(indmap), valmap, shape)
        # replicate
        subtensors = [copy.copy(subtensor) for _ in range(r)]
        all_subtensors += subtensors
        mats[(slice(None),)+indices] = np.array(subtensors, dtype=IRSubTensor)
    return GridLayout(ftensor, all_subtensors, mats)


def togrid(ftensor: IRFullTensor, subtensors: List[IRSubTensor]) -> Optional[GridLayout]:
    _replica: int = None
    _value: int = None
    _dims: List[int] = [None] * len(ftensor.shape)
    _tindex: Dict[int, List[int]] = dict()

    ndims = len(ftensor.shape)

    replicas: Dict[int, List[IRSubTensor]] = dict()
    vchunks: set = set()
    dchunks: List[set] = [set() for _ in range(ndims)]
    
    for subtensor in subtensors:
        tid = id(subtensor)
        # set up replica
        if subtensor._id not in replicas:
            replicas[subtensor._id] = []
        _tindex[tid] = [len(replicas[subtensor._id])]
        replicas[subtensor._id].append(subtensor)
        # setup value
        _tindex[tid].append(subtensor.valmap.idx)
        vchunks.add(subtensor.valmap.chunk_num)
        # setup dimensions
        for dim in range(ndims):
            snele = subtensor.shape[dim]
            start = subtensor.indmap.get()[dim].start
            fnele = ftensor.shape[dim]
            if fnele % snele != 0 or start % snele != 0:
                raise RuntimeError(f"dimension split error: full nele: {fnele}, sub nele: {snele}, start: {start}")
            dchunks[dim].add(fnele // snele)
            _tindex[tid].append(start // snele)
    # replica (R)
    nreplicas = set(len(ts) for ts in replicas.values())
    if len(nreplicas) != 1:
        raise RuntimeError(f"different replicas: {nreplicas}")
    _replica = list(nreplicas)[0]
    # value (V)
    nchunks = set(t.valmap.chunk_num for t in subtensors)
    if len(nchunks) != 1:
        raise RuntimeError(f"different value split: {nchunks}")
    _value = list(nchunks)[0]
    # dimension (D)
    for dim in range(ndims):
        if len(dchunks[dim]) != 1:
            raise RuntimeError(f"different dimension split: {dchunks[dim]}")
        _dims[dim] = list(dchunks[dim])[0]
    
    # set matrix
    mats = np.empty([_replica, _value] + _dims, dtype=IRSubTensor)
    for subtensor in subtensors:
        idx = tuple(_tindex[id(subtensor)])
        assert mats[idx] is None, f"repeating entry. mutiple same {subtensor}"
        mats[tuple(idx)] = subtensor
    assert not (mats == None).any(), "at least one entry not set"
    return GridLayout(ftensor, subtensors, mats)
