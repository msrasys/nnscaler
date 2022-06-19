from typing import Dict, List, Tuple
import copy
import numpy as np

from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.ir.tensor import IndexMap, ValueMap

from cube.ir.adapter.prim import AllGatherPrim      # d2r
from cube.ir.adapter.prim import AllToAllPrim       # d2d
from cube.ir.adapter.prim import AllReducePrim      # v2r
from cube.ir.adapter.prim import ReduceScatterPrim  # v2d
from cube.ir.adapter.prim import ChunkPrim          # r2d


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

    @property
    def mat(self):
        return self._mats

    # ====== primitives ===== #

    def d2r(self, dim: int, chunks: int):
        """
        dimension to replica: allgather
        """
        layout = list(self.vec)
        assert layout[2+dim] % chunks == 0, f"not dividable dim: {layout[2+dim]} // {chunks}"
        layout[0] = layout[0] * chunks
        layout[2+dim] = layout[2+dim] // chunks
        glayout = GridLayout.grid(self.ftensor,
                       r=layout[0], v=layout[1], dims=layout[2:])
        # set device
        imat = GridLayout.transpose(self.mat, 0, 2+dim)
        omat = GridLayout.transpose(glayout.mat, 2+dim, 0)
        for itensor, otensor in zip(imat.flatten(), omat.flatten()):
            otensor._cell = itensor._cell
        prims = []
        for itensors, otensors in zip(imat.reshape(-1, chunks), omat.reshape(-1, chunks)):
            prims.append(AllGatherPrim(itensors, otensors, dim))
        return glayout, prims

    def d2d(self, from_dim: int, to_dim: int, chunks: int):
        """
        dimension to dimension: all-to-all
        """
        layout = list(self.vec)
        assert layout[2+from_dim] % chunks == 0, f"not dividable dim: {layout[2+from_dim]} // {chunks}"
        layout[2+from_dim] = layout[2+from_dim] // chunks
        layout[2+to_dim] = layout[2+to_dim] * chunks
        glayout = GridLayout.grid(self.ftensor,
                       r=layout[0], v=layout[1], dims=layout[2:])
        # set device
        imat = GridLayout.transpose(self.mat, 2+to_dim, 2+from_dim)
        omat = GridLayout.transpose(glayout.mat, 2+from_dim, 2+to_dim)
        for itensor, otensor in zip(imat.flatten(), omat.flatten()):
            otensor._cell = itensor._cell
        prims = []
        for itensors, otensors in zip(imat.reshape(-1, chunks), omat.reshape(-1, chunks)):
            prims.append(AllToAllPrim(itensors, otensors, from_dim, to_dim))
        return glayout, prims

    def v2r(self, chunks: int):
        """
        value to replica: all-reduce
        """
        layout = list(self.vec)
        assert layout[1] % chunks == 0, f"not dividable value chunks: {layout[1]} // {chunks}"
        layout[1] = layout[1] // chunks
        layout[0] = layout[0] * chunks
        glayout = GridLayout.grid(self.ftensor,
                       r=layout[0], v=layout[1], dims=layout[2:])
        # set device
        imat = GridLayout.transpose(self.mat, 0, 1)
        omat = GridLayout.transpose(glayout.mat, 1, 0)
        for itensor, otensor in zip(imat.flatten(), omat.flatten()):
            otensor._cell = itensor._cell
        prims = []
        for itensors, otensors in zip(imat.reshape(-1, chunks), omat.reshape(-1, chunks)):
            prims.append(AllReducePrim(itensors, otensors))
        return glayout, prims

    def v2d(self, dim: int, chunks: int):
        """
        value to dimension: reduce-scatter 
        """
        layout = list(self.vec)
        assert layout[1] % chunks == 0, f"not dividable value chunks: {layout[0]} // {chunks}"
        layout[1] = layout[1] // chunks
        layout[2+dim] = layout[2+dim] * chunks
        glayout = GridLayout.grid(self.ftensor,
                       r=layout[0], v=layout[1], dims=layout[2:])
        # set device
        imat = GridLayout.transpose(self.mat, 2+dim, 1)
        omat = GridLayout.transpose(glayout.mat, 1, 2+dim)
        for itensor, otensor in zip(imat.flatten(), omat.flatten()):
            otensor._cell = itensor._cell
        prims = []
        for itensors, otensors in zip(imat.reshape(-1, chunks), omat.reshape(-1, chunks)):
            prims.append(ReduceScatterPrim(itensors, otensors, dim))
        return glayout, prims

    def r2d(self, dim: int, chunks: int):
        """
        replica to dimension: split
        """
        layout = list(self.vec)
        assert layout[0] % chunks == 0, f"not dividable replica: {layout[0]} // {chunks}"
        layout[0] = layout[0] // chunks
        layout[2+dim] = layout[2+dim] * chunks
        glayout = GridLayout.grid(self.ftensor,
                       r=layout[0], v=layout[1], dims=layout[2:])
        # set device
        imat = GridLayout.transpose(self.mat, 2+dim, 0)
        omat = GridLayout.transpose(glayout.mat, 0, 2+dim)
        for itensor, otensor in zip(imat.flatten(), omat.flatten()):
            otensor._cell = itensor._cell
        prims = []
        for itensors, otensors in zip(imat.reshape(-1, chunks), omat.reshape(-1, chunks)):
            prims.append(ChunkPrim(itensors, otensors, dim))
            # ranks = tuple(t.device[0] for t in itensors)
            # for idx, (itensor, otensor) in enumerate(zip(itensors, otensors)):
            #     prims.append(ChunkPrim(itensor, otensor, dim, ranks))
        return glayout, prims

    # ================ solution ============= #

    def path(self, dst) -> Tuple:
        """
        Find a path from self to destination GridLayout using
        primitivies. This implementation uses search order of
        R -> V -> S.

        Args:
            dst: GridLayout
            auto_replace: bool
                If true, the consumer operator may be replaced
                to match the device assignment.

        Return:
            paths: List[GridLayout]
                the search path from source GridLayout (self)
                to destination GridLayout (self)
            comm_prims: List[IRAdapterPrim]
                communication primitives for translation
        """
        def step(ilayout: GridLayout, dec_idx: int, inc_idx: int, chunks: int) -> GridLayout:
            if dec_idx >= 2 and inc_idx == 0:  # d2r
                return ilayout.d2r(dec_idx-2, chunks)
            if dec_idx >= 2 and inc_idx >= 2:  # d2d
                return ilayout.d2d(dec_idx-2, inc_idx-2, chunks)
            if dec_idx == 1 and inc_idx == 0:  # v2r
                return ilayout.v2r(chunks)
            if dec_idx == 1 and inc_idx >= 2:  # v2d
                return ilayout.v2d(inc_idx-2, chunks)
            if dec_idx == 0 and inc_idx >= 2:  # r2d
                return ilayout.r2d(inc_idx-2, chunks)
            raise RuntimeError("Cannot find primitive. Report as a bug")
        
        comm_prims = []
        paths: List[GridLayout] = [self]
        dst: GridLayout = dst
        while paths[-1].vec != dst.vec:
            src: GridLayout = paths[-1]
            inc_idx, dec_idx = None, None
            for idx, (schunk, dchunk) in enumerate(zip(src.vec, dst.vec)):
                if schunk != dchunk:
                    # print(f'src: {src.vec}, dst: {dst.vec}')
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
                    # print(chunks, need_chunks)
                    olayout, oprims = step(src, dec_idx, inc_idx, chunks)
                    paths.append(olayout)
                    comm_prims += oprims
                    break
        return paths, comm_prims

    def __repr__(self):
        dscp = f'T{self.ftensor._id}<R({self.R}),V({self.V}),D({self.D})>'
        return dscp

    def print_dev_tensors(self):
        """
        print each device hold tensors.
        """
        devices: Dict[int, List[IRSubTensor]] = dict()
        for tensor in self.subtensors:
            assert len(tensor.device) == 1, f"got tensor device: {tensor.device}"
            if tensor.device[0] not in devices:
                devices[tensor.device[0]] = []
            devices[tensor.device[0]].append(tensor)
        devs = list(devices.keys())
        devs.sort()
        for dev in devs:
            print(f'dev{dev}:')
            for tensor in devices[dev]:
                print(f'\t{tensor.extra_repr()}')

    @staticmethod
    def transpose(mat: np.ndarray, dim0: int, dim1: int):
        """
        put the dim0 and dim1 of the mat to the last two dims
        """
        ndims = len(mat.shape)
        axes = list(range(ndims))
        assert dim0 < ndims and dim1 < ndims, "dim0 or dim1 out of index"
        axes.pop(max(dim0, dim1))
        axes.pop(min(dim0, dim1))
        axes += [dim0, dim1]
        return np.transpose(mat, axes)

    @staticmethod
    def grid(ftensor: IRFullTensor, r: int, v: int, dims: Tuple[int]):
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

    @staticmethod
    def togrid(ftensor: IRFullTensor, subtensors: List[IRSubTensor]):
        """
        convert ftensor and subtensors into a GridLayout.

        If failed, raise error
        """
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
                    print(subtensor, dim)
                    raise RuntimeError(
                        f"dimension split error:\n"
                        f"Full Tensor: {ftensor}\n"
                        f"full nele: {fnele}, sub nele: {snele}, start: {start}"
                    )
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
