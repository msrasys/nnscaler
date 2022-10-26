from typing import Callable, Dict, List, Tuple, Optional
import copy
import numpy as np
from regex import R
from cube.ir.cten import IRCell

from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.ir.tensor import IndexMap, ValueMap

from cube.ir.adapter.prim import IRAdapterPrim
from cube.ir.adapter.prim import AllGatherPrim      # d2r
from cube.ir.adapter.prim import AllToAllPrim       # d2d
from cube.ir.adapter.prim import AllReducePrim      # v2r
from cube.ir.adapter.prim import ReduceScatterPrim  # v2d
from cube.ir.adapter.prim import ChunkPrim          # r2d

from cube.ir.adapter.prim import MovePrim           # p2p
from cube.ir.adapter.prim import BroadcastPrim
from cube.ir.adapter.prim import RDScatterPrim, RVScatterPrim
from cube.ir.adapter.prim import RDGatherPrim, RVGatherPrim
from cube.runtime.device import DeviceGroup


TShape = Tuple[int, ...]
TRVD = Tuple[int, ...]


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
    def ndevs(self):
        return len(self.subtensors)

    @property
    def mat(self):
        return self._mats

    def tensor(self, r: int, v: int, d: List[int]) -> IRSubTensor:
        """
        Get subtenor indexed by RVD position.
        """
        assert r <= self.R and v <= self.V and len(d) == len(self.D), "out of scope"
        indices = [r, v] + list(d)
        return self._mats[tuple(indices)]

    def __repr__(self):
        dscp = f'T{self.ftensor._id}<R({self.R}),V({self.V}),D({self.D})>'
        return dscp

    # ====== inshard transformation primitives ===== #

    def d2r(self, dim: int, chunks: int):
        """
        RVD Primitive: dimension to replica
        collective: allgather
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
        RVD Primitive: dimension to dimension
        collective: all-to-all
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
        RVD Prmitive: value to replica
        collective: all-reduce
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
        RVD Primitive: value to dimension
        collective: reduce-scatter 
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
        RVD Primitive: replica to dimension
        collective: split
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

    def incr(self, chunks: int, devices: Optional[np.ndarray] = None):
        """
        RVD+ Prmitive: increase replica
        collective: broadcast
        """
        layout = list(self.vec)
        layout[0] = layout[0] * chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # set device
        if devices is not None:
            assert devices.size == len(self.subtensors) * chunks
            for tensor, devid in zip(glayout.mat.flatten(), devices.flatten()):
                tensor.cell = IRCell('dummy', '', 0, 0, init_outputs=False)
                tensor.cell.device = int(devid)
        # setup prims
        imat = GridLayout.dims2last(self.mat, [0]).flatten()
        omat = GridLayout.dims2last(glayout.mat, [0]).reshape(-1, chunks)
        prims = []
        for src, dsts in zip(imat, omat):
            if chunks == 1:
                prims.append(MovePrim([src], dsts))
            else:
                prims.append(BroadcastPrim([src], [src] + list(dsts)))
        return glayout, prims

    def decr(self, chunks: int, devices: Optional[np.ndarray] = None):
        """
        RVD+ Prmitive: decrease replica
        collective: move
        """
        layout = list(self.vec)
        assert layout[0] % chunks == 0, f"not divisible replica: {layout[0]} // {chunks}"
        layout[0] = layout[0] // chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # set device
        if devices is not None:
            assert devices.size == len(self.subtensors) // chunks
            for tensor, devid in zip(glayout.mat.flatten(), devices.flatten()):
                tensor.cell = IRCell('dummy', '', 0, 0, init_outputs=False)
                tensor.cell.device = int(devid)
        # setup prims
        imat = GridLayout.dims2last(self.mat, [0]).reshape(-1, chunks)
        omat = GridLayout.dims2last(glayout.mat, [0]).flatten()
        prims = []
        for srcs, dst in zip(imat, omat):
            prims.append(MovePrim([srcs[0]], [dst]))
        return glayout, prims

    def incd(self, chunks: int, dim: int, devices: Optional[np.ndarray] = None):
        """
        RVD+ Prmitive: increase dimension
        collective: rdscatter
        """
        layout = list(self.vec)
        layout[2+dim] = layout[2+dim] * chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # set device
        if devices is not None:
            assert devices.size == len(self.subtensors) * chunks
            for tensor, devid in zip(glayout.mat.flatten(), devices.flatten()):
                tensor.cell = IRCell('dummy', '', 0, 0, init_outputs=False)
                tensor.cell.device = int(devid)
        # setup prims
        imat = GridLayout.dims2last(self.mat, [2+dim]).flatten()
        omat = GridLayout.dims2last(glayout.mat, [2+dim]).reshape(-1, chunks)
        prims = []
        for src, dsts in zip(imat, omat):
            prims.append(RDScatterPrim([src], dsts, dim=dim))
        return glayout, prims

    def decd(self, chunks: int, dim: int, devices: Optional[np.ndarray] = None):
        """
        RVD+ Prmitive: increase dimension
        collective: rdgather
        """
        layout = list(self.vec)
        assert layout[2+dim] % chunks == 0, f"not divisible dim: {self.D[dim]} % {chunks} != 0"
        layout[2+dim] = layout[2+dim] // chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # set device
        if devices is not None:
            assert devices.size == len(self.subtensors) // chunks
            for tensor, devid in zip(glayout.mat.flatten(), devices.flatten()):
                tensor.cell = IRCell('dummy', '', 0, 0, init_outputs=False)
                tensor.cell.device = int(devid)
        # setup prims
        imat = GridLayout.dims2last(self.mat, [2+dim]).reshape(-1, chunks)
        omat = GridLayout.dims2last(glayout.mat, [2+dim]).flatten()
        prims = []
        for srcs, dst in zip(imat, omat):
            prims.append(RDGatherPrim(srcs, [dst], dim=dim))
        return glayout, prims

    def incv(self, chunks: int, devices: Optional[np.ndarray] = None):
        """
        RVD+ Primitive: increase value partition
        collective: rvscatter
        """
        layout = list(self.vec)
        layout[1] = layout[1] * chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # set device
        if devices is not None:
            assert devices.size == len(self.subtensors) * chunks
            for tensor, devid in zip(glayout.mat.flatten(), devices.flatten()):
                tensor.cell = IRCell('dummy', '', 0, 0, init_outputs=False)
                tensor.cell.device = int(devid)
        # setup prims
        imat = GridLayout.dims2last(self.mat, [1]).flatten()
        omat = GridLayout.dims2last(glayout.mat, [1]).reshape(-1, chunks)
        prims = []
        for src, dsts in zip(imat, omat):
            prims.append(RVScatterPrim([src], dsts))
        return glayout, prims

    def decv(self, chunks: int, devices: Optional[np.ndarray] = None):
        """
        RVD+ Primitive: decrease value partition
        collective: rvgather
        """
        layout = list(self.vec)
        assert layout[1] % chunks == 0, f"not divisable value split: {self.V} % {chunks} != 0"
        layout[1] = layout[1] * chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # set device
        if devices is not None:
            assert devices.size == len(self.subtensors) // chunks
            for tensor, devid in zip(glayout.mat.flatten(), devices.flatten()):
                tensor.cell = IRCell('dummy', '', 0, 0, init_outputs=False)
                tensor.cell.device = int(devid)
        # setup prims
        imat = GridLayout.dims2last(self.mat, [1]).reshape(-1, chunks)
        omat = GridLayout.dims2last(glayout.mat, [1]).flatten()
        prims = []
        for srcs, dst in zip(imat, omat):
            prims.append(RVGatherPrim(srcs, [dst]))
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
    def dims2last(mat: np.ndarray, dims: List[int]) -> np.ndarray:
        """
        Permute a matrix by putting dimensions to the last.
        """
        axes = list(range(len(mat.shape)))
        for dim in dims:
            axes.remove(dim)
        axes += list(dims)
        return np.transpose(mat, axes)

    @staticmethod
    def dims2orig(mat: np.ndarray, last_dims: List[int]) -> np.ndarray:
        axes = list(range(len(mat.shape)))
        for dim in last_dims:
            axes.remove(dim)
        axes += list(last_dims)
        axes = np.argsort(np.array(axes))
        return np.transpose(mat, axes)

    @staticmethod
    def changed_dims(src: TRVD, dst: TRVD) -> Tuple[List[int], List[int]]:
        """
        Get changed dimensions

        @param src Tuple[int]: the source RVD layout
        @param dst Tuple[int]: the destination RVD layout
        
        @return inc_dims Tuple[int]: the dimensions that need to increase
        @return dec_dims Tuple[int]: the dimensions that need to decrease
        """
        assert len(src) == len(dst)
        inc_dims, dec_dims = [], []
        for dim, (slen, dlen) in enumerate(zip(src, dst)):
            if slen < dlen:
                inc_dims.append(dim)
            elif slen > dlen:
                dec_dims.append(dim)
        return inc_dims, dec_dims

    @staticmethod
    def grid(ftensor: IRFullTensor, r: int, v: int, dims: Tuple[int], devices: Optional[Tuple[int]] = None):
        """
        partition a ftensor using grid layout of <r, v, *dims>
        """
        def dummy_assign(tensor: IRSubTensor, devid: int):
            tensor.cell = IRCell('dummy', '', 0, 0, init_outputs=False)
            tensor.cell.device = devid

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
            valmap = ValueMap((indices[0], v))
            indmap = []
            shape = []
            for dim, (nchunk, index) in enumerate(zip(dims, indices[1:])):
                assert ftensor.shape[dim] % nchunk == 0, f"not dividable for {nchunk} chunks over dim {dim}. ftensor shape: {ftensor.shape}"
                csize = ftensor.shape[dim] // nchunk
                start = csize * index
                indmap.append((start, start+csize))
                shape.append(csize)
            subtensor = ftensor.select(tuple(indmap), valmap)
            # replicate
            subtensors = [copy.copy(subtensor) for _ in range(r)]
            all_subtensors += subtensors
            mats[(slice(None),)+indices] = np.array(subtensors, dtype=IRSubTensor)

        # devices
        if devices is not None:
            assert len(devices) == len(all_subtensors), f"devices number {len(devices)} not match with RVD number {len(all_subtensors)}"
            for tensor, devid in zip(mats.flatten(), devices):
                dummy_assign(tensor, int(devid))

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
            if subtensor.tid not in replicas:
                replicas[subtensor.tid] = []
            _tindex[tid] = [len(replicas[subtensor.tid])]
            replicas[subtensor.tid].append(subtensor)
            # setup value
            _tindex[tid].append(subtensor.valmap[0])
            vchunks.add(subtensor.valmap[1])
            # setup dimensions
            for dim in range(ndims):
                snele = subtensor.shape[dim]
                start = subtensor.indmap[dim][0]
                fnele = ftensor.shape[dim]
                if fnele % snele != 0 or start % snele != 0:
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
        nchunks = set(t.valmap[1] for t in subtensors)
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


class PathFinder:
    """
    Pathfinder for generating communication plans for GridLayout
    """

    # intra-shard: cached nodes. paths[shape][i][j] = List[int] of indices from (src -> dst]
    _cached_intra_nodes: Dict[Tuple[TShape, int], Tuple[TRVD]] = {}
    _cached_intra_edges: Dict[Tuple[TShape, int], np.ndarray] = {}
    _cached_intra_paths: Dict[Tuple[TShape, int], Dict[TRVD, List[List[int]]]] = {}

    # inter-shard: cached nodes. paths[(shape1, shape2)][i][j] = List[int]
    _cached_inter_nodes: Dict[Tuple[TShape, int, int], Tuple[Tuple[TRVD]]] = {}
    _cached_inter_edges: Dict[Tuple[TShape, int, int], Tuple[np.ndarray]] = {}
    _cached_inter_paths: Dict[Tuple[TShape, int, int], Dict[TRVD, List[List[int]]]] = {}

    @staticmethod
    def intra_path(ftensor: IRFullTensor, ilayout: GridLayout, olayout: GridLayout, cost_fn: Optional[Callable] = None) -> Tuple[List[GridLayout], List[IRAdapterPrim]]:
        """
        Get primitive path of transforming ilayout into olayout.
        ilayout has the same device set with olayout

        @param ftensor IRFullTensor: The fulltensor
        @param ilayout GridLayout: input tensor layout
        @param olayout GridLayout: output tensor layout
        @param cost_fn Optional[Callable]: cost function of each primitive.
            Default (None) will use transmission volume as metrics

        @return layouts List[GridLayout]: each transformation.
        @return prims List[IRAdapterPrim]: the primitives to perform transformation.
        """
        cost_fn = PathFinder.default_cost_fn if cost_fn is None else cost_fn
        shape = tuple(ftensor.shape)
        key = (shape, ilayout.ndevs)
        src = (ilayout.R, ilayout.V) + tuple(ilayout.D)
        dst = (olayout.R, olayout.V) + tuple(olayout.D)
        if src == dst: return [], []
        
        # get paths using dijkstra algorithm or cached
        if key in PathFinder._cached_intra_paths and src in PathFinder._cached_intra_paths[key]:
            paths = PathFinder._cached_intra_paths[key][src]
        else:
            # initialize the graph if not cached
            if key not in PathFinder._cached_intra_nodes:
                nodes, edges = PathFinder.init_intra_graph(ftensor, ilayout.ndevs, cost_fn)
                PathFinder._cached_intra_nodes[key] = nodes
                PathFinder._cached_intra_edges[key] = edges
                PathFinder._cached_intra_paths[key] = {}
            nodes = PathFinder._cached_intra_nodes[key]
            edges = PathFinder._cached_intra_edges[key]
            # build and initialize cost table
            cost = np.full((len(nodes),), np.inf)
            cost[nodes.index(src)] = 0
            # setup unvisited and visited set
            unvisited = set(range(len(nodes)))
            visited = set()
            paths = [[] for _ in range(len(nodes))]
            paths[nodes.index(src)] = [nodes.index(src)]
            # dijkstra body
            while len(unvisited) > 0:
                min_cost, visit = np.inf, None
                for idx in unvisited:
                    if cost[idx] < min_cost:
                        min_cost = idx
                        visit = idx
                if visit is None: break
                for neighbor in np.where(edges[visit] != np.inf)[0]:
                    if neighbor in visited: continue
                    new_cost = cost[visit] + edges[visit, neighbor]
                    if cost[neighbor] == np.inf or new_cost < cost[neighbor]:
                        cost[neighbor] = new_cost
                        paths[neighbor] = paths[visit] + [neighbor]
                    cost[neighbor] = min(cost[neighbor], cost[visit] + edges[visit, neighbor])
                unvisited.remove(visit)
                visited.add(visit)
            PathFinder._cached_intra_paths[key][src] = paths

        # print for debug
        for idx, path in enumerate(paths):
            print(f"{src} -> {nodes[idx]}: {' -> '.join([str(nodes[i]) for i in path])} | cost: {cost[idx]}")
        
        # get layout
        nodes = PathFinder._cached_intra_nodes[key]
        path = paths[nodes.index(dst)]
        assert len(path) > 0, f"Un-reachable src RVD ({src}) -> dst RVD ({dst})"

        layouts = [ilayout]
        all_prims = []
        curr_rvd = src
        for hop in path[1:]:
            hop_rvd = nodes[hop]
            ret, layout, prims = PathFinder.intra_transform(ftensor, curr_rvd, hop_rvd, layouts[-1])
            assert ret, "Internal Error."
            layouts.append(layout)
            all_prims += prims
            curr_rvd = hop_rvd
        return layouts, all_prims


    @staticmethod
    def inter_path(ftensor: IRFullTensor, ilayout: GridLayout, olayout: GridLayout, cost_fn: Optional[Callable] = None) -> Tuple[List[GridLayout], List[IRAdapterPrim]]:
        """
        Get primitives for transforming ilayout into olayout. ilayout has the different device set
        to olayout. And number of device of ilayout and olayout must be divisable by each other.

        @param ftensor IRFullTensor: The fulltensor
        @param ilayout GridLayout: input tensor layout
        @param olayout GridLayout: output tensor layout
        @param cost_fn Optional[Callable]: cost function of each primitive.
            Default (None) will use transmission volume as metrics

        @return layouts List[GridLayout]: each transformation.
        @return prims List[IRAdapterPrim]: the primitives to perform transformation.
        """
        cost_fn = PathFinder.default_cost_fn if cost_fn is None else cost_fn
        shape = tuple(ftensor.shape)
        key = (shape, ilayout.ndevs, olayout.ndevs)

        src = ('p',) + (ilayout.R, ilayout.V) + tuple(ilayout.D)
        dst = ('c',) + (olayout.R, olayout.V) + tuple(olayout.D)

        if key in PathFinder._cached_inter_nodes and src in PathFinder._cached_inter_paths[key]:
            nodes = PathFinder._cached_inter_nodes[key]
            paths = PathFinder._cached_inter_paths[key][src]
        else:
            if key in PathFinder._cached_inter_nodes:
                nodes = PathFinder._cached_inter_nodes[key]
                edges = PathFinder._cached_inter_edges[key]
            else:
                nodes, edges = PathFinder.init_inter_graph(ftensor, ilayout.ndevs, olayout.ndevs, cost_fn)
                PathFinder._cached_inter_nodes[key] = nodes
                PathFinder._cached_inter_edges[key] = edges
                PathFinder._cached_inter_paths[key] = {}
            # build cost
            cost = np.full((len(nodes),), np.inf)
            cost[nodes.index(src)] = 0
            # setup unvisited and visited set
            unvisited = set(range(len(nodes)))
            visited = set()
            paths = [[] for _ in range(len(nodes))]
            paths[nodes.index(src)] = [nodes.index(src)]
            # dijkstra body
            while len(unvisited) > 0:
                min_cost, visit = np.inf, None
                for idx in unvisited:
                    if cost[idx] < min_cost:
                        min_cost = idx
                        visit = idx
                if visit is None: break
                for neighbor in np.where(edges[visit] != np.inf)[0]:
                    if neighbor in visited: continue
                    new_cost = cost[visit] + edges[visit, neighbor]
                    if cost[neighbor] == np.inf or new_cost < cost[neighbor]:
                        cost[neighbor] = new_cost
                        paths[neighbor] = paths[visit] + [neighbor]
                    cost[neighbor] = min(cost[neighbor], cost[visit] + edges[visit, neighbor])
                unvisited.remove(visit)
                visited.add(visit)
            PathFinder._cached_inter_paths[key][src] = paths
        
        # print for debug
        # for idx, path in enumerate(paths):
        #     print(f"{src} -> {nodes[idx]}: {' -> '.join([str(nodes[i]) for i in path])} | cost: {cost[idx]}")
        
        path = paths[nodes.index(dst)]
        # print(f"Find path: {' -> '.join(str(nodes[i]) for i in path)}")
        assert len(path) > 0, f"Un-reachable src RVD ({src}) -> dst RVD ({dst})"

        # setup consumer begining devices
        cpaths = tuple(idx for idx in path if nodes[idx][0] == 'c')
        curr_devs = np.array([t.device[0] for t in olayout.mat.flatten()]).reshape(dst[1:])
        curr_node = dst[1:]
        # print('result device map:', list(cdevs.flatten()))
        for hop in cpaths[:-1][::-1]:
            hop_rvd = nodes[hop][1:]
            curr_devs = PathFinder.intra_devmap(curr_node, hop_rvd, curr_devs)
            curr_node = hop_rvd
        consumer_entry_devs = curr_devs
        # print('calculated consumer device map: ', list(cdevs.flatten()))
        # setup primitives for communication
        side, layouts, all_prims = 'p', [ilayout], []
        curr_rvd = src[1:]
        for hop in path[1:]:
            use_inter_step = side != nodes[hop][0]
            hop_rvd = nodes[hop][1:]
            if not use_inter_step:
                ret, layout, prims = PathFinder.intra_transform(ftensor, curr_rvd, hop_rvd, layouts[-1])
                assert ret, "Internal Error"
            else:
                ret, layout, prims = PathFinder.inter_transform(ftensor, curr_rvd, hop_rvd, layouts[-1], consumer_entry_devs)
            layouts.append(layout)
            all_prims += prims
            curr_rvd = hop_rvd
            side = nodes[hop][0]
        return layouts, all_prims

    @staticmethod
    def intra_transform(ftensor: IRFullTensor, src_rvd: TRVD, dst_rvd: TRVD, ilayout: Optional[GridLayout] = None) -> Tuple[GridLayout, List[IRAdapterPrim]]:
        """
        Get output layout and transform primitives from a source rvd layout to dst_rvd layout, 
        
        @param ftensor IRFullTensor
        @param src_rvd Tuple[int]
        @param dst_rvd Tuple[int]
        @param ilayout Optional[GridLayout]

        @return ret bool: True if there is a primitive performed 
        @return layout Optonal[GridLayout]: the RVD layout if ilayout is not None
        @return prims Optional[List[IRAdapterPrim]]: the prmitives in transformation
        """
        if ilayout is not None:
            assert src_rvd == tuple(ilayout.vec)
        inc_dims, dec_dims = GridLayout.changed_dims(src_rvd, dst_rvd)
        if len(inc_dims) != 1 or len(dec_dims) != 1:
            return False, None, None
        inc_idx, dec_idx = inc_dims[0], dec_dims[0]
        if src_rvd[dec_idx] % dst_rvd[dec_idx] != 0:
            return False, None, None
        if inc_idx == 1:
            return False, None, None
        src = ilayout if ilayout is not None else GridLayout.grid(ftensor, src_rvd[0], src_rvd[1], list(src_rvd[2:]))
        chunks = src_rvd[dec_idx] // dst_rvd[dec_idx]
        if dec_idx >= 2 and inc_idx == 0:  # d2r
            olayout, prims = src.d2r(dec_idx-2, chunks)
        elif dec_idx >= 2 and inc_idx >= 2:  # d2d
            olayout, prims = src.d2d(dec_idx-2, inc_idx-2, chunks)
        elif dec_idx == 1 and inc_idx == 0:  # v2r
            olayout, prims = src.v2r(chunks)
        elif dec_idx == 1 and inc_idx >= 2:  # v2d
            olayout, prims = src.v2d(inc_idx-2, chunks)
        elif dec_idx == 0 and inc_idx >= 2:  # r2d
            olayout, prims = src.r2d(inc_idx-2, chunks)
        else:
            raise RuntimeError(f"Cannot find primitive. Report as a bug. dec-idx: {dec_idx}, inc-idx: {inc_idx}")
        return True, (olayout if ilayout is not None else None), prims

    @staticmethod
    def intra_devmap(src_rvd: TRVD, dst_rvd: TRVD, src_devs: np.ndarray):
        """
        Infer device from source rvd to destination rvd
        """
        assert tuple(src_rvd) == tuple(src_devs.shape), f"RVD mis-matches with device shape, {src_rvd} != {src_devs.shape}"
        # get changed dimensions
        inc_idx, dec_idx = GridLayout.changed_dims(src_rvd, dst_rvd)
        assert len(inc_idx) == 1 and len(dec_idx) == 1
        inc_idx, dec_idx = inc_idx[0], dec_idx[0]
        assert src_rvd[dec_idx] % dst_rvd[dec_idx] == 0
        chunks = src_rvd[dec_idx] // dst_rvd[dec_idx]
        # reshape array to match devices
        dst_devs = np.full(dst_rvd, -1, dtype=int)
        src_devs = GridLayout.dims2last(src_devs, [inc_idx, dec_idx]).reshape(-1, chunks)
        dst_devs = GridLayout.dims2last(dst_devs, [dec_idx, inc_idx])
        dshape = dst_devs.shape
        # set up device
        dst_devs = dst_devs.reshape(-1, chunks)
        for rid, devs in enumerate(src_devs):
            dst_devs[rid] = devs
        dst_devs = dst_devs.reshape(dshape)
        # permute to original shape
        dst_devs = GridLayout.dims2orig(dst_devs, [dec_idx, inc_idx])
        return dst_devs

    @staticmethod
    def inter_transform(ftensor, src_rvd: TRVD, dst_rvd: TRVD, ilayout: Optional[GridLayout] = None, dst_devs: Optional[np.array] = None):
        """
        Get output layout and transform primitives from a source rvd layout to dst_rvd layout, 
        
        @param ftensor IRFullTensor
        @param src_rvd Tuple[int]
        @param dst_rvd Tuple[int]
        @param ilayout Optional[GridLayout]

        @return ret bool: True if there is a primitive performed 
        @return layout Optonal[GridLayout]: the RVD layout if ilayout is not None
        @return prims Optional[List[IRAdapterPrim]]: the prmitives in transformation
        """
        inc_dims, dec_dims = GridLayout.changed_dims(src_rvd, dst_rvd)
        if len(inc_dims) == 0 and len(dec_dims) == 0:
            inc_dims = [0]
        if not ((len(inc_dims) == 1 and len(dec_dims) == 0) or (len(inc_dims) == 0 and len(dec_dims) == 1)):
            return False, None, None
        inc_idx = inc_dims[0] if len(inc_dims) == 1 else None
        dec_idx = dec_dims[0] if len(dec_dims) == 1 else None
        src = ilayout if ilayout is not None else GridLayout.grid(ftensor, src_rvd[0], src_rvd[1], list(src_rvd[2:]))
        if isinstance(inc_idx, int):
            if not (dst_rvd[inc_idx] % src_rvd[inc_idx] == 0):
                return False, None, None
            chunks = dst_rvd[inc_idx] // src_rvd[inc_idx]
            if inc_idx == 0:
                olayout, prims = src.incr(chunks, dst_devs)
            elif inc_idx == 1:
                olayout, prims = src.incv(chunks, dst_devs)
            elif inc_idx > 1:
                olayout, prims = src.incd(chunks, inc_idx-2, dst_devs)
            else:
                raise RuntimeError(f"Cannot find primitive. Report as a bug. dec-idx: {dec_idx}, inc-idx: {inc_idx}")
        else:
            if not (src_rvd[dec_idx] % dst_rvd[dec_idx] == 0):
                return False, None, None
            chunks = src_rvd[dec_idx] // dst_rvd[dec_idx]
            if dec_idx == 0:
                olayout, prims = src.decr(chunks, dst_devs)
            elif dec_idx == 1:
                olayout, prims = src.decv(chunks, dst_devs)
            elif dec_idx > 1:
                olayout, prims = src.decd(chunks, dec_idx-2, dst_devs)
            else:
                raise RuntimeError(f"Cannot find primitive. Report as a bug. dec-idx: {dec_idx}, inc-idx: {inc_idx}")
        return True, (olayout if ilayout is not None else None), prims

    @staticmethod
    def init_intra_graph(ftensor: IRFullTensor, ndevs: int, cost_fn: Optional[Callable]) -> Tuple[List[TRVD], np.ndarray]:
        """
        Initialize the graph of RVD status graph.

        @param ftensor IRFullTensor: the full tensor
        @param ndevs int: total device number

        @return nodes Tuple[TRVD]
        @return edges np.ndarray: edges among nodes
        """
        nodes = tuple(PathFinder.get_inshard_space(ftensor, ndevs))
        edges = np.full((len(nodes), len(nodes)), np.inf)
        # initialize the cost
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if i == j: continue
                src, dst = nodes[i], nodes[j]
                ret, _, prims = PathFinder.intra_transform(ftensor, src, dst)
                if not ret: continue
                edges[i, j] = cost_fn(prims[0])
        return nodes, edges

    @staticmethod
    def init_inter_graph(ftensor: IRFullTensor, idevs: int, odevs: int, cost_fn: Callable) -> Tuple[List[TRVD], np.ndarray]:
        """
        Initialize the graph of RVD status graph.

        An additional positition tage is append to at the first element of each node, i.e.,
            For source (producer) layout: ('p', 2,1,1,2) means <R(2), V(1), D(1,2)>
            For dest (consumer) layout: ('c', 2,1,1,2) means <R(2), V(1), D(1,2)>

        @param ftensor IRFullTensor: the full tensor
        @param idevs int: total device number of source tensor

        @return nodes Tuple[TRVD]
        @return edges np.ndarray: edges among nodes
        """
        shape = tuple(ftensor.shape)
        if (shape, idevs) in PathFinder._cached_intra_nodes:
            src_nodes = PathFinder._cached_intra_nodes[(shape, idevs)]
            src_edges = PathFinder._cached_intra_edges[(shape, idevs)]
        else:
            src_nodes, src_edges = PathFinder.init_intra_graph(ftensor, idevs, cost_fn)
            PathFinder._cached_intra_nodes[(shape, idevs)] = src_nodes
            PathFinder._cached_intra_edges[(shape, idevs)] = src_edges
            PathFinder._cached_intra_paths[(shape, idevs)] = {}
        if (shape, odevs) in PathFinder._cached_inter_edges:
            dst_nodes = PathFinder._cached_intra_nodes[(shape, odevs)]
            dst_edges = PathFinder._cached_intra_edges[(shape, odevs)]
        else:
            dst_nodes, dst_edges = PathFinder.init_intra_graph(ftensor, odevs, cost_fn)
            PathFinder._cached_intra_nodes[(shape, odevs)] = dst_nodes
            PathFinder._cached_intra_edges[(shape, odevs)] = dst_edges
            PathFinder._cached_intra_paths[(shape, odevs)] = {}
        nodes = tuple(('p',) + n for n in src_nodes ) + tuple(('c',) + n for n in dst_nodes)
        edges = np.full((len(nodes), len(nodes)), np.inf)
        edges[:len(src_nodes), :len(src_nodes)] = src_edges
        edges[len(src_nodes):,len(src_nodes):] = dst_edges
        # NVLink: 300GBps Inter-node: 100Gbps
        comm_factor = 24
        for i in range(len(src_nodes)):
            for j in range(len(dst_nodes)):
                src, dst = src_nodes[i], dst_nodes[j]
                # set for [i, len(src_nodes) + j]
                ret, _, prims = PathFinder.inter_transform(ftensor, src, dst)
                if not ret: continue
                edges[i, len(src_nodes) + j] = cost_fn(prims[0]) * comm_factor
                # set for [len(src_nodes) + j, i]
                ret, _, prims = PathFinder.inter_transform(ftensor, dst, src)
                assert ret
                edges[len(src_nodes) + j, i] = cost_fn(prims[0]) * comm_factor
        return nodes, edges

    # utility function
    @staticmethod
    def get_inshard_space(ftensor: IRSubTensor, ndevs: int) -> List[Tuple[int, ...]]:
        """
        Get all possible space that can be transformed from layout.

        This space is pruned by limiting partition number of each RVD dimension
        in the range of [min(ilayout[dim], olayout[dim]), max(ilayout[dim], olayout[dim])]

        @param ftensor IRFullTensor
        @param ilayout GridLayout: input layout
        @param olayout GridLayout: output layout

        @return layouts List[GridLayout]: 
        """
        all_layouts: List[int] = []
        
        def factors(ndevs: int, length: int):
            if length == 1: yield [ndevs]
            else:
                for i in range(1, ndevs + 1):
                    if ndevs % i == 0:
                        for res in factors(ndevs // i, length - 1):
                            yield [i] + res
        
        for rvd in factors(ndevs, 2+len(ftensor.shape)):
            skip = False
            for dimlen, pnum in zip(ftensor.shape, rvd[2:]):
                if dimlen % pnum != 0:
                    skip = True
                    break
            if not skip:
                all_layouts.append(tuple(rvd))
        return all_layouts

    @staticmethod
    def default_cost_fn(prim: IRAdapterPrim) -> int:
        return prim.volume() + 1 # 1 is hop penalty
