from typing import Callable, Dict, List, Tuple, Optional
import copy
import numpy as np
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

    def incr(self, chunks: int, devices: List[int]):
        """
        RVD+ Prmitive: increase replica
        collective: broadcast
        """
        layout = list(self.vec)
        layout[0] = layout[0] * chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # set device
        imat = GridLayout.dims2last(self.mat, [0]).flatten()
        omat = GridLayout.dims2last(glayout.mat, [0]).reshape(-1, chunks)
        prims = []
        for src, dsts in zip(imat, omat):
            prims.append(BroadcastPrim(src, [src] + list(dsts)))
        return glayout, prims


    def decr(self, chunks: int, devices: List[int]):
        """
        RVD+ Prmitive: decrease replica
        collective: move
        """
        layout = list(self.vec)
        assert layout[0] % chunks == 0, f"not divisible replica: {layout[0]} // {chunks}"
        layout[0] = layout[0] // chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # set device
        imat = GridLayout.dims2last(self.mat, [0]).reshape(-1, chunks)
        omat = GridLayout.dims2last(glayout.mat, [0]).flatten()
        prims = []
        for srcs, dst in zip(imat, omat):
            prims.append(MovePrim(srcs[0], dst))
        return glayout, prims


    def incd(self, chunks: int, dim: int, devices: List[int]):
        """
        RVD+ Prmitive: increase dimension
        collective: rdscatter
        """
        layout = list(self.vec)
        layout[2+dim] = layout[2+dim] * chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # TODO: set device
        imat = GridLayout.dims2last(glayout.mat, [2+dim]).flatten()
        omat = GridLayout.dims2last(glayout.mat, [2+dim]).reshape(-1, chunks)
        prims = []
        for src, dsts in zip(imat, omat):
            prims.append(RDScatterPrim(src, dsts, dim=dim))
        return glayout, prims


    def decd(self, chunks: int, dim: int, devices: List[int]):
        """
        RVD+ Prmitive: increase dimension
        collective: rdgather
        """
        layout = list(self.vec)
        assert layout[2+dim] % chunks == 0, f"not divisible dim: {self.D[dim]} % {chunks} != 0"
        layout[2+dim] = layout[2+dim] // chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # set device
        imat = GridLayout.dims2last(self.mat, [2+dim]).reshape(-1, chunks)
        omat = GridLayout.dims2last(glayout.mat, [2+dim]).flatten()
        prims = []
        for srcs, dst in zip(imat, omat):
            prims.append(RDGatherPrim(srcs, dst, dim=dim))
        return glayout, prims


    def incv(self, chunks: int, devices: List[int]):
        """
        RVD+ Primitive: increase value partition
        collective: rvscatter
        """
        layout = list(self.vec)
        layout[1] = layout[1] * chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # TODO: set device
        imat = GridLayout.dims2last(glayout.mat, [1]).flatten()
        omat = GridLayout.dims2last(glayout.mat, [1]).reshape(-1, chunks)
        prims = []
        for src, dsts in zip(imat, omat):
            prims.append(RVScatterPrim(src, dsts))
        return glayout, prims

    def decv(self, chunks: int, devices: List[int]):
        """
        RVD+ Primitive: decrease value partition
        collective: rvgather
        """
        layout = list(self.vec)
        assert layout[1] % chunks == 0, f"not divisable value split: {self.V} % {chunks} != 0"
        layout[1] = layout[1] * chunks
        glayout = GridLayout.grid(self.ftensor, layout[0], layout[1], layout[2:])
        # TODO: set device
        imat = GridLayout.dims2last(self.mat, [1]).reshape(-1, chunks)
        omat = GridLayout.dims2last(glayout.mat, [1]).flatten()
        prims = []
        for srcs, dst in zip(imat, omat):
            prims.append(RVGatherPrim(srcs, dst))
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
                assert ftensor.shape[dim] % nchunk == 0, f"not dividable for {nchunk} chunks over dim {dim}"
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
            for tensor, devid in zip(all_subtensors, devices):
                dummy_assign(tensor, devid)

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


TShape = Tuple[int, ...]
TRVD = Tuple[int, ...]


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
    def intra_shard(ftensor: IRFullTensor, ilayout: GridLayout, olayout: GridLayout, cost_fn: Optional[Callable] = None) -> Tuple[List[GridLayout], List[IRAdapterPrim]]:
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
            inc_dim, dec_dim = None, None
            for dim, (ipnum, opnum) in enumerate(zip(curr_rvd, hop_rvd)):
                if ipnum > opnum:
                    assert dec_dim is None
                    dec_dim = dim
                    continue
                if opnum > ipnum:
                    assert inc_dim is None
                    inc_dim = dim
                    continue
            nchunks = curr_rvd[dec_dim] // hop_rvd[dec_dim]
            layout, prims = PathFinder.intra_step(layouts[-1], dec_dim, inc_dim, nchunks)
            layouts.append(layout)
            all_prims += prims
            curr_rvd = hop_rvd
        return layouts, all_prims


    @staticmethod
    def inter_shard(ftensor: IRFullTensor, ilayout: GridLayout, olayout: GridLayout, cost_fn: Optional[Callable] = None) -> Tuple[List[GridLayout], List[IRAdapterPrim]]:
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
        for idx, path in enumerate(paths):
            print(f"{src} -> {nodes[idx]}: {' -> '.join([str(nodes[i]) for i in path])} | cost: {cost[idx]}")
        

    @staticmethod
    def intra_step(ilayout: GridLayout, dec_idx: int, inc_idx: int, chunks: int) -> Tuple[GridLayout, List[IRAdapterPrim]]:
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
        raise RuntimeError(f"Cannot find primitive. Report as a bug. dec-idx: {dec_idx}, inc-idx: {inc_idx}")

    @staticmethod
    def inter_step(ilayout: GridLayout, dec_idx: Optional[int], inc_idx: Optional[int], chunks: int):
        assert dec_idx is None or inc_idx is None
        if isinstance(inc_idx, int):
            if inc_idx == 0:
                return ilayout.incr(chunks, [])
            if inc_idx == 1:
                return ilayout.incv(chunks, [])
            if inc_idx > 1:
                return ilayout.incd(chunks, inc_idx-2, [])
            raise RuntimeError(f"Cannot find primitive. Report as a bug. dec-idx: {dec_idx}, inc-idx: {inc_idx}")
        else:
            if dec_idx == 0:
                return ilayout.decr(chunks, [])
            if dec_idx == 1:
                return ilayout.decv(chunks, [])
            if dec_idx > 1:
                return ilayout.decd(chunks, dec_idx-2, [])
            raise RuntimeError(f"Cannot find primitive. Report as a bug. dec-idx: {dec_idx}, inc-idx: {inc_idx}")


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
                isrc, idst = nodes[i], nodes[j]
                inc_dim, dec_dim = [], []
                for dim, (pnum_src, pnum_dst) in enumerate(zip(isrc, idst)):
                    if pnum_src > pnum_dst:
                        dec_dim.append(dim)
                    elif pnum_src < pnum_dst:
                        inc_dim.append(dim)
                if len(inc_dim) != 1 or len(dec_dim) != 1:
                    continue  # not direct
                inc_dim, dec_dim = inc_dim[0], dec_dim[0]
                if idst[inc_dim] % isrc[inc_dim] != 0 or isrc[dec_dim] % idst[dec_dim] != 0:
                    continue  # not direct
                if inc_dim == 1:
                    continue  # not consider increasing value partition
                nchunks = isrc[dec_dim] // idst[dec_dim]
                isrc_layout = GridLayout.grid(ftensor, isrc[0], isrc[1], list(isrc[2:]))
                _, prims = PathFinder.intra_step(isrc_layout, dec_dim, inc_dim, nchunks)
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
        for node in nodes:
            print(node)
        edges = np.full((len(nodes), len(nodes)), np.inf)
        edges[:len(src_nodes), :len(src_nodes)] = src_edges
        edges[len(src_nodes):,len(src_nodes):] = dst_edges
        # NVLink: 300GBps Inter-node: 100Gbps
        comm_factor = 24
        for i in range(len(src_nodes)):
            for j in range(len(dst_nodes)):
                src, dst = src_nodes[i], dst_nodes[j]
                diff_dim = []
                for dim, (pnum_src, pnum_dst) in enumerate(zip(src, dst)):
                    if pnum_src != pnum_dst:
                        diff_dim.append(dim)
                diff_dim = [0] if len(diff_dim) == 0 else diff_dim
                if len(diff_dim) != 1:
                    continue # not direct
                diff_dim = diff_dim[0]
                if (src[diff_dim] % dst[diff_dim] != 0) and (dst[diff_dim] % src[diff_dim] != 0):
                   continue # not divisible -> not direct
                nchunks = src[diff_dim] // dst[diff_dim] if src[diff_dim] > dst[diff_dim] else dst[diff_dim] // src[diff_dim]
                # set for [i, len(src_nodes) + j]
                src_layout = GridLayout.grid(ftensor, src[0], src[1], list(src[2:]))
                dec_dim = diff_dim if src[diff_dim] > dst[diff_dim] else None
                inc_dim = diff_dim if dec_dim is None else None
                _, prims = PathFinder.inter_step(src_layout, dec_dim, inc_dim, nchunks)
                edges[i, len(src_nodes) + j] = cost_fn(prims[0]) * comm_factor
                # set for [len(src_nodes) + j, i]
                dst_layout = GridLayout.grid(ftensor, dst[0], dst[1], list(dst[2:]))
                dec_dim, inc_dim = inc_dim, dec_dim
                _, prims = PathFinder.inter_step(dst_layout, dec_dim, inc_dim, nchunks)
                # NVLink: 300GBps Inter-node: 100Gbps
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
            for dimlen, pnum in zip(ftensor.shape, rvd[2:]):
                if dimlen % pnum != 0:
                    continue
            all_layouts.append(tuple(rvd))
        return all_layouts

    @staticmethod
    def default_cost_fn(prim: IRAdapterPrim) -> int:
        return prim.volume() + 1 # 1 is hop penalty
