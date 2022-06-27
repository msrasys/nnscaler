from typing import List
from cube.graph.graph import IRSegment

from cube.ir.adapter import IRAdapter

from cube.execplan import ExecutionPlan
from cube.execplan.planpass.planpass import PlanPass

from cube.ir.adapter.prim import IRAdapterPrim
from cube.ir.adapter.prim import AllReducePrim, AllGatherPrim, ReduceScatterPrim, AllToAllPrim
from cube.ir.adapter.prim import IdentityPrim, ChunkPrim
from cube.ir.adapter.prim import IdentityAllreducePrim, AllReduceIdentityPrim, AllReduceAllReducePrim
from cube.ir.adapter.prim import AllGatherReduceScatterPrim, ReduceScatterAllGatherPrim
from cube.ir.adapter.prim import SplitAllGatherPrim, AllGatherSplitPrim
from cube.ir.adapter.prim import AllToAllAllToAllPrim


class DiffFusion(PlanPass):

    @staticmethod
    def apply(execplan: ExecutionPlan) -> ExecutionPlan:
        """
        Fuse the non-differentiable adapters into differentiable adapters.
        """
        cnt = 0
        for devid in execplan.devices():
            for node in execplan.seq(devid):
                if isinstance(node, IRAdapter):
                    if node.forward:
                        ret = DiffFusion.nnfuse(node)
                        cnt = cnt+1 if ret else cnt
                if isinstance(node, IRSegment) and node.forward:
                    for fnode in node.nodes():
                        if isinstance(fnode, IRAdapter):
                            if node.forward:
                                ret = DiffFusion.nnfuse(fnode)
                                if not ret:
                                    raise NotImplementedError(
                                        f"adapter within IRSegment cannot fuse to differientiable adapter"
                                        f"\nforward: {fnode.extra_repr()}"
                                        f"\nbackward: {fnode.mirror.extra_repr()}"
                                    )
                                cnt = cnt + 1
        print(f'successfully generate {cnt} differentiable adapters')
        return execplan

    @staticmethod
    def nnfuse(fadapter: IRAdapter) -> bool:
        """
        Fuse the forward adapter with its backward adapter into differentiable
        communications. Note this is an inplacement update

        Return:
            success: boolean
        """
        if not isinstance(fadapter.mirror, IRAdapter):
            return False
        badapter: IRAdapter = fadapter.mirror
        fprims, bprims = fadapter.prims, badapter.prims

        def is_allreduce(prims: List[IRAdapterPrim]) -> bool:
            return len(prims) == 1 and all(isinstance(prim, AllReducePrim) for prim in prims)

        def is_identity(prims: List[IRAdapterPrim]) -> bool:
            return len(prims) == 1 and all(isinstance(prim, IdentityPrim) for prim in prims)

        def is_redsca(prims: List[IRAdapterPrim]) -> bool:  # reduce-scatter
            return len(prims) == 1 and all(isinstance(prim, ReduceScatterPrim) for prim in prims)

        def is_allgather(prims: List[IRAdapterPrim]) -> bool:
            return len(prims) == 1 and all(isinstance(prim, AllGatherPrim) for prim in prims)

        def is_chunk(prims: List[IRAdapterPrim]) -> bool:
            return len(prims) == 1 and all(isinstance(prim, ChunkPrim) for prim in prims)

        def is_alltoall(prims: List[IRAdapterPrim]) -> bool:
            return len(prims) == 1 and all(isinstance(prim, AllToAllPrim) for prim in prims)

        prims = None
        # allreduce-identity
        if is_allreduce(fprims) and is_identity(bprims):
            prims = [AllReduceIdentityPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # identity-allreduce
        elif is_identity(fprims) and is_allreduce(bprims):
            prims = [IdentityAllreducePrim(p.inputs(), p.outputs(), **bprims[0].kwargs) for p in fprims]
        # allreduce-allreduce
        elif is_allreduce(fprims) and is_allreduce(bprims):
            prims = [AllReduceAllReducePrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # allgather-reducescatter
        elif is_allgather(fprims) and is_redsca(bprims):
            prims = [AllGatherReduceScatterPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # reducescatter-allgather
        elif is_redsca(fprims) and is_allgather(bprims):
            prims = [ReduceScatterAllGatherPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # allgather-chunk
        elif is_allgather(fprims) and is_chunk(bprims):
            prims = [AllGatherSplitPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # chunk-allgather
        elif is_chunk(fprims) and is_allgather(bprims):
            prims = [SplitAllGatherPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        # all-to-all
        elif is_alltoall(fprims) and is_alltoall(bprims):
            prims = [AllToAllAllToAllPrim(p.inputs(), p.outputs(), **p.kwargs) for p in fprims]
        
        if prims is not None:
            fadapter.prims = prims
            badapter.prims = prims
            fadapter.custom = False
            fadapter.differentiable = True
            badapter.custom = False
            badapter.differentiable = True
            return True
        return False
