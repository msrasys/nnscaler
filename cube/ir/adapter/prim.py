"""
The primitive used for IRAdapter
"""

from typing import List, Optional, Union
import copy

from cube.ir.tensor import IRSubTensor, IndexMap, ValueMap


# the general adapter primitive class
class IRAdapterPrim:

    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor], **kwargs):
        self._inputs = list(inputs)
        self._outputs = list(outputs)
        self._device = []
        self.kwargs = dict()
        for arg, val in kwargs.items():
            self.kwargs[arg] = val
        self.signature = None

    def input(self, idx:int):
        return self._inputs[idx]

    def inputs(self):
        return copy.copy(self._inputs)

    def output(self, idx:int):
        return self._outputs[idx]

    def outputs(self):
        return copy.copy(self._outputs)

    def dispatch(self, devid: int):
        if devid not in self.device:
            return None
        return self

    def volume(self) -> int:
        """
        Communication volume of the primitive. The total elements
        transferred in the network.

        @return nele int: the number of elements go through network
        """
        raise NotImplementedError("The communication cost is not implemented")

    @property
    def device(self) -> List[int]:
        return copy.copy(self._device)

    @device.setter
    def device(self, devs: Union[int, List[int]]):
        if isinstance(devs, int):
            devs = [devs]
        self._device = devs

# spatial abstract primitive
class SpatialPrim(IRAdapterPrim):
    """
    basic class for representing spatial primitives
    """
    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor], **kwargs):
        super().__init__(inputs, outputs, **kwargs)
        self.device = list(set(t.device[0] for t in inputs))


# numerical abstract primitive
class ValuePrim(IRAdapterPrim):
    """
    basic class for representing numerical primitives
    """
    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor]):
        super().__init__(inputs, outputs)
        self.device = list(set(t.device[0] for t in inputs))


# communication abstract primitive
class CommPrim(IRAdapterPrim):
    """
    communication primitive
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        devices = []
        for t in list(itensors) + list(otensors):
            devices += t.device
        self.device = list(set(devices))

    def dispatch(self, devid: int):
        """
        dispatch to a given device
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        dscp = f'{self.outputs()} = {self.signature}({self.inputs()})'
        return dscp

# ======================================================

class IdentityPrim(SpatialPrim):

    def __init__(self, itensor: IRSubTensor):
        super().__init__([itensor], [itensor])
        self.signature = 'cube.runtime.adapter.identity'

    def __repr__(self):
        dscp = f"{self.output(0)} = identity({self.input(0)})"
        return dscp

    def volume(self) -> int:
        return 0


class SelectPrim(SpatialPrim):

    def __init__(self,
                 itensor: IRSubTensor,
                 indmap: IndexMap, valmap: ValueMap,
                 otensor: IRSubTensor):
        indmap = IndexMap(indmap).indices
        indmap = tuple(slice(s, e) for s, e in indmap)
        valmap = ValueMap(valmap).weight[1]
        super().__init__([itensor], [otensor], indmap=indmap, valmap=valmap)
        self.signature = f"cube.runtime.adapter.select"

    def __repr__(self):
        dscp = f"{self.output(0)} = select({self.input(0)}, indmap={self.kwargs['indmap']}, valmap={self.kwargs['valmap']})"
        return dscp

    def volume(self) -> int:
        return 0


class MergeDimPrim(SpatialPrim):
    """
    concatenate dimension
    """
    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor, dim: int) -> None:
        assert all(itensor.device == itensors[0].device for itensor in itensors), "device not same"
        super().__init__(itensors, [otensor], dim=dim)
        self.signature = 'cube.runtime.adapter.smerge'

    def __repr__(self) -> str:
        return f"dev{self.device}: {self.output(0)} = concat({self.inputs()}, dim={self.kwargs['dim']})"

    def volume(self) -> int:
        return 0

# numerical primitive

class SumPrim(ValuePrim):

    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor):
        assert all(itensor.device == itensors[0].device for itensor in itensors), "device not same"
        super().__init__(itensors, [otensor])
        self.signature = 'cube.runtime.adapter.vmerge'

    def volume(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"dev{self.device}: {self.output(0)} = add({self.inputs()})"


# communication primitive

class SendPrim(CommPrim):
    """
    P2P send prim
    """
    def __init__(self, tensor, dst: int):
        super().__init__([tensor], [tensor], dst=dst)
        self.signature = 'cube.runtime.adapter.send'

    def volume(self) -> int:
        return self.input(0).nelement()

    def __repr__(self) -> str:
        return f"{self.input(0)} = send[{self.device}]({self.input(0)}, dst={self.kwargs['dst']}"


class RecvPrim(CommPrim):
    """
    P2P recv prim
    """
    def __init__(self, tensor: IRSubTensor, src: int):
        super().__init__([], [tensor],
                         shape=tensor.shape, dtype='torch.'+tensor.dtype.value, src=src)
        self.signature = 'cube.runtime.adapter.recv'

    def volume(self) -> int:
        return self.input(0).nelement()

    def __repr__(self) -> str:
        return f"{self.output(0)} = recv[{self.device}](shape={self.kwargs['shape']}, dtype={self.kwargs['dtype']}, src={self.kwargs['src']}"


class MovePrim(CommPrim):
    """
    P2P send/recv, non-differentiable
    """
    def __init__(self, itensor: IRSubTensor, otensor: IRSubTensor):
        src: int = itensor.device[0] if len(itensor.device) > 0 else None
        dst: int = otensor.device[0] if len(otensor.device) > 0 else None
        super().__init__([itensor], [otensor], src=src, dst=dst)

    def dispatch(self, devid: int) -> Union[SendPrim, RecvPrim]:
        if devid == self.kwargs['src']:
            return SendPrim(self.input(0), self.kwargs['dst'])
        if devid == self.kwargs['dst']:
            return RecvPrim(self.output(0), self.kwargs['src'])
        return None

    def volume(self) -> int:
        return self.input(0).nelement()

    def __repr__(self):
        dscp = f"move[{self.device}]({self.input(0)}, src={self.kwargs['src']}, dst={self.kwargs['dst']})"
        return dscp


class RDScatterPrim(CommPrim):
    """
    P2P Cross-device dimension scatter, non-differentiable.

    Tensor[Tile0, Tile1]: device 0 -> Tensor[Tile0]: device0, Tensor[Tile1]: device1
    """
    def __init__(self, itensor: IRSubTensor, otensors: List[IRSubTensor], dim: int):
        """
        @param itensors List[IRSubTensor]: one tensor at device of `src`.
        @param otensors List[IRSubTensor]: each ran hosts one tenor partitioned by dim.
        @param dim int: the dimension that itensor will be partitioned
        """
        src: int = itensor.device[0] if len(itensor.device) > 0 else None
        dst: List[int] = [t.device[0] if len(t.device) > 0 else None for t in otensors]
        super().__init__([itensor], otensors, dim=dim, src=src, dst=dst)
        self.signature = 'cube.runtime.adapter.rdscatter'

    def volume(self) -> int:
        return self.input(0).nelement()

    def __repr__(self) -> str:
        return f"{self.outputs()} = rdscatter[{self.device}]({self.inputs()}, dim={self.kwargs['dim']})"


class RVScatterPrim(CommPrim):
    """
    P2P Cross-device dimension scatter, non-differentiable.

    Tensor[Tile0, Tile1]: device 0 -> Tensor[Tile0]: device0, Tensor[Tile1]: device1
    """
    def __init__(self, itensor: IRSubTensor, otensors: List[IRSubTensor]):
        """
        @param itensors List[IRSubTensor]: one tensor at device of `src`.
        @param otensors List[IRSubTensor]: each ran hosts one tenor partitioned by dim.
        @param dim int: the dimension that itensor will be partitioned
        """
        src: int = itensor.device[0] if len(itensor.device) > 0 else None
        dst: List[int] = [t.device[0] if len(t.device) > 0 else None for t in otensors]
        super().__init__([itensor], otensors, src=src, dst=dst)
        self.signature = 'cube.runtime.adapter.rvscatter'

    def volume(self) -> int:
        return self.input(0).nelement() * len(self.outputs())

    def __repr__(self) -> str:
        return f"{self.outputs()} = rvscatter[{self.device}]({self.inputs()})"


class RDGatherPrim(CommPrim):
    """
    Gather tensors from remote devices to a local device.
    The local device doesn't have any tensor
    """
    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor, dim: int):
        src = [t.device[0] if len(t.device) > 0 else None for t in itensors]
        dst = otensor.device[0] if len(otensor.device) > 0 else None
        super().__init__(itensors, [otensor], src=src, dst=dst, dim=dim)
        self.signature = 'cube.runtime.adapter.rdgather'

    def volume(self) -> int:
        return self.output(0).nelement()
    
    def __repr__(self) -> str:
        return (
            f"rdgather[{self.device}]("
            f"{self.inputs()}, dim={self.kwargs['dim']}, "
            f"src={self.kwargs['src']}, dst={self.kwargs['dst']})"
        )

class RVGatherPrim(CommPrim):
    """
    Gather tensors from remote devices and sum in the local device.
    The local device doesn't have any tensor
    """
    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor):
        src = [t.device[0] if len(t.device) > 0 else None for t in itensors]
        dst = otensor.device[0] if len(otensor.device) > 0 else None
        super().__init__(itensors, [otensor], src=src, dst=dst)
        self.signature = 'cube.runtime.adapter.rvgather'

    def volume(self) -> int:
        return self.output(0).nelement() * len(self.inputs())

    def __repr__(self) -> str:
        src = self.kwargs['src']
        dst = self.kwargs['dst']
        return f"{self.outputs()} = rvgather[{self.device}]({self.inputs()}, src={src}, dst={dst})"


class CollectivePrim(CommPrim):
    """
    Collective primitive, non-differentiable
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        if 'ranks' not in self.kwargs:
            self.kwargs['ranks'] = self.device

    def dispatch(self, devid: int) -> Optional[CommPrim]:
        """
        dispatch to a given device
        """
        if devid not in self.device:
            return None
        assert devid in self.device, f"device {devid} not applied for this comm primitive"
        itensors = [itensor for itensor in self.inputs() if devid in itensor.device]
        otensors = [otensor for otensor in self.outputs() if devid in otensor.device]
        prim = type(self)(itensors, otensors, **self.kwargs)
        prim.signature = self.signature
        return prim


class AllReducePrim(CollectivePrim):
    """
    non-differentiable allreduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'cube.runtime.adapter.all_reduce'

    def volume(self) -> int:
        """
        Use ring-allreduce communication cost
        """
        ndevs = len(self.inputs())
        return 2 * (ndevs - 1) * self.input(0).nelement() // ndevs

    def __repr__(self) -> str:
        return f'{self.outputs()} = all_reduce[{self.device}]({self.inputs()})'


class AllGatherPrim(CollectivePrim):
    """
    non-differentiabl all-to-all
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim=dim, **kwargs)
        self.signature = 'cube.runtime.adapter.all_gather'

    def volume(self) -> int:
        """
        Use ring-based communication cost
        """
        ndevs = len(self.inputs())
        return (ndevs - 1) * self.input(0).nelement()

    def __repr__(self) -> str:
        return f'{self.outputs()} = all_gather[{self.device}]({self.inputs()})'


class ReduceScatterPrim(CollectivePrim):
    """
    non-differential reduce-scatter
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim=dim, **kwargs)
        self.signature = 'cube.runtime.adapter.reduce_scatter'

    def volume(self) -> int:
        """
        Use ring-based communication cost
        """
        ndevs = len(self.inputs())
        return (ndevs - 1) * self.input(0).nelement() // ndevs

    def __repr__(self) -> str:
        return f'{self.outputs()} = reduce_scatter[{self.device}]({self.inputs()})'


class BroadcastPrim(CollectivePrim):
    """
    non-differential reduce-scatter
    """
    def __init__(self, itensor: IRSubTensor, otensors: List[IRSubTensor], **kwargs):
        src: int = itensor.device[0] if len(itensor.device) > 0 else None
        super().__init__([itensor], otensors, src=src, **kwargs)
        self.signature = 'cube.runtime.adapter.broadcast'

    def volume(self) -> int:
        ndevs = len(self.outputs())
        return self.input(0).nelement() * (ndevs-1)

    def __repr__(self) -> str:
        return f"{self.outputs()} = broadcast[{self.device}]({self.inputs()}, src={self.kwargs['src']})"


class ReducePrim(CollectivePrim):
    """
    non-differential reduce prim
    """
    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor, **kwargs):
        super().__init__(itensors, [otensor], dst=otensor.device[0], **kwargs)
        self.signature = 'cube.runtime.adapter.reduce'

    def volume(self) -> int:
        ndevs = len(self.inputs())
        return self.input(0).nelement() * ndevs

    def __repr__(self) -> str:
        return f"{self.outputs()} = reduce[{self.device}]({self.inputs()}, dst={self.kwargs['dst']})"


class AllToAllPrim(CollectivePrim):
    """
    non-differentiable all-to-all
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], idim: int, odim: int, **kwargs):
        """
        itensors: each rank hosts one tensor splitted by idim
        otensors: each rank hosts one tensor splitted by odim
        idim != odim
        """
        super().__init__(itensors, otensors, idim=idim, odim=odim, **kwargs)
        self.signature = 'cube.runtime.adapter.all_to_all'

    def volume(self) -> int:
        ndevs = len(self.inputs())
        return self.input(0).nelement() * (ndevs - 1) // ndevs

    def __repr__(self) -> str:
        return f"{self.outputs()} = all_to_all[{self.device}]({self.inputs()}, idim={self.kwargs['idim']}, odim={self.kwargs['odim']})"


class ChunkPrim(CollectivePrim):
    """
    split dimension in n chunks and take idx-th chunk
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim=dim, **kwargs)
        self.signature = 'cube.runtime.adapter.chunk'

    def volume(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"{self.outputs()} = split[{self.device}]({self.inputs()}, dim={self.kwargs['dim']})"


class VChunkPrim(CollectivePrim):
    """
    split value in n chunks and take idx-th chunk
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'cube.runtime.adapter.vchunk'
    
    def volume(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"{self.outputs()} = vsplit[{self.device}]({self.inputs()})"


class AllReduceIdentityPrim(AllReducePrim):
    """
    forward: allreduce.
    backward: identity
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'cube.runtime.adapter.nn.allreduce_identity'

    def __repr__(self) -> str:
        return f"{self.outputs()} = allreduce_identity[{self.device}]({self.inputs()})"


class IdentityAllreducePrim(AllReducePrim):
    """
    forward: identity
    backward: allreduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'cube.runtime.adapter.nn.identity_allreduce'

    def __repr__(self) -> str:
        return f"{self.outputs()} = identity_allreduce[{self.device}]({self.inputs()})"


class AllReduceAllReducePrim(AllReducePrim):
    """
    forward: allreduce
    backward: allreduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors, **kwargs)
        self.signature = 'cube.runtime.adapter.nn.allreduce_allreduce'

    def __repr__(self) -> str:
        return f"{self.outputs} = nn.allreduce_allreduce[{self.device}]({self.inputs()}"


class ReduceScatterAllGatherPrim(ReduceScatterPrim):
    """
    forward: reduce-scatter
    backward: all-gather
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim, **kwargs)
        self.signature = 'cube.runtime.adapter.nn.reducescatter_allgather'


class AllGatherReduceScatterPrim(AllGatherPrim):
    """
    forward: all-gather
    backward: reduce-scatter
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim, **kwargs)
        self.signature = 'cube.runtime.adapter.nn.allgather_reducescatter'


class AllGatherSplitPrim(AllGatherPrim):
    """
    forward: all-gather
    backward: split
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim, **kwargs)
        self.signature = 'cube.runtime.adapter.nn.allgather_split'


class SplitAllGatherPrim(AllGatherPrim):
    """
    forward: split
    backward: all-gather
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int, **kwargs):
        super().__init__(itensors, otensors, dim, **kwargs)
        self.signature = 'cube.runtime.adapter.nn.split_allgather'


class AllToAllAllToAllPrim(AllToAllPrim):
    """
    forward: all-to-all
    backward: all-to-all
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], idim: int, odim: int, **kwargs):
        super().__init__(itensors, otensors, idim, odim, **kwargs)
        self.signature = 'cube.runtime.adapter.nn.alltoall_alltoall'


class ReduceBroadcastPrim(CollectivePrim):
    """
    forward: broadcast
    backward: reduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dst: int, **kwargs):
        super().__init__(itensors, otensors, dst=dst, **kwargs)


class BroadcastRedducePrim(CollectivePrim):
    """
    forward: broadcast
    backward: reduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], src: int, **kwargs):
        super().__init__(itensors, otensors, src=src, **kwargs)
