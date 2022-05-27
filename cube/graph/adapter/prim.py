"""
The primitive used for IRAdapter
"""

from typing import List, Optional, Tuple, Union
import copy

from cube.graph.tensor import IRSubTensor, IndexMap, ValueMap

# the general adapter primitive class
class IRAdapterPrim:

    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor]):
        self._inputs = list(inputs)
        self._outputs = list(outputs)
        self._device = []
        self.kwargs = dict()
        self.signature = None

    def inputs(self, idx: Optional[int] = None):
        assert idx is None or isinstance(idx, int),  "expected idx to be None or int"
        if idx is None:
            return copy.copy(self._inputs)
        else:
            return self._inputs[idx]

    def outputs(self, idx: Optional[int] = None):
        assert idx is None or isinstance(idx, int),  "expected idx to be None or int"
        if idx is None:
            return copy.copy(self._outputs)
        else:
            return self._outputs[idx]

    def dispatch(self, devid: int):
        if devid not in self.device:
            return None
        return self

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
    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor]):
        super().__init__(inputs, outputs)
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
    def __init__(self,
                 itensors: List[IRSubTensor],
                 otensors: List[IRSubTensor]):
        super().__init__(itensors, otensors)
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
        dscp = f"{self.outputs(0)} = identity({self.inputs(0)})"
        return dscp


class SelectPrim(SpatialPrim):

    def __init__(self,
                 itensor: IRSubTensor,
                 indmap: IndexMap, valmap: ValueMap,
                 otensor: IRSubTensor):
        super().__init__([itensor], [otensor])
        self.kwargs['indmap'] = indmap
        self.kwargs['valmap'] = (valmap.idx, valmap.chunk_num)
        self.signature = f"cube.runtime.adapter.select"

    def __repr__(self):
        dscp = f"{self.outputs(0)} = select({self.inputs(0)}, indmap={self.kwargs['indmap']}, valmap={self.kwargs['valmap']})"
        return dscp


class ChunkPrim(SpatialPrim):
    """
    split dimension in n chunks and take idx-th chunk
    """
    def __init__(self, itensor: IRSubTensor, otensor: IRSubTensor,
                 dim: int, ranks: Tuple[int]):
        assert itensor.device[0] in ranks, "idx out of scope"
        super().__init__([itensor], [otensor])
        self.kwargs['dim'] = dim
        self.kwargs['ranks'] = ranks
        self.signature = 'cube.runtime.adapter.chunk'

    def __repr__(self) -> str:
        chunks = len(self.kwargs['ranks'])
        idx = self.kwargs['ranks'].index(self.device[0])
        return f"dev{self.device}: {self.outputs(0)} = split(dim={self.kwargs['dim']}, chunks={chunks}, idx={idx})"


class MergeDimPrim(SpatialPrim):
    """
    concatenate dimension
    """
    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor, dim: int) -> None:
        assert all(itensor.device == itensors[0].device for itensor in itensors), "device not same"
        super().__init__(itensors, [otensor])
        self.kwargs['dim'] = dim
        self.signature = 'cube.runtime.adapter.smerge'

    def __repr__(self) -> str:
        return f"dev{self.device}: {self.outputs(0)} = concat({self.inputs()}, dim={self.kwargs['dim']})"

# numerical primitive

class SumPrim(ValuePrim):

    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor):
        assert all(itensor.device == itensors[0].device for itensor in itensors), "device not same"
        super().__init__(itensors, [otensor])
        self.signature = 'cube.runtime.adapter.vmerge'

    def __repr__(self) -> str:
        return f"dev{self.device}: {self.outputs(0)} = add({self.inputs()})"

# communication primitive

class SendPrim(CommPrim):
    """
    P2P send prim
    """
    def __init__(self, tensor, dst: int):
        super().__init__([tensor], [tensor])
        self.kwargs['dst'] = dst
        self.device = tensor.device
        self.signature = 'cube.runtime.adapter.send'

    def __repr__(self) -> str:
        return f"{self.inputs(0)} = send({self.inputs(0)}, dst={self.kwargs['dst']}"


class RecvPrim(CommPrim):
    """
    P2P recv prim
    """
    def __init__(self, tensor: IRSubTensor, src: int):
        super().__init__([], [tensor])
        self.kwargs['shape'] = tensor.shape
        self.kwargs['dtype'] = 'torch.' + tensor.dtype.value
        self.kwargs['src'] = src
        self.device = tensor.device
        self.signature = 'cube.runtime.adapter.recv'

    def __repr__(self) -> str:
        return f"{self.outputs(0)} = recv(shape={self.kwargs['shape']}, dtype={self.kwargs['dtype']}, src={self.kwargs['src']}"


class MovePrim(CommPrim):
    """
    P2P send/recv, non-differentiable
    """
    def __init__(self, itensor: IRSubTensor, otensor: IRSubTensor):
        super().__init__([itensor], [otensor])
        self.kwargs['src'] = itensor.device[0]
        self.kwargs['dst'] = otensor.device[0]
        self.device = itensor.device + otensor.device

    def dispatch(self, devid: int) -> Union[SendPrim, RecvPrim]:
        if devid == self.kwargs['src']:
            return SendPrim(self.inputs(0), self.kwargs['dst'])
        if devid == self.kwargs['dst']:
            return RecvPrim(self.outputs(0), self.kwargs['src'])
        return None

    def __repr__(self):
        dscp = f"move({self.inputs(0)}, src={self.kwargs['src']}, dst={self.kwargs['dst']})"
        return dscp


class CollectivePrim(CommPrim):
    """
    Collective primitive, non-differentiable
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        super().__init__(itensors, otensors)
        self.kwargs['ranks'] = self.device
        for arg, val in kwargs.items():
            self.kwargs[arg] = val

    def dispatch(self, devid: int) -> Optional[CommPrim]:
        """
        dispatch to a given device
        """
        if devid not in self.device:
            return None
        assert devid in self.device, f"device {devid} not applied for this comm primitive"
        itensors = [itensor for itensor in self.inputs() if devid in itensor.device]
        otensors = [otensor for otensor in self.outputs() if devid in otensor.device]
        prim = CollectivePrim(itensors, otensors, **self.kwargs)
        prim.signature = self.signature
        return prim


class AllReducePrim(CollectivePrim):
    """
    non-differentiable allreduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor]):
        super().__init__(itensors, otensors)
        self.signature = 'cube.runtime.adapter.all_reduce'

    def __repr__(self) -> str:
        return f'dev{self.device}: {self.outputs()} = all_reduce({self.inputs()}'


class AllGatherPrim(CollectivePrim):
    """
    non-differentiabl all-to-all
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int):
        super().__init__(itensors, otensors, dim=dim)
        self.signature = 'cube.runtime.adapter.all_gather'

    def __repr__(self) -> str:
        return f'dev{self.device}: {self.outputs()} = all_gather({self.inputs()})'


class ReduceScatterPrim(CollectivePrim):
    """
    non-differential reduce-scatter
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int):
        super().__init__(itensors, otensors, dim=dim)
        self.signature = 'cube.runtime.adapter.reduce_scatter'

    def __repr__(self) -> str:
        return f'dev{self.device}: {self.outputs()} = reduce_scatter({self.inputs()})'


class BroadcastPrim(CollectivePrim):
    """
    non-differential reduce-scatter
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], src: int):
        super().__init__(itensors, otensors, src=src)


class ReducePrim(CollectivePrim):
    """
    non-differential reduce prim
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dst: int):
        super().__init__(itensors, otensors, dst=dst)


class AllToAllPrim(CollectivePrim):
    """
    non-differentiable all-to-all
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], idim: int, odim: int):
        """
        itensors: each rank hosts one tensor splitted by idim
        otensors: each rank hosts one tensor splitted by odim
        idim != odim
        """
        super().__init__(itensors, otensors, idim=idim, odim=odim)
        self.signature = 'cube.runtime.adapter.all_to_all'

    def __repr__(self) -> str:
        return f"dev{self.device}: {self.outputs()} = all_to_all({self.inputs}, idim={self.kwargs['idm']}, odim={self.kwargs['odim']})"


class DiffCollectivePrim(CollectivePrim):
    """
    Differentiable collective primitive
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], **kwargs):
        """
        differentiable collectives 
        """
        super().__init__(itensors, otensors, **kwargs)


class AllReduceIdentityPrim(DiffCollectivePrim):
    """
    forward: allreduce.
    backward: identity
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor]):
        super().__init__(itensors, otensors)


class IdentityAllreducePrim(DiffCollectivePrim):
    """
    forward: identity
    backward: allreduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor]):
        super().__init__(itensors, otensors)


class ReduceScatterAllGatherPrim(DiffCollectivePrim):
    """
    forward: reduce-scatter
    backward: all-gather
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int):
        super().__init__(itensors, otensors, dim=dim)


class AllGatherSplitPrim(DiffCollectivePrim):
    """
    forward: all-gather
    backward: split
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int):
        super().__init__(itensors, otensors, dim=dim)


class SplitAllGatherPrim(DiffCollectivePrim):
    """
    forward: split
    backward: all-gather
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dim: int):
        super().__init__(itensors, otensors, dim=dim)


class ReduceBroadcastPrim(DiffCollectivePrim):
    """
    forward: broadcast
    backward: reduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], dst: int):
        super().__init__(itensors, otensors, dst=dst)


class BroadcastRedducePrim(DiffCollectivePrim):
    """
    forward: broadcast
    backward: reduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor], src: int):
        super().__init__(itensors, otensors, src=src)
