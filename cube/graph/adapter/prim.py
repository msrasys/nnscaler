"""
The primitive used for IRAdapter
"""

from typing import Callable, List, Optional, Union
import copy

from cube.graph.tensor import IRSubTensor, IndexMap, ValueMap

# the general adapter primitive class
class IRAdapterPrim:

    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor]):
        self._inputs = inputs
        self._outputs = outputs
        self._device = []
        self.kwargs = dict()

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


# numerical abstract primitive
class ValuePrim(IRAdapterPrim):
    """
    basic class for representing numerical primitives
    """
    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor]):
        super().__init__(inputs, outputs)

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
        for t in itensors + otensors:
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

class SelectPrim(SpatialPrim):

    def __init__(self,
                 itensor: IRSubTensor,
                 indmap: IndexMap, valmap: ValueMap,
                 otensor: IRSubTensor):
        super().__init__([itensor], [otensor])
        self.indmap = indmap
        self.valmap = valmap
        self.device = itensor.device

    def __repr__(self):
        dscp = f'{self.outputs(0)} = select({self.inputs(0)})'
        return dscp


class SplitDimPrim(SpatialPrim):
    """
    split dimension
    """
    def __init__(self, itensor: IRSubTensor, dim: int,
                 otensors: List[IRSubTensor]):
        super().__init__([itensor], otensors)
        self.dim = dim
        self.device = itensor.device


class MergeDimPrim(SpatialPrim):
    """
    concatenate dimension
    """
    def __init__(self, itensors: List[IRSubTensor], dim: int,
                 otensor: IRSubTensor) -> None:
        assert all(itensor.device == itensors[0].device for itensor in itensors), "device not same"
        super().__init__(itensors, [otensor])
        self.dim = dim
        self.device = itensors[0].device

# numerical primitive

class ReducePrim(ValuePrim):

    def __init__(self, itensors: List[IRSubTensor], otensor: IRSubTensor):
        assert all(itensor.device == itensors[0].device for itensor in itensors), "device not same"
        super().__init__(itensors, [otensor])
        self.reduce = '+'
        self.device = itensors[0].device

# communication primitive

class SendPrim(CommPrim):
    """
    P2P send prim
    """
    def __init__(self, tensor, dst: int):
        super().__init__([tensor], [tensor])
        self.kwargs['dst'] = dst
    
    def dispatch(self, devid: int):
        assert devid == self.device[0], f"device {devid} not applied for this comm primitive"
        return SendPrim(self.inputs(0), self.kwargs['dst'])

    def __repr__(self) -> str:
        return f"{self.inputs(0)} = send({self.inputs(0)}, dst={self.kwargs['dst']}"


class RecvPrim(CommPrim):
    """
    P2P recv prim
    """
    def __init__(self, tensor, src: int):
        super().__init__([], [tensor])
        self.kwargs['src'] = src
        self.kwargs['shape'] = tensor.shape
        self.kwargs['dtype'] = tensor.dtype
    
    def dispatch(self, devid: int):
        assert devid == self.device[0], f"device {devid} not applied for this comm primitive"
        return RecvPrim(self.outputs(0), self.kwargs['src'])

    def __repr__(self) -> str:
        return f"{self.outputs(0)} = recv(shape={self.kwargs['shape']}, dtype={self.kwargs['dtype']}, dst={self.kwargs['dst']}"


class MovePrim(CommPrim):
    """
    P2P send/recv, non-differentiable
    """
    def __init__(self, tensor: IRSubTensor, src: int, dst: int):
        super().__init__([tensor], [tensor])
        self.kwargs['src'] = src
        self.kwargs['dst'] = dst

    def dispatch(self, devid: int) -> Union[SendPrim, RecvPrim]:
        if devid == self.kwargs['src']:
            return SendPrim(self.inputs(0), self.kwargs['devid'])
        if devid == self.kwargs['dst']:
            return RecvPrim(self.inputs(0), self.kwargs['src'])
        raise ValueError(f"device {devid} is not src ({self.kwargs['src']}) or ({self.kwargs['dst']})")

    def __repr__(self):
        dscp = f'move({self.inputs(0)}, from={self.src}, to={self.dst})'
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

    def dispatch(self, devid: int, init_method: Callable):
        """
        dispatch to a given device
        """
        assert devid in self.device, f"device {devid} not applied for this comm primitive"
        itensors = [itensor for itensor in self.inputs() if devid in itensor.device]
        otensors = [otensor for otensor in self.outputs() if devid in otensor.device]
        prim = init_method(itensors, otensors, **self.kwargs)
        return prim


class AllReducePrim(CollectivePrim):
    """
    non-differentiable allreduce
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor]):
        super().__init__(itensors, otensors)


class AllGatherPrim(CollectivePrim):
    """
    non-differentiabl all-to-all
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor]):
        super().__init__(itensors, otensors)


class ReduceScatterPrim(CollectivePrim):
    """
    non-differential reduce-scatter
    """
    def __init__(self, itensors: List[IRSubTensor], otensors: List[IRSubTensor]):
        super().__init__(itensors, otensors)


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
