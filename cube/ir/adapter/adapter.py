from typing import List, Optional
import copy

from cube.ir.adapter.prim import IRAdapterPrim, IdentityPrim
from cube.ir.tensor import IRSubTensor
from cube.ir.cten import IRCell


class IRAdapter(IRCell):

    def __init__(self, inputs: List[IRSubTensor], outputs: List[IRSubTensor]):
        super().__init__(
            name='adapter', signature='adapter',
            input_length=len(inputs),
            output_length=len(outputs),
            init_outputs=False
        )
        # we don't use input and output setter as this will
        # change tensor device info
        self._inputs = inputs
        self._outputs = outputs

        self._prims: Optional[List[IRAdapterPrim]] = None
        self._differentiable = False

        device = set()
        for tensor in inputs + outputs:
            device.update(set(tensor.device))
        self.device = list(device)

        # setup whether this adapter is for forward stage
        is_fw = any(not t.is_grad() for t in self.inputs() + self.outputs())
        is_bw = any(t.is_grad() for t in self.inputs() + self.outputs())
        assert not (is_fw and is_bw), "An IRAdapter cannot serve for both forward and backward stage"
        self._forward = is_fw

    @property
    def prims(self) -> List[IRAdapterPrim]:
        if self.is_forward:
            if self.differentiable():
                return self.diffcolls
            else:
                return self.forward
        else:
            if self.differentiable():
                # not able to see
                return []
            else:
                return self.backward

    @property
    def prims(self) -> List[IRAdapterPrim]:
        return copy.copy(self._prims)

    @prims.setter
    def prims(self, prims: List[IRAdapterPrim]):
        assert all(isinstance(prim, IRAdapterPrim) for prim in prims), "Expect List[IRAdapterPrim]"
        self._prims = prims

    @property
    def differentiable(self) -> bool:
        """
        return if the adapter is using differentiable primitives
        """
        return self._differentiable

    @differentiable.setter
    def differentiable(self, val: bool):
        self._differentiable = val

    @property
    def forward(self) -> bool:
        """
        return True if this adapter serves in forward stage.
        """
        return self._forward

    def dispatch(self, devid: int):
        """
        Get Adapter for a specific rank

        Returns:
            IRAdapter
        """
        assert isinstance(devid, int), f"Expect devid to be int but got {devid}"
        prims = [prim.dispatch(devid) for prim in self.prims]
        prims = [prim for prim in prims if prim is not None]
        # get inputs
        inputs = []
        for itensor in self.inputs():
            if devid in itensor.device:
                inputs.append(itensor) 
        outputs = []
        for otensor in self.outputs():
            if devid in otensor.device:
                outputs.append(otensor)
        # insert identity prims
        if len(prims) == 0:
            assert len(inputs) == len(outputs) and all(itensor in outputs for itensor in inputs), \
                "input/output tensor not match for empty prims"
            for itensor in inputs:
                prims.append(IdentityPrim(itensor))
        # dispatch
        adapter = IRAdapter(inputs, outputs)
        adapter.prims = prims
        adapter.name = self.name
        adapter._id = self._id
        return adapter

    def __repr__(self):
        return f'Adapter-{self._id}{self.device}(inputs={self.inputs()}, outputs={self.outputs()})'

    def extra_repr(self) -> str:
        dscp = f'Adapter-{self._id}[{self.device}](inputs={self.inputs()}, outputs={self.outputs()})\n'
        for prim in self.prims:
            dscp += repr(prim) + '\n'
        return dscp


class IRWeightReducer(IRCell):

    def __init__(self, weights: List[IRSubTensor], name='reducer'):
        if not all(isinstance(w, IRSubTensor) and w.is_param() for w in weights):
            raise RuntimeError("Expected a list of gradient IRSubTensor")
        signature = None
        super().__init__(name, signature, len(weights), 0)
        for idx, weight in enumerate(weights):
            self.set_input(idx, weight)

    def __repr__(self):
        dscp = f'WReducer{self._id}-{self.device}(inputs={self.inputs()})'
        return dscp

    def module_repr(self) -> str:
        return repr(self)
