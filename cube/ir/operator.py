from typing import Optional, Tuple
import copy

from cube.ir.cten import IRCell, IRTensor
from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.algorithm.factory import DistAlgorithmFactory
from cube.ir.unique import IDGenerator
from cube.ir.dtype import IRDType, DTypeInferRule


class IRFwOperation(IRCell):
    """
    Forward operation
    """

    def __init__(self,
                 name: str, 
                 signature: str,
                 input_length: int,
                 output_length: int):
        """!
        Create a forward operation.

        @param name str: the name of forward operation
        @param signature str: the signature of the forward operation
        @param input_length int: number of inputs
        @param output_length int: number of outputs
        """
        # additional argument
        self.kwargs = dict()
        # recompute schedule
        self._recompute = None
        super().__init__(name, signature, input_length, output_length, init_outputs=False)
        outputs = [IRFullTensor() for _ in range(output_length)]
        for idx, output in enumerate(outputs):
            self.set_output(idx, output)

    def infer_dtype(self):
        """
        Infer output value dtype.

        By default will follow the same dtype promotion rule with PyTorch.
        """
        itensors = [t for t in self.inputs() if isinstance(t, IRTensor)]
        assert len(itensors) > 0, "Missing input tensors, need to customize the infer rule"
        odtype = DTypeInferRule.infer(self, [t.dtype for t in itensors])
        assert odtype != IRDType.unknown, f"{self} : {[t.dtype for t in itensors]}"
        otensors = [t for t in self.outputs() if isinstance(t, IRTensor)]
        for tensor in otensors:
            tensor.dtype = odtype

    def infer_shape(self):
        """
        Infer output value shape
        """
        raise NotImplementedError

    @property
    def recompute(self) -> Optional[int]:
        """!
        Get recompute group id.
        To enable recompute, a recompute group refers to a sequence of operators that
        will perform recompute optimization.

        @return group_id Optional[int]: None if no recompute, else a group id.
        """
        return self._recompute

    @recompute.setter
    def recompute(self, group_id: Optional[int]):
        """!
        Set recompute group

        @param group_id Optional[int]: recompute group id. None indicates no group is applied
        """
        assert group_id is None or isinstance(group_id, int), "Expect None or int"
        if isinstance(group_id, int) and self._recompute is not None:
            assert self._recompute == group_id, "The operator is set to recompute in another recompute group."
        self._recompute = group_id

    def algorithms(self, tag: Optional[str] = None):
        """
        get algorithm from algorithm factory

        Args:
            tag: str or None. If None, return all 
        """
        factory = DistAlgorithmFactory()
        if tag is None:
            templates = list()
            if factory.exist(type(self)):
                templates = factory.algorithms(type(self))
            algos = list()
            for template in templates:
                algos.append(template(self))
            return algos
        else:
            if not factory.exist(type(self), tag):
                return None
            template = factory.algorithms(type(self), tag)
            return template(self)

    def replicate(self):
        """!
        Replicate the forward operation.
        The operator id, recompute and comment attribute will also be replicated.

        @return replica IRFwOperation: the replicated operator
        """
        cpy = copy.copy(self)
        cpy._device = list()
        # reset input and output
        cpy.reset_inputs(len(self.inputs()))
        for idx, input in enumerate(self.inputs()):
            cpy.set_input(idx, input)
        cpy.reset_outputs(len(self.outputs()))
        for idx, output in enumerate(self.outputs()):
            cpy.set_output(idx, output)
        cpy._mirror = None
        cpy.recompute = self.recompute
        cpy.clear_predecessor()
        cpy.clear_successor()
        return cpy

    def gen_backward(self) -> IRCell:
        """!
        Generate backward operator for this forward operator.

        Note by calling this API, this forward operator must be
        attached into any of one IRGraph, or will lead to reference
        count 0 error on gradient calcaultion.

        return: IRBpOperation
        """
        if self.mirror is not None:
            raise RuntimeError(
                "Backward Op already generated. Use self.mirror.update() instead.")
        bnode = IRBpOperation(self)
        return bnode

    def __repr__(self) -> str:
        sign = self.signature.split('.')[-1]
        ins = [t for t in self.inputs() if isinstance(t, IRTensor) and not t.is_attr()]
        dscp = (f"FwOp{self._id}-{self.device}(sign={sign}, "
                f"inputs={ins}, "
                f"outputs={self.outputs()})")
        return dscp

    def extra_repr(self) -> str:
        sign = self.signature.split('.')[-1]
        ins = [t for t in self.inputs()]
        dscp = (f"FwOp{self._id}-{self.device}(sign={sign}, "
                f"inputs={ins}, "
                f"outputs={self.outputs()})")
        return dscp


class IRBpOperation(IRCell):
    """
    Backward operation
    """

    def __init__(self, fwop: IRFwOperation):
        """
        Create dummy backward node for forward inputs and forward outputs
        
        @param fwop IRFwOperation: forward operator
        """
        assert isinstance(fwop, IRFwOperation), "Expected IRFwOperation"
        finputs, foutputs = fwop.inputs(), fwop.outputs()
        super().__init__(
            'backward', 'torch.autograd.grad',
            len(foutputs), len(finputs), init_outputs=False
        )
        # pair forward op and backward op
        IRCell.make_pair(self, fwop)
        # set inputs and outputs
        self.update()

    def update(self):
        """
        Update this backward operator.
        This is neccessary when op is partitioned and reference count is changed.

        Note in order to update produced and consumed tensor list, this call should be
        wrapped with IRGraph detach and attach:

        ```
        with graph.update(node):
            node.update()
        ```
        """
        fnode: IRFwOperation = self.mirror
        assert isinstance(fnode, IRFwOperation), "Cannot find corresponding IRFwOperation"
        for idx, itensor in enumerate(fnode.inputs()):
            grad = itensor.grad if isinstance(itensor, IRSubTensor) else None
            self.set_output(idx, grad)
        for idx, otensor in enumerate(fnode.outputs()):
            grad = otensor.grad if isinstance(otensor, IRSubTensor) else None
            self.set_input(idx, grad)

    def replicate(self):
        """
        Replicate the backward op
        """
        cpy = copy.copy(self)
        cpy._device = list()
        cpy._id = IDGenerator().gen_cell_id()
        # reset input and output
        cpy.reset_inputs(len(self.inputs()))
        for idx, input in enumerate(self.inputs()):
            cpy.set_input(idx, input)
        cpy.reset_outputs(len(self.outputs()))
        for idx, output in enumerate(self.outputs()):
            cpy.set_output(idx, output)
        cpy._mirror = None
        cpy.clear_predecessor()
        cpy.clear_successor()
        return cpy

    def __repr__(self) -> str:
        dscp = (f"BwOp{self._id}-{self.device}(FwOp{self.mirror._id}, "
                f"inputs={self.inputs()}, "
                f"outputs={self.outputs()})")
        return dscp


class IRDataOperation(IRCell):

    def __init__(self, data_num: int, batch_dims: Tuple[int], name='dataloader'):
        if len(batch_dims) != data_num:
            raise RuntimeError("Expected each output data has a specified batch dim")
        signature = 'dataloader.__next__'
        super().__init__(name, signature, 0, data_num)
        self.batch_dims = tuple(batch_dims)

    def replicate(self):
        """
        Replicate the Operation
        """
        cpy = copy.copy(self)
        cpy._device = list()
        cpy._id = IDGenerator().gen_cell_id()
        # reset input and output
        cpy.reset_inputs(len(self.inputs()))
        for idx, input in enumerate(self.inputs()):
            cpy.set_input(idx, input)
        cpy.reset_outputs(len(self.outputs()))
        for idx, output in enumerate(self.outputs()):
            cpy.set_output(idx, output)
        cpy._mirror = None
        cpy.clear_predecessor()
        cpy.clear_successor()
        return cpy

    def get_batch_dims(self):
        return copy.copy(self.batch_dims)

    def infer_shape(self):
        """
        Infer output value shape
        """
        return True

    def algorithms(self, tag: Optional[str] = None):
        """
        get algorithm from algorithm factory

        Args:
            tag: str or None. If None, return all 
        """
        factory = DistAlgorithmFactory()
        if tag is None:
            templates = list()
            if factory.exist(type(self)):
                templates = factory.algorithms(type(self))
            algos = list()
            for template in templates:
                algos.append(template(self))
            return algos
        else:
            if not factory.exist(type(self), tag):
                return None
            template = factory.algorithms(type(self), tag)
            return template(self)
    
    def __repr__(self):
        dscp = (f"DataLoader{self._id}-{self.device}(outputs={self.outputs()})")
        return dscp

    def module_repr(self) -> str:
        return repr(self)
