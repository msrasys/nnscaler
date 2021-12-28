from typing import Any, Optional, Union, List
import copy

from cube.ir.cten import IRCell
from cube.graph.tensor import IRFullTensor, IRSubTensor
from cube.algorithm.factory import DistAlgorithmFactory


__all__ = ['IRFwOperation', 'IRBpOperation', 'IRDataOperation', 'IROptimOperation']


class IRFwOperation(IRCell):

    def __init__(self,
                 name: str, 
                 signature: str,
                 input_length: int,
                 output_length: int):
        """
        Create a node with name (variable name) and module type (module_name)

        Args:
            name (str): the op semantic name
            signature (str): the op signature, e.g., torch.functional.nn.linear
            input_length (int): the number of inputs for the op
            output_length (int): the number of outputs for the op
        """
        # additional argument
        self.kwargs = dict()
        super().__init__(name, signature, input_length, output_length, init_outputs=False)
        outputs = [IRFullTensor() for _ in range(output_length)]
        for idx, output in enumerate(outputs):
            self.set_output(idx, output)

    def infer_shape(self):
        """
        Infer output value shape
        """
        raise NotImplementedError

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

    def set_input(self, input_index: int, val: Any):
        # remove the consumer
        old_val = self.inputs(input_index)
        if isinstance(old_val, IRSubTensor):
            old_val.parent.rm_consumer(self)
        # add the consumer
        val = super().set_input(input_index, val)
        if isinstance(val, IRSubTensor):
            val.parent.add_consumer(self, val)
        return val

    def set_output(self, output_index: int, val: Any):
        # remove the producer
        old_val = self.outputs(output_index)
        if isinstance(old_val, IRSubTensor):
            old_val.parent.rm_producer(self)
        # add the producer
        val = super().set_output(output_index, val)
        if isinstance(val, IRSubTensor):
            val.parent.add_producer(self, val)
        return val

    def replicate(self):
        """
        Replicate the Operation
        """
        cpy = copy.copy(self)
        cpy._device = list()
        cpy._inputs = copy.copy(self._inputs)
        cpy._outputs = copy.copy(self._outputs)
        cpy._mirror = None
        cpy._tag = None
        cpy.clear_predecessor()
        cpy.clear_successor()
        return cpy

    def gen_backward(self):
        if self.mirror is not None:
            raise RuntimeError(
                "Backward Op already generated. Use self.mirror.update() instead.")
        bnode = IRBpOperation(
            data_num=len(self.inputs()),
            grad_num=len(self.outputs())
        )
        for idx, input in enumerate(self.inputs()):
            grad = None
            if isinstance(input, IRSubTensor):
                grad = input.get_grad(self)
                input.grad = grad
            bnode.set_data(idx, input)
            bnode.set_output(idx, grad)
        for idx, output in enumerate(self.outputs()):
            grad = output.get_grad(self)
            output.grad = grad
            bnode.set_input(idx, grad)
        IRCell.make_pair(self, bnode)
        return bnode

    def __repr__(self):
        sign = self.signature.split('.')[-1]
        dscp = f'FwOp{self._id}-{self.device}(sign={sign}, inputs={self.inputs()}, outputs={self.outputs()})'
        return dscp

    def module_repr(self) -> str:
        """
        Weight-hidden string representation
        """
        sign = self.signature.split('.')[-1]
        ins = [t for t in self.inputs() if isinstance(t, IRSubTensor) and not t.is_param()]
        dscp = f'FwOp{self._id}-{self.device}(sign={sign}, inputs={ins}, outputs={self.outputs()})'
        return dscp


class IRBpOperation(IRCell):

    def __init__(self, data_num: int, grad_num, name='backward'):
        """
        Args:
            data_num (int): corresponding forward input length
            grad_num (int): corresponding forward output length
        """
        signature = 'torch.autograd.backward'
        self.data_num = data_num
        self.grad_num = grad_num
        self._datas = [None] * data_num
        super().__init__(
            name, signature,
            input_length=grad_num,
            output_length=data_num,
            init_outputs=False
        )

    def replicate(self):
        """
        Replicate the backward op
        """
        cpy = copy.copy(self)
        cpy._device = list()
        cpy._inputs = copy.copy(self._inputs)
        cpy._outputs = copy.copy(self._outputs)
        cpy._mirror = None
        cpy._tag = None
        cpy.clear_predecessor()
        cpy.clear_successor()
        return cpy

    def datas(self, index: Optional[int] = None) -> Union[List[Any], Any]:
        """
        Forward inputs
        """
        if index is None:
            return copy.copy(self._datas[:self.data_num])
        if index >= self.data_num:
            raise RuntimeError(
                f"Set the input out of range ({index} >= {self.data_num})"
            )
        return self._datas[index]

    def set_data(self, data_index: int, val: Any):
        """
        Set the node inputs[input_index] with the tensor

        Args:
            val: Union[IRTensor, Any]

        Return:
            the set tensor
        """
        if data_index >= self.data_num:
            raise RuntimeError(
                f"Set the input out of range ({data_index} >= {self.data_num})"
            )
        val = copy.copy(val)
        val.attach_cell(self)
        self._datas[data_index] = val
        return val

    def set_input(self, input_index: int, val: Any):
        """
        Set the node input gradient
        (i.e., output gradient in forward) at input index.
        The grad is same order with corresponding output tensor
        of it's forward tensor

        Args:
            input_idx: input index
            val: Union[IRTensor, Any]

        Return:
            The set val
        """
        # remove the consumer
        old_val = self.inputs(input_index)
        if isinstance(old_val, IRSubTensor):
            old_val.parent.rm_consumer(self)
        # add the consumer
        val = super().set_input(input_index, val)
        if isinstance(val, IRSubTensor):
            val.parent.add_consumer(self, val)
        return val

    def set_output(self, output_index: int, val: Any):
        """
        Set op output grad (Forward input gradient)
        """
        # remove the producer
        old_val = self.outputs(output_index)
        if isinstance(old_val, IRSubTensor):
            old_val.parent.rm_producer(self)
        # add the producer
        val = super().set_output(output_index, val)
        if isinstance(val, IRSubTensor):
            val.parent.add_producer(self, val)
        return val

    def update(self):
        """
        Update this backward operator.
        This neccessary when op is partitioned and reference count is changed.
        """
        fnode = self.mirror
        for idx, input in enumerate(fnode.inputs()):
            grad = None
            if isinstance(input, IRSubTensor):
                grad = input.get_grad(fnode)
            self.set_data(idx, input)
            self.set_output(idx, grad)
        for idx, output in enumerate(fnode.outputs()):
            grad = output.get_grad(fnode)
            self.set_input(idx, grad)

    def __repr__(self):
        dscp = f'BwOp{self._id}-{self.device}(FwOp{self.mirror._id}, inputs={self.inputs()}, datas={self.datas()}, outputs={self.outputs()})'
        return dscp

    def module_repr(self) -> str:
        """
        Weight-hidden string representation
        """
        ins = [t for t in self.datas() if isinstance(t, IRSubTensor) and not t.is_param()]
        outs = [t.grad for t in ins]
        assert all([out in self.outputs() for out in outs])
        dscp = f'BwOp{self._id}-{self.device}(FwOp{self.mirror._id}, inputs={self.inputs()}, outputs={outs})'
        return dscp


class IRDataOperation(IRCell):

    def __init__(self, data_num: int, batch_dims: List[int], name='dataloader'):
        if not isinstance(batch_dims, list):
            raise RuntimeError("Expected batch dims to be a list")
        if len(batch_dims) != data_num:
            raise RuntimeError("Expected each output data has a specified batch dim")
        signature = 'dataloader.__next__'
        super().__init__(name, signature, 0, data_num)
        self.batch_dims = batch_dims

    def replicate(self):
        """
        Replicate the Operation
        """
        cpy = copy.copy(self)
        cpy._device = list()
        cpy._inputs = copy.copy(self._inputs)
        cpy._outputs = copy.copy(self._outputs)
        cpy._mirror = None
        cpy._tag = None
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

    def set_output(self, output_index: int, val: Any):
        # remove the producer
        old_val = self.outputs(output_index)
        if isinstance(old_val, IRSubTensor):
            old_val.parent.rm_producer(self)
        # add the producer
        val = super().set_output(output_index, val)
        if isinstance(val, IRSubTensor):
            val.parent.add_producer(self, val)
        return val

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
        dscp = f'DataLoader{self._id}-{self.device}(outputs={self.outputs()})'
        return dscp

    def module_repr(self) -> str:
        return repr(self)


class IROptimOperation(IRCell):

    def __init__(self, weights: List[IRSubTensor], ranks: List[int], name='optimizer'):
        if not all([isinstance(w, IRSubTensor) and w.is_param() for w in weights]):
            raise RuntimeError("Expected a list of gradient IRSubTensor")
        if not all([isinstance(rank, int) for rank in ranks]):
            raise RuntimeError("Expected a list of int")
        signature = None
        self._ranks = ranks

        super().__init__(name, signature, len(weights), 0)
        for idx, weight in enumerate(weights):
            self.set_input(idx, weight)

    @property
    def ranks(self):
        return copy.copy(self._ranks)
