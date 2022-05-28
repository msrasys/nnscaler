from typing import Any, Optional, Tuple, Union, List
import copy

from cube.ir.cten import IRCell, IRTensor
from cube.ir.tensor import IRFullTensor, IRSubTensor
from cube.algorithm.factory import DistAlgorithmFactory
from cube.ir.unique import IDGenerator


class BaseOperator:

    def __init__(self, name: str, signature: str,
                 input_length: int, output_length: int,
                 init_outputs=False):
        super().__init__(name, signature,
                         input_length, output_length,
                         init_outputs=init_outputs)

    def infer_shape(self):
        """
        Infer output value shape
        """
        raise NotImplementedError

    def replicate(self):
        """
        Replicate the Operation
        """
        cpy = copy.copy(self)
        cpy._device = list()
        cpy._id = IDGenerator().gen_cell_id()
        # reset input and output
        cpy._inputs = [None] * len(self.inputs())
        for idx, input in enumerate(self.inputs()):
            cpy.set_input(idx, input)
        cpy._outputs = [None] * len(self.outputs())
        for idx, output in enumerate(self.outputs()):
            cpy.set_output(idx, output)
        cpy._mirror = None
        cpy._tag = None
        cpy.clear_predecessor()
        cpy.clear_successor()
        return cpy


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

    def replicate(self):
        """
        Replicate the Operation
        """
        cpy = copy.copy(self)
        cpy._device = list()
        # cpy._id = IDGenerator().gen_cell_id()
        # reset input and output
        cpy._inputs = [None] * len(self.inputs())
        for idx, input in enumerate(self.inputs()):
            cpy.set_input(idx, input)
        cpy._outputs = [None] * len(self.outputs())
        for idx, output in enumerate(self.outputs()):
            cpy.set_output(idx, output)
        cpy._mirror = None
        cpy._tag = None
        cpy.clear_predecessor()
        cpy.clear_successor()
        return cpy

    def gen_backward(self):
        """
        Generate backward operator for this forward operator.

        Note by calling this API, this forward operator must be
        attached into any of one IRGraph, or will lead to reference
        count 0 error on gradient calcaultion.

        return: IRBpOperation
        """
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
        cpy._id = IDGenerator().gen_cell_id()
        # reset input and output
        cpy._inputs = [None] * len(self.inputs())
        for idx, input in enumerate(self.inputs()):
            cpy.set_input(idx, input)
        cpy._outputs = [None] * len(self.outputs())
        for idx, output in enumerate(self.outputs()):
            cpy.set_output(idx, output)
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
        if isinstance(val, IRTensor):
            val.attach_cell(self)
        self._datas[data_index] = val
        return val

    def update(self):
        """
        Update this backward operator.
        This is neccessary when op is partitioned and reference count is changed.

        Note in order to update produced and consumed tensor list, this call should be
        wrapped with IRGraph detach and attach:

        ```
        idx = graph.detach(node)
        node.update()
        graph.attach(node, idx)
        ```
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

    def __init__(self, data_num: int, batch_dims: Tuple[int], name='dataloader'):
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
        cpy._id = IDGenerator().gen_cell_id()
        # reset input and output
        cpy._inputs = [None] * len(self.inputs())
        for idx, input in enumerate(self.inputs()):
            cpy.set_input(idx, input)
        cpy._outputs = [None] * len(self.outputs())
        for idx, output in enumerate(self.outputs()):
            cpy.set_output(idx, output)
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
