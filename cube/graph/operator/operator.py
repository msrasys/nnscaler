from typing import Any, Optional, Union, List
import copy

from cube.ir.cten import IRTensor, IRCell
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
        super().__init__(name, signature, input_length, output_length)
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
        old_val = self.inputs(input_index)
        # remove the old one
        if isinstance(old_val, IRSubTensor):
            old_val.parent._rm_fdst_cell(self)
        if isinstance(val, IRSubTensor):
            val.parent._add_fdst_cell(self)
        return super().set_input(input_index, val)

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

    def __repr__(self):
        inputs = list()
        for tensor in self.inputs():
            if isinstance(tensor, IRTensor):
                anno = 't'
                if tensor.is_param():
                    anno = 'w'
                if tensor.is_grad():
                    anno = 'g'
                if isinstance(tensor, IRFullTensor):
                    pid = tensor._id
                    valmap = (0,1)
                else:
                    pid = tensor.parent._id
                    valmap = tensor.val_map
                inputs.append(f'{anno}{tensor._id}(p{pid},{tensor.shape},{valmap})')
            else:
                inputs.append(tensor)
        
        outputs = list()
        for tensor in self.outputs():
            if isinstance(tensor, IRTensor):
                anno = 't'
                if tensor.is_param():
                    anno = 'w'
                if tensor.is_grad():
                    anno = 'g'
                if isinstance(tensor, IRFullTensor):
                    pid = tensor._id
                    valmap = (0,1)
                else:
                    pid = tensor.parent._id
                    valmap = tensor.val_map
                pid = tensor.parent._id if hasattr(tensor, 'parent') else tensor._id
                outputs.append(f'{anno}{tensor._id}(p{pid},{tensor.shape},{valmap})')
            else:
                outputs.append(tensor)

        sign = self.signature.split('.')[-1]
        dscp = f'Op{self._id}(sign={sign}, inputs={inputs}, outputs={outputs})'
        return dscp


class IRBpOperation(IRCell):

    def __init__(self, data_num, grad_num, name='backward'):
        signature = 'torch.autograd.backward'
        self.data_num = data_num
        self.grad_num = grad_num
        super().__init__(
            name, signature,
            input_length=data_num + grad_num,
            output_length=data_num
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
        if index is None:
            return self.inputs()[:self.data_num]
        if index >= self.data_num:
            raise RuntimeError(
                f"Set the input out of range ({index} >= {self.data_num})"
            )
        return self.inputs(index)

    def grads(self, index: Optional[int] = None) -> Union[List[Any], Any]:
        if index is None:
            return self.inputs()[self.data_num:]
        elif index >= self.grad_num:
            raise RuntimeError(
                f"Set the input out of range ({index} >= {self.grad_num})"
            )
        return self.inputs(index + self.data_num)

    def set_data(self, input_index: int, val: Any):
        """
        Set the node inputs[input_index] with the tensor

        Args:
            val: Union[IRTensor, Any]

        Return:
            the set tensor
        """
        if input_index >= self.data_num:
            raise RuntimeError(
                f"Set the input out of range ({input_index} >= {self.data_num})"
            )
        return self.set_input(input_index, val)

    def set_grad(self, input_index: int, val: Any):
        """
        Set the node gradient at input index

        Args:
            input_idx: input index
            val: Union[IRTensor, Any]

        Return:
            The set val
        """
        if input_index >= self.grad_num:
            raise RuntimeError(
                f"Set the grad out of range ({input_index} >= {self.grad_num})"
            )
        return self.set_input(input_index + self.data_num, val)

    def __repr__(self):
        datas = list()
        for tensor in self.datas():
            if isinstance(tensor, IRTensor):
                anno = 't'
                if tensor.is_param():
                    anno = 'w'
                if tensor.is_grad():
                    anno = 'g'
                # datas.append(f'{anno}{tensor._id}')
                datas.append(f'{anno}{tensor._id}(p{tensor.parent._id},{tensor.shape},{tensor.val_map})')
            else:
                datas.append(tensor)

        grads = list()
        for tensor in self.grads():
            if isinstance(tensor, IRTensor):
                anno = 't'
                if tensor.is_param():
                    anno = 'w'
                if tensor.is_grad():
                    anno = 'g'
                # grads.append(f'{anno}{tensor._id}')
                grads.append(f'{anno}{tensor._id}(p{tensor.parent._id},{tensor.shape},{tensor.val_map})')
            else:
                grads.append(tensor)
        
        outputs = list()
        for tensor in self.outputs():
            if isinstance(tensor, IRTensor):
                anno = 't'
                if tensor.is_param():
                    anno = 'w'
                if tensor.is_grad():
                    anno = 'g'
                # outputs.append(f'{anno}{tensor._id}')
                outputs.append(f'{anno}{tensor._id}(p{tensor.parent._id},{tensor.shape},{tensor.val_map})')
            else:
                outputs.append(tensor)

        sign = self.signature.split('.')[-1]
        dscp = f'bOp{self._id}(sign={sign}, grads={grads}, datas={datas}, outputs={outputs})'
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
