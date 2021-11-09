from typing import Any, Optional, Union, List

from cube.ir.cten import IRTensor, IRCell
from cube.graph.tensor import IRFullTensor, IRSubTensor
from cube.algorithm.factory import DistAlgorithmFactory


__call__ = ['IRFwOperation', 'IRBpOperation']


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

    def __repr__(self):
        inputs = list()
        for tensor in self.inputs():
            if isinstance(tensor, IRTensor):
                anno = 't'
                if tensor.is_param():
                    anno = 'w'
                if tensor.is_grad():
                    anno = 'g'
                inputs.append(f'{anno}{tensor._id}')
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
                outputs.append(f'{anno}{tensor._id}')
            else:
                outputs.append(tensor)

        dscp = f'Op(id={self._id}, signature={self.signature}, device={self.device}, inputs={inputs}, outputs={outputs})'
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
                datas.append(f'{anno}{tensor._id}')
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
                grads.append(f'{anno}{tensor._id}')
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
                outputs.append(f'{anno}{tensor._id}')
            else:
                outputs.append(tensor)

        dscp = f'bOp(id={self._id}, signature={self.signature}, device={self.device}, grads={grads}, datas={datas}, outputs={outputs})'
        return dscp


class IRDataOperation(IRCell):

    def __init__(self, data_num: int, name='dataloader'):

        signature = 'dataloader.__next__'
        super().__init__(name, signature, 0, data_num)

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
