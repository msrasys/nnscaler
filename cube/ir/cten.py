r"""
IRCell:
    a graph node component serving for different purpose,
    e.g., operator, device graph, graph

IRTensor:
    Tensor representation serving for edges to connect IRCells

The input of IRCell are IRTensors or any deterministic values (e.g., int).
If an IRTensor is the input of Cell, then Cell.device \in IRTensor.deivce

The output of IRCell are IRTensors or any deterministic values (e.g., int)
If an IRTensor is the output of Cell, then Cell.device == IRTensor.device
"""


from typing import List, Union, Optional, Any
import copy

from cube.ir.unique import IDGenerator
from cube.ir.dtype import IRDType


__all__ = ['IRCell', 'IRDType', 'IRTensor']


class IRCell:
    r"""
    IRCell serves as a general node for different purpose
    """

    def __init__(self,
                 name: str,
                 signature: str,
                 input_length: int,
                 output_length: int,
                 init_outputs = True):
        """
        Create a node with name (variable name) and module type (module_name)

        Args:
            name (str): the cell name
            signature (str): the cell function signature,
                e.g., torch.functional.nn.linear
            input_length (int): the number of inputs for the op
            output_length (int): the number of outputs for the op
        """
        # node info
        self._id: int = IDGenerator().gen_cell_id()
        self.name: str = name
        self.signature = signature

        self._dtype = IRDType.unknown
        self._device = list()

        # source tensors
        self._inputs: List[Any] = [None] * input_length
        
        # destination tensors
        self._outputs: List[IRTensor] = [None] * output_length
        if init_outputs:
            self._outputs: List[IRTensor] = [IRTensor() for _ in range(output_length)]
            for tensor in self._outputs:
                tensor.attach_cell(self)

        # destination cells. [-1] for control dependency
        self._successors: List[List[IRCell]] = [list() for _ in range(output_length+1)]
        # source cells. [-1] for control dependency
        self._predecessors: List[List[IRCell]] = [list() for _ in range(input_length+1)]

        self._mirror = None
        self._tag = None

    def __eq__(self, other):
        if isinstance(other, IRCell):
            return self._id == other._id
        return False

    @property
    def device(self):
        return copy.copy(self._device)

    @device.setter
    def device(self, device_id: Union[int, List[int]]):
        """
        Set the operation device.
        """
        if isinstance(device_id, int):
            device_id = [device_id]
        if not all([isinstance(devid, int) for devid in device_id]):
            raise KeyError("Require device Union[int, List[int]]")
        self._device = copy.copy(list(device_id))

    @property
    def mirror(self):
        """
        The mirror cell. E.g., forward op / backward op.
        """
        return self._mirror

    @mirror.setter
    def mirror(self, other):
        raise RuntimeError("Use IRCell.make_pair instead")

    @staticmethod
    def make_pair(cell1, cell2):
        if not isinstance(cell1, IRCell):
            raise TypeError("Expected cell1 to be IRCell")
        if not isinstance(cell2, IRCell):
            raise TypeError("Expected cell2 to be IRCell")
        cell1._mirror = cell2
        cell2._mirror = cell1

    def on_device(self, device_id: int):
        """
        Check whether the operation is on device_id

        Returns:
            Boolean
        """
        if not isinstance(device_id, int):
            raise TypeError(f"Expected device id to be int but got {type(device_id)}")
        return device_id in self.device

    def inputs(self, index: Optional[int] = None) -> Union[List[Any], Any]:
        """
        Get input tensor at input index

        Args:
            index (int or None): 
                index of the inputs, None will return the nodes
                for all the inputs

        Returns:
            values: Union[List[Any], Any]
        """
        if isinstance(index, int):
            if index >= len(self._inputs):
                raise RuntimeError(
                    f"Get the input out of range ({index} >= {len(self._inputs)}"
                )
            return self._inputs[index]
        elif index is None:
            return copy.copy(self._inputs)
        else:
            raise TypeError("Expected index to be None or int")

    def predecessors(self, index: Optional[int] = None) -> List:
        """
        Get input operator at input index
        (or index = -1 for control dependency)

        Returns:
            cell(s): Union[List[IRCell], IRCell]
        """
        if isinstance(index, int):
            if index >= len(self._inputs):
                raise RuntimeError(
                    f"Get the input out of range ({index} >= {len(self._inputs)}"
                )
            return copy.copy(self._predecessors[index])
        elif index is None:
            predecessors = list()
            for pre_cells in self._predecessors:
                predecessors += pre_cells
            return predecessors
        else:
            raise TypeError("Expected index to be None or int")

    def outputs(self, index: Optional[int] = None) -> Union[List[Any], Any]:
        """
        Get output tensor at output index

        Args:
            index (int or None): 
                index of the outputs, None will return the nodes
                for all the outputs

        Returns:
            values: Union[List[Any], Any]
        """
        if isinstance(index, int):
            if index >= len(self._outputs):
                raise RuntimeError(
                    f"Get the output out of range ({index} >= {len(self._outputs)}"
                )
            return self._outputs[index]
        elif index is None:
            return copy.copy(self._outputs)
        else:
            raise TypeError("Expected index to be None or int")

    def successors(self, index: Optional[int] = None) -> List:
        """
        Get output operator at output index

        Args:
            index (int or None): 
                index of the outputs (or -1 for control dependency),
                None will return the nodes for all the outputs
        """
        if isinstance(index, int):
            if index >= len(self._outputs):
                raise RuntimeError(
                    f"Get the output out of range ({index} >= {len(self._outputs)}"
                )
            return copy.copy(self._successors[index])
        elif index is None:
            successors = list()
            for post_cells in self._successors:
                successors += post_cells
            return successors
        else:
            raise TypeError("Expected index to be None or int")

    def set_input(self, input_index: int, val: Any):
        """
        Set the node inputs[input_index] with the tensor

        Args:
            val: Union[IRTensor, Any]

        Return:
            the set tensor
        """
        if input_index >= len(self.inputs()):
            raise RuntimeError(
                f"Set the input out of range ({input_index} >= {len(self._inputs)})"
            )
        if isinstance(val, IRTensor):
            # copy the val
            val = copy.copy(val)
            # set tensor dst
            val.attach_cell(self)
            # set input value dtype
            if self._dtype != IRDType.unknown:
                val.dtype = self._dtype
            # set cell dtype
            elif val.dtype != IRDType.unknown:
                self._dtype = val.dtype
        self._inputs[input_index] = val
        return val

    def set_output(self, output_index: int, val: Any):
        """
        Set the node inputs[output_index] with the tensor

        Args:
            val: Union[IRTensor, Any]
                IRTensor or any deterministic value (int, bool, str, etc)
        """
        if output_index >= len(self.outputs()):
            raise RuntimeError(
                f"Set the input out of range ({output_index} >= {len(self._inputs)})"
            )
        if isinstance(val, IRTensor):
            val = copy.copy(val)
            val.attach_cell(self)
            # set output value dtype
            if self._dtype != IRDType.unknown:
                val.dtype = self._dtype
            # set cell dtype
            elif val.dtype != IRDType.unknown:
                self._dtype = val.dtype
        self._outputs[output_index] = val
        return val

    def add_predecessor(self, input_index: int, cell):
        """
        Add a predecessor cell in the input_index slot. 
        
        Note this won't add successor if caller cell to the node

        To add control dependency, use `input_index=-1`
        """
        if not isinstance(cell, IRCell):
            raise TypeError("Expected node to be IRCell")
        if input_index >= len(self.inputs()):
            raise RuntimeError(
                f"Set the input out of range ({input_index} >= {len(self._inputs)})"
            )
        if cell not in self._predecessors[input_index]:
            self._predecessors[input_index].append(cell)

    def clear_predecessor(self):
        """
        Clear all predecessors
        """
        self._predecessors = [
            list() for _ in range(len(self.inputs()) + 1)
        ]

    def add_successor(self, output_index: int, cell):
        """
        Set self node the output index node. 
        `node` will take the self.outputs(index) as the input

        To add control dependency, use `output_index=-1`
        """
        if not isinstance(cell, IRCell):
            raise TypeError("Expected node to be IRCell")
        if cell not in self._successors[output_index]:
            self._successors[output_index].append(cell)

    def clear_successor(self):
        """
        Clear all successors
        """
        self._successors = [
            list() for _ in range(len(self.outputs()) + 1)
        ]

    def make_empty(self):
        """
        Clear all inputs, outputs of this Cell
        """
        for idx in range(len(self.inputs())):
            self.set_input(idx, None)
        for idx in range(len(self.outputs())):
            self.set_output(idx, None)

    @staticmethod
    def get_inputs(cells):
        """
        Get all the input tensors the is not generated by nodes

        Inputs

        Returns:
            List[IRTensor]
        """
        all_outputs = list()
        for cell in cells:
            all_outputs += cell.outputs()
        inputs = list()
        for cell in cells:
            for input in cell.inputs():
                if isinstance(input, IRTensor):
                    if input not in all_outputs:
                        if input not in inputs:
                            inputs.append(input)
        return inputs

    @staticmethod
    def get_outputs(cells):
        """
        Get all the input tensors the is not generated by nodes

        Returns:
            List[IRTensor]
        """
        all_inputs = list()
        for node in cells:
            all_inputs += node.inputs()
        outputs = list()
        for node in cells:
            for output in node.outputs():
                if isinstance(output, IRTensor):
                    if output not in all_inputs:
                        if output not in outputs:
                            outputs.append(output)
        return outputs

    @property
    def tag(self) -> Any:
        return self._tag

    @tag.setter
    def tag(self, info: Any):
        """
        Tag an info to the cell
        """
        self._tag = info

    def __repr__(self):
        """
        Cell string presentation
        """
        inputs = list()
        for tensor in self.inputs():
            if isinstance(tensor, IRTensor):
                inputs.append(f't{tensor._id}-dev{tensor.device}')
            else:
                inputs.append(tensor)

        outputs = list()
        for tensor in self.outputs():
            if isinstance(tensor, IRTensor):
                outputs.append(f't{tensor._id}-dev{tensor.device}')
            else:
                outputs.append(tensor)
        dcsp = f'Cell-{self._id}({self.signature}, device={self.device})'\
               f'({inputs}) -> {outputs}'
        return dcsp


class IRTensor:
    """
    IRTensor serves as IRGraph edge

    Note by setting IRTensor name to "None" indicates this tensor holds nothing
    and will be translated to None in code generation. 
    """

    _attr = ['name', '_is_param', '_requires_grad', '_is_grad', '_grad', '_dtype']

    def __init__(self, shape=None, name='tensor', dtype=IRDType.unknown):

        self._id: int = IDGenerator().gen_tensor_id()
        self._shape: Optional(List[int]) = shape
        self.name = name if name else 'tensor'

        # device
        self._cell: Optional[IRCell] = None

        self._dtype: IRDType = dtype

        self._requires_grad = True
        self._is_param = False

        self._is_grad = False
        self._grad = None  # the gradient of this tensor
        self._data = None  # the tensor of this gradient belongs to

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._requires_grad = val

    @property
    def dtype(self):
        """
        Data type
        """
        return self._dtype

    @dtype.setter
    def dtype(self, val: IRDType):
        """
        Set data type
        """
        if not isinstance(val, IRDType):
            raise TypeError(f"Expected IRDType but got {val}")
        self._dtype = val

    def attach_cell(self, cell: IRCell):
        """
        Attach to a cell, to be with input or output
        """
        if not isinstance(cell, IRCell):
            raise TypeError("Expected an IRCell")
        self._cell = cell

    def detach_cell(self):
        """
        Detach from a cell
        """
        self._cell = None

    @property
    def device(self) -> List[int]:
        return self._cell.device

    @device.setter
    def device(self, val: Union[int, List[int]]):
        raise RuntimeError(
            "tensor placement is not allowed to set manually"
        )

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, requires: bool):
        if not isinstance(requires, bool):
            raise TypeError("Expected bool")
        self._requires_grad = requires
        if not requires:
            self.grad = None

    def as_param(self):
        """
        Set the tensor as trainable parameter
        """
        self.requires_grad = True
        self._is_grad = False
        self._is_param = True
        return self

    def is_param(self):
        """
        Check if the tensor is parameter
        """
        return self._is_param

    @property
    def data(self):
        return self._data

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, grad):
        if grad is None:
            self._grad = grad
            return
        elif not isinstance(grad, IRTensor):
            raise TypeError("grad can only be None or Tensor")
        self.requires_grad = True
        self._grad = grad
        grad._data = self

    def as_grad(self):
        self._is_param = False
        self._is_grad = True
        return self

    def is_grad(self):
        return self._is_grad

    def renew(self):
        """
        Renew a new tensor with same name and shape,
        but with a different new id

        Returns:
            tensor
        """
        tensor = IRTensor(self._shape, self.name)
        new_id = tensor._id
        for key in self.__dict__:
            setattr(tensor, key, getattr(self, key))
        # clear attached cells
        tensor._cell = list()
        tensor._id = new_id
        return tensor

    def __copy__(self):
        """
        Copy the tensor that will have the exactly same id
        except the empty attached cell

        Returns:
            tensor
        """
        tensor = IRTensor(self._shape, self.name)
        for key in self.__dict__:
            setattr(tensor, key, getattr(self, key))
        # clear attached cells
        tensor._cell = list()
        return tensor

    def __eq__(self, tensor):
        if not isinstance(tensor, IRTensor):
            return False
        return self._id == tensor._id

    @property
    def shape(self):
        return copy.copy(self._shape)

    @shape.setter
    def shape(self, val):
        if self._shape is not None and self._shape != val:
            raise RuntimeError("Try to change shape")
        if not isinstance(val, list) or \
           not all([isinstance(size, int) for size in val]):
            raise RuntimeError("Expected shape to be list[int]")
        self._shape = copy.copy(list(val))

    def src(self, cells: List[IRCell]) -> List[IRCell]:
        """
        Return all the cells that will generate this tensor
        """
        src_cells = list()
        for cell in cells:
            if not isinstance(cell, IRCell):
                raise TypeError("Expected cells to be List[IRCell]")
            if self in cell.outputs():
                src_cells.append(cell)
        return src_cells

    def dst(self, cells: List[IRCell]) -> List[IRCell]:
        """
        Return all the cells that will generate this tensor
        """
        dst_cells = list()
        for cell in cells:
            if not isinstance(cell, IRCell):
                raise TypeError("Expected cells to be List[IRCell]")
            if self in cell.inputs():
                dst_cells.append(cell)
        return dst_cells

    def is_leaf(self, cells: List[IRCell]):
        """
        Check if it is a leaf tensor (parameter or input data)
        """
        return len(self.src(cells)) == 0

    def backward(self):
        """
        Autograd backward on the tensor
        """
        from cube.logics.translator import LogicTranslator
        return LogicTranslator.backward(self)

    def __repr__(self):
        dscp = f'Tensor(id={self._id}, shape={self.shape}, device={self.device})'
        return dscp
