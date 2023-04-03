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


from functools import lru_cache
from typing import Iterable, List, Tuple, Union, Optional, Any
import copy

from cube.ir.unique import IDGenerator
from cube.ir.dtype import IRDType, dtype2byte_size


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

        self._device: Tuple[int] = ()

        # source tensors
        self._inputs: List[Optional[IRTensor]] = [None,] * input_length
        
        # destination tensors
        self._outputs: List[Optional[IRTensor]] = [None,] * output_length
        if init_outputs:
            self._outputs = [IRTensor() for _ in range(output_length)]
            for tensor in self._outputs:
                tensor.cell = self

        # destination cells. [-1] for control dependency
        self._successors: List[List[IRCell]] = [list() for _ in range(output_length+1)]
        # source cells. [-1] for control dependency
        self._predecessors: List[List[IRCell]] = [list() for _ in range(input_length+1)]

        self._mirror: Optional[IRCell] = None

        # the comment for code generation
        self._comment: Optional[str] = None

    @property
    def cid(self) -> int:
        """
        Get cell id

        @return cid int: the cell id.
        """
        return self._id

    @property
    def device(self) -> Tuple[int]:
        return self._device

    @device.setter
    def device(self, device_id: Union[int, List[int]]):
        """
        Set the operation device.
        """
        if isinstance(device_id, int):
            device_id = (device_id,)
        if not all([isinstance(devid, int) for devid in device_id]):
            raise KeyError("Require device Union[int, List[int]]")
        self._device = tuple(device_id)

    def dispatch(self, device: int):
        """
        Instantiate this node to a specified device. Its mirror node will also
        be dispatched and paired with this node.

        For single operators, the mirror node will be reserved.
        For nodes that cover multiple devices, e.g., IRSegment and IRAdapter,
        the mirror node will be removed and require additional `make_pair` elsewhere.
        
        @param device int: device id
        @return dispatched_node IRCell: the node that only has one device placement.
        """
        assert len(self.device) == 1, \
            f"Require dispatch implementation for node type: {type(self)}"
        if isinstance(self.mirror, IRCell):
            assert len(self.mirror.device) == 1, \
                f"IRCell got unexpected mirro node that has multiple device placement.\n{self.mirror}"
        assert device in self.device, f"Fail to dispatch to device {device}. node: {self}"
        return self

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
        if isinstance(cell1, IRCell):
            cell1._mirror = cell2
        elif cell1 is not None:
            raise TypeError("Expected cell1 to be IRCell or None")
        if isinstance(cell2, IRCell):
            cell2._mirror = cell1
        elif cell2 is not None:
            raise TypeError("Expected cell2 to be IRCell or None")

    def isfw(self) -> bool:
        """
        Return if the IRCell is executed fully in forward phase.
        This needs to be overrided by derived classes
        """
        return True

    def input(self, index:int):
        # type: (int) -> Optional[IRTensor]
        """
        Get the input tensor at input index

        Args:
            index (int): 
                index of the inputs

        Returns:
            values: Optional[IRTensor]
        """
        return self._inputs[index]

    # 'maxsize=None' set no limit on cache growth, but it's ok since we have no args
    @lru_cache(maxsize=None)
    def inputs(self):
        # type: () -> Tuple[Optional[IRTensor], ...]
        """
        Get all input tensors

        Returns:
            values: Tuple[Optional[IRTensor], ...]
        """

        return tuple(self._inputs)

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

    def output(self, index:int):
        # type: (int) -> Optional[IRTensor]
        """
        Get the output tensor at output index

        Args:
            index (int): 
                index of the outputs

        Returns:
            values: Optional[IRTensor]
        """
        return self._outputs[index]

    # 'maxsize=None' set no limit on cache growth, but it's ok since we have no args
    @lru_cache(maxsize=None)
    def outputs(self):
        # type: () -> Tuple[Optional[IRTensor], ...]
        """
        Get all output tensors

        Returns:
            values: Tuple[Optional[IRTensor], ...]
        """

        return tuple(self._outputs)

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

    def reset_inputs(self, length:int) -> None:
        """
        Resize the inputs list to the new length and reset all input items to None.
        """
        self._inputs = [None] * length
        self.inputs.cache_clear()

    def set_input(self, input_index: int, val):
        # type: (int, Optional[IRTensor]) -> Optional[IRTensor]
        """
        Set the node inputs[input_index] with the tensor

        Args:
            val: Optional[IRTensor]

        Return:
            the set tensor
        """
        c = len(self._inputs)
        if input_index >= c or input_index < -c:
            raise RuntimeError(
                f"Set the input out of range ({input_index} >= {c} or {input_index} < {-c})"
            )
        if isinstance(val, IRObject):
            # copy the val
            val = copy.copy(val)
            # set tensor dst
            val.cell = self

        self._inputs[input_index] = val

        self.inputs.cache_clear()

        return val

    def reset_outputs(self, length:int) -> None:
        """
        Resize the outputs list to the new length and reset all output items to None.
        """
        self._outputs = [None] * length
        self.outputs.cache_clear()

    def set_output(self, output_index: int, val):
        # type: (int, Optional[IRTensor]) -> Optional[IRTensor]
        """
        Set the node inputs[output_index] with the tensor

        Args:
            val: Optional[IRTensor]
                IRTensor or any deterministic value (int, bool, str, etc)
        """
        c = len(self._outputs)
        if output_index >= c or output_index < -c:
            raise RuntimeError(
                f"Set the input out of range ({output_index} >= {c} or {output_index} < {-c})"
            )
        if isinstance(val, IRObject):
            val = copy.copy(val)
            val.cell = self

        self._outputs[output_index] = val
        self.outputs.cache_clear()

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
        `node` will take the self.output(index) as the input

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
        # type: (Iterable[IRCell]) -> list[IRCell]
        """
        Get all the input tensors the is not generated by nodes

        Inputs

        Returns:
            List[IRTensor]
        """
        all_outputs = list()
        for cell in cells:
            all_outputs.extend(cell.outputs())
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
        # type: (Iterable[IRCell]) -> list[IRCell]
        """
        Get all the input tensors the is not generated by nodes

        Returns:
            List[IRTensor]
        """
        all_inputs = list()
        for node in cells:
            all_inputs.extend(node.inputs())
        outputs = list()
        for node in cells:
            for output in node.outputs():
                if isinstance(output, IRTensor):
                    if output not in all_inputs:
                        if output not in outputs:
                            outputs.append(output)
        return outputs

    @property
    def comment(self) -> Any:
        return self._comment

    @comment.setter
    def comment(self, info: str):
        """
        Tag an info to the cell
        """
        assert isinstance(info, str), "comment only allowed to be string"
        self._comment = info 

    def __repr__(self) -> str:
        """
        Cell string presentation
        """
        ins = [t for t in self.inputs() if isinstance(t, IRTensor)]
        dscp = (f"Cell{self._id}-{self.device}(sign={self.signature}, "
                f"inputs={ins}, "
                f"outputs={self.outputs()})")
        return dscp


class IRObject:
    """
    IRObject serves as general data of IRGraph edge
    """

    def __init__(self, name: Optional[str] = None, tid: Optional[int] = None):
        """
        @param name str: object name
        @param tid int: object unique id
        """
        self._id: int = tid if isinstance(tid, int) else IDGenerator().gen_tensor_id()
        self.name: str = name if name else 'obj'
        self._cell: Optional[IRCell] = None
        self._is_attr: bool = False

    def __eq__(self, obj):
        if not isinstance(obj, IRObject):
            return False
        return self._id == obj.tid

    def __hash__(self) -> int:
        return self._id

    def getstate_for_dump(self):
        """
        __getstate__ method for pickle dump

        @warning: dump an IRObject will disconnect the tensor to its cell
        """
        state = self.__dict__.copy()
        # this will decouple the interconnected object and cell during dump.
        state['_cell'] = None
        return state

    @property
    def tid(self) -> int:
        """Get object id"""
        return self._id

    @property
    def cell(self) -> IRCell:
        return self._cell
    
    @cell.setter
    def cell(self, val: Optional[IRCell]):
        assert isinstance(val, IRCell) or val is None, "Expected cell to be Optional[IRCell]"
        self._cell = val

    @property
    def device(self) -> Tuple[int]:
        if self._cell:
            return tuple(self._cell.device)
        else:
            return ()

    @device.setter
    def device(self, val: Union[int, List[int]]):
        raise RuntimeError(
            "IRObject placement is not allowed to set manually"
        )
    
    @property
    def parent(self):
        """Get parent"""
        return self

    def __eq__(self, obj) -> bool:
        if not isinstance(obj, IRObject):
            return False
        return self._id == obj.tid

    def __copy__(self):
        """Copy this object but remove the cell information"""
        return IRObject(self.name, self._id)

    def as_attr(self):
        """
        Set the obj as graph attributes
        """
        self._is_attr = True
        return self

    def is_attr(self) -> bool:
        """!
        Check if the object is graph attribute.

        @return is_attr boolean: True if is graph attribute (buffer or parameter or gradient of parameter)
        """
        return self._is_attr

    def overlap(self, other: Any) -> bool:
        """!
        Check whether two object can be overlapped
        """
        if isinstance(other, IRObject):
            return other.tid == self._id
        else:
            return False

    def __repr__(self):
        return f'Object({self.name}{self.tid})'


class IRTensor(IRObject):
    """
    IRTensor serves as tensor data of IRGraph edge

    Note by setting IRTensor name to "None" indicates this tensor holds nothing
    and will be translated to None in code generation. 
    """

    _meta = ['name', '_is_attr', '_is_grad', '_requires_grad', '_dtype']

    def __init__(self, shape=None, name='tensor', dtype=IRDType.unknown, tid=None):

        super().__init__(name, tid)
        self._shape: Tuple[int] = () if shape is None else tuple(shape)
        self._cell: Optional[IRCell] = None
        assert isinstance(dtype, IRDType), f'expect IRDType, get {dtype} with type {type(dtype)}'
        self._dtype: IRDType = dtype
        # tensor gradient
        self._is_grad: bool = False
        self._requires_grad: bool = False
        self._grad: Optional[Union[IRTensor, float]] = None

    @property
    def dtype(self) -> IRDType:
        """
        Tensor data type
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
        if isinstance(self._grad, IRTensor):
            self._grad._dtype = val

    def is_param(self) -> bool:
        """!
        Check if the tensor is parameter

        @return is_param boolean: True if is parameter.
        """
        return self._is_attr and self.requires_grad

    def is_buffer(self) -> bool:
        """!
        Check if the tensor is buffer.

        @return is_buffer boolean: True if is buffer.
        """
        return self._is_attr and not self.requires_grad

    def is_grad(self) -> bool:
        """!
        Check if the tensor is gradient

        @return is_grad boolean: True if is gradient
        """
        return self._is_grad

    def as_param(self):
        """
        Set the tensor as trainable parameter
        """
        assert self._grad is not None, "missing grad tensor"
        self._requires_grad = True
        self._is_attr = True
        self._is_grad = False
        return self

    def as_buffer(self):
        """
        Set the tensor as un-trainable buffer
        """
        self._requires_grad = False
        self._is_attr = True
        self._is_grad = False
        return self

    def as_grad(self):
        """
        Set the tensor as gradient
        """
        self._is_param = False
        self._is_attr = False
        self._is_grad = True
        return self

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad

    def __copy__(self):
        """
        Copy the tensor that will have the exactly same id
        except the empty attached cell

        Returns:
            tensor
        """
        tensor = IRTensor(self._shape, self.name, tid=self._id)
        for key in self.__dict__:
            setattr(tensor, key, getattr(self, key))
        # clear attached cells
        tensor.cell = None
        return tensor

    @property
    def shape(self) -> Tuple[int]:
        return list(self._shape)

    @shape.setter
    def shape(self, val: Tuple[int]):
        self._shape = tuple(val)
        if isinstance(self._grad, IRTensor):
            self._grad.shape = tuple(val)

    def nelement(self) -> int:
        """
        Get total number of element in the tensor.
        """
        if self.shape is None:
            raise RuntimeError("Tensor shape is not set")
        cnt = 1
        for num in self.shape:
            cnt *= num
        return cnt

    def byte_size(self) -> int:
        return self.nelement() * dtype2byte_size(self.dtype)

    def backward(self) -> None:
        """
        Autograd backward on the tensor

        The backward will apply on the program graph

        @return None
        """
        from cube.program import Program
        graph = Program().get_graph()
        return graph.backward(self)

    def __repr__(self):
        dscp = f'Tensor(id={self._id}, shape={self.shape}, device={self.device})'
        return dscp
