"""
IROperation:

    Semantic operation representation (node) in IRGraph.
    An operation is of Computation (Comp) or Communication (Comm) type.

    A Comp type operation can be assigned to multiple devices for redundant computation.
    A Comm type operation can be assigned to multiple devices (List[int]).

    Each IROperation can have (multiple) input args and (multiple) output args.

IRTensor:

    Semantic tensor representation (edge) in IRGraph.

    IRTensor can be assigned (deploy) to multiple devices (List[int])

    The IRTensor is a logical tensor that
    
    1). can be generated from multipe operations (i.e., different operators
    can generate different part of the IRTensor).
    
    => multiple source IROperation.

    2). can be used as input for multiple operations.
    
    => multiple destination IROperation


IROperation can accept tensors that are placed on the different devices.

Set the operation device will in default change the output tensor placement
and input leaf tensor placement to match with the operation.
"""


from cube.graph.unique import IDGenerator
from cube.graph.mapping import IR2LogicOp

from enum import Enum
from typing import List, Optional, Any, Union
import copy


__all__ = ['OperationType', 'IROperation', 'IRTensor']


class OperationType(Enum):

    Comp = 1  # computation
    Comm = 2  # communication


class IROperation:
    """
    IROperation serves as IRGraph node
    """

    def __init__(self,
                 name: str, 
                 signature: str,
                 input_length: int,
                 output_length: int,
                 type=OperationType.Comp):
        """
        Create a node with name (variable name) and module type (module_name)

        Args:
            name (str): the op semantic name
            signature (str): the op signature, e.g., torch.functional.nn.linear
            input_length (int): the number of inputs for the op
            output_length (int): the number of outputs for the op
        """
        # node info
        self._id: int = IDGenerator().gen_op_id()
        self.name: str = name

        # op signature and op class
        self.signature: str = signature
        self.semantic = IR2LogicOp.map(self.signature)
        self._type = type
        self._device = list()

        # source operations
        self._inputs: List[IRTensor] = [None] * input_length
        self._predecessors: List[List[IROperation]] = [list() for _ in range(input_length)]
        
        # destination operations
        self._outputs: List[IRTensor] = [IRTensor() for _ in range(output_length)]
        for tensor in self._outputs:
            tensor.add_src_node(self)
        self._successors: List[List[IROperation]] = [list() for _ in range(output_length)]

    @property
    def type(self) -> OperationType:
        return self._type

    @type.setter
    def type(self, _):
        raise RuntimeError("Not allowed to set type except initialization")

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device_id: Union[int, List[int]]):
        """
        Set the operation device.

        For computation operators, they are only allowed
        to happen on one device (int)

        For communication operators (e.g., move, all-reduce),
        they are allowed to happend on multiple devices
        """
        if isinstance(device_id, int):
            device_id = [device_id]
        if not all([isinstance(devid, int) for devid in device_id]):
            raise KeyError("Require device Union[int, List[int]]")
        self._device = device_id
        for input in self._inputs:
            # in default, parameters will be placed on all devices
            # that needs it
            if isinstance(input, IRTensor) and input.is_leaf():
                devices = set()
                for node in input.dst():
                    devices.update(node.device)
                input.device = list(devices)
        for output in self._outputs:
            if isinstance(output, IRTensor):
                output.device = device_id

    def on_device(self, device_id: int):
        """
        Check whether the operation is on device_id

        Returns:
            Boolean
        """
        if not isinstance(device_id, int):
            raise TypeError("Expected device id to be int")
        return device_id in self.device

    def inputs(self, index: Optional[int] = None):
        """
        Get input tensor at input index

        Args:
            index (int or None): 
                index of the inputs, None will return the nodes
                for all the inputs
        """
        if isinstance(index, int):
            if index >= len(self._inputs):
                raise RuntimeError(
                    f"Get the input out of range ({index} >= {len(self._inputs)}"
                )
            return self._inputs[index]
        elif index is None:
            return self._inputs
        else:
            raise TypeError("Expected index to be None or int")

    def predecessors(self, index: Optional[int] = None):
        """
        Get input operator at input index
        """
        if isinstance(index, int):
            if index >= len(self._inputs):
                raise RuntimeError(
                    f"Get the input out of range ({index} >= {len(self._inputs)}"
                )
            return self._predecessors[index]
        elif index is None:
            return self._predecessors
        else:
            raise TypeError("Expected index to be None or int")

    def outputs(self, index: Optional[int] = None):
        """
        Get output tensor at output index

        Args:
            index (int or None): 
                index of the outputs, None will return the nodes
                for all the outputs
        """
        if isinstance(index, int):
            if index >= len(self._outputs):
                raise RuntimeError(
                    f"Get the output out of range ({index} >= {len(self._outputs)}"
                )
            return self._outputs[index]
        elif index is None:
            return self._outputs
        else:
            raise TypeError("Expected index to be None or int")

    def successors(self, index: Optional[int] = None):
        """
        Get output operator at output index

        Args:
            index (int or None): 
                index of the outputs, None will return the nodes
                for all the outputs
        """
        if isinstance(index, int):
            if index >= len(self._outputs):
                raise RuntimeError(
                    f"Get the output out of range ({index} >= {len(self._outputs)}"
                )
            return self._successors[index]
        elif index is None:
            return self._successors
        else:
            raise TypeError("Expected index to be None or int")

    def set_input(self, input_index: int, val: Any):
        """
        Set the node inputs[input_index] with the tensor

        val: IRTensor or any deterministic value (int, bool, str, etc)
        """
        if input_index >= len(self.inputs()):
            raise RuntimeError(
                f"Set the input out of range ({input_index} >= {len(self._inputs)})"
            )
        # set tensor
        self._inputs[input_index] = val
        if isinstance(val, IRTensor):
            # set tensor dst
            val.add_dst_node(self)
            # set predecessor
            self._predecessors[input_index] = val.src()
            # set the source node successor
            for node in val.src():
                if isinstance(node, IROperation):
                    node._add_successor(val, self)

    def set_output(self, output_index: int, val: Any):
        """
        Set the node inputs[output_index] with the tensor

        val: IRTensor or any deterministic value (int, bool, str, etc)
        """
        if output_index >= len(self.outputs()):
            raise RuntimeError(
                f"Set the input out of range ({output_index} >= {len(self._inputs)})"
            )
        # set tensor
        self._outputs[output_index] = val
        if isinstance(val, IRTensor):
            # set predecessor
            for node in val.src():
                if isinstance(node, IROperation):
                    self._successors[output_index].append(node)
            # set the source node
            if self not in val.src():
                val.add_src_node(self)

    def set_predecessor(self, input_index: int, node, out_index: int):
        """
        Set self node the input node. self.input[input_index] = node.output[out_index]
        """
        if not isinstance(node, IROperation):
            raise TypeError("Expected node to be IROperation")
        if input_index >= len(self.inputs()):
            raise RuntimeError(
                f"Set the input out of range ({input_index} >= {len(self._inputs)})"
            )
        self._inputs[input_index] = node.outputs(out_index)
        self._predecessors[input_index] = node
        node.set_successor(out_index, self)

    def _add_successor(self, tensor, node):
        """
        Set self node the output index node. 
        `node` will take the self.outputs(index) as the input
        """
        out_index = self._outputs.index(tensor)
        if out_index < 0:
            raise RuntimeError("Fail to find output tensor")
        self._successors[out_index].append(node)

    def infer_shape(self):
        """
        Infer output value shape
        """
        shapes = list()
        for input in self.inputs():
            if isinstance(input, IRTensor):
                if input.shape is None:
                    return False
                shapes.append(input.shape)
            else:
                shapes.append([1,])
        shapes = tuple(shapes)
        out_shapes = self.semantic.shape_infer(*shapes)
        if len(out_shapes) != len(self._outputs):
            raise RuntimeError(
                "The logical op semantic doesn't match with parsed op"
            )
        for shape, val in zip(out_shapes, self._outputs):
            if isinstance(val, IRTensor):
                val.shape = shape
        return True

    def __repr__(self):
        inputs = list()
        for tensor in self.inputs():
            if isinstance(tensor, IRTensor):
                inputs.append(f't{tensor._id}')
            else:
                inputs.append(tensor)
        
        outputs = list()
        for tensor in self.outputs():
            if isinstance(tensor, IRTensor):
                outputs.append(f't{tensor._id}')
            else:
                outputs.append(tensor)

        dscp = f'Op(id={self._id}, signature={self.signature}, device={self.device}, inputs={inputs}, outputs={outputs})'
        return dscp


class IRTensor:
    """
    IRTensor serves as IRGraph edge
    """
    def __init__(self, shape=None, name=None):

        self._id: int = IDGenerator().gen_tensor_id()
        self._shape: Optional(List[int]) = shape
        self.name = name
        self._device = list()

        # connected to IROperation
        self._src_nodes: List[IROperation] = list() # -> output of the node
        self._dst_nodes: List[IROperation] = list() # -> input of the nodes

        # forward graph
        self.requires_grad = True
        self.forward_graph = None

    def set_forward_graph(self, graph):
        """
        Set forward graph (IRGraph)
        """
        self.forward_graph = graph

    def __copy__(self):
        """
        Copy the tensor that will be same except a new id
        """
        tensor = IRTensor(self._shape, self.name)
        new_id = tensor._id
        for key in self.__dict__:
            setattr(tensor, key, getattr(self, key))
        tensor._id = new_id
        return tensor

    def __deepcopy__(self, memo):
        """
        Deep Copy will copy the exactly same tensor with same tensor id
        """
        tensor = IRTensor(self._shape, self.name)
        for key in self.__dict__:
            val = getattr(self, key)
            if isinstance(val, IRTensor):
                pass
            if isinstance(val, list) and all([isinstance(v, IRTensor) for v in val]):
                pass
            else:
                val = copy.copy(val)
            setattr(tensor, key, val)
        return tensor

    def __eq__(self, tensor):
        if not isinstance(tensor, IRTensor):
            return False
        return self._id == tensor._id

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        if self._shape is not None and self._shape != val:
            raise RuntimeError("Try to change shape")
        if not isinstance(val, list) or \
           not all([isinstance(size, int) for size in val]):
            raise RuntimeError("Expected shape to be list[int]")
        self._shape = val

    @property
    def device(self) -> List[int]:
        return self._device

    @device.setter
    def device(self, device_id: List[int]):
        """
        Set placement of the tensor

        A tensor can be placed on multiple devices as input
        for multiple operations on different devices
        """
        if isinstance(device_id, int):
            device_id = [device_id]
        if not all([isinstance(devid, int) for devid in device_id]) :
            raise TypeError(f"Expected device id to be int or List[int]")
        self._device = device_id

    def src(self) -> List[IROperation]:
        return self._src_nodes

    def dst(self, index: Optional[int] = None):
        if index is None:
            return self._dst_nodes
        elif index >= len(self._dst_nodes):
            raise RuntimeError("get tensor dst out of range")
        return self._dst_nodes[index]

    def add_src_node(self, node: IROperation):
        if not isinstance(node, IROperation):
            raise TypeError("IRTensor source node should be IROperation")
        self._src_nodes.append(node)

    def add_dst_node(self, node: IROperation):
        if not isinstance(node, IROperation):
            raise TypeError("IRTensor destination node should be IROperation")
        self._dst_nodes.append(node)

    def is_leaf(self):
        """
        Check if it is a leaf tensor (parameter)
        """
        return len(self.src()) == 0

    def backward(self):
        """
        Backward will generate a backward action scheduling pool

        Construct a reverse graph of forward and seperate to actions
        """
        if self.forward_graph is None:
            raise RuntimeError("Backward on a tensor without forward graph")
        self.forward_graph.backward()


    def __repr__(self):
        dscp = f'Tensor(id={self._id}, shape={self.shape}, device={self.device})'
        return dscp

