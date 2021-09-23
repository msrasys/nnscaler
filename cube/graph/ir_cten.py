from typing import List, Union, Optional, Any
import copy

from cube.graph.unique import IDGenerator


__all__ = ['IRCell', 'IRTensor']


class IRCell:
    """
    IRCell serves as a general node for different purpose
    """

    def __init__(self,
                 name: str,
                 signature: str,
                 input_length: int,
                 output_length: int):
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

        # device
        self._device = list()

        # source tensors
        self._inputs: List[Any] = [None] * input_length
        # source cells
        self._predecessors: List[List[IRCell]] = [list() for _ in range(input_length)]
        
        # destination tensors
        self._outputs: List[IRTensor] = [IRTensor() for _ in range(output_length)]
        for output in self._outputs:
            output.add_src_node(self)
        # destination cells
        self._successors: List[List[IRCell]] = [list() for _ in range(output_length)]

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device_id: Union[int, List[int]]):
        """
        Set the operation device.
        """
        if isinstance(device_id, int):
            device_id = [device_id]
        if not all([isinstance(devid, int) for devid in device_id]):
            raise KeyError("Require device Union[int, List[int]]")
        self._device = device_id

    def on_device(self, device_id: int):
        """
        Check whether the operation is on device_id

        Returns:
            Boolean
        """
        if not isinstance(device_id, int):
            raise TypeError(f"Expected device id to be int but got {type(device_id)}")
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
            return copy.copy(self._inputs)
        else:
            raise TypeError("Expected index to be None or int")

    def predecessors(self, index: Optional[int] = None) -> List:
        """
        Get input operator at input index
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
            return copy.copy(self._outputs)
        else:
            raise TypeError("Expected index to be None or int")

    def successors(self, index: Optional[int] = None) -> List:
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
            return copy.copy(self._successors[index])
        elif index is None:
            successors = list()
            for post_cells in self._successors:
                successors += post_cells
            return post_cells
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
                if isinstance(node, IRCell):
                    node.add_successor(val, self)

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
                if isinstance(node, IRCell):
                    self._successors[output_index].append(node)
            # set the source node
            if self not in val.src():
                val.add_src_node(self)

    def add_predecessor(self, input_index: int, node, out_index: int):
        """
        Set self node the input node. self.input[input_index] = node.output[out_index]
        """
        if not isinstance(node, IRCell):
            raise TypeError("Expected node to be IRCell")
        if input_index >= len(self.inputs()):
            raise RuntimeError(
                f"Set the input out of range ({input_index} >= {len(self._inputs)})"
            )
        self._inputs[input_index] = node.outputs(out_index)
        self._predecessors[input_index].append(node)
        node.add_successor(out_index, self)

    def add_successor(self, tensor, node):
        """
        Set self node the output index node. 
        `node` will take the self.outputs(index) as the input
        """
        if not isinstance(node, IRCell):
            raise TypeError("Expected node to be IRCell")
        out_index = self._outputs.index(tensor)
        if out_index < 0:
            raise RuntimeError("Fail to find output tensor")
        self._successors[out_index].append(node)

    def __repr__(self):
        """
        Cell string presentation
        """
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
        dcsp = f'Cell-{self._id}({self.signature}, device={self.device})'\
               f'({inputs}) -> {outputs}'
        return dcsp


class IRTensor:
    """
    IRTensor serves as IRGraph edge
    """
    def __init__(self, shape=None, name=None):

        self._id: int = IDGenerator().gen_tensor_id()
        self._shape: Optional(List[int]) = shape
        self.name = name
        self._device = list()

        # connected to IRCell
        self._src_nodes: List[IRCell] = list() # -> output of the node
        self._dst_nodes: List[IRCell] = list() # -> input of the nodes

        # forward graph
        self.requires_grad = True
        self.gen_graph = None

    def set_gen_graph(self, graph):
        """
        Set forward graph (IRGraph)
        """
        self.gen_graph = graph

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

    def src(self) -> List[IRCell]:
        return self._src_nodes

    def dst(self, index: Optional[int] = None):
        if index is None:
            return self._dst_nodes
        elif index >= len(self._dst_nodes):
            raise RuntimeError("get tensor dst out of range")
        return self._dst_nodes[index]

    def add_src_node(self, node: IRCell):
        if not isinstance(node, IRCell):
            raise TypeError("IRTensor source node should be IRCell")
        self._src_nodes.append(node)

    def add_dst_node(self, node: IRCell):
        if not isinstance(node, IRCell):
            raise TypeError("IRTensor destination node should be IRCell")
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
        if self.gen_graph is None:
            raise RuntimeError("Backward on a tensor without forward graph")
        self.gen_graph.backward(self)


    def __repr__(self):
        dscp = f'Tensor(id={self._id}, shape={self.shape}, device={self.device})'
        return dscp
