from cube.graph.unique import IDGenerator
from cube.graph.mapping import IR2LogicOp

from cube.tschedule.action import Action
from cube.tschedule.pool import TSchedulePool

from typing import Union, Tuple, List, Optional, Any
import copy


__all__ = ['IROperation', 'IRTensor', 'IRGraph']


class IROperation:
    """
    IROperation serves as IRGraph node
    """

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
        # node info
        self._id: int = IDGenerator().gen_op_id()
        self.name: str = name

        # op signature and op class
        self.signature: str = signature
        self.semantic = IR2LogicOp.map(self.signature)
        self.device = None
        
        # edge (dataflow info)
        self._inputs: List[IRTensor] = [None] * input_length
        self._predecessors: List[IROperation] = [None] * input_length
        # todo for outputs
        self._outputs: List[IRTensor] = [IRTensor() for _ in range(output_length)]
        for tensor in self._outputs:
            tensor.set_src_node(self)
        self._successors: List[List(IROperation)] = [list() for _ in range(output_length)]

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
            # set predecessor
            self._predecessors[input_index] = val.src()
            # set the source node successor
            if isinstance(val.src(), IROperation):
                val.src()._add_successor(val, self)

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
        dscp = f'Op(id={self._id}, signature={self.signature}, inputs={self._inputs}, outputs={self._outputs})'
        return dscp


class IRTensor:
    """
    IRTensor serves as IRGraph edge
    """
    def __init__(self, shape=None, name=None):

        self._id: int = IDGenerator().gen_tensor_id()
        self._shape: Optional(List[int]) = shape
        self.name = name
        self.device = -1

        # connected to IROperation
        self._src_nodes: IROperation = None # -> output of the node
        self._dst_nodes: List[IROperation] = list() # -> input of the nodes

    def __copy__(self):
        """
        Copy the tensor that will be same except a new id
        """
        tensor = IRTensor(self._shape, self.name)
        tensor.device = self.device
        tensor._id = IDGenerator().gen_tensor_id()
        return tensor

    def __deepcopy__(self):
        raise RuntimeError("DeepCopy is not allowed.")

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

    def src(self) -> Optional[IROperation]:
        return self._src_nodes

    def dst(self, index: Optional[int] = None):
        if index >= len(self._dst_nodes):
            raise RuntimeError("get tensor dst out of range")
        return self._dst_nodes[index]

    def set_src_node(self, node: IROperation):
        if not isinstance(node, IROperation):
            raise TypeError("IRTensor source node should be IROperation")
        self._src_nodes = node

    def add_dst_nodes(self, node: IROperation):
        if not isinstance(node, IROperation):
            raise TypeError("IRTensor destination node should be IROperation")
        self._dst_nodes.append(IROperation)

    def is_leaf(self):
        """
        Check if it is a leaf tensor (parameter)
        """
        return self.src() is None

    def backward(self, tensors = None):
        raise NotImplementedError

    def __repr__(self):
        dscp = f'Tensor(id={self._id}, shape={self.shape})'
        return dscp


class IRGraph:
    """
    PyTorch IR Graph

    The IR Graph only contains forward graph
    """

    def __init__(self, 
                 nodes: List[IROperation],
                 input_tensors: List[IRTensor], 
                 output_tensors: List[IRTensor], 
                 module_name: str):
        self.module_name = module_name
        self._nodes: List[IROperation] = nodes
        self._inputs = input_tensors
        self._outputs = output_tensors

    def add_node(self, node: IROperation):
        if not isinstance(node, IROperation):
            raise TypeError("Expected node to be IROperation")
        self._nodes.append(node)

    def nodes(self, index: Optional[int] = None):
        """
        Get node at position index
        """
        if isinstance(index, int):
            if index >= len(self._nodes):
                raise RuntimeError(
                    f"Get node out of range ({index} >= {len(self._nodes)})"
                )
            return self._nodes[index]
        elif index is None:
            return self._nodes
        else:
            raise TypeError("Expected index to be None or int")

    def inputs(self, index: Optional[int] = None):
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

    def replace(self, target: IROperation, nodes: List[IROperation]):
        """
        Replace the node with new nodes (IRGraph)
        """
        raise NotImplementedError

    def forward(self, *args, **kwargs) -> Union[IRTensor, Tuple[IRTensor]]:
        """
        forward will divide the graph into Actions according to
        node device assignment

        Currently each forward call will result in a new flow
        even if the input is same

        Returns:
            List[Action]
        """
        if len(self._outputs) == 1:
            return copy.copy(self._outputs[0])
        else:
            return tuple([copy.copy(output) for output in self._outputs])

    def __call__(self, *args, **kwargs):
        """
        Register forward action
        """
        curr_nodes: List[IROperation] = list()
        curr_device = None

        def _wrap_to_action():
            sub_graph = IRGraph(
                curr_nodes, self._inputs, self._outputs,
                module_name=self.module_name
            )
            action = Action(sub_graph, device=curr_device)
            action.tag('forward')
            return action

        for node in self.nodes():
            device = node.device
            if device is None:
                raise RuntimeError("All the node should be assigned to devices")
            if device != curr_device and curr_device is not None:
                # note we still use same input and output to make consistency
                action = _wrap_to_action()
                # register to schedule space
                TSchedulePool().add_action(action)
                curr_nodes = list()
            curr_device = device
            curr_nodes.append(node)
        if curr_device is not None:
            action = _wrap_to_action()
            TSchedulePool().add_action(action)

        return self.forward(*args, **kwargs)

    def __repr__(self):
        dscp = ''
        # inputs
        dscp += f'Inputs: {self._inputs}\n'
        # nodes
        for node in self._nodes:
            succ_node_ids = [None] * len(node.outputs())
            for out_idx, node_list in enumerate(node.successors()):
                node_list = [snode._id for snode in node_list]
                succ_node_ids[out_idx] = node_list
            dscp += f"\n{node._id}: {node} -> node id {succ_node_ids}\n"
        # outputs
        dscp += f'\nOutputs: {self._outputs}'
        return dscp
