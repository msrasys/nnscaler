"""
Convert PyTorch nn.Module to our IRGraph
"""
from typing import List, Optional


__all__ = ['IROperation', 'IRTensor', 'IRGraph']


class IROperation:
    """
    IROperation serves as IRGraph node
    """

    def __init__(self, node_id: int, 
                 op_name: str,
                 input_length: int, output_length: int,
                 label: str,):
        """
        Create a node with name (variable name) and module type (module_name)

        Args:
            node_id (int): the int 
            label (str): the var name of the module
            type: the type name of the module

        Example:
            init code:
                self.linear1 = torch.nn.Linear(input_feats, output_feats)
            forward code:
                output = self.linear1(input)
            =>
                label = linear1;
                op = torch._C._nn.linear
        """
        # node info
        self._id: int = node_id
        self.label: str = label

        # node type
        self.op: str = op_name
        
        # edge (dataflow info)
        self._inputs: List[IRTensor] = [None] * input_length
        self._predecessors: List[IROperation] = [None] * input_length
        # todo for outputs
        self._outputs: List[IRTensor] = [None] * output_length
        self._successors: List[IROperation] = [None] * output_length

    def inputs(self, index: Optional(None, int) = None):
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

    def predecessors(self, index: Optional(None, int) = None):
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

    def outputs(self, index: Optional(None, int) = None):
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

    def successors(self, index: Optional(None, int) = None):
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

    def set_predecessor(self, input_index: int, node: IROperation, out_index: int):
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

    def set_successor(self, out_index: int, node: IROperation):
        """
        Set self node the output index node. 
        `node` will take the self.outputs(index) as the input
        """
        if out_index >= len(self._outputs):
            raise RuntimeError(
                f"Set output index out of range ({out_index} >= {len(self._outputs)}"
            )
        self._successors[out_index] = node


class IRTensor:
    """
    IRTensor serves as IRGraph edge
    """
    def __init__(self, edge_id: int, shape: List[int], label: str):
        self._id = edge_id
        self.shape = shape
        self.label = label


class IRGraph:
    """
    PyTorch IR Graph

    The IR Graph only contains forward graph
    """

    def __init__(self, module_name: str):
        self.module_name = module_name
        self._nodes: List[IROperation] = list()

    def add_node(self, node: IROperation):
        if not isinstance(node, IROperation):
            raise TypeError("Expected node to be IROperation")
        self._nodes.append(node)

    def nodes(self, index: Optional(None, int)):
        """
        Get node at position index
        """
        if index >= len(self._nodes):
            raise RuntimeError(
                f"Get node out of range ({index} >= {len(self._nodes)})"
            )
        return self._nodes[index]

    def replace(self, target: IROperation, nodes: List[IROperation]):
        """
        Replace the node with new nodes (IRGraph)
        """
        raise NotImplementedError
