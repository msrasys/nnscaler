from cube.graph.ir_opten import IROperation, IRTensor
from cube.tschedule.action import Action
from cube.tschedule.pool import TSchedulePool

from typing import Union, Tuple, List, Optional
import copy


__all__ = ['IRGraph', 'IRLocalGraph']


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


class IRLocalGraph(IRGraph):

    def __init__(self, graph: IRGraph, device: int):

        if not isinstance(graph, IRGraph):
            raise TypeError(f"Expected graph: IRGraph but go {type(graph)}")
        if not isinstance(device, int):
            raise TypeError(f"Expected device: int but not {type(device)}")
        
        self.global_graph = graph
        self.device = device
        self.send_tensors = list()
        self.recv_tensors = list()
        # get nodes belong to this graph
        nodes = list()
        all_tensors = set()
        for node in self.global_graph.nodes():
            # collect on device node, inputs and outputs
            if node.on_device(self.device):
                nodes.append(node)
                # collect send tensors and recv tensors
                if node.semantic == 'move':
                    if device in node.inputs(0).device:
                        self.send_tensors.append(node.inputs(0))
                    if device in node.outputs(0).device:
                        self.recv_tensors.append(node.outputs(0))
                all_tensors.update(node.inputs())
                all_tensors.update(node.outputs())

        # model inputs and outputs
        model_inputs = list()
        model_outputs = list()
        for input in self.global_graph.inputs():
            if input in all_tensors:
                model_inputs.append(input)
        for output in self.global_graph.outputs():
            if output in all_tensors:
                model_outputs.append(output)

        super().__init__(
            nodes,
            model_inputs + self.recv_tensors,  # input tensors
            model_outputs + self.send_tensors,  # output tensors
            self.global_graph.module_name + f'Rank{self.device}'
        )
