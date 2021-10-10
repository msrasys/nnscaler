from typing import Union, List, Optional
import copy
from enum import Enum

from cube.graph.ir_comm import IRCommunication

from cube.graph.ir_cten import IRCell


class SUType(Enum):

    Forward = 'forward'
    Backward = 'backward'
    Adapter = 'adapter'


class ScheduleUnit(IRCell):
    """
    Action recv tensors must be inside of Action inputs,
    and can be mapped to Action.graph.inputs

    """

    # outputs = cube.runtime.temporal.forward(model, *args)
    _forward_signature = 'cube.runtime.temporal.forward'
    # grads = cube.runtime.temporal.backward(
    #   input_tensors, output_tensors, output_grads
    # )
    _backward_signature = 'cube.runtime.temporal.backward'
    # cube.runtime.collectives.sendrecv(send_tensors, send_ranks,
    #   recv_shapes, from_ranks
    # )
    _adapter_signature = 'cube.runtime.collectives.sendrecv'

    def __init__(self, sub_nodes, graph, devices: Union[List[int], int]):

        if all([isinstance(node, IRCommunication) for node in sub_nodes]):
            self.tag = 'adapter'
        else:
            self.tag = graph.tag

        self.global_graph = graph

        if self.tag == 'forward':
            signature = ScheduleUnit._forward_signature
        elif self.tag == 'backward':
            signature = ScheduleUnit._backward_signature
        elif self.tag == 'adapter':
            signature = ScheduleUnit._adapter_signature
        else:
            raise RuntimeError(f"Unsupported graph tag: {self.tag}")

        subgraph = graph.subgraph(sub_nodes)
        self._nodes = sub_nodes

        super().__init__(
            name          = self.tag,
            signature     = signature,
            input_length  = len(subgraph.inputs()),
            output_length = len(subgraph.outputs())
        )

        for idx, input in enumerate(subgraph.inputs()):
            self.set_input(idx, input)
        for idx, output in enumerate(subgraph.outputs()):
            self.set_output(idx, output)

        # set su device
        self.device = devices

        # additional control dependency for add_flow
        self._ctrl_predecessors = list()
        self._ctrl_successors = list()

        self.mirror = None

    def set_mirror(self, su):
        """
        Create a mirrored ScheduleUnit: the 
        inputs and outputs are reversed
        """
        if not isinstance(su, ScheduleUnit):
            raise TypeError("Expected mirror to be ScheduleUnit")
        self.mirror = su

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
            return copy.copy(self._nodes)
        else:
            raise TypeError("Expected index to be None or int")

    def add_predecessor(self, input_index: int, su):
        """
        Add a predecessor cell in the input_index slot. 
        self.input[input_index] = node.output[out_index]
        """
        if input_index == -1:
            self._predecessors.append(su)
        else:
            super().add_predecessor(input_index, su)

    def predecessors(self, index: Optional[int] = None) -> List:
        """
        Get 1-hop predecessor cells including control predecessors

        Args:
            index (Optional[int]):
                -1: return control predecessors
                None: return all predecessors including index
                >0 : return input SUs at input index

        Returns:
            cell(s): List[IRCell]
        """
        if isinstance(index, int):
            if index == -1:
                return copy.copy(self._ctrl_predecessors)
            if index >= len(self._inputs):
                raise RuntimeError(
                    f"Get the input out of range ({index} >= {len(self._inputs)}"
                )
            return copy.copy(self._predecessors[index])
        elif index is None:
            predecessors = list()
            for pre_cells in self._predecessors:
                predecessors += pre_cells
            predecessors += self._ctrl_predecessors
            return predecessors
        else:
            raise TypeError("Expected index to be None or int")

    def add_successor(self, output_index: int, su):
        """
        Set self node the output index node. 
        `node` will take the self.outputs(index) as the input
        """
        if output_index == -1:
            self._successors.append(su)
        else:
            super().add_successor(output_index, su)

    def successors(self, index: Optional[int] = None) -> List:
        """
        Get 1-hop successor cells including control successors

        Args:
            index (Optional[int]):
                -1: return control successors
                None: return all successors including index
                >0 : return output SUs at output index

        Returns:
            cells: List[ScheduleUnit]
        """
        if isinstance(index, int):
            if index == -1:
                return copy.copy*self._ctrl_successors
            if index >= len(self._outputs):
                raise RuntimeError(
                    f"Get the output out of range ({index} >= {len(self._outputs)}"
                )
            return copy.copy(self._successors[index])
        elif index is None:
            successors = list()
            for post_cells in self._successors:
                successors += post_cells
            successors += self._ctrl_successors
            return successors
        else:
            raise TypeError("Expected index to be None or int")

    def __repr__(self):
        su_inputs = [f't{tensor._id}-dev{tensor.device}' for tensor in self.inputs()]
        su_outputs = [f't{tensor._id}-dev{tensor.device}' for tensor in self.outputs()]
        dscp = f'SU({self.name}, nodes={len(self.nodes())})-dev{self.device}: {su_inputs} -> {su_outputs}'
        return dscp


def forward_convert(graph) -> List[ScheduleUnit]:
    sus = list()
    for node in graph.nodes():
        devices = node.device
        for device in devices:
            su = ScheduleUnit([node], graph, device)
            sus.append(su)
    return sus
