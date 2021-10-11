from typing import Union, List, Optional
import copy
from enum import Enum
from cube.graph.ir_comm import IRCommunication

from cube.graph.ir_cten import IRCell


class SUType(Enum):

    # outputs = cube.runtime.temporal.forward(model, *args)
    Forward = 'cube.runtime.temporal.forward'

    # grads = cube.runtime.temporal.backward(
    #   input_tensors, output_tensors, output_grads
    # )
    Backward = 'cube.runtime.temporal.backward'

    # cube.runtime.collectives.sendrecv(send_tensors, send_ranks,
    #   recv_shapes, from_ranks
    # )
    Adapter = 'cube.runtime.collectives.sendrecv'

    Dataloader = 'next(dataloader)'


class ScheduleUnit(IRCell):
    """
    Action recv tensors must be inside of Action inputs,
    and can be mapped to Action.graph.inputs

    """

    def __init__(self, sub_nodes, graph, devices: Union[List[int], int], stype: SUType):

        if not isinstance(stype, SUType):
            raise TypeError("Expected stype be SUType")
        
        self.stype = stype
        self.global_graph = graph

        subgraph = graph.subgraph(sub_nodes)
        inputs = subgraph.inputs()
        outputs = subgraph.outputs()

        super().__init__(
            name          = graph.name,
            signature     = stype.value,
            input_length  = len(inputs),
            output_length = len(outputs)
        )

        self._nodes = sub_nodes
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        for idx, output in enumerate(outputs):
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
        dscp = f'SU({self.stype}, nodes={len(self.nodes())})-dev{self.device}: {su_inputs} -> {su_outputs}'
        return dscp


def logic_translator(graph, su_type: SUType) -> List[ScheduleUnit]:
    if not isinstance(su_type, SUType):
        raise TypeError("Expected SU Type")
    sus = list()
    for node in graph.nodes():
        stype = su_type
        if isinstance(node, IRCommunication):
            stype = SUType.Adapter
        devices = node.device
        for device in devices:
            su = ScheduleUnit([node], graph, device, stype)
            sus.append(su)
    return sus
