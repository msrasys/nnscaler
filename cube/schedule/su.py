from typing import List, Optional, Tuple
import copy
from enum import Enum

from cube.ir.cten import IRCell


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

    def __init__(self, nodes: List[IRCell], stype: SUType, name='su'):

        if not all([isinstance(node, IRCell) for node in nodes]):
            raise ValueError("Expected each nodes to be List[IRCell]")
        if not isinstance(stype, SUType):
            raise TypeError("Expected stype be SUType")

        # get inputs and outputs
        inputs = IRCell.get_inputs(nodes)
        inputs = [input for input in inputs if not input.is_param()]
        outputs = IRCell.get_outputs(nodes)
        super().__init__(
            name          = name,
            signature     = stype.value,
            input_length  = len(inputs),
            output_length = len(outputs)
        )

        self.stype = stype

        self._nodes = nodes
        for idx, input in enumerate(inputs):
            self.set_input(idx, input)
        for idx, output in enumerate(outputs):
            self.set_output(idx, output)

        # each input is associated with
        # send adapters and recv adapters (send + recv)
        self._send_in_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(inputs))
        ]
        self._recv_in_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(inputs))
        ]

        # each output is associated with
        # send adapters and recv adapters (send + recv)
        self._send_out_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(outputs))
        ]
        self._recv_out_adapters: List[List[ScheduleUnit]] = [
            list() for _ in range(len(outputs))
        ]

        # additional control dependency for add_flow
        self._ctrl_predecessors = list()
        self._ctrl_successors = list()

        self.mirror = None

    def __copy__(self):
        """
        Copy the SU. Note the mirror su is also copied
        """
        su = ScheduleUnit(self._nodes, self.stype, self.name)
        #TODO: adapter copy
        if self.mirror is not None:
            mirror_su = self.mirror
            mirror_su = ScheduleUnit(
                mirror_su._nodes, mirror_su.stype, mirror_su.name
            )
            su.set_mirror(mirror_su)
            mirror_su.set_mirror(su)
        return su

    def set_mirror(self, su):
        """
        Create a mirrored ScheduleUnit: the 
        inputs and outputs are reversed
        """
        if not isinstance(su, ScheduleUnit):
            raise TypeError("Expected mirror to be ScheduleUnit")
        self.mirror = su

    def in_adapters(self, index: Optional[int] = None) -> List:
        """
        Get adapter for the input tensor at index

        Returns:
            Tuple[List[ScheduleUnit], List[ScheduleUnit]]:
                the send_adapters and recv_adapters
        """
        if isinstance(index, int):
            if index >= len(self._inputs):
                raise RuntimeError(
                    f"Get index out of range ({index} >= {len(self._inputs)})"
                )
            send_adapters = copy.copy(self._send_in_adapters[index])
            recv_adapters = copy.copy(self._recv_in_adapters[index])
            return send_adapters, recv_adapters
        elif index is None:
            all_send_adapters = list()
            all_recv_adapters = list()
            for adapters in self._send_in_adapters:
                all_send_adapters += adapters
            for adapters in self._recv_in_adapters:
                all_recv_adapters += adapters
            return all_send_adapters, all_recv_adapters
        else:
            raise TypeError("Expected index to be None or int")

    def out_adapters(self, index: Optional[int] = None) -> Tuple[List, List]:
        """
        Get adapter for the output tensor at index

        Returns:
            Tuple[List[ScheduleUnit], List[ScheduleUnit]]:
                the send_adapters and recv_adapters
        """
        if isinstance(index, int):
            if index >= len(self._outputs):
                raise RuntimeError(
                    f"Get index out of range ({index} >= {len(self._outputs)})"
                )
            send_adapters = copy.copy(self._send_out_adapters[index])
            recv_adapters = copy.copy(self._recv_out_adapters[index])
            return send_adapters, recv_adapters
        elif index is None:
            all_send_adapters = list()
            all_recv_adapters = list()
            for adapters in self._send_out_adapters:
                all_send_adapters += adapters
            for adapters in self._recv_out_adapters:
                all_recv_adapters += adapters
            return all_send_adapters, all_recv_adapters
        else:
            raise TypeError("Expected index to be None or int")

    def _add_in_adapter(self, index: int, send_adapter, recv_adapter):
        """
        Add adapters to the input tensor of this SU

        Args:
            index (int): the input index
            send_adapter (ScheduleUnit)
            recv_adapter (ScheduleUnit)
        """
        if index >= len(self._inputs):
            raise ValueError(f"index {index} out of range {len(self._inputs)}")
        if not isinstance(send_adapter, ScheduleUnit):
            raise TypeError("Expected send adapter to be ScheduleUnit")
        if not isinstance(recv_adapter, ScheduleUnit):
            raise TypeError("Expected recv adapter to be ScheduleUnit")
        self._send_in_adapters[index].append(send_adapter)
        self._recv_in_adapters[index].append(recv_adapter)

    def _add_out_adapter(self, index: int, send_adapter, recv_adapter):
        """
        Add adapters to the output tensor of this SU

        Args:
            index (int): the output index
            send_adapter (ScheduleUnit)
            recv_adapter (ScheduleUnit)
        """
        if index >= len(self._outputs):
            raise ValueError(f"index {index} out of range {len(self._outputs)}")
        if not isinstance(send_adapter, ScheduleUnit):
            raise TypeError("Expected send adapter to be ScheduleUnit")
        if not isinstance(recv_adapter, ScheduleUnit):
            raise TypeError("Expected recv adapter to be ScheduleUnit")
        self._send_out_adapters[index].append(send_adapter)
        self._recv_out_adapters[index].append(recv_adapter)

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
            self._ctrl_predecessors.append(su)
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
            self._ctrl_successors.append(su)
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
